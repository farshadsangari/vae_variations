import torch
from torch import nn
from .model_utils import Encoder, Decoder
from utils import kl_normal, sample_gaussian, log_bernoulli_with_logits


class VAE(nn.Module):
    def __init__(self, z_dim, beta, conditional, device):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        self.device = device
        self.enc = Encoder(self.z_dim, conditional, device)
        self.dec = Decoder(self.z_dim, conditional, device)
        self.y_dim = 10 if conditional else 0

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y=None):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs
        Args:
            x: tensor: (batch, dim): Observations
        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        sample_size = x.size()[0]
        # Pass data through Encoder to get Gaussian parameters
        q_mean, q_var = self.enc.encode(x, y)
        # First term - Closed form of KL, because both p and q distributions come from Normal distribution
        p_mean = torch.zeros([sample_size, self.z_dim], dtype=torch.float).to(
            self.device
        )
        p_var = torch.ones([sample_size, self.z_dim], dtype=torch.float).to(self.device)
        kl = torch.mean(kl_normal(q_mean, q_var, p_mean, p_var))

        # Second term Reconstruction loss - using Monte-carlo
        latent_sample = sample_gaussian(q_mean, q_var)
        probs = self.dec.decode(latent_sample, y)
        log_probs = log_bernoulli_with_logits(x, probs)
        rec = -1 * torch.mean(log_probs)
        # Negetive ELBO
        nelbo = self.beta * kl + rec
        return nelbo, kl, rec

    def loss(self, x, y=None):
        nelbo, kl, rec = self.negative_elbo_bound(x, y)
        loss = nelbo
        summaries = dict(
            (
                ("train/loss", nelbo),
                ("gen/elbo", -nelbo),
                ("gen/kl_z", kl),
                ("gen/rec", rec),
            )
        )
        return loss, summaries

    def sample_sigmoid(self, batch, y=None):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z, y)

    def compute_sigmoid_given(self, z, y=None):
        logits = self.dec.decode(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim),
        )

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z, y=None):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
