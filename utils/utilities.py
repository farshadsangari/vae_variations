"""
Utilities of Project
"""

import argparse
import torch
import os
import torch.nn.utils as utils
from yaml import parse
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def get_args():
    """
    The argument that we have defined, will be used in training modes
    """
    parser = argparse.ArgumentParser(
        description="Arguemnt Parser of `Train`of VAE and it's variation networks"
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        choices=["vae", "bvae", "cvae"],
        default="vae",
        type=str,
        help="Model name",
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        dest="batch_size",
        default=1000,
        type=int,
        help="Batch size",
    )

    parser.add_argument(
        "-b", "--beta", dest="beta", default=1, type=float, help="Beta coefficient"
    )

    parser.add_argument(
        "--latent-dimension",
        dest="latent_dimension",
        default=10,
        type=int,
        help="Latent dimension",
    )

    parser.add_argument(
        "-e",
        "--num-epochs",
        dest="num_epochs",
        default=200,
        type=int,
        help="Number of epochs",
    )

    parser.add_argument(
        "-l",
        "--learning-rate",
        dest="learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate",
    )

    parser.add_argument(
        "--load-saved-model",
        dest="load_saved_model",
        action="store_true",
        help="Whether load model or not",
    )

    parser.add_argument(
        "-c",
        "--condtional",
        dest="conditional",
        action="store_true",
        help="Weather use conditional-VAE or not",
    )

    parser.add_argument(
        "--ckpt-save-freq",
        dest="ckpt_save_freq",
        default=50,
        type=int,
        help="Checkpoint saving frequency",
    )

    parser.add_argument(
        "--ckpt-save-path",
        default="./ckpts",
        dest="ckpt_save_path",
        type=str,
        help="Path to save checkpoint",
    )
    parser.add_argument(
        "--ckpt-path",
        dest="ckpt_path",
        default="./ckpts",
        type=str,
        help="Checkpoint path for load model",
    )

    parser.add_argument(
        "--report-root",
        dest="report_root",
        default="./reports",
        type=str,
        help="Path for save report",
    )

    options = parser.parse_args()

    return options


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group["params"], max_norm, norm_type)


def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """

    normal_noise = torch.randn_like(m)
    z = m + normal_noise * torch.sqrt(v)
    return z


bce = torch.nn.BCEWithLogitsLoss(reduction="none")


def log_bernoulli_with_logits(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits
    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits
    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """

    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    Return:
        kl: tensor: (batch,): kl between each sample
    """

    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


def index2onehot(label, y_dim=10):
    assert torch.max(label).item() < y_dim
    if label.dim() == 1:
        label = label.unsqueeze(1)
    onehot = torch.zeros(label.size(0), y_dim).to(label.device)
    onehot.scatter_(1, label, 1)
    return onehot
