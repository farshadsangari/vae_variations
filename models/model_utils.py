from torch import nn
import torch
from utils import index2onehot, gaussian_parameters


class Encoder(nn.Module):
    def __init__(self, z_dim, conditional, device):
        super().__init__()
        self.z_dim = z_dim
        y_dim = 10 if conditional else 0
        self.conditional = conditional
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(784 + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def encode(self, x, y=None):
        if self.conditional:
            onehot_labels = index2onehot(y).to(self.device)
            x = torch.cat((x, onehot_labels), dim=-1)
        h = self.net(x)
        m, v = gaussian_parameters(h, dim=1)
        return m, v


class Decoder(nn.Module):
    def __init__(self, z_dim, conditional, device):
        super().__init__()
        self.z_dim = z_dim
        y_dim = 10 if conditional else 0
        self.conditional = conditional
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 784),
        )

    def decode(self, z, y=None):
        if self.conditional:
            onehot_labels = index2onehot(y).to(self.device)
            z = torch.cat((z, onehot_labels), dim=1)
        return self.net(z)
