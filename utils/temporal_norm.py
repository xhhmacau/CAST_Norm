import torch
import torch.nn as nn


class TemporalNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x_bnl: torch.Tensor):
        if x_bnl.ndim != 3:
            raise ValueError('Expected input with shape [B, N, L]')

        mu = x_bnl.mean(dim=-1, keepdim=True)
        var = (x_bnl - mu).pow(2).mean(dim=-1, keepdim=True)
        theta = torch.sqrt(var + self.eps)
        x_clean = (x_bnl - mu) / theta
        return x_clean, mu, theta
