import math

import torch
import torch.nn as nn
import gpytorch

from utils import softplus_tril


class RFExpansion(gpytorch.Module):
    def __init__(self, x_dim, n_units, kernel, mvn=True,
                 fix_freq=False, residual=False, fix_ls=False, activation=nn.Tanh()):
        super().__init__()

        self.kernel = kernel
        self.n_units = n_units
        self.residual = residual
        self.mvn = mvn

        fixed_freq = torch.randn(x_dim, n_units)
        if fix_freq:
            self.freq = fixed_freq
        else:
            self.freq = nn.Parameter(fixed_freq)

        if fix_ls:
            self.lengthscales = kernel.base_kernel.lengthscale
        else:
            self.lengthscales = nn.Parameter(kernel.base_kernel.lengthscale)

        self.w_mean = nn.Parameter(torch.Tensor(2 * n_units, 1))
        self.w_mean.data.normal_(std=0.001)

        self.w_cov_raw = nn.Parameter(torch.eye(2 * n_units))

        if self.residual:
            self.mlp = nn.Sequential(
                nn.Linear(x_dim, n_units),
                activation,
                nn.Linear(n_units, 1)
            )

    def forward(self, x, full_cov=True):
        h = x.mm(self.freq / self.lengthscales)
        h = torch.sqrt(self.kernel.outputscale) / math.sqrt(self.n_units) * torch.cat([torch.cos(h), torch.sin(h)], -1)

        f_mean = h.mm(self.w_mean)
        if self.residual:
            f_mean += self.mlp(x)
        f_mean = f_mean.squeeze(-1)

        w_cov_tril = softplus_tril(self.w_cov_raw)
        f_cov_half = h.mm(w_cov_tril)

        if full_cov:
            f_cov = f_cov_half.mm(f_cov_half.t())
            f_cov = gpytorch.add_jitter(f_cov)
            if self.mvn:
                f_dist = gpytorch.distributions.MultivariateNormal(f_mean, f_cov)
                return f_dist
            else:
                return f_mean, f_cov
        else:
            f_var = f_cov_half.pow(2).sum(-1)
            f_var += 1e-6
            return f_mean, f_var
