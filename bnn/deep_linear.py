import torch
import torch.nn as nn
import gpytorch

from utils import softplus_tril, truncated_normal_


class DeepKernel(gpytorch.Module):
    def __init__(self, layer_sizes, mvn=True, activation=nn.Tanh(), ):
        super().__init__()

        input_sizes = layer_sizes[:-1]
        output_sizes = layer_sizes[1:]
        n_units = layer_sizes[-1]

        layers = []
        for input_size, output_size in zip(input_sizes, output_sizes):
            layers.extend([nn.Linear(input_size, output_size), activation])

        self.layers = nn.Sequential(*layers)

        self.w_mean = nn.Parameter(torch.Tensor(n_units, 1))
        truncated_normal_(self.w_mean, std=0.001)
        self.w_cov_raw = nn.Parameter(torch.eye(n_units))

        self.mvn = mvn

    def forward(self, x, full_cov=True):
        w_cov_tril = softplus_tril(self.w_cov_raw)

        h = self.layers(x)

        f_mean = h.mm(self.w_mean).squeeze(-1)

        f_cov_half = h.mm(w_cov_tril)

        if full_cov:
            f_cov = f_cov_half.mm(f_cov_half.t())
            f_cov = gpytorch.add_jitter(f_cov)

            if self.mvn:
                return gpytorch.distributions.MultivariateNormal(f_mean, f_cov)
            else:
                return f_mean, f_cov
        else:
            hw_cov = f_cov_half.mm(w_cov_tril.t())
            f_var = torch.sum(hw_cov * h, -1)
            f_var += 1e-6
            return f_mean, f_var
