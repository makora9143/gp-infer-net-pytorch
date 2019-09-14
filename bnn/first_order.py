from functools import partial

import torch
import torch.nn as nn
import gpytorch
from gpytorch.distributions import MultivariateNormal

from utils import truncated_normal_


def jacobian(x, y, create_graph=True):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


class CustomLinear(gpytorch.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.w_mean = nn.Parameter(torch.Tensor(output_features, input_features))
        truncated_normal_(self.w_mean, std=0.01)
        self.w_logstd = nn.Parameter(torch.ones(output_features, input_features) * -2)

        self.b_mean = nn.Parameter(torch.zeros(output_features))
        self.b_logstd = nn.Parameter(torch.ones(output_features) * -2)

    def forward(self, x):
        output = x.mm(self.w_mean.t())
        output += self.b_mean.unsqueeze(0).expand_as(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features
        )


class FirstOrder(nn.Module):
    def __init__(self, layer_sizes, periodic=False, mvn=True, activation=nn.Tanh()):
        super().__init__()

        self.periodic = periodic
        self.mvn = mvn
        if periodic:
            self.periodic_fc = nn.Linear(layer_sizes[0], layer_sizes[0])

        layers = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.extend([
                CustomLinear(n_in, n_out),
                activation
            ])
        layers.append(CustomLinear(layer_sizes[-1], 1))
        self.layers = nn.Sequential(*layers)

        self.means = []
        self.stds = []
        for name, params in self.named_parameters():
            if 'mean' in name:
                self.means.append(params)
            elif 'std' in name:
                self.stds.append(torch.exp(params))

    def forward(self, x, full_cov=True):
        batch_size = x.size(0)
        if self.periodic:
            x = torch.cat([x, torch.sin(self.periodic_fc(x))], -1)

        h = self.layers(x)

        f_mean = h.squeeze(-1)

        grad_w_means = map(partial(jacobian, y=f_mean), self.means)

        f_cov_half = [(grad_w_mean * w_std).reshape(batch_size, -1)
                      for (grad_w_mean, w_std) in zip(grad_w_means, self.stds)]

        if full_cov:
            f_cov = sum([i.mm(i.t()) for i in f_cov_half])
            f_cov = gpytorch.add_jitter(f_cov)
            if self.mvn:
                return MultivariateNormal(f_mean, f_cov)
            else:
                return f_mean, f_cov
        else:
            f_var = sum([torch.sum(i.pow(2), -1) for i in f_cov_half])
            f_var += 1e-6
            return f_mean, f_var
