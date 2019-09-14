import torch
import torch.optim as optim

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

from utils import TestStats


class ExactGPModel(ExactGP):
    def __init__(self, x, y, likelihood, kernel=None):
        super().__init__(x, y, likelihood)
        self.mean_module = ConstantMean()
        if kernel is None:
            kernel = RBFKernel()
        self.covar_module = ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def exact_gp(data, n_epochs, kernel=None):
    train_x, train_y, test_x = data
    N = train_x.size(0)

    if N > 1000:
        train_x = train_x[:1000]
        train_y = train_y[:1000]

    # p(y|x, f)
    likelihood = GaussianLikelihood()
    # p(f|X, Y)
    model = ExactGPModel(train_x, train_y, likelihood, kernel)

    model.train()
    likelihood.train()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        if epoch % 50 == 0:
            print(
                'Iter {}/{} - Loss: {:.3f}   lengthscale: {:.3f}   noise: {:.3f}'
                .format(epoch, n_epochs, loss.item(),
                        model.covar_module.base_kernel.lengthscale.item(),
                        model.likelihood.noise.item()))

        optimizer.step()

    # test
    test_stats = TestStats(None, None)
    if N <= 1000:
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            test_y_mean = observed_pred.mean
            test_y_var = observed_pred.variance
        test_stats = test_stats._replace(test_y_mean=test_y_mean,
                                         test_y_var=test_y_var)

    return model, test_stats
