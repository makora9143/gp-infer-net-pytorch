from scipy.cluster.vq import kmeans2

import torch
import torch.optim as optim

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.models import AbstractVariationalGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from utils import TestStats


class SVGP(AbstractVariationalGP):
    def __init__(self, inducing_points, kernel=None):
        # q(u)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-1))

        # q(f|x) = ∫q(f, u)du = ∫q(f|u, x)q(u)du
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()

        if kernel is None:
            kernel = RBFKernel()
        self.covar_module = ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred


def svgp(args, dataloader, test_x, kernel=None):
    N = len(dataloader.dataset)

    inducing_points, _ = kmeans2(dataloader.dataset.train_x.numpy(), args.n_inducing, minit='points')
    inducing_points = torch.from_numpy(inducing_points).squeeze(-1)

    model = SVGP(inducing_points, kernel)
    # p(y|f)
    likelihood = GaussianLikelihood()

    model.train()
    likelihood.train()

    optimizer = optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=args.learning_rate)

    mll = VariationalELBO(likelihood, model, N, combine_terms=False)

    for epoch in range(args.n_iters):
        for train_x, train_y in dataloader:
            train_x, train_y = train_x.squeeze(), train_y.squeeze()
            optimizer.zero_grad()
            output = model(train_x)

            log_ll, kl_div, log_prior = mll(output, train_y)
            loss = -(log_ll - kl_div + log_prior)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print("Iter {}, lower bound = {:.4f}, obs_var = {:.4f}"
                        .format(epoch, -loss.item(), likelihood.noise.item()))

    test_stats = TestStats(None, None)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        test_y_mean = observed_pred.mean
        test_y_var = observed_pred.variance
    test_stats = test_stats._replace(test_y_mean=test_y_mean,
                                     test_y_var=test_y_var)

    return test_stats
