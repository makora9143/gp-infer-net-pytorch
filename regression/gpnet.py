import math

import torch
import torch.optim as optim

import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.distributions.multivariate_normal import kl_mvn_mvn as kl_div

from fastprogress import master_bar, progress_bar

from bnn.first_order import FirstOrder
from bnn.rf_expansion import RFExpansion
from bnn.deep_linear import DeepKernel
from utils import cholesky_solve, TestStats


def gpnet(args, dataloader, test_x, prior_gp):
    N = len(dataloader.dataset)
    x_dim = 1
    prior_gp.train()

    if args.net == 'tangent':
        kernel = prior_gp.covar_module
        bnn_prev = FirstOrder([x_dim] + [args.n_hidden] * args.n_layer, mvn=False)
        bnn = FirstOrder([x_dim] + [args.n_hidden] * args.n_layer, mvn=True)
    elif args.net == 'deep':
        kernel = prior_gp.covar_module
        bnn_prev = DeepKernel([x_dim] + [args.n_hidden] * args.n_layer, mvn=False)
        bnn = DeepKernel([x_dim] + [args.n_hidden] * args.n_layer, mvn=True)
    elif args.net == 'rf':
        kernel = ScaleKernel(RBFKernel())
        kernel_prev = ScaleKernel(RBFKernel())
        bnn_prev = RFExpansion(x_dim, args.n_hidden, kernel_prev, mvn=False, fix_ls=args.fix_rf_ls, residual=args.residual)
        bnn = RFExpansion(x_dim, args.n_hidden, kernel, fix_ls=args.fix_rf_ls, residual=args.residual)
        bnn_prev.load_state_dict(bnn.state_dict())
    else:
        raise NotImplementedError('Unknown inference net')

    infer_gpnet_optimizer = optim.Adam(bnn.parameters(), lr=args.learning_rate)
    hyper_opt_optimizer = optim.Adam(prior_gp.parameters(), lr=args.hyper_rate)

    x_min, x_max = dataloader.dataset.range
    n = dataloader.batch_size

    bnn.train()
    bnn_prev.train()
    prior_gp.train()

    mb = master_bar(range(1, args.n_iters + 1))

    for t in mb:
        beta = args.beta0 * 1. / (1. + args.gamma * math.sqrt(t - 1))
        dl_bar = progress_bar(dataloader, parent=mb)
        for x, y in dl_bar:
            n = x.size(0)
            x_star = torch.Tensor(args.measurement_size, x_dim).uniform_(x_min, x_max)
            xx = torch.cat([x, x_star], 0)

            # inference net
            infer_gpnet_optimizer.zero_grad()
            hyper_opt_optimizer.zero_grad()

            qf_star = bnn(x_star)
            qff_mean_prev, K_prox = bnn_prev(xx)

            # Eq.(8)
            K_prior = kernel(xx, xx).add_jitter(1e-6)

            K_sum = K_prior * (1 - beta) + K_prox * beta

            # \Sigma_3 = \Sigma_1(\Sigma_1 + \Sigma_2)^{-1}\Sigma_2
            K_adapt = K_prior.matmul(K_sum.inv_matmul(K_prox))
            Kn, Knm, Km = K_adapt[:n, :n], K_adapt[:n, n:], K_adapt[n:, n:]

            # \mu_3 = \Sigma_3\Sigma_2^{-1}\mu_2
            mean_adapt = K_adapt.mm(torch.solve(qff_mean_prev[..., None], K_prox)[0]) * (1 - beta)
            mean_n, mean_m = mean_adapt[:n, :], mean_adapt[n:, :]

            # Eq.(10) and Eq.(2)
            Ky = Kn + torch.eye(n) * prior_gp.likelihood.noise / (N / n * beta)
            Ky_tril = torch.cholesky(Ky)

            mean_target = Knm.t().mm(cholesky_solve(y - mean_n, Ky_tril)) + mean_m
            mean_target = mean_target.squeeze(-1)
            K_target = gpytorch.add_jitter(Km - Knm.t().mm(cholesky_solve(Knm, Ky_tril)))

            target_pf_star = MultivariateNormal(mean_target, K_target)

            kl_obj = kl_div(qf_star, target_pf_star).sum()

            kl_obj.backward(retain_graph=True)
            infer_gpnet_optimizer.step()

            # Hyper paramter update
            Kn_prior = K_prior[:n, :n]
            pf = MultivariateNormal(torch.zeros(n), Kn_prior)
            Kn_prox = K_prox[:n, :n]
            qf_prev_mean = qff_mean_prev[:n]
            qf_prev = MultivariateNormal(qf_prev_mean, Kn_prox)

            hyper_obj = prior_gp.likelihood.expected_log_prob(y.squeeze(-1), qf_prev) - kl_div(qf_prev, pf)
            hyper_obj = - hyper_obj
            hyper_obj.backward(retain_graph=True)
            hyper_opt_optimizer.step()

            mb.child.comment = "kl_obj = {:.3f}, obs_var={:.3f}".format(
                kl_obj.item(), prior_gp.likelihood.noise.item())

        # update q_{\gamma_t} to q_{\gamma_{t+1}}
        bnn_prev.load_state_dict(bnn.state_dict())
        if args.net == 'rf':
            kernel_prev.load_state_dict(kernel.state_dict())
        if t % 50 == 0:
            mb.write(
                "Iter {}/{}, kl_obj = {:.4f}, noise = {:.4f}".format(
                    t, args.n_iters, kl_obj.item(), prior_gp.likelihood.noise.item()))

    test_stats = evaluate(bnn, prior_gp.likelihood, test_x, args.net == 'tangent')
    return test_stats


def evaluate(bnn, likelihood, x, requires_grad=False):
    bnn.eval()

    if requires_grad:
        test_y_means, test_y_vars = bnn(x, full_cov=False)
        test_y_vars += likelihood.noise
        return TestStats(test_y_means.detach().unsqueeze(-1), test_y_vars.detach().unsqueeze(-1))

    with torch.no_grad():
        test_y_means, test_y_vars = bnn(x, full_cov=False)
        test_y_vars += likelihood.noise

    return TestStats(test_y_means.unsqueeze(-1), test_y_vars.unsqueeze(-1))
