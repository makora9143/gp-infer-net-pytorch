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
    bnn = bnn.to(args.device)
    bnn_prev = bnn_prev.to(args.device)
    prior_gp = prior_gp.to(args.device)

    infer_gpnet_optimizer = optim.Adam(bnn.parameters(), lr=args.learning_rate)
    hyper_opt_optimizer = optim.Adam(prior_gp.parameters(), lr=args.hyper_rate)

    x_min, x_max = dataloader.dataset.range

    bnn.train()
    bnn_prev.train()
    prior_gp.train()

    mb = master_bar(range(1, args.n_iters + 1))

    for t in mb:
        # Hyperparameter selection
        beta = args.beta0 * 1. / (1. + args.gamma * math.sqrt(t - 1))
        dl_bar = progress_bar(dataloader, parent=mb)
        for x, y in dl_bar:
            observed_size = x.size(0)
            x, y = x.to(args.device), y.to(args.device)
            x_star = torch.Tensor(args.measurement_size, x_dim).uniform_(x_min, x_max).to(args.device)
            # [Batch + Measurement Points x x_dims]
            xx = torch.cat([x, x_star], 0)

            infer_gpnet_optimizer.zero_grad()
            hyper_opt_optimizer.zero_grad()

            # inference net
            # Eq.(6) Prior p(f)
            # \mu_1=0, \Sigma_1
            mean_prior = torch.zeros(observed_size).to(args.device)
            K_prior = kernel(xx, xx).add_jitter(1e-6)

            # q_{\gamma_t}(f_M, f_n) = Normal(mu_2, sigma_2|x_n, x_m)
            # \mu_2, \Sigma_2
            qff_mean_prev, K_prox = bnn_prev(xx)

            # Eq.(8) adapt prior; p(f)^\beta x q(f)^{1 - \beta}
            mean_adapt, K_adapt = product_gaussians(mu1=mean_prior, sigma1=K_prior,
                                                    mu2=qff_mean_prev, sigma2=K_prox,
                                                    beta=beta)

            # Eq.(8)
            (mean_n, mean_m), (Knn, Knm, Kmm) = split_gaussian(mean_adapt, K_adapt, observed_size)

            # Eq.(2) K_{D,D} + noise / (N\beta_t)
            Ky = Knn + torch.eye(observed_size).to(args.device) * prior_gp.likelihood.noise / (N / observed_size * beta)
            Ky_tril = torch.cholesky(Ky)

            # Eq.(2)
            mean_target = Knm.t().mm(cholesky_solve(y - mean_n, Ky_tril)) + mean_m
            mean_target = mean_target.squeeze(-1)
            K_target = gpytorch.add_jitter(Kmm - Knm.t().mm(cholesky_solve(Knm, Ky_tril)), 1e-6)
            # \hat{q}_{t+1} (f_M)
            target_pf_star = MultivariateNormal(mean_target, K_target)

            # q_\gamma (f_M)
            qf_star = bnn(x_star)

            # Eq. (11)
            kl_obj = kl_div(qf_star, target_pf_star).sum()

            kl_obj.backward(retain_graph=True)
            infer_gpnet_optimizer.step()

            # Hyper paramter update
            (mean_n_prior, _), (Kn_prior, _, _) = split_gaussian(mean_prior, K_prior, observed_size)
            pf = MultivariateNormal(mean_n_prior, Kn_prior)

            (qf_prev_mean, _), (Kn_prox, _, _) = split_gaussian(qff_mean_prev, K_prox, observed_size)
            qf_prev = MultivariateNormal(qf_prev_mean, Kn_prox)

            hyper_obj = - (prior_gp.likelihood.expected_log_prob(y.squeeze(-1), qf_prev) - kl_div(qf_prev, pf))
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

    test_x = test_x.to(args.device)
    test_stats = evaluate(bnn, prior_gp.likelihood, test_x, args.net == 'tangent')
    return test_stats


def split_gaussian(mean, variance, n):
    """Separate gaussian by n
     -----      ---------   -----
    | m_n |    |  Knn    | | Knm |
    |     |    |         | |     |
     -----      ---------   -----
     -----  ,   ---------   -----
    | m_m |    |  Kmn    | | Kmm |
     -----      ---------   -----
    """
    mean_n, mean_m = mean[:n], mean[n:]
    variance_nn, variance_nm, variance_mm = variance[:n, :n], variance[:n, n:], variance[n:, n:]
    return (mean_n, mean_m), (variance_nn, variance_nm, variance_mm)


def product_gaussians(mu1, sigma1, mu2, sigma2, beta=0.0):
    # https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
    # sigma1 = prior, sigma2 = neural networks
    # \Sigma_1 + \Sigma_2
    K_sum = sigma1 * (1 - beta) + sigma2 * beta

    # \Sigma_3 = \Sigma_1(\Sigma_1 + \Sigma_2)^{-1}\Sigma_2
    K_adapt = sigma1.matmul(K_sum.inv_matmul(sigma2))

    # \mu_3 = \Sigma_3\Sigma_2^{-1}\mu_2
    # \mu_3 = \Sigma_3\Sigma_1^{-1}\mu_1 + \Sigma_3\Sigma_2^{-1}\mu_2
    mean_adapt = K_adapt.mm(torch.solve(mu2.unsqueeze(-1), sigma2)[0]) * (1 - beta)
    return mean_adapt, K_adapt


def gpnet_nonconj(args, dataloader, test_x, prior_gp):
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

            qff = bnn(xx)
            qff_mean_prev, K_prox = bnn_prev(xx)
            qf_mean, qf_var = bnn(x, full_cov=False)

            # Eq.(8)
            K_prior = kernel(xx, xx).add_jitter(1e-6)
            pff = MultivariateNormal(torch.zeros(xx.size(0)), K_prior)

            f_term = expected_log_prob(prior_gp.likelihood, qf_mean, qf_var, y.squeeze(-1))
            f_term = torch.sum(expected_log_prob(prior_gp.likelihood, qf_mean, qf_var, y.squeeze(-1)))
            f_term *= N / x.size(0) * beta

            prior_term = -beta * cross_entropy(qff, pff)

            qff_prev = MultivariateNormal(qff_mean_prev, K_prox)
            prox_term = - (1 - beta) * cross_entropy(qff, qff_prev)

            entropy_term = entropy(qff)

            lower_bound = f_term + prior_term + prox_term + entropy_term
            loss = - lower_bound / n

            loss.backward(retain_graph=True)

            infer_gpnet_optimizer.step()

            # Hyper-parameter update
            Kn_prior = K_prior[:n, :n]
            pf = MultivariateNormal(torch.zeros(n), Kn_prior)
            Kn_prox = K_prox[:n, :n]
            qf_prev_mean = qff_mean_prev[:n]
            qf_prev_var = torch.diagonal(Kn_prox)
            qf_prev = MultivariateNormal(qf_prev_mean, Kn_prior)
            hyper_obj = expected_log_prob(prior_gp.likelihood, qf_prev_mean, qf_prev_var, y.squeeze(-1)).sum() - kl_div(qf_prev, pf)
            hyper_obj = -hyper_obj
            hyper_obj.backward()
            hyper_opt_optimizer.step()

        bnn_prev.load_state_dict(bnn.state_dict())
        if args.net == 'rf':
            kernel_prev.load_state_dict(kernel.state_dict())
        if t % 50 == 0:
            mb.write(
                "Iter {}/{}, kl_obj = {:.4f}, noise = {:.4f}".format(
                    t, args.n_iters, lower_bound.item(), prior_gp.likelihood.noise.item()))
    test_x = test_x.to(args.device)
    test_stats = evaluate(bnn, prior_gp.likelihood, test_x, args.net == 'tangent')

    return test_stats


def expected_log_prob(likelihood, mean, variance, target):
    noise = likelihood.noise
    res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
    return res.mul(-0.5).sum(-1)


def cross_entropy(p, q):
    d = p.mean.size(-1)
    ret = 0.5 * d * math.log(2 * math.pi)
    ret += q.variance.log().sum(-1)
    q_cov_tril = torch.cholesky(q.covariance_matrix)
    p_cov_tril = torch.cholesky(p.covariance_matrix)
    q_inv_p = torch.triangular_solve(p_cov_tril, q_cov_tril)[0]
    ret += 0.5 * q_inv_p.pow(2).sum(-1).sum(-1)
    Kinv_m = cholesky_solve((p.mean - q.mean).unsqueeze(-1), q_cov_tril)
    ret += 0.5 * torch.sum((p.mean - q.mean) * Kinv_m.squeeze(-1), -1)
    return ret


def entropy(p):
    d = p.mean.size(-1)
    ret = 0.5 * d * math.log(2 * math.pi)
    ret += p.variance.log().sum(-1)
    ret += 0.5 * d
    return ret


def evaluate(bnn, likelihood, x, requires_grad=False):
    bnn.eval()

    if requires_grad:
        test_y_means, test_y_vars = bnn(x, full_cov=False)
        test_y_vars += likelihood.noise
        return TestStats(test_y_means.detach().unsqueeze(-1), test_y_vars.detach().unsqueeze(-1))

    with torch.no_grad():
        test_y_means, test_y_vars = bnn(x, full_cov=False)
        test_y_vars += likelihood.noise

    return TestStats(test_y_means.cpu().unsqueeze(-1), test_y_vars.cpu().unsqueeze(-1))
