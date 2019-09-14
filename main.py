import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from regression import exact_gp, svgp, gpnet
from dataset import Snelson

torch.manual_seed(1234)
np.random.seed(1234)


def set_up_figure(data, test_stats):
    train_x, train_y, test_x = data

    plt.figure(figsize=(12, 8))
    plt.scatter(
        train_x.squeeze(-1), train_y.squeeze(-1), c="k")
    plot_ground_truth(test_x.squeeze(-1),
                      test_stats.test_y_mean.squeeze(-1),
                      test_stats.test_y_var.squeeze(-1))


def plot_ground_truth(test_x, test_y_mean_, test_y_var_):
    plt.plot(test_x, test_y_mean_, c="k", linewidth=2)
    plt.plot(test_x, test_y_mean_ + 3. * np.sqrt(test_y_var_), '--',
             color="k", linewidth=2)
    plt.plot(test_x, test_y_mean_ - 3. * np.sqrt(test_y_var_), '--',
             color="k", linewidth=2)


def plot_method(data, test_stats, color):
    _, _, test_x = data
    test_x = test_x.squeeze(-1)
    test_y_mean = test_stats.test_y_mean.squeeze(-1)
    test_y_var = test_stats.test_y_var.squeeze(-1)
    plt.plot(test_x, test_y_mean, c=color, linewidth=2)
    plt.fill_between(test_x,
                     test_y_mean + 3. * np.sqrt(test_y_var),
                     test_y_mean - 3. * np.sqrt(test_y_var),
                     alpha=0.2, color=color)


def main():
    dataset = Snelson("data")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    data = (dataset.train_x.squeeze(), dataset.train_y.squeeze(), dataset.test_x.squeeze())

    if not args.pretrain:
        _, true_stats = exact_gp(data, n_epochs=300)
        gp, _ = exact_gp(data, n_epochs=0)
    else:
        gp, true_stats = exact_gp(data, n_epochs=300)

    set_up_figure(data, true_stats)

    if args.method == 'svgp':
        test_stats = svgp(args, dataloader, dataset.test_x.unsqueeze(-1))
        plot_method(data, test_stats, 'b')

    if args.method == 'gpnet':
        test_stats = gpnet(args, dataloader, dataset.test_x, prior_gp=gp)
        plot_method(data, test_stats, 'g')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='gpnet', choices=['svgp', 'gpnet'],
                        help='Inference algorithm')
    parser.add_argument('--batch-size', '-B', type=int, default=20,
                        help='Total batch size')
    parser.add_argument('--measurement-size', '-M', type=int, default=20,
                        help='Measurement set size')
    parser.add_argument('--learning-rate', '-LR', type=float, default=0.003,
                        help='Learning rate')
    parser.add_argument('--n-hidden', '-nH', type=int, default=20,
                        help='Hidden layer size')
    parser.add_argument('--n-layer', '-nL', type=int, default=1,
                        help='Numbewr of hidden layer')
    parser.add_argument('--n-inducing', '-nP', type=int, default=20,
                        help='Number of inducing points')
    parser.add_argument('--n-iters', '-nI', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--test_freq', '-T', type=int, default=50,
                        help='Test frequency')
    parser.add_argument('--measure', type=str, default='uniform',
                        help='Measurement set')
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Iters of pretraining GP priors')
    parser.add_argument('--net', type=str, default='rf',
                        help='Inference network')
    parser.add_argument('--residual', default=False, action='store_true', help='Use residual connection')
    parser.add_argument('--beta0', type=float, default=1.0, help='Initial beta value')
    parser.add_argument('--gamma', type=float, default=0.1, help='Beta schedule')
    parser.add_argument('--hyper-rate', type=float, default=0.003, help='Hyperparameter update rate.')
    parser.add_argument('--hyper-anneal', default=False, action='store_true', help='Hyper_rate annealed by beta')
    parser.add_argument('--lr-anneal', default=False, action='store_true', help='learning rate annealed by beta')
    parser.add_argument('--fix-rf-ls', default=False, action='store_true', help='fix the lengthscales of rf as prior')

    args = parser.parse_args()
    main()
