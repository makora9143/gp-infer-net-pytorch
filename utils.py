from collections import namedtuple
import torch
from torch.nn.functional import softplus


def cholesky_solve(b, u):
    "Like :func:`torch.cholesky_solve` but supports gradients."
    if not b.requires_grad and not u.requires_grad:
        return b.cholesky_solve(u)
    x = b.triangular_solve(u, upper=False).solution
    return x.triangular_solve(u, upper=False, transpose=True).solution


def softplus_tril(input):
    tril = torch.tril(input.float(), -1)
    softplus_diag = torch.diag_embed(softplus(torch.diagonal(input.float())))
    return tril + softplus_diag


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


TestStats = namedtuple("TestStats", ["test_y_mean", "test_y_var"])