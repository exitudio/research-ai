import torch
import torch.nn.functional as F


def append(a, b, dim=0):
    b = b.unsqueeze(dim)
    if a is None:
        shape = list(b.shape)
        shape[dim] = 0
        a = b.new_empty(shape)
    return torch.cat([a, b], dim=dim)


def resize(img, size):
    return F.adaptive_avg_pool2d(img, size)
