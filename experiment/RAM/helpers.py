import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def append(a, b, dim=0):
    b = b.unsqueeze(dim)
    if a is None:
        shape = list(b.shape)
        shape[dim] = 0
        a = b.new_empty(shape)
    return torch.cat([a, b], dim=dim)


def resize(img, size):
    return F.adaptive_avg_pool2d(img, size)
