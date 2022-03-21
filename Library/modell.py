import torch.nn as nn
from functools import partial
from collections import namedtuple
import torch

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y

class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return

class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size=size

    def forward(self, x):
        return x[:, :, :self.size, :self.size]


batch_norm = partial(BatchNorm, weight_init=None, bias_init=None)
union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}
