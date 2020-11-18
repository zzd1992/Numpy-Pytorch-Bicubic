import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pyResize import utils


def cubic_down(s):
    if s == 2:
        b = 7 / float(4)
        B = np.abs(np.linspace(-b, b, 8))
    elif s == 3:
        b = 5 / float(3)
        B = np.linspace(-b, b, 11)
    elif s == 4:
        b = 15 / float(8)
        B = np.linspace(-b, b, 16)

    A = utils.cubic(B)

    return A / np.sum(A)


def cubic_up(s):
    A = np.zeros((s, 5), 'float32')
    delta = 1 / float(s)
    k = [-(s - 1) / float(s) / 2]
    for i in range(0, s - 1):
        k.append(k[-1] + delta)
    for i, b in enumerate(k[::-1]):
        B = np.array([b - 2, b - 1, b, b + 1, b + 2])
        A[i] = utils.cubic(B)
        A[i] /= np.sum(A[i])

    return A


class DownSample(nn.Module):
    def __init__(self, scale: int):
        assert scale in [2, 3, 4], "scale must be 2, 3, or 4!"
        super(DownSample, self).__init__()
        self.scale = scale
        self._get_kernel()
        pad = (self.weights_x.size(-1) + 1) // 2 - 1 - (scale - 1) // 2
        self.padder = nn.ReplicationPad2d(pad)

    def _get_kernel(self):
        weights = cubic_down(self.scale)
        weights = torch.FloatTensor(weights)
        self.register_buffer("weights_x", weights.view(1, 1, 1, -1))
        self.register_buffer("weights_y", weights.view(1, 1, -1, 1))

    def forward(self, x):
        x = self.padder(x)
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        x = F.conv2d(x, self.weights_x, None, (1, self.scale), 0, 1, 1)
        x = F.conv2d(x, self.weights_y, None, (self.scale, 1), 0, 1, 1)
        x = x.view(b, c, x.size(2), x.size(3))

        return x


class UpSample(nn.Module):
    def __init__(self, scale: int):
        assert scale in [2, 3, 4], "scale must be 2, 3, or 4!"
        super(UpSample, self).__init__()
        self.scale = scale
        self._get_kernel()
        # the width of cubic kernel for up sample is always 5.
        self.padder = nn.ReplicationPad2d(2)

    def _get_kernel(self):
        w = cubic_up(self.scale)  # (scale, width)
        weights = []
        for i in range(self.scale):
            for j in range(self.scale):
                weights.append(np.outer(w[i], w[j]))
        weights = np.stack(weights, 0)
        weights = torch.FloatTensor(weights)
        weights = weights.view(self.scale ** 2, 1, w.shape[1], w.shape[1])
        self.register_buffer("weights", weights)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * c, 1, h, w)
        x = self.padder(x)
        x = F.conv2d(x, self.weights, None, 1, 0, 1, 1)
        x = F.pixel_shuffle(x, self.scale)
        x = x.view(b, c, h * self.scale, w * self.scale)

        return x


