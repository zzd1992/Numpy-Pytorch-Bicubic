import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from _jit_internal import weak_module, weak_script_method


def bicubic_down(s, a=-0.5):
    '''
    This function has anti-aliasing function whose outputs are the same as Matlab's 'imresize'.

    :param s: down sample scale -> (int in {2, 3, 4})
    :param a: hyper-parameter of bicubic kernel
    :return: bicubic kernel
    '''
    if s==2:
        b = 7 / float(4)
        B = np.abs(np.linspace(-b, b, 8))
    elif s==3:
        b = 5 / float(3)
        B = np.linspace(-b, b, 11)
    elif s==4:
        b = 15 / float(8)
        B = np.linspace(-b, b, 16)
    B = np.abs(B)
    A = ((a+2) * B ** 3 - (a+3) * B ** 2 + 1) * (B<=1) + \
        (a * B ** 3 - 5 * a * B ** 2 + 8 * a * B - 4 * a) * (B>1)

    return A / np.sum(A)


def bicubic_up(s, a=-0.5):
    A = np.zeros((s, 5), 'float32')
    delta = 1 / float(s)
    k = [-(s-1)/float(s)/2]
    for i in range(0, s-1):
        k.append(k[-1] + delta)
    for i, b in enumerate(k[::-1]):
        B = np.array([b-2, b-1, b, b+1, b+2])
        B = np.abs(B)
        A[i] = ((a + 2) * B ** 3 - (a + 3) * B ** 2 + 1) * (B <= 1) + \
            (a * B ** 3 - 5 * a * B ** 2 + 8 * a * B - 4 * a) * (B > 1) * (B < 2)
        A[i] /= np.sum(A[i])

    return A


@weak_module
class BicubicDown(nn.Module):
    def __init__(self, scale=2, channel=3, cuda=False):
        super(BicubicDown, self).__init__()
        self.scale = scale
        self.channel = channel
        w = bicubic_down(scale).astype('float32')
        n = w.shape[0]
        w = torch.tensor(w)
        pad = (n+1)//2 - 1 - (scale-1)//2

        self.weight_x = torch.zeros(channel, 1, 1, n)
        self.weight_y = torch.zeros(channel, 1, n, 1)
        for i in range(channel):
            self.weight_x[i, 0, 0] = w
            self.weight_y[i, 0, :, 0] = w
        if cuda and torch.cuda.is_available():
            self.weight_x = self.weight_x.cuda()
            self.weight_y = self.weight_y.cuda()

        self.pader = nn.ReplicationPad2d(pad)

    @weak_script_method
    def forward(self, x):
        x = self.pader(x)
        x = F.conv2d(x, self.weight_x, None, (1, self.scale), 0, 1, self.channel)
        x = F.conv2d(x, self.weight_y, None, (self.scale, 1), 0, 1, self.channel)
        return x


@weak_module
class BicubicUp(nn.Module):
    def __init__(self, scale, channel=3, cuda=False):
        super(BicubicUp, self).__init__()
        self.scale = scale
        self.channel = channel
        w = bicubic_up(scale)
        w = torch.tensor(w)
        self.weight_x = torch.zeros((channel*scale, 1, 1, 5))
        self.weight_y = torch.zeros((channel*scale, 1, 5, 1))
        for i in range(scale):
            for j in range(channel):
                self.weight_x[i*channel+j, 0, 0] = w[i]
                self.weight_y[i*channel+j, 0, :, 0] = w[i]
        if cuda and torch.cuda.is_available():
            self.weight_x = self.weight_x.cuda()
            self.weight_y = self.weight_y.cuda()

        self.pader = nn.ReplicationPad2d(2)

    @weak_script_method
    def forward(self, x):
        b, c, h, w = x.size()
        x = self.pader(x)

        x = F.conv2d(x, self.weight_x, None, 1, 0, 1, self.channel)
        x = x.contiguous().view(b, self.scale, c, h+4, w)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.contiguous().view(b, c, h+4, w * self.scale)

        x = F.conv2d(x, self.weight_y, None, 1, 0, 1, self.channel)
        x = x.contiguous().view(b, self.scale, c, h, w * self.scale)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.contiguous().view(b, c, h * self.scale, w * self.scale)

        return x


if __name__ == '__main__':
    x = torch.rand(16, 3, 64, 64)

    m_down = BicubicDown(scale=2, channel=3)
    y_down = m_down(x)
    print(y_down.size())

    m_up = BicubicUp(scale=2, channel=3)
    y_up = m_up(x)
    print(y_up.size())
