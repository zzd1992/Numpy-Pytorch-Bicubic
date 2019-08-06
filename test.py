from scipy import io
import numpy as np
import time
import torch
from torch.nn import Upsample
from bicubic import BicubicUp, BicubicDown


def compare_matlab(scale=2):
    mat = io.loadmat("bicubic.mat")
    # d->data, d2->down sample by factor 'scale' ...
    d, d_scale = mat['d'], mat['d{}'.format(scale)]
    # u->data, u2->up sample by factor 'scale' ...
    u, u_scale = mat['u'], mat['u{}'.format(scale)]
    d = torch.tensor(d).unsqueeze(0).unsqueeze(0)
    u = torch.tensor(u).unsqueeze(0).unsqueeze(0)

    m_up = BicubicUp(scale, channel=1)
    out_u = m_up(u)
    out_u = out_u[0, 0].numpy()
    err_u = np.sum((out_u - u_scale) ** 2)
    print("Error of up sample scale {}:\t{:.3f}".format(scale, err_u))

    m_down = BicubicDown(scale, channel=1)
    out_d = m_down(d)
    out_d = out_d[0, 0].numpy()
    err_d = np.sum((out_d - d_scale) ** 2)
    print("Error of down sample scale {}:\t{:.3f}\n".format(scale, err_d))


def compare_speed(cuda=False):
    x = torch.rand(64, 3, 256, 256)
    m_lazy = BicubicUp(2, channel=3, cuda=cuda)
    m_official = Upsample(scale_factor=2, mode='bicubic', align_corners=False)
    if cuda:
        x = x.cuda()
        m_official.cuda()

    t = time.time()
    for i in range(10):
        y = m_lazy(x)
    print("lazy implementation on {}:\t{:.4f}s".format("GPU" if cuda else "CPU", time.time()-t))

    t = time.time()
    for i in range(10):
        y = m_official(x)
    print("official implementation on {}:\t{:.4f}s\n".format("GPU" if cuda else "CPU", time.time() - t))


if __name__ == '__main__':
    # whether the output is same as matlab's
    compare_matlab(2)
    compare_matlab(3)
    compare_matlab(4)

    if torch.__version__ >= "1.1.0":
        # compare the speed of up sample with official implementation
        compare_speed(False)
        if torch.cuda.is_available():
            compare_speed(True)
