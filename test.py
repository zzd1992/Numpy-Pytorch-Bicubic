import argparse
from pyResize import DownSample, UpSample, imresize
import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser("scale must be int")
parser.add_argument("img_o", help="path of original image")
parser.add_argument("img_r", help="path of Matlab resized image")
args = parser.parse_args()


def main():
    img_o = cv2.imread(args.img_o).astype("float64")
    img_r = cv2.imread(args.img_r).astype("float64")
    down = img_o.shape[0] > img_r.shape[0]

    if down:
        scale = img_o.shape[0] // img_r.shape[0]
        img_numpy = imresize(img_o, 1.0 / scale)
        m = DownSample(scale)
    else:
        scale = img_r.shape[0] // img_o.shape[0]
        img_numpy = imresize(img_o, scale)
        m = UpSample(scale)

    img_numpy = np.clip(img_numpy, 0, 255)
    err_numpy = np.abs(img_r - img_numpy).max()

    x = torch.FloatTensor(img_o)
    x = x.permute(2, 0, 1).contiguous().unsqueeze(0)
    img_torch = m(x)[0].numpy()
    img_torch = np.transpose(img_torch, (1, 2, 0))
    img_torch = np.clip(img_torch, 0, 255)
    err_torch = np.abs(img_r - img_torch).max()

    print("************************************************************************")
    print("Maximum difference (0-255) between Matlab's and our's is shown below.")
    print("It should be smaller than 1.0. The difference of 0.5 is caused by round.")

    print("Difference of numpy imresize: {:.5f}".format(err_numpy))
    if down:
        print("Difference of torch DownSample: {:.5f}".format(err_torch))
    else:
        print("Difference of torch UpSample: {:.5f}".format(err_torch))


if __name__ == '__main__':
    main()
