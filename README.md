# PyTorch Bicubic Interpolation with Anti-aliasing

## Introduction
*Bicubic* is an interpolating method for data points on a two-dimensional regular grid. See [Wiki](https://en.wikipedia.org/wiki/Bicubic_interpolation) for more information. There are two reasons why we implement *Bicubic* on PyTorch:]

1. Current Python implementations don't have the function of anti-aliasing. Thus the outputs of bicubic downsample are different from Matlab's.
2. Some people use *Bicubic* inside deep neural networks.

*Bicubic* can be accomplished using 1D cubic convolution algorithm two times. Thus our implementation is built at the top of `nn.Conv2d`(this is a lazy way). 

## Anti-aliasing
When applying bicubic downsample, anti-aliasing can improve the interpolation quality. Anti-aliasing is the default setting of Matlab while most of the Python implementations haven't this function. This makes their outputs of bicubic downsample different.

The basic idea of anti-aliasing is expanding the kernel size by factor S, where S is the down sample factor. Or equivalently, the scale of coordinate is shrunk by factor S. For example, there are 4 neighboring points involved in cubic convolution *without* anti-aliasing. When S=2, there are 8 neighboring points involved n cubic convolution *with* anti-aliasing.

With anti-aliasing, the outputs of our implementation are exactly the same as Matlab's (check it by running test.py).

## Notice

1. The hyper-parameter **a** (see [Wiki](https://en.wikipedia.org/wiki/Bicubic_interpolation)) is set to -0.5 in Matlab and -0.75 in Python. Here, we set **a** to -0.5.
2. Our implementation only supports integer scale factor. For non-integer factor, it can't be implemented via `nn.Conv2d`.
3. Bicubic upsample is officially implemented in the latest version of PyTorch(1.1.0) while bicubic downsample is not. We find our implementation is faster than official's when using CPU. When using GPU, our implementation is slower than official's.

