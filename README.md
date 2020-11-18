# Numpy and PyTorch Bicubic Interpolation with Anti-aliasing

## Introduction
Bicubic is an interpolating method for data points on a two-dimensional regular grid. See [Wiki](https://en.wikipedia.org/wiki/Bicubic_interpolation) for more information. However, current Numpy and  PyTorch implementations don't have the function of anti-aliasing. Thus their outputs are different from Matlab's.

This project implements anti-aliasing bicubic interpolation using Numpy and PyTorch. **Our results are the same as Matlab's**.

- Numpy implementation supports any scale factors.
- PyTorch implementation only supports scale factors of 2, 3, and 4. This is based on `nn.Conv2d` thus it is efficient.


## Example and Test
First import our module.
```python
from pyResize import DownSample, UpSample, imresize
```

This is a Numpy example:
```python 
x = np.random.rand(100, 100, 3)
y = imresize(x, 0.3)
```

This is a PyTorch example:
```python
down = DownSample(2) 
up = UpSample(2)

x = torch.rand(1, 3, 100, 100)
y_l = down(x)
y_h = up(x)
```

If you want to test whether the results are the same as Matlab's, please run
```
python test.py {path of original image} {path of Matlab resized image}
```


