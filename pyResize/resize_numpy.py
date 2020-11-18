import numpy as np
from pyResize.utils import kernel_info, fix_scale_and_size


def imresize(im, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
    scale_factor, output_shape = fix_scale_and_size(im.shape, output_shape, scale_factor)

    method, kernel_width = kernel_info(kernel)
    antialiasing *= (scale_factor[0] < 1)

    sorted_dims = np.argsort(np.array(scale_factor)).tolist()
    out_im = np.copy(im)
    for dim in sorted_dims:
        if scale_factor[dim] == 1.0:
            continue

        weights, field_of_view = contributions(im.shape[dim], output_shape[dim], scale_factor[dim],
                                               method, kernel_width, antialiasing)
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    out_coordinates = np.arange(1, out_length + 1)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    left_boundary = np.floor(match_coordinates - kernel_width / 2)
    expanded_kernel_width = np.ceil(kernel_width)

    field_of_view = np.expand_dims(left_boundary, axis=1) + np.arange(1, expanded_kernel_width+1)
    field_of_view = field_of_view.astype("int64")

    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view)
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    sum_weights[sum_weights == 0] = 1.0
    weights = np.divide(weights, sum_weights)
    field_of_view = np.clip(field_of_view - 1, 0, in_length-1)

    return weights, field_of_view


def resize_along_dim(im, dim, weights, field_of_view):
    tmp_im = np.swapaxes(im, dim, 1)
    if im.ndim == 2:
        tmp_out_im = np.sum(tmp_im[:, field_of_view] * weights, axis=-1)
    elif im.ndim == 3:
        weights = np.expand_dims(weights, -1)
        tmp_out_im = np.sum(tmp_im[:, field_of_view] * weights, axis=-2)

    return np.swapaxes(tmp_out_im, dim, 1)

