# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Sophie Ribes <sophie.ribes@inria.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------
import numpy as np
from math import log10

np_square, np_array, np_reshape, np_sum = np.square, np.array, np.reshape, np.sum
try:
    from timagetk.components import SpatialImage
    from timagetk.components import try_spatial_image
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))

__all__ = ['mean_squared_error', 'psnr']


def mean_squared_error(sp_img_1, sp_img_2):
    """
    Mean squared error - measure of image quality

    Parameters
    ----------
    sp_img_1: ``SpatialImage``
        ``SpatialImage`` image
    sp_img_2: ``SpatialImage``
        ``SpatialImage`` image

    Returns
    -------
    mse: float
        mean sqared error value
    """
    # - Check if both input images are `SpatialImage`:
    try_spatial_image(sp_img_1, obj_name='sp_img_1')
    try_spatial_image(sp_img_2, obj_name='sp_img_2')

    try:
        assert sp_img_1.shape == sp_img_2.shape
    except AssertionError:
        msg = 'sp_img_1 and sp_img_2 does not have the same shape'
        raise TypeError(msg)

    img_1 = sp_img_1.get_array().astype(np.float16)  # np.ndarray instance
    img_2 = sp_img_2.get_array().astype(np.float16)  # np.ndarray instance
    tmp = np_square((img_1 - img_2))  # np.ndarray instance
    tmp = np_array(np_reshape(tmp, (-1, 1))).tolist()
    mse = np_sum(tmp) / len(tmp)

    return mse


def psnr(sp_img_1, sp_img_2):
    """
    Peak Signal To Noise Ratio - measure of image quality

    Parameters
    ----------
    sp_img_1: ``SpatialImage``
        ``SpatialImage`` image
    sp_img_2: ``SpatialImage``
        ``SpatialImage`` image

    Returns
    -------
    psnr: float
        psnr value (dB)
    """
    # - Check if both input images are `SpatialImage`:
    try_spatial_image(sp_img_1, obj_name='sp_img_1')
    try_spatial_image(sp_img_2, obj_name='sp_img_2')

    try:
        assert sp_img_1.itemsize == sp_img_2.itemsize
    except AssertionError:
        msg = 'sp_img_1 and sp_img_2 does not have the same type'
        raise TypeError(msg)

    maxi = 2 ** (sp_img_1.itemsize * 8) - 1
    mse = mean_squared_error(sp_img_1, sp_img_2)
    if mse != 0:
        psnr = 20.0 * log10(maxi) - 10 * log10(mse)
    else:
        psnr = np.inf

    return psnr
