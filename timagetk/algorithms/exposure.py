# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2018 INRIA
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------

"""
This module gether some of the functionalities from the 'exposure' module of
`scikit-image <https://scikit-image.org/>`_.

These algorithms are useful to stretch intensity images to span the whole range
of value accessible to a given bit-depth.
"""

import numpy as np
from timagetk.components import SpatialImage

try:
    from skimage import exposure
except ImportError:
    msg = "Missing package 'scikit-image' (skimage), please install it!"
    msg += "\n"
    msg += "Use `conda install scikit-image` or `pip install scikit-image -U`."
    raise ImportError(msg)


def type_to_range(img):
    """
    Returns the minimum and maximum values of a dtype according to image.

    Parameters
    ----------
    image : np.array or SpatialImage
        Image from which to extract the slice

    Returns
    -------
    min, max : int, int
        the minimum and maximum values depending on the array type encoding
    """
    try:
        assert hasattr(img, 'dtype')
    except:
        raise ValueError("Input 'img' has no attribute 'dtype', please check!")

    if img.dtype == 'uint8':
        return 0, 2 ** 8 - 1
    elif img.dtype == 'uint16':
        return 0, 2 ** 16 - 1
    else:
        msg = "Does not know what to do with such type: '{}'!".format(img.dtype)
        raise NotImplementedError(msg)


# ------------------------------------------------------------------------------
# - Contrast stretching:
# ------------------------------------------------------------------------------
def _contrast_stretch(image, pc_min=2, pc_max=99):
    """
    Return image after stretching its intensity levels from the lower / upper
    percentile values to the "dtype range" of the `image` (eg. to [0, 65535]
    if `image` is of type 'uint16').

    Parameters
    ----------
    image : np.array or SpatialImage
        Image from which to extract the slice
    pc_min : int
        Lower percentile use to define the lower range of the input image for
        image stretching
    pc_max : int
        Upper percentile use to define the upper range of the input image for
        image stretching

    Returns
    -------
    np.array
        stretched greylevel array
    """
    # Contrast stretching
    pcmin = np.percentile(image, pc_min)
    pcmax = np.percentile(image, pc_max)
    return exposure.rescale_intensity(image, in_range=(pcmin, pcmax))


def x_slice_contrast_stretch(image, pc_min=2, pc_max=99):
    """
    Performs slice by slice contrast stretching in z direction.
    Contrast stretching is here performed using lower and upper percentile of
    the image values to the min and max value of the image dtype.

    Parameters
    ----------
    image : np.array or SpatialImage
        Image from which to extract the slice
    pc_min : int
        Lower percentile use to define the lower range of the input image for
        image stretching
    pc_max : int
        Upper percentile use to define the upper range of the input image for
        image stretching

    Returns
    -------
    SpatialImage
        stretched SpatialImage
    """
    # TODO: add a min threshold to reach (eg. 5000) for each slice in order to apply intensity rescaling, otherwise don't rescale (i.e. no signal there!)
    # Slice by slice contrast stretching
    sh = image.shape
    im = np.array([_contrast_stretch(image[n, :, :], pc_min, pc_max) for n in
                   range(0, sh[0])]).transpose([0, 1, 2])
    if isinstance(image, SpatialImage):
        return SpatialImage(im, voxelsize=image.voxelsize, origin=image.origin,
                            metadata_dict=image.metadata)
    else:
        return im


def y_slice_contrast_stretch(image, pc_min=2, pc_max=99):
    """
    Performs slice by slice contrast stretching in z direction.
    Contrast stretching is here performed using lower and upper percentile of
    the image values to the min and max value of the image dtype.

    Parameters
    ----------
    image : np.array or SpatialImage
        Image from which to extract the slice
    pc_min : int
        Lower percentile use to define the lower range of the input image for
        image stretching
    pc_max : int
        Upper percentile use to define the upper range of the input image for
        image stretching

    Returns
    -------
    SpatialImage
        stretched SpatialImage
    """
    # TODO: add a min threshold to reach (eg. 5000) for each slice in order to apply intensity rescaling, otherwise don't rescale (i.e. no signal there!)
    # Slice by slice contrast stretching
    sh = image.shape
    im = np.array([_contrast_stretch(image[:, n, :], pc_min, pc_max) for n in
                   range(0, sh[1])]).transpose([1, 0, 2])
    if isinstance(image, SpatialImage):
        return SpatialImage(im, voxelsize=image.voxelsize, origin=image.origin,
                            metadata_dict=image.metadata)
    else:
        return im


def z_slice_contrast_stretch(image, pc_min=2, pc_max=99):
    """
    Performs slice by slice contrast stretching in z direction.
    Contrast stretching is here performed using lower and upper percentile of
    the image values to the min and max value of the image dtype.

    Parameters
    ----------
    image : np.array or SpatialImage
        Image from which to extract the slice
    pc_min : int
        Lower percentile use to define the lower range of the input image for
        image stretching
    pc_max : int
        Upper percentile use to define the upper range of the input image for
        image stretching

    Returns
    -------
    SpatialImage
        stretched SpatialImage
   """
    # TODO: add a min threshold to reach (eg. 5000) for each slice in order to apply intensity rescaling, otherwise don't rescale (i.e. no signal there!)
    # Slice by slice contrast stretching
    sh = image.shape
    im = np.array([_contrast_stretch(image[:, :, n], pc_min, pc_max) for n in
                   range(0, sh[2])]).transpose([1, 2, 0])
    if isinstance(image, SpatialImage):
        return SpatialImage(im, voxelsize=image.voxelsize, origin=image.origin,
                            metadata_dict=image.metadata)
    else:
        return im


# ------------------------------------------------------------------------------
# - Adaptive histogram equalisation:
# ------------------------------------------------------------------------------
def _equalize_adapthist(image, kernel_size=None, clip_limit=None, n_bins=256):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image.
    Local details can therefore be enhanced even in regions that are darker or
    lighter than most of the image.

    WARNING: 'equalize_adapthist' from 'exposure' module work on 2D images!!
    It should not be used on whole 3D images.

    By default, kernel_size is 1/8 of image height by 1/8 of its width.

    Parameters
    ----------
    image : np.array or SpatialImage
        image from which to extract the slice
    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm.
        If iterable is passed, it must have the same number of elements as
        image.ndim (without color channel).
        If integer, it is broadcasted to each image dimension.
        By default, kernel_size is 1/8 of image height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more contrast).
    n_bins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    np.array
        equalized greylevel array
    """
    # Adaptive Equalization
    mini, maxi = type_to_range(image)
    if clip_limit is None:
        clip_limit = 0.01
    if n_bins == 'dtype':
        n_bins = maxi + 1
    else:
        assert isinstance(n_bins, int)
        assert n_bins <= maxi + 1
    # Performs adaptative histogram equalisation:
    # WARNING: returns a float array with values in [0., 1.]
    float_arr = exposure.equalize_adapthist(image, kernel_size=kernel_size,
                                            clip_limit=clip_limit, nbins=n_bins)
    # Re-scale data in original range:
    if maxi == 2 ** 8 - 1:
        return np.array(float_arr * (maxi)).astype(np.uint8)
    elif maxi == 2 ** 16 - 1:
        return np.array(float_arr * (maxi)).astype(np.uint16)
    else:
        raise NotImplementedError("Only accept 'uint8' and 'uint16' types...")


def z_slice_equalize_adapthist(image, kernel_size=None, clip_limit=None,
                               n_bins=256):
    """
    Performs slice by slice adaptive histogram qualization in z direction.

    Parameters
    ----------
    image : np.array or SpatialImage
        image from which to extract the slice
    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm.
        If iterable is passed, it must have the same number of elements as
        image.ndim (without color channel).
        If integer, it is broadcasted to each image dimension.
        By default, kernel_size is 1/8 of image height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more contrast).
    n_bins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    SpatialImage
        equalized intensisty image

    Notes
    -----
    For color images, the following steps are performed:
        - The image is converted to HSV color space
        - The CLAHE algorithm is run on the V (Value) channel
        - The image is converted back to RGB space and returned
    For RGBA images, the original alpha channel is removed.
    """
    # TODO: add a min threshold to reach (eg. 5000) for each slice in order to apply intensity rescaling, otherwise don't rescale (i.e. no signal there!)
    sh = image.get_shape()
    im = np.array(
        [_equalize_adapthist(image[:, :, n], kernel_size, clip_limit, n_bins)
         for n in range(0, sh[2])]).transpose([1, 2, 0])
    if isinstance(image, SpatialImage):
        return SpatialImage(im, voxelsize=image.voxelsize, origin=image.origin,
                            metadata_dict=image.metadata)
    else:
        return im
