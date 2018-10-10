# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Gregoire Malandain <gregoire.malandain@inria.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------

# --- Aug. 2016
import numpy as np

try:
    from timagetk.components import SpatialImage
    from timagetk.algorithms import apply_trsf
    from timagetk.components import try_spatial_image
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))

__all__ = ['resample_isotropic', 'subsample']

POSS_OPTION = ['gray', 'label']
POSS_METHODS = ['min', 'max']


def resample(image, voxelsize, option='gray', **kwargs):
    """
    Resample an image to the given voxelsize.
    Use 'option' to control type of interpolation applied to the image.

    Parameters
    ----------
    image: SpatialImage
        an image to resample
    voxelsize: list
        the voxelsize to which the image should be resampled
    option: str, optional
        'gray' (default) indicate the image will be interpolated with a linear
        function, when 'label' use the nearest function.

    Returns
    -------
    out_img: SpatialImage
        the resampled image
    """
    verbose = kwargs.get('verbose', False)
    ndim = image.get_dim()
    try:
        assert image.voxelsize != []
    except AssertionError:
        raise ValueError("Input image has an EMPTY voxelsize attribute!")
    try:
        assert len(voxelsize) == ndim
    except AssertionError:
        msg = "Given 'voxelsize' ({}) ".format(voxelsize)
        msg += "does not match the dimension of the image ({}).".format(ndim)
        raise ValueError(msg)

    # - Compute the new shape of the object using image extent & new voxelsize:
    ext = image.extent
    new_shape = [int(round(ext[i] / float(voxelsize[i]))) for i in range(ndim)]
    # - Initialise a template array with the new shape:
    tmp_img = np.zeros(tuple(new_shape), dtype=image.dtype)
    # - Initialise a new metadata dictionary matching the template array:
    new_md = image.metadata
    # -- Remove the keys and values of 'NEW' properties: 'shape' & 'voxelsize'
    try:
        new_md.pop('voxelsize')  # updated later during SpatialImage.__init__
    except KeyError:
        pass
    try:
        new_md.pop('shape')  # updated later during SpatialImage.__init__
    except KeyError:
        pass

    # - Initialise a SpatialImage from the template array, new voxelsize and metadata dictionary:
    tmp_img = SpatialImage(tmp_img, voxelsize=voxelsize, origin=image.origin,
                           metadata_dict=new_md)

    param_str_2 = ' -resize'
    if option == 'gray' or option == 'grey':
        param_str_2 += ' -interpolation linear'
    elif option == 'label':
        param_str_2 += ' -interpolation nearest'
    else:
        msg = "Given 'option' ({}) is not available!\n".format(option)
        msg += "Choose among: {}".format(POSS_OPTION)
        raise NotImplementedError(msg)

    if verbose:
        is_iso = tmp_img.is_isometric()
        print "Image {}resampling:".format("isometric " if is_iso else "")
        print "  - 'shape': {} -> {}".format(image.shape, new_shape)
        print "  - 'voxelsize': {} -> {}".format(image.voxelsize, voxelsize)

    # - Performs resampling using 'apply_trsf':
    out_img = apply_trsf(image, trsf=None, template_img=tmp_img,
                         param_str_2=param_str_2)

    # - Since '' only works on 3D images, it might have converted it to 3D:
    if 1 in out_img.shape:
        out_img = out_img.to_2D()

    return out_img


def resample_isotropic(image, voxelsize, option='gray', **kwargs):
    """
    Resample into an isotropic dataset

    Parameters
    ----------
    sp_img: ``SpatialImage``
        input image
    voxelsize: float
        voxelsize value
    option: str, optional
        option can be either 'gray' or 'label'

    Returns
    -------
    ``SpatialImage``
        output image and metadata

    Example
    -------
    >>> output_image = resample_isotropic(input_image, voxelsize=0.4)
    """
    # - Make sure the given voxelsize is a float:
    if not isinstance(voxelsize, float):
        voxelsize = float(voxelsize)

    # - Create the new voxelsize list based on image dimensionality:
    new_vox = [voxelsize] * image.get_dim()
    return resample(image, new_vox, option, **kwargs)


def isometric_resampling(input_im, method='min', option='gray', **kwargs):
    """
    Transform the image to an isometric version according to a method or a given voxelsize.

    Parameters
    ----------
    input_im: SpatialImage
        image to resample
    method: str|float, optional
        change voxelsize to 'min' (default), 'max' of original voxelsize or to a
        given value.
    option: str, optional
        option can be either 'gray' or 'label'

    Returns
    -------
    ``SpatialImage``
        output image and metadata
    """
    try:
        assert input_im.voxelsize != []
    except AssertionError:
        raise ValueError("Input image has an EMPTY voxelsize attribute!")

    if method not in POSS_METHODS and not isinstance(method, float):
        raise ValueError(
            "Possible values for 'methods' are a float, 'min' or 'max'.")
    if method == 'min':
        vxs = np.min(input_im.voxelsize)
    elif method == 'max':
        vxs = np.max(input_im.voxelsize)
    else:
        vxs = method

    if np.allclose(input_im.voxelsize, [vxs] * input_im.get_dim()):
        print "Image is already isometric!"
        return input_im

    return resample_isotropic(input_im, vxs, option, **kwargs)


def subsample(image, factor=[2, 2, 1], option='gray'):
    """
    Subsample a ``SpatialImage`` (2D/3D, grayscale/label)

    Parameters
    ----------
    image: ``SpatialImage``
        input ``SpatialImage``
    factor: list, optional
        int|float or xyz-list of subsampling values
    option: str, optional
        option can be either 'gray' or 'label'

    Returns
    -------
    ``SpatialImage``
        output image and metadata

    Example
    -------
    >>> output_image = subsample(input_image)
    """
    try_spatial_image(image, obj_name='image')

    poss_opt = ['gray', 'label']
    if option not in poss_opt:
        option = 'gray'

    try:
        assert image.is2D() or image.is3D()
    except AssertionError:
        raise ValueError("Image should be 2D or 3D.")

    n_dim = image.get_dim()
    if isinstance(factor, int) or isinstance(factor, float):
        factor = [factor] * n_dim

    if n_dim == 2:
        image = image.to_3D()
        factor.append(1)

    shape, extent = image.shape, image.extent
    # new_shape = [int(np.ceil(shape[ind] / factor[ind])) for ind in
    #              range(image.get_dim())]
    # Smaller approximation error with round than np.ceil ?!
    new_shape = [int(round(shape[ind] / factor[ind])) for ind in
                 range(image.get_dim())]
    new_vox = [extent[ind] / new_shape[ind] for ind in
               range(image.get_dim())]
    tmp_img = np.zeros((new_shape[0], new_shape[1], new_shape[2]),
                       dtype=image.dtype)
    tmp_img = SpatialImage(tmp_img, voxelsize=new_vox,
                           origin=image.origin,
                           metadata_dict=image.metadata)

    if option == 'gray':
        param_str_2 = '-resize -interpolation linear'
    elif option == 'label':
        param_str_2 = '-resize -interpolation nearest'

    out_img = apply_trsf(image, trsf=None, template_img=tmp_img,
                         param_str_2=param_str_2)
    if 1 in out_img.shape:
        out_img = out_img.to_2D()

    return out_img
