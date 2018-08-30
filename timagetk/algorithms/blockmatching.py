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
from ctypes import pointer
import numpy as np

try:
    from timagetk.wrapping.clib import add_doc, libblockmatching
    from timagetk.wrapping.balImage import BAL_IMAGE
    from timagetk.wrapping.bal_image import BalImage, init_c_bal_image
    from timagetk.wrapping.bal_image import allocate_c_bal_image, \
        spatial_image_to_bal_image_fields
    from timagetk.wrapping.bal_trsf import BalTransformation
    from timagetk.components import SpatialImage
    from timagetk.util import try_spatial_image
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))

__all__ = ['BLOCKMATCHING_DEFAULT', 'blockmatching']
BLOCKMATCHING_DEFAULT = '-trsf-type rigid'


# previous API - FRED
# Three plugins (rigid, affine and deformable registration) were implemented
# def blockmatching(floating_image,
#                  reference_image,
#                  transformation_type = 'rigid',
#                  init_result_transformation=None,
#                  left_transformation=None,
#                  param_str=''):


def blockmatching(floating_image, reference_image,
                  init_result_transformation=None, left_transformation=None,
                  param_str_1=BLOCKMATCHING_DEFAULT, param_str_2=None,
                  dtype=None):
    """
    Blockmatching registration algorithm.
    Registers a floating_image onto a reference_image.

    Parameters
    ----------
    floating_image: ``SpatialImage``
        image to register
    reference_image: ``SpatialImage``
        reference image to use for registration of floating_image
    init_result_transformation: BalTransformation, optional
        if given (default=None) it is used to initialise the registration and
        the returned transformation will contain it (composition)
    left_transformation: BalTransformation, optional
        if given (default=None) it is used to initialise the registration but
        the returned transformation will NOT contain it (no composition)
    param_str1: str, optional
        string of parameters used by blockmatching API (default='-trsf-type rigid')
    param_str_2: str, optional
        string of EXTRA parameters used by blockmatching API (default=None)
    dtype: np.dtype, optional
        output image type, by default is equal to the input type.

    Returns
    -------
    trsf_out: ``BalTransformation``
        transformation matrix
    res_img: ``SpatialImage`
        registered image and metadata

    Example
    -------
    >>> from timagetk.util import data_path
    >>> from timagetk.components import imread
    >>> flo_path = data_path('time_0_cut.inr')
    >>> floating_image = imread(flo_path)
    >>> ref_path = data_path('time_1_cut.inr')
    >>> reference_image = imread(ref_path)
    >>> trsf_rig, res_rig = blockmatching(floating_image, reference_image) # rigid registration
    >>> param_str_2 = '-trsf-type vectorfield'
    >>> trsf_def, res_def = blockmatching(floating_image, reference_image,
                                          init_result_transformation=trsf_rig,
                                          param_str_2=param_str_2) # deformable registration
    """
    # - Check if both input images are `SpatialImage`:
    try_spatial_image(floating_image, obj_name='floating_image')
    try_spatial_image(reference_image, obj_name='reference_image')

    # - Initialise BalImage:
    bal_floating_image = BalImage(floating_image)
    bal_reference_image = BalImage(reference_image)
    # - Get keyword arguments to initialise result image:
    kwargs = spatial_image_to_bal_image_fields(reference_image)
    if dtype:
        kwargs['np_type'] = dtype
    #  - Initialise result image:
    c_img_res = BAL_IMAGE()
    init_c_bal_image(c_img_res, **kwargs)
    allocate_c_bal_image(c_img_res,
                         np.ndarray(kwargs['shape'], kwargs['np_type']))
    img_res = BalImage(c_bal_image=c_img_res)

    # --- old API FRED, see plugins
    #    if transformation_type:
    #         param_str_2 = '-trsf-type '+transformation_type+' '+param_str
    #    else:
    #         param_str_2 = param_str
    # - Performs blockmatching registration:
    trsf_out_ptr = libblockmatching.API_blockmatching(bal_floating_image.c_ptr,
                                                      bal_reference_image.c_ptr,
                                                      pointer(c_img_res),
                                                      left_transformation.c_ptr if left_transformation else None,
                                                      init_result_transformation.c_ptr if init_result_transformation else None,
                                                      param_str_1, param_str_2)
    if init_result_transformation:
        # If init_result_transformation is given, this transformation is modified
        # during registration and trsf_out is init_result_transformation
        trsf_out = init_result_transformation
    else:
        trsf_out = BalTransformation(c_bal_trsf=trsf_out_ptr.contents)
    # - Transform result BalImage to SpatialImage:
    sp_img = img_res.to_spatial_image()
    #  - Free memory:
    bal_floating_image.free(), bal_reference_image.free(), img_res.free()

    return trsf_out, sp_img


add_doc(blockmatching, libblockmatching.API_Help_blockmatching)
