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
try:
    from timagetk.wrapping.clib import libvtexec, add_doc, return_value
    from timagetk.wrapping.vt_image import vt_image, new_vt_image
    from timagetk.components import SpatialImage
    from timagetk.util import try_spatial_image
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))

__all__ = ['REGIONALEXT_DEFAULT', 'regionalext']
REGIONALEXT_DEFAULT = '-minima -connectivity 26 -h 3'  # --- of course, stupid !


def regionalext(image, param_str_1=REGIONALEXT_DEFAULT, param_str_2=None,
                dtype=None):
    """
    H-transform algorithms.

    Parameters
    ----------
    image: ``SpatialImage``
        input image
    param_str_1: str, optional
        REGIONALEXT_DEFAULT, by default a h_min transform is computed
    param_str_2: str, optional
        optional parameters
    dtype: *np.dtype*, optional
        output image type. By default, the output type is equal to the input type.

    Returns
    -------
    ``SpatialImage``
        output image and metadata

    Example
    -------
    >>> from timagetk.util import data_path
    >>> from timagetk.components import imread
    >>> from timagetk.algorithms import regionalext
    >>> img_path = data_path('time_0_cut.inr')
    >>> input_image = imread(img_path)
    >>> output_image = regionalext(input_image)
    >>> param_str_2 = '-minima -connectivity 26 -h 5'
    >>> output_image = regionalext(input_image, param_str_2=param_str_2)
    """
    try_spatial_image(image)

    if dtype is None:
        dtype = image.dtype
    vt_input, vt_res = vt_image(image), new_vt_image(image, dtype=dtype)
    rvalue = libvtexec.API_regionalext(vt_input.c_ptr, vt_res.c_ptr,
                                       param_str_1, param_str_2)
    out_sp_img = return_value(vt_res.get_spatial_image(), rvalue)
    vt_input.free(), vt_res.free()
    return out_sp_img


add_doc(regionalext, libvtexec.API_Help_regionalext)
