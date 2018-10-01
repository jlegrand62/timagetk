# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>

#       See accompanying file LICENSE.txt
# -----------------------------------------------------------------------------

import warnings

try:
    from timagetk.components.spatial_image import SpatialImage
except ImportError:
    raise ImportError('Unable to import SpatialImage class.')

try:
    from timagetk.components.labelled_image import LabelledImage
except ImportError:
    raise ImportError('Unable to import LabelledImage class.')

try:
    from timagetk.components.tissue_image import TissueImage
    from timagetk.components.tissue_image import TissueImage2D
    from timagetk.components.tissue_image import TissueImage3D
except ImportError:
    raise ImportError('Unable to import TissueImage class.')

try:
    from timagetk.components.io import imread, imsave
except ImportError:
    raise ImportError('Unable to import imread and imsave.')


def try_spatial_image(obj, obj_name=None):
    """
    Tests whether given instance is a SpatialImage.

    Parameters
    ----------
    obj: instance
        instance to test
    obj_name: str, optional
        if given used as object name for TypeError printing
    """
    if obj_name is None:
        obj_name = type(obj)

    err = "Input '{}' is not a SpatialImage instance."
    try:
        assert isinstance(obj, SpatialImage)
    except AssertionError:
        raise TypeError(err.format(obj_name))

    return


def _input_img_check(input_image, real=False):
    """
    Used to check `input_image` type and method units.
    If not real, check that the given input image is isometric.

    Parameters
    ----------
    input_image: SpatialImage
        tested input type
    real: bool, optional
        indicate if the method works on real or voxel units
    """
    from timagetk.util import clean_warning
    # - Check the `input_image` is indeed a `SpatialImage`
    try_spatial_image(input_image, obj_name='input_image')

    # - Check the isometry of the image when using voxel units:
    if not real and not input_image.is_isometric():
        warnings.formatwarning = clean_warning
        warnings.simplefilter("always")
        msg = "The image is NOT isometric, this method operates on voxels!"
        warnings.warn(msg)
    return
