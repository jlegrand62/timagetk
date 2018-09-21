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
