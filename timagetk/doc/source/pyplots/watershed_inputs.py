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
This exemple shows how to performs seeded watershed segmentation on grey level images.
"""
import numpy as np
from timagetk.util import data_path
from timagetk.components import imread
from timagetk.plugins import linear_filtering, morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling, segmentation
from timagetk.algorithms.exposure import z_slice_contrast_stretch

from pylab import show, imshow, title, subplot, figure, tight_layout, suptitle, \
    colorbar
from matplotlib import gridspec

# we consider an input image
# SpatialImage instance
input_img = imread(data_path('input.tif'))

# - Get the middle z-slice:
n_zslices = input_img.shape[2]
# middle_z = int(n_zslices / 2)
middle_z = 3

# - We take only one slice to be sure to have the seeds in this "plane":
input_img = input_img.get_z_slice(middle_z)

# - SEED extraction:
# - Contrast stretching of the original image:
contrast_img = z_slice_contrast_stretch(input_img.to_3D())
contrast_img = contrast_img.to_2D()

# - Smoothing the contrasted image:
sigma = 1.0
smooth_img = linear_filtering(contrast_img, std_dev=sigma, real=False,
                              method='gaussian_smoothing')

# - Perform height-transform:
h_min = 100
ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')

# - Performs local minima detection and labelling:
low_th = 1
high_th = h_min
con_img = region_labeling(ext_img, low_threshold=low_th,
                          high_threshold=high_th,
                          method='connected_components')
wat_control = "most"
n_col = 3
n_row = 5
figure(figsize=[4 * n_col, 4 * n_row])
gs = gridspec.GridSpec(n_row, n_col)

# ------------------------------------------------------------------------------
# - ORIGINAL IMAGE input:
# ------------------------------------------------------------------------------
# - Create a view of the original z-slice:
subplot(gs[0, 0])
imshow(input_img[20:70, 20:70], cmap="gray", vmin=0, vmax=2 ** 16)
title("original z-slice (z {}/{})".format(middle_z, n_zslices))

# - Create a view of the local minima detection and labelling:
subplot(gs[0, 1])
imshow(con_img[20:70, 20:70], cmap="prism")
title('Local minima & labelling ({}-{})'.format(low_th, high_th))

# - Create a view of the segmented image:
seg_img = segmentation(input_img, con_img, control=wat_control,
                       method='seeded_watershed')
subplot(gs[0, 2])
imshow(seg_img[20:70, 20:70], cmap="prism")
title('Seeded watershed (control={})'.format(wat_control))

# ------------------------------------------------------------------------------
# - CONTRASTED IMAGE input:
# ------------------------------------------------------------------------------
# - Create a view of the z-slice contrast stretching:
subplot(gs[1, 0])
imshow(contrast_img[20:70, 20:70], cmap="gray", vmin=0, vmax=2 ** 16)
title('Contrast stretched')

# - Create a view of the local minima detection and labelling:
subplot(gs[1, 1])
imshow(con_img[20:70, 20:70], cmap="prism")
title('Local minima & labelling ({}-{})'.format(low_th, high_th))

# - Create a view of the segmented image:
seg_img = segmentation(contrast_img, con_img, control=wat_control,
                       method='seeded_watershed')
subplot(gs[1, 2])
imshow(seg_img[20:70, 20:70], cmap="prism")
title('Seeded watershed (control={})'.format(wat_control))

# ------------------------------------------------------------------------------
# - ORIGINAL SMOOTHED IMAGE input:
# ------------------------------------------------------------------------------
# - Create a view of the smoothed image:
subplot(gs[2, 0])
imshow(smooth_img[20:70, 20:70], cmap="gray", vmin=0, vmax=2 ** 16)
title('Gaussian smoothing (std_dev={})'.format(sigma))

# - Create a view of the local minima detection and labelling:
subplot(gs[2, 1])
imshow(con_img[20:70, 20:70], cmap="prism")
title('Local minima & labelling ({}-{})'.format(low_th, high_th))

# - Create a view of the segmented image:
seg_img = segmentation(smooth_img, con_img, control=wat_control,
                       method='seeded_watershed')
subplot(gs[2, 2])
imshow(seg_img[20:70, 20:70], cmap="prism")
title('Seeded watershed (control={})'.format(wat_control))

# ------------------------------------------------------------------------------
# - CONTRASTED & SMOOTHED IMAGE input:
# ------------------------------------------------------------------------------
# - Create a view of the smoothed image:
sigma = 1.0
smooth_img = linear_filtering(contrast_img, std_dev=sigma, real=False,
                              method='gaussian_smoothing')
subplot(gs[3, 0])
imshow(smooth_img[20:70, 20:70], cmap="gray", vmin=0, vmax=2 ** 16)
title('Contrasted & smoothed')

# - Create a view of the local minima detection and labelling:
subplot(gs[3, 1])
imshow(con_img[20:70, 20:70], cmap="prism")
title('Local minima & labelling ({}-{})'.format(low_th, high_th))

# - Create a view of the segmented image:
seg_img = segmentation(smooth_img, con_img, control=wat_control,
                       method='seeded_watershed')
subplot(gs[3, 2])
imshow(seg_img[20:70, 20:70], cmap="prism")
title('Seeded watershed (control={})'.format(wat_control))

# ------------------------------------------------------------------------------
# - CONTRASTED & SMOOTHED & CLOSING/OPENING ASF IMAGE input:
# ------------------------------------------------------------------------------
# - Create a view of the grayscale image after Closing-Opening ASF:
max_rad = 1
asf_img = morphology(smooth_img, max_radius=max_rad,
                     method='co_alternate_sequential_filter')
subplot(gs[4, 0])
imshow(smooth_img[20:70, 20:70], cmap="gray", vmin=0, vmax=2 ** 16)
title('Constrast & smooth & co-asf')

# - Create a view of the local minima detection and labelling:
subplot(gs[4, 1])
imshow(con_img[20:70, 20:70], cmap="prism")
title('Local minima & labelling ({}-{})'.format(low_th, high_th))

# - Create a view of the segmented image:
seg_img = segmentation(asf_img, con_img, control=wat_control,
                       method='seeded_watershed')
subplot(gs[4, 2])
imshow(seg_img[20:70, 20:70], cmap="prism")
title('Seeded watershed (control={})'.format(wat_control))

tight_layout()
show()
