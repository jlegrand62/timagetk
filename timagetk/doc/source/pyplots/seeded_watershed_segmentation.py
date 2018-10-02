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

n_col = 3
n_row = 3
figure(figsize=[4.5 * n_col, 4 * n_row])
gs = gridspec.GridSpec(n_row, n_col)

# - Create a view of the original z-slice:
subplot(gs[0, 0])
imshow(input_img[:, :, middle_z], cmap="gray", vmin=0, vmax=2 ** 16)
title("original z-slice (z {}/{})".format(middle_z, n_zslices))

# - Create a view of the z-slice contrast stretching:
input_img = z_slice_contrast_stretch(input_img)

subplot(gs[0, 1])
imshow(input_img[:, :, middle_z], cmap="gray", vmin=0, vmax=2 ** 16)
title('Contrast stretched')

# - Create a view of the smoothed image:
sigma = 1.0
smooth_img = linear_filtering(input_img, std_dev=sigma, real=False,
                              method='gaussian_smoothing')

subplot(gs[0, 2])
imshow(smooth_img[:, :, middle_z], cmap="gray", vmin=0, vmax=2 ** 16)
title('Gaussian smoothing (std_dev={})'.format(sigma))

# - Create a view of the grayscale image after Closing-Opening ASF:
max_rad = 1
asf_img = morphology(smooth_img, max_radius=max_rad,
                     method='co_alternate_sequential_filter')

subplot(gs[1, 0])
imshow(asf_img[:, :, middle_z], cmap="gray", vmin=0, vmax=2 ** 16)
title('Closing-Opening ASF (max_radius={})'.format(max_rad))

# - Create a view of the Height-transform:
h_min = 100
ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')

subplot(gs[1, 1])
imshow(ext_img[:, :, middle_z], cmap="viridis")
colorbar()
title('Height-transform (h_min={})'.format(h_min))

# - Create a view of the local minima detection and labelling:
low_th = 1
high_th = h_min
con_img = region_labeling(ext_img, low_threshold=low_th,
                          high_threshold=high_th,
                          method='connected_components')

subplot(gs[1, 2])
imshow(con_img[:, :, middle_z], cmap="prism")
title('Local minima & labelling ({}-{})'.format(low_th, high_th))

# ------------------------------------------------------------------------------
# - SEEDED WATERSHED
# ------------------------------------------------------------------------------
# - Create a view of the segmented image based on contrast stretched image:
control = 'most'
seg_img1 = segmentation(input_img, con_img, control=control,
                        method='seeded_watershed')

subplot(gs[2, 0])
imshow(seg_img1[:, :, middle_z], cmap="prism")
title('Watershed (contrast, {})'.format(control))

# - Create a view of the segmented image based on smoothed image:
seg_img2 = segmentation(smooth_img, con_img, control=control,
                        method='seeded_watershed')

subplot(gs[2, 1])
imshow(seg_img2[:, :, middle_z], cmap="prism")
title('Watershed (smoothed, {})'.format(control))

# - Create a view of the segmented image based on C/O ASF image:
seg_img3 = segmentation(asf_img, con_img, control=control,
                        method='seeded_watershed')

subplot(gs[2, 2])
imshow(seg_img3[:, :, middle_z], cmap="prism")
title('Watershed (co-asf, {})'.format(control))

tight_layout()
show()
