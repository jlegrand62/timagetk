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
from timagetk.util import data_path
from timagetk.components import imread
from timagetk.plugins import linear_filtering, morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling, segmentation
from timagetk.algorithms.exposure import z_slice_contrast_stretch

from pylab import show, imshow, title, subplot, figure, tight_layout, suptitle
from matplotlib import gridspec

# we consider an input image
# SpatialImage instance
input_img = imread(data_path('input.tif'))

# - Get the middle z-slice:
n_zslices = input_img.shape[2]
# middle_z = int(n_zslices / 2)
middle_z = 3

# - Performs z-slice contrast stretching:
input_img = z_slice_contrast_stretch(input_img)

# - Performs Gaussian smoothing:
sigma = 1.0
smooth_img = linear_filtering(input_img, std_dev=sigma, real=False,
                              method='gaussian_smoothing')

# - Performs Closing-Opening ASF:
max_rad = 1
asf_img = morphology(smooth_img, max_radius=max_rad,
                     method='co_alternate_sequential_filter')

# - Performs the Height-transform:
h_min = 100
ext_img = h_transform(smooth_img, h=h_min, method='h_transform_min')

# - Performs local minima detection and labelling:
low_th = 1
high_th = h_min
con_img = region_labeling(ext_img, low_threshold=low_th,
                          high_threshold=high_th,
                          method='connected_components')

# ------------------------------------------------------------------------------
# - Show the effects of changing 'control' parameter in 'seeded_watershed':
# ------------------------------------------------------------------------------
from timagetk.plugins.segmentation import POSS_CONTROLS

# List possible control methods of seeded watershed algortihm:
print POSS_CONTROLS

n_col = len(POSS_CONTROLS) + 1
figure(figsize=[4 * n_col, 4])
suptitle('ZOOM-IN [20:70, 20:70]')
gs = gridspec.GridSpec(1, n_col)

subplot(gs[0, 0])
imshow(input_img[20:70, 20:70, middle_z], cmap="gray", vmin=0, vmax=2 ** 16)
title('Contrast stretched')

for n, control in enumerate(POSS_CONTROLS):
    # - Create a view of the segmented image:
    seg_img = segmentation(smooth_img, con_img, control=control,
                           method='seeded_watershed')
    subplot(gs[0, n + 1])
    imshow(seg_img[20:70, 20:70, middle_z], cmap="prism")
    title('Seeded watershed (control={})'.format(control))

tight_layout()
show()
