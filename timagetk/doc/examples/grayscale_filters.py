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
These exemples shows the effect of some filters on grayscale images.
"""
import numpy as np
from timagetk.components import imread
from timagetk.util import data_path
from timagetk.algorithms.resample import isometric_resampling
from timagetk.plugins.linear_filtering import linear_filtering
from timagetk.plugins.linear_filtering import list_linear_methods

from pylab import show, imshow, title, subplot, figure, tight_layout
from matplotlib import gridspec

# - Example input image path
img_path = data_path('time_0_cut.inr')
# - Load the image:
sp_img = imread(img_path)

# - Make the grayscale image isometric:
# gaussian smoothing methods assumes image isometry
# (by applying the same parameter 'std_dev' in every direction)
sp_img = isometric_resampling(sp_img)

# - Get the middle z-slice:
n_zslices = sp_img.shape[2]
middle_z = int(n_zslices / 2)

# - To get the list of defined linear filtering methods:
linear_methods = list_linear_methods()
print linear_methods

# Define list of increasing standard deviation to apply on images:
sigmas = range(1, 4)
n_fig = len(sigmas) + 1  # don't forget original image

# - We create a grid to assemble all figures:
figure(figsize=[3.5 * n_fig, 4])
gs = gridspec.GridSpec(1, n_fig)

# - Get the middle z-slice of the original grayscale image and add it as first image:
subplot(gs[0, 0])
imshow(sp_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
title("original z-slice (z {}/{})".format(middle_z, n_zslices))

# - Gaussian smoothing example with increasing standard deviation:
for n, sigma in enumerate(sigmas):
    # Define where to add the image in the plot:
    subplot(gs[0, n + 1])
    # Perform gaussian smoothing:
    gauss_filter_img = linear_filtering(sp_img, 'gaussian_smoothing',
                                        std_dev=sigma)
    imshow(gauss_filter_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
    title('gaussian_smoothing (std_dev={})'.format(sigma))

# - Display the images:
tight_layout()
show()
