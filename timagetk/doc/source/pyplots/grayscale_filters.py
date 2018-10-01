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

# Make image isometric: most methods assumes image isometry
sp_img = isometric_resampling(sp_img)

# - Get the middle z-slice:
n_zslices = sp_img.shape[2]
middle_z = int(n_zslices / 2)

# get list of defined linear filtering methods:
linear_methods = list_linear_methods()

# Need to define the standard deviation to apply for Gaussian smoothing
sigma = 1

# Define the display grid
n_fig = len(linear_methods) + 1
n_col = 3
n_row = int(np.ceil(n_fig / float(n_col)))
figure(figsize=[3.5 * n_col, 5 * n_row])
gs = gridspec.GridSpec(n_row, n_col)

ax = subplot(gs[0, 0])
# Get the middle z-slice of the image and display it
imshow(sp_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
title("original z-slice (z {}/{})".format(middle_z, n_zslices))

i_row = 0
n_fig_row = 1
for n, method in enumerate(linear_methods):
    print i_row, n_fig_row
    ax = subplot(gs[i_row, n_fig_row])
    # 'std_dev' keyword argument is only used by method='gaussian_smoothing':
    filter_img = linear_filtering(sp_img, method, std_dev=sigma)
    # Get the middle z-slice of the filtered image and display it
    imshow(filter_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
    title(method)
    # update coordinate at wich the next plot should be (in the grid)
    if n_fig_row == n_col - 1:
        i_row += 1
        n_fig_row = 0
    else:
        n_fig_row += 1

tight_layout()
show()
