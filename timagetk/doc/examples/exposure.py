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
These exemples shows the effect of intensity rescaling on grey level images.
"""

from timagetk.components import imread
from timagetk.util import data_path

from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.exposure import z_slice_equalize_adapthist
import matplotlib.pyplot as plt
from timagetk.util import slice_n_hist

# - Example input image path
img_path = data_path('time_0_cut.inr')
# - Load the image:
sp_img = imread(img_path)
# - Get the middle z-slice:
n_zslices = sp_img.shape[2]
middle_z = int(n_zslices / 2)

# - Create a view of the original z-slice and the associated histograms:
slice_n_hist(sp_img[:, :, middle_z],
             "original z-slice (z {}/{})".format(middle_z, n_zslices),
             'time_0_cut.inr')
# - Display interactive maptlotlib viewer:
plt.show()

# - Performs z-slice by z-slice contrast stretching:
st_img = z_slice_contrast_stretch(sp_img)
# - Create a view of the contrasted z-slice and the associated histograms:
slice_n_hist(st_img[:, :, middle_z],
             "z-slice contrast stretching (z {}/{})".format(middle_z,
                                                            n_zslices),
             'time_0_cut.inr')
# - Display interactive maptlotlib viewer:
plt.show()

# - Performs z-slice by z-slice adaptative histogram equalization:
eq_img = z_slice_equalize_adapthist(sp_img)
# - Create a view of the equalized z-slice and the associated histograms:
slice_n_hist(eq_img[:, :, middle_z],
             "z-slice adaptative histogram equalization (z {}/{})".format(
                 middle_z, n_zslices), 'time_0_cut.inr')
# - Display interactive maptlotlib viewer:
plt.show()
