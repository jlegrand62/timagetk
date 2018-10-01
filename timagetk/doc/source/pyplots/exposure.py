from timagetk.util import data_path
from timagetk.util import slice_n_hist
from timagetk.components import imread
from timagetk.algorithms.exposure import z_slice_contrast_stretch
from timagetk.algorithms.exposure import z_slice_equalize_adapthist
from pylab import show, imshow, title, tight_layout, subplot
from matplotlib import gridspec

# input image path
img_path = data_path('time_0_cut.inr')
# .inr format to SpatialImage
sp_img = imread(img_path)

middle_z = int(sp_img.shape[2] / 2)

# - Performs z-slice by z-slice contrast stretching:
st_img = z_slice_contrast_stretch(sp_img)
# - Performs z-slice by z-slice adaptative histogram equalization:
eq_img = z_slice_equalize_adapthist(sp_img)

plt.figure(figsize=[3.5 * 3, 4])
gs = gridspec.GridSpec(1, 3)
# - Create a view of the original z-slice and the associated histograms:
ax = subplot(gs[0, 0])
imshow(sp_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
title("original z-slice")
tight_layout()
ax = subplot(gs[0, 1])
imshow(st_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
title("contrast stretched")
tight_layout()
ax = subplot(gs[0, 2])
imshow(eq_img[:, :, middle_z], cmap="gray", vmin=0, vmax=255)
title("adaptative equalization")
tight_layout()
show()
