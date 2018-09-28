# imports
import os

from timagetk.components import SpatialImage
from timagetk.components import imread, imsave
from timagetk.util import data_path

out_path = './results/'  # to save results
if not os.path.isdir(out_path):
    new_fold = os.path.join(os.getcwd(), 'results')
    os.mkdir(new_fold)

# input image path
img_path = data_path('time_0_cut.inr')
# .inr format to SpatialImage
sp_img = imread(img_path)
# sp_img is now a SpatialImage instance

try:
    assert isinstance(sp_img, SpatialImage)
except AssertionError:
    raise TypeError("Object 'sp_img' is not a SpatialImage.")

# get the SpatialImage metadata property
metadata = sp_img.metadata  # dictionary
# print the SpatialImage metadata
print('Metadata :'), metadata
# there are two ways to modify metadata property
# 1. access the '' metadata directly
sp_img.metadata['filename'] = img_path
# print the SpatialImage metadata
print('New metadata :'), sp_img.metadata
# 2. update the whole dictionary with a new one:
sp_img.metadata = {'filename': img_path.split('/')[-1]}
# print the SpatialImage metadata
print('New metadata :'), sp_img.metadata

# get the SpatialImage voxelsize (list of floating numbers)
vox = sp_img.voxelsize
# print the SpatialImage voxelsize
print('Voxelsize :'), vox
# voxelsize modification
new_vox = []  # empty list
for ind, val in enumerate(vox):
    new_vox.append(2.0 * val)
# set the SpatialImage voxelsize property
sp_img.voxelsize = new_vox
print('New voxelsize :'), sp_img.voxelsize

# output
# filename
res_name = 'example_input_output.tif'
# SpatialImage to .tif format
imsave(out_path + res_name, sp_img)
# filename
res_name = 'example_input_output.inr'
# SpatialImage to .inr format
imsave(out_path + res_name, sp_img)
