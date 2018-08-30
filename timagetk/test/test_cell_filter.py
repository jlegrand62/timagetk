# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------

import unittest
import numpy as np

try:
    from timagetk.util import data_path
    from timagetk.components import imread
    from timagetk.algorithms import cell_filter
    from timagetk.plugins.cell_filtering import cell_filtering
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))


class TestCellFilter(unittest.TestCase):

    def test_erosion(self):
        """
        Tests segmented image erosion using wrapped MORPHEME library VT.
        """
        sp_img_ref = imread(data_path('eroded_segmentation.mha'))
        sp_img = imread(data_path('segmentation_seeded_watershed.inr'))
        # ./cellfilter segmentation_seeded_watershed.inr eroded_segmentation.mha -erosion -sphere -radius 1 -iterations 1
        output = cell_filter(sp_img, param_str_2='-erosion -R 1')
        np.testing.assert_array_equal(output, sp_img_ref)

    def test_dilation(self):
        """
        Tests segmented image dilation using wrapped MORPHEME library VT.
        """
        sp_img_ref = imread(data_path('dilated_eroded_segmentation.mha'))
        sp_img = imread(data_path('eroded_segmentation.mha'))
        # ./cellfilter segmentation_seeded_watershed.inr dilated_eroded_segmentation.mha -dilation -sphere -radius 1 -iterations 1
        output = cell_filter(sp_img, param_str_2='-dilation -R 1')
        np.testing.assert_array_equal(output, sp_img_ref)

    def test_plugin_erosion(self):
        """
        Tests segmented image erosion using cell_filtering plugin.
        """
        sp_img_ref = imread(data_path('eroded_segmentation.mha'))
        sp_img = imread(data_path('segmentation_seeded_watershed.inr'))
        output = cell_filtering(sp_img, method='erosion')
        np.testing.assert_array_equal(output, sp_img_ref)

    def test_plugin_dilation(self):
        """
        Tests segmented image dilation using cell_filtering plugin.
        """
        sp_img_ref = imread(data_path('dilated_eroded_segmentation.mha'))
        sp_img = imread(data_path('eroded_segmentation.mha'))
        output = cell_filtering(sp_img, method='dilation')
        np.testing.assert_array_equal(output, sp_img_ref)
