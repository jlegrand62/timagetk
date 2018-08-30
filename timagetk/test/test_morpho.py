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
    from timagetk.algorithms import morpho
    from timagetk.plugins import morphology
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))


class TestMorpho(unittest.TestCase):

    def test_dilation(self):
        """
        Tests grayscale image dilation using wrapped MORPHEME library VT.
        """
        im = imread(data_path('filtering_src.inr'))
        im_ref = imread(data_path('morpho_dilation_default.inr.gz'))
        # $ ./morpho filtering_src.inr morpho_dilation_default.inr.gz -dilation -sphere -radius 1 -iterations 1
        output = morpho(im, param_str_2='-dilation')
        np.testing.assert_array_equal(output, im_ref)

    def test_erosion(self):
        """
        Tests grayscale image erosion using wrapped MORPHEME library VT.
        """
        im = imread(data_path('filtering_src.inr'))
        im_ref = imread(data_path('morpho_erosion_default.inr.gz'))
        # $ ./morpho filtering_src.inr morpho_dilation_default.inr.gz -erosion -sphere -radius 1 -iterations 1
        output = morpho(im, param_str_2='-erosion')
        np.testing.assert_array_equal(output, im_ref)

    def test_plugin_dilation(self):
        """
        Tests grayscale image dilation using morphology plugin.
        """
        im = imread(data_path('filtering_src.inr'))
        im_ref = imread(data_path('morpho_dilation_default.inr.gz'))
        output = morphology(im, method='dilation')
        np.testing.assert_array_equal(output, im_ref)

    def test_plugin_erosion(self):
        """
        Tests grayscale image erosion using morphology plugin.
        """
        im = imread(data_path('filtering_src.inr'))
        im_ref = imread(data_path('morpho_erosion_default.inr.gz'))
        output = morphology(im, method='erosion')
        np.testing.assert_array_equal(output, im_ref)
