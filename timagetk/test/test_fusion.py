# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
# -----------------------------------------------------------------------------

import unittest

from timagetk.components import imread
from timagetk.algorithms.fusion import fusion

filepath = "/home/jonathan/Projects/TissueAnalysis/timagetk/timagetk/share/data/"
files = ["time_0_cut.inr",
"time_0_cut_rotated1.mha",
"time_0_cut_rotated2.mha"]
ref_fusion_file = "time_0_cut_fused.mha"

class TestFusion(unittest.TestCase):

    def test_fusion(self):
        """
        Test fusion script.
        """
        img_list = []
        for f in files:
            print f
            img = imread(filepath+f)
            img_list.append(img)

        fus_img = fusion(img_list, 0)

        ref_fus_img = imread(filepath + ref_fusion_file)
        self.assertTrue(fus_img.equal(ref_fus_img))

    def test_fusion_manual_init(self):
        """
        Test fusion script, with a manual initialisation.
        """
        pass