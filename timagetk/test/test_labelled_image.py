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
import numpy as np

from timagetk.components import LabelledImage

array_2D = np.array([[1, 2, 7, 7, 1, 1],
                     [1, 6, 5, 7, 3, 3],
                     [2, 2, 1, 7, 3, 3],
                     [1, 1, 1, 4, 1, 1]])


class TestLabelledImage(unittest.TestCase):

    def test_init(self):
        """
        Test correct initialisation of 'background' & 'no_label_id' properties.
        """
        back_id = 1
        no_id = 0

        lab_im_2D = LabelledImage(array_2D)
        # 'background' & 'no_label_id' should be None
        self.assertIsNone(lab_im_2D.background)
        self.assertIsNone(lab_im_2D.no_label_id)

        lab_im_2D = LabelledImage(array_2D, background=back_id)
        # 'background' & 'no_label_id' should be 1 and None
        self.assertEqual(lab_im_2D.background, back_id)
        self.assertIsNone(lab_im_2D.no_label_id)

        lab_im_2D = LabelledImage(array_2D, background=back_id,
                                  no_label_id=no_id)
        # 'background' & 'no_label_id' should be 1 and 0
        self.assertEqual(lab_im_2D.background, back_id)
        self.assertEqual(lab_im_2D.no_label_id, no_id)

    def test_labels(self):
        """
        Test the labels related methods.
        """
        uniq = list(np.unique(array_2D))
        back_id = 1
        no_id = 0

        lab_im_2D = LabelledImage(array_2D)
        self.assertEqual(lab_im_2D.labels(), uniq)
        self.assertEqual(lab_im_2D.nb_labels(), len(uniq))
        self.assertTrue(lab_im_2D.is_label_in_image(2))
        self.assertFalse(lab_im_2D.is_label_in_image(no_id))

        uniq = list(set(uniq) - {back_id})
        lab_im_2D = LabelledImage(array_2D, background=back_id)
        self.assertEqual(lab_im_2D.labels(), uniq)
        self.assertEqual(lab_im_2D.nb_labels(), len(uniq))
        self.assertTrue(lab_im_2D.is_label_in_image(2))
        self.assertFalse(lab_im_2D.is_label_in_image(no_id))

        lab_im_2D = LabelledImage(array_2D, background=back_id,
                                  no_label_id=no_id)
        self.assertEqual(lab_im_2D.labels(), uniq)
        self.assertEqual(lab_im_2D.nb_labels(), len(uniq))
        self.assertTrue(lab_im_2D.is_label_in_image(2))
        self.assertFalse(lab_im_2D.is_label_in_image(no_id))

    def test_remove_labels_from_image(self):
        """
        Test label deletion.
        """
        back_id = 1
        no_id = 0
        lab_im_2D = LabelledImage(array_2D, background=back_id,
                                  no_label_id=no_id)
        # - Replace 2 by 0:
        lab_im_2D.remove_labels_from_image([2], verbose=False)
        self.assertFalse(lab_im_2D.is_label_in_image(2))
        self.assertTrue(lab_im_2D.is_label_in_image(no_id))
        self.assertFalse(2 in lab_im_2D.labels())
