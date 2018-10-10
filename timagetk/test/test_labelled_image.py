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

from timagetk.components.labelled_image import LabelledImage

array_2D = np.array([[1, 2, 7, 7, 1, 1],
                     [1, 6, 5, 7, 3, 3],
                     [2, 2, 1, 7, 3, 3],
                     [1, 1, 1, 4, 1, 1]])

# used to set 'no_label_id'
no_id = 0
# list of unique labels found in the array
uniq = list(np.unique(array_2D))
# we invert the label of two neighbors (error prone!)
mapping = {3: 7,
           7: 3}


class TestLabelledImage(unittest.TestCase):

    def test_init(self):
        """
        Test correct initialisation of 'no_label_id' property.
        """
        # - Test init of 'no_label_id' to None:
        lab_im_2D = LabelledImage(array_2D)
        self.assertIsNone(lab_im_2D.no_label_id)

        # - Test init of 'no_label_id' to 'no_id':
        lab_im_2D = LabelledImage(array_2D, no_label_id=no_id)
        self.assertEqual(lab_im_2D.no_label_id, no_id)

        # - Test init of 'no_label_id' to value obtained from a LabelledImage as input:
        lab_im_2D = LabelledImage(lab_im_2D)
        self.assertEqual(lab_im_2D.no_label_id, no_id)

    def test_labels(self):
        """
        Test the labels related methods.
        """
        lab_im_2D = LabelledImage(array_2D, no_label_id=no_id)
        self.assertItemsEqual(lab_im_2D.labels(), uniq)
        self.assertEqual(lab_im_2D.nb_labels(), len(uniq))
        self.assertTrue(lab_im_2D.is_label_in_image(2))
        self.assertFalse(lab_im_2D.is_label_in_image(no_id))

    def test_remove_labels_from_image(self):
        """
        Test label deletion.
        """
        lab_im_2D = LabelledImage(array_2D, no_label_id=no_id)
        # - Replace 2 by 0:
        lab_im_2D.remove_labels_from_image([2], verbose=False)
        # - Check we did not changed the object class (a LabelledImage):
        self.assertTrue(isinstance(lab_im_2D, LabelledImage))
        # - Check we have removed '2' from the array:
        self.assertFalse(2 in lab_im_2D.get_array())
        # - Two ways of checking '2' have been removed from labels attribute:
        self.assertFalse(lab_im_2D.is_label_in_image(2))
        self.assertFalse(2 in lab_im_2D.labels())
        # - Check the 'no_label_id' is now in the array:
        self.assertTrue(lab_im_2D.is_label_in_image(no_id))

    def test_relabelling_1(self):
        """
        Test relabelling method WITHOUT CLEARING UNMAPPED LABELS.
        """
        # - Manual definition of the relabelled array with 'clear_unmapped=False':
        relab = np.array([[1, 2, 3, 3, 1, 1],
                          [1, 6, 5, 3, 7, 7],
                          [2, 2, 1, 3, 7, 7],
                          [1, 1, 1, 4, 1, 1]])

        lab_im_2D = LabelledImage(array_2D, no_label_id=no_id)
        # - Relabel the array using the mapping:
        lab_im_2D.relabel_from_mapping(mapping, clear_unmapped=False)
        # --------
        # - TESTS:
        # --------
        # - Check the relabelling is correct:
        assert lab_im_2D.equal_array(LabelledImage(relab, no_label_id=no_id))
        # - We should have the same labels list:
        self.assertItemsEqual(lab_im_2D.labels(), uniq)

    def test_relabelling_2(self):
        """
        Test relabelling method WITH CLEARING UNMAPPED LABELS.
        """
        # - Manual definition of the relabelled array with 'clear_unmapped=True':
        relab = np.array([[0, 0, 3, 3, 0, 0],
                          [0, 0, 0, 3, 7, 7],
                          [0, 0, 0, 3, 7, 7],
                          [0, 0, 0, 0, 0, 0]])

        lab_im_2D = LabelledImage(array_2D, no_label_id=no_id)
        # - Relabel the array using the mapping, WITH CLEARING:
        lab_im_2D.relabel_from_mapping(mapping, clear_unmapped=True)
        # --------
        # - TESTS:
        # --------
        # - Check the relabelling is correct:
        assert lab_im_2D.equal_array(LabelledImage(relab, no_label_id=no_id))
        # - We should have only the mapping values:
        self.assertItemsEqual(lab_im_2D.labels(), mapping.values())
