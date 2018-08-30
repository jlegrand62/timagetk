# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
# -----------------------------------------------------------------------------

__license__ = "Private/UnderDev"
__revision__ = " $Id$ "

import time
import numpy as np
import scipy.ndimage as nd

from timagetk.util import elapsed_time
from timagetk.util import percent_progress

from timagetk.components import SpatialImage
from timagetk.components import imread
from timagetk.components.slices import dilation
from timagetk.components.slices import real_indices

__all__ = ['LabelledImage']


# ------------------------------------------------------------------------------
#
# Morphology functions, array based (not to use with VT algorithms):
#
# ------------------------------------------------------------------------------
def connectivity_4():
    """
    Create a 2D structuring element (array) of radius 1 with a 4-neighborhood.

    Returns
    -------
    np.array
        a boolean numpy.array defining the 2D structuring element
    """
    return nd.generate_binary_structure(2, 1)


def connectivity_6():
    """
    Create a 3D structuring element (array) of radius 1 with a 6-neighborhood.

    Returns
    -------
    np.array
        a boolean numpy.array defining the 3D structuring element
    """
    return nd.generate_binary_structure(3, 1)


def connectivity_8():
    """
    Create a 2D structuring element (array) of radius 1 with a 8-neighborhood.

    Returns
    -------
    np.array
        a boolean numpy.array defining the 2D structuring element
    """
    return nd.generate_binary_structure(2, 2)


def connectivity_18():
    """
   Create a 3D structuring element (array) of radius 1 with a 18-neighborhood.

   Returns
   -------
   np.array
       a boolean numpy.array defining the 3D structuring element
   """
    return nd.generate_binary_structure(3, 2)


def connectivity_26():
    """
    Create a 3D structuring element (array) of radius 1 with a 26-neighborhood.

    Returns
    -------
    np.array
        a boolean numpy.array defining the 3D structuring element
    """
    return nd.generate_binary_structure(3, 3)


def structuring_element(connectivity=26):
    """
    Create a structuring element (array)
    Connectivity is among the 4-, 6-, 8-, 18-, 26-neighborhoods.
    4 and 8 are 2-D elements, the others being 3-D (default = 26).

    Parameters
    ----------
    connectivity: int, optional
        connectivity or neighborhood of the structuring element, default 26

    Returns
    -------
    np.array
        a boolean numpy.array defining the required structuring element
    """
    assert connectivity in [4, 6, 8, 18, 26]
    if connectivity == 4:
        struct = connectivity_4()
    elif connectivity == 6:
        struct = connectivity_6()
    elif connectivity == 8:
        struct = connectivity_8()
    elif connectivity == 18:
        struct = connectivity_18()
    else:
        struct = connectivity_26()
    return struct


def default_structuring_element2d():
    """
    Default 2D structuring element.
    This is a square (6-connectivity) of radius 1.
    """
    return connectivity_6()


def default_structuring_element3d():
    """
    Default 3D structuring element.
    This is a cube (26-connectivity) of radius 1.
    """
    return connectivity_26()


# ------------------------------------------------------------------------------
#
# LABEL based functions:
#
# ------------------------------------------------------------------------------
def label_inner_wall(labelled_img, label_id, struct=None, connectivity_order=1):
    """
    Detect inner-wall position of a given 'label_id' within a segmented image
    'labelled_img'.

    Parameters
    ----------
    labelled_img: np.array|LabelledImage
        a labelled image containing 'label_id'
    label_id: int
        label to use for outer-wall detection
    struct: np.array, optional
        a binary structure to use for erosion
    connectivity_order: int, optional
        connectivity order determines which elements of the output array belong to the
        structure, i.e. are considered as neighbors of the central element.
        Elements up to a squared distance of connectivity from the center are
        considered neighbors, thus it may range from 1 (no diagonal elements are
        neighbors) to rank (all elements are neighbors), with rank the number of
        dimensions of the image.

    Returns
    -------
    np.array|LabelledImage: a labelled array with only the inner-wall position
    as non-null value
    """
    if struct is None:
        rank = labelled_img.ndim
        struct = nd.generate_binary_structure(rank, connectivity_order)
    # Create boolean mask of 'label_id' position in the image
    mask_img = labelled_img == label_id
    # Binary dilation of the mask
    er_mask_img = nd.binary_erosion(mask_img, structure=struct)
    # Define a mask giving outer-wall position for 'label_id'
    inner_wall = mask_img ^ er_mask_img
    # return the labelled array with only the outer-wall position:
    return labelled_img[inner_wall]


def label_outer_wall(labelled_img, label_id, struct=None, connectivity_order=1):
    """
    Detect outer-wall position of a given 'label_id' within a segmented image
    'labelled_img'.

    Parameters
    ----------
    labelled_img: np.array|LabelledImage
        a labelled image containing 'label_id'
    label_id: int
        label to use for outer-wall detection
    struct: np.array, optional
        a binary structure to use for dilation
    connectivity_order: int, optional
        connectivity order determines which elements of the output array belong to the
        structure, i.e. are considered as neighbors of the central element.
        Elements up to a squared distance of connectivity from the center are
        considered neighbors, thus it may range from 1 (no diagonal elements are
        neighbors) to rank (all elements are neighbors), with rank the number of
        dimensions of the image.

    Returns
    -------
    np.array|LabelledImage: a labelled array with only the outer-wall position
    as non-null value
    """
    if struct is None:
        dim = labelled_img.ndim
        struct = nd.generate_binary_structure(dim, connectivity_order)
    # Create boolean mask of 'label_id' position in the image
    mask_img = labelled_img == label_id
    # Binary dilation of the mask
    dil_mask_img = nd.binary_dilation(mask_img, structure=struct)
    # Define a mask giving outer-wall position for 'label_id'
    outer_wall = dil_mask_img ^ mask_img
    # return the labelled array with only the outer-wall position:
    return labelled_img[outer_wall]


def neighbors_from_image(labelled_img, label_id):
    """
    List neighbors of 'label_id' in 'labelled_img' as found in its outer-wall.

    Parameters
    ----------
    labelled_img: np.array|LabelledImage
        a labelled image containing 'label_id'
    label_id: int|list(int)
        label to use for neighbors detection

    Returns
    -------
    list: neighbors label of 'label_id'
    """
    if isinstance(label_id, int):
        # Get outer-wall position & return unique list of labels:
        return list(np.unique(label_outer_wall(labelled_img, label_id)))
    elif isinstance(label_id, list):
        return {l: list(np.unique(label_outer_wall(labelled_img, l))) for l in
                label_id}
    else:
        raise TypeError(
            "Parameter 'label_id' should be an integer or a list, got '{}'.".format(
                type(label_id)))


# ------------------------------------------------------------------------------
#
# WHOLE LABELLED IMAGE functions:
#
# ------------------------------------------------------------------------------
def image_with_labels(image, labels, boundingbox, erase_value=0):
    """
    Create a new image containing only the given labels.
    Use image as template (shape, origin, voxelsize & metadata).

    Parameters
    ----------
    image: SpatialImage
        labelled spatial image to use as template for labels extraction
    labels: list
        list of labels to keep in the image
    boundingbox: dict
        dictionary of labels' boundingbox
    erase_value: int, optional
        value use to use in place of discarded labels

    Returns
    -------
    template_im: SpatialImage
        the image containing only the given 'labels'
    """
    print "Initialising empty template image..."
    if erase_value == 0:
        template_im = np.zeros_like(image.get_array())
    else:
        template_im = np.ones_like(image.get_array()) * erase_value

    # - Now we can add 'labels' using bounding boxes to speed-up computation:
    nb_labels = len(labels)
    array = image.get_array()
    ndim = image.get_dim()
    no_bbox = []
    progress = 0
    print "Adding {} labels to the empty template image...".format(nb_labels)
    for n, label in enumerate(labels):
        progress = percent_progress(progress, n, nb_labels)
        try:
            bbox = boundingbox[label]
            xyz = np.array(np.where((array[bbox]) == label)).T
            xyz = tuple([xyz[:, n] + bbox[n].start for n in range(ndim)])
            template_im[xyz] = label
        except ValueError:
            no_bbox.append(label)
            template_im[array == label] = label

    # - If some boundingbox were missing, print about it:
    if no_bbox:
        n = len(no_bbox)
        print "Could not find boundingbox for {} labels: {}".format(n, no_bbox)

    return SpatialImage(template_im, voxelsize=image.voxelsize,
                        origin=image.origin,
                        metadata_dict=image.metadata)


def image_without_labels(image, labels, boundingbox, erase_value=0):
    """
    Create a new image without the given labels.
    Use self.image as template (shape, origin, voxelsize & metadata).

    Parameters
    ----------
    image: SpatialImage
        labelled spatial image to use as template for labels deletion
    labels: list
        list of labels to remove from the image
    boundingbox: dict
        dictionary of labels' boundingbox
    erase_value: int, optional
        value use to use in place of discarded labels

    Returns
    -------
    template_im: SpatialImage
        an image without the given 'labels'
    """
    print "Initialising template image from self.image..."
    template_im = image.get_array()
    # - Now we can add 'labels' using bounding boxes to speed-up computation:
    nb_labels = len(labels)
    no_bbox = []
    progress = 0
    print "Removing the {} labels from the template image...".format(
        nb_labels)
    for n, label in enumerate(labels):
        progress = percent_progress(progress, n, nb_labels)
        # Try to get the label's boundingbox::
        try:
            bbox = boundingbox[label]
        except KeyError:
            no_bbox.append(label)
            bbox = None
        # Performs value replacement:
        template_im = array_replace_label(template_im, label, erase_value, bbox)

    # - If some boundingbox were missing, print about it:
    if no_bbox:
        n = len(no_bbox)
        print "Could not find boundingbox for {} labels: {}".format(n, no_bbox)

    return SpatialImage(template_im, voxelsize=image.voxelsize,
                        origin=image.origin,
                        metadata_dict=image.metadata)


def array_replace_label(array, label, new_label, bbox=None):
    """
    Replace a label by a new one in a numpy array, may use boundingbox if
    provided.

    Parameters
    ----------
    array: np.array
        labelled array with integer values
    label: int
        label value to replace
    new_label: int
        label value to use for replacement
    bbox: tuple(slice), optional
        tuple of slice indicating the location of the label within the image

    Returns
    -------
    array: np.array
        the modified array
    """
    if bbox is not None:
        xyz = np.array(np.where((array[bbox]) == label)).T
        xyz = tuple([xyz[:, n] + bbox[n].start for n in range(array.ndim)])
        array[xyz] = new_label
    else:
        array[array == label] = new_label
    return array


# - GLOBAL VARIABLES:
MISS_LABEL = "The following label{} {} not found in the image: {}"  # ''/'s'; 'is'/'are'; labels


class AbstractLabelledImage(SpatialImage):
    """
    This class allows to manipulate labelled SpatialImage, eg. segmented image
    of a biological tissue where each cell has a different label or id.
    """

    def __init__(self, image, background=None, no_label_id=None, **kwargs):
        """
        Parameters
        ----------
        image: SpatialImage
            the SpatialImage containing the labelled tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        no_label_id: int, optional
            if given define the "unknown label" (ie. not a cell)
        """
        # - Get variable for SpatialImage initialisation:
        input_array = image.get_array()
        origin = image.origin
        voxelsize = image.voxelsize
        dtype = image.type
        metadata_dict = image.metadata
        # - Inherit SpatialImage class:
        SpatialImage.__init__(self, input_array, origin=origin,
                              voxelsize=voxelsize,
                              dtype=dtype, metadata_dict=metadata_dict)
        # - Initializing EMPTY hidden attributes:
        # -- Useful label lists:
        self._labels = None  # list of image-labels referring to cell-labels
        self._cell_layer1 = None  # contain the list of L1 cells
        self._cell_layer2 = None  # contain the list of L2 cells

        # -- Useful & recurrent label properties:
        self._bbox = []  # list of bounding boxes (indexed 'self._labels - 1')
        self._bbox_dict = {}  # dict of bounding boxes
        self._neighbors = {}  # unfiltered neighborhood label-dict {vid_i: neighbors(vid_i)}

        # - Define object hidden attributes:
        # -- Define the background value, if any (can be None):
        self._background_id = background
        if self._background_id is not None:
            print "The background position is defined by value '{}'!".format(
                self._background_id)

        # -- Define the "no_label_id" value, if any (can be None):
        self._no_label_id = no_label_id
        if self._no_label_id is not None:
            print "The absence of label is defined by value '{}'!".format(
                self._no_label_id)

        # - Compute useful & recurrent label properties
        # -- Get the list of labels found in the image:
        self._labels = self.labels()
        # -- Get the boundingbox dictionary:
        self.boundingbox()

    @property
    def background(self):
        """
        Get the background label, can be None.

        Returns
        -------
        background: int
            the label value for the background
        """
        if self._background_id is None:
            print "WARNING: no value defined for the background id!"
        return self._background_id

    @background.setter
    def background(self, label):
        """
        Set the background label.

        Parameters
        ----------
        label: int
            integer defining the background id in the image.
        """
        if not isinstance(label, int):
            print "Provided label '{}' is not an integer!".format(label)
            return
        elif label not in self._labels:
            print "Provided label '{}' is not found in the image!".format(label)
            return
        else:
            self._background_id = label

    @property
    def no_label_id(self):
        """
        Get the value associated to no label.
        This is used as "unknown label" or "erase value".

        Returns
        -------
        no_label_id: int
            the value defined as the "no_label_id"
        """
        if self._no_label_id is None:
            print "WARNING: no value defined for the 'no label' id!"
        return self._no_label_id

    @no_label_id.setter
    def no_label_id(self, value):
        """
        Set the value of the label indicating "unknown" label.

        Parameters
        ----------
        value: int
            value to be defined as the "no_label_id"
        """
        if not isinstance(value, int):
            print "Provided value '{}' is not an integer!".format(value)
            return
        else:
            self._no_label_id = value

    def labels(self, labels=None):
        """
        Get the list of labels found in the image, or make sure provided labels
        exists.

        Parameters
        ----------
        labels: None|list

        Returns
        -------
        labels: list
            list of label found in the image, except for 'background' and
            'no_label_id' if defined!

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components.labelled_image import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1,2,3,4,5,6,7]
        >>> im = LabelledImage(a, background=1)
        >>> im.labels()
        [2,3,4,5,6,7]
        """
        if self._labels is None:
            self._labels = np.unique(self.get_array())

        unwanted_set = {self._background_id} - {self._no_label_id}
        label_set = set(self._labels) - unwanted_set

        if isinstance(labels, int):
            labels = [labels]

        if labels:
            return list(label_set & set(labels))
        else:
            return list(label_set)

    def nb_labels(self):
        """
        Return the number of labels.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components.labelled_image import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.nb_labels()
        7
        >>> im = LabelledImage(a, background=1)
        >>> im.nb_labels()
        6
        """
        return len(self.labels())

    def is_label_in_image(self, label):
        """
        Returns True if the label is found in the image, else False.

        Parameters
        ----------
        label: int
            label to check
        """
        return label in self.get_array()

    def boundingbox(self, labels=None, real=False, verbose=False):
        """
        Return the bounding-box of a cell for given 'labels'.

        Parameters
        ----------
        labels: None|int|list(int)|str, optional
            if None (default) returns all labels.
            if an integer, make sure it is in self.labels()
            if a list of integers, make sure they are in self.labels()
            if a string, should be in LABEL_STR to get corresponding
            list of cells (case insensitive)
        real: bool, optional
            if False (default), return the bounding-boxes in voxel units, else
            in real units.
        verbose: bool, optional
            control verbosity of the function

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components.labelled_image import LabelledImage
        >>> im = LabelledImage(a)

        >>> im.boundingbox(7)
        (slice(0, 3), slice(2, 4), slice(0, 1))

        >>> im.boundingbox([7,2])
        [(slice(0, 3), slice(2, 4), slice(0, 1)), (slice(0, 3), slice(0, 2), slice(0, 1))]

        >>> im.boundingbox()
        [(slice(0, 4), slice(0, 6), slice(0, 1)),
        (slice(0, 3), slice(0, 2), slice(0, 1)),
        (slice(1, 3), slice(4, 6), slice(0, 1)),
        (slice(3, 4), slice(3, 4), slice(0, 1)),
        (slice(1, 2), slice(2, 3), slice(0, 1)),
        (slice(1, 2), slice(1, 2), slice(0, 1)),
        (slice(0, 3), slice(2, 4), slice(0, 1))]
        """

        # - Starts with integer case since it is the easiest:
        if isinstance(labels, int):
            try:
                assert self._bbox_dict.has_key(labels)
            except AssertionError:
                image = self.get_array()
                bbox = nd.find_objects(image == labels, max_label=1)[0]
                self._bbox_dict[labels] = bbox
            return self._bbox_dict[labels]
        else:
            # !! remove 'self._background_id' & 'self._no_label_id'
            labels = self.labels()

        # - Create a dict of bounding-boxes using 'scipy.ndimage.find_objects':
        known_bbox = [l in self._bbox_dict for l in labels]
        image = self.get_array()
        if self._bbox_dict is None or not all(known_bbox):
            max_lab = max(labels)
            if verbose:
                print "Computing {} bounding-boxes...".format(max_lab),
            bbox = nd.find_objects(image, max_label=max_lab)
            # NB: scipy.ndimage.find_objects start at 1 (and python index at 0), hence to access i-th element, we have to use (i-1)-th index!
            self._bbox_dict = {n: bbox[n - 1] for n in range(1, max_lab + 1)}

        # - Filter returned bounding-boxes to the (cleaned) given list of labels
        bboxes = {l: self._bbox_dict[l] for l in labels}
        if real:
            vxs = self.voxelsize
            bboxes = {l: real_indices(bbox, vxs) for l, bbox in bboxes.items()}

        return bboxes

    def label_array(self, label):
        """
        Returns an array made from the labelled image cropped around the label
        bounding box.

        Parameters
        ----------
        label: int
            label for which to extract the neighbors

        Returns
        -------
        crop_im: np.array
            labelled array cropped around the label bounding box
        """
        # - Get the slice for given label:
        slices = self.boundingbox(label)
        # - Create the cropped image when possible:
        if slice is None:
            crop_img = self.get_array()
        else:
            ex_slices = dilation(slices)
            crop_img = self.get_array()[ex_slices]
        return crop_img

    def _neighbors_with_mask(self, label):
        """
        Sub-function called when only one label is given to self.neighbors

        Parameters
        ----------
        label: int
            the label for which to compute the neighborhood

        Returns
        -------
        neighbors: list
            list of neighbors for given label
        """
        # - Compute the neighbors and update the unfiltered neighbors dict:
        if not self._neighbors.has_key(label):
            crop_img = self.label_array(label)
            self._neighbors[label] = neighbors_from_image(crop_img, label)

        return self._neighbors[label]

    def _neighborhood_with_mask(self, labels, **kwargs):
        """
        Sub-function called when a list of 'labels' is given to self.neighbors()

        Parameters
        ----------
        label: int
            the label for which to compute the neighborhood

        Returns
        -------
        neighborhood: dict
            neighborhood dictionary for given list of labels
        """
        verbose = kwargs.get('verbose', False)
        # - Check we have all necessary bounding boxes...
        self.boundingbox(labels, verbose=verbose)

        # - Try a shortcut: 'self._neighbors' might have all required 'labels'...
        miss_labels = [l for l in labels if not self._neighbors.has_key(l)]
        n_miss = len(miss_labels)
        # - Compute the neighborhood for labels without (unfiltered) neighbors list:
        if miss_labels:
            t_start = time.time()
            if verbose:
                print "-- Computing the neighbors list for {} labels...".format(
                    n_miss)
            progress = 0
            nb_labels = len(miss_labels)
            for n, label in enumerate(miss_labels):
                progress = percent_progress(progress, n, nb_labels)
                # compute the neighborhood for the given label
                self._neighbors[label] = neighbors_from_image(
                    self.label_array(label), label)

            print elapsed_time(t_start)

        neighborhood = {l: self._neighbors[l] for l in labels}
        return neighborhood

    def neighbors(self, labels=None, verbose=True):
        """
        Return the dictionary of neighbors of each label.
        Except for 'self.background' & 'self.no_label_id', which are filtered
        out from the 'labels' list!

        Parameters
        ----------
        labels: None|int|list(int)|str, optional
            if None (default) returns all labels.
            if an integer, make sure it is in self.labels()
            if a list of integers, make sure they are in self.labels()
            if a string, should be in LABEL_STR to get corresponding
            list of cells (case insensitive)
        verbose: bool, optional
            control verbosity

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components.labelled_image import LabelledImage
        >>> im = LabelledImage(a)

        >>> im.neighbors(7)
        [1, 2, 3, 4, 5]

        >>> im.neighbors([7,2])
        {7: [1, 2, 3, 4, 5], 2: [1, 6, 7] }

        >>> im.neighbors()
        {1: [2, 3, 4, 5, 6, 7],
         2: [1, 6, 7],
         3: [1, 7],
         4: [1, 7],
         5: [1, 6, 7],
         6: [1, 2, 5],
         7: [1, 2, 3, 4, 5] }
        """
        # - Transform length-1 list to integers
        if isinstance(labels, list) and len(labels) == 1:
            labels = labels[0]

        # - Neighborhood computing:
        if isinstance(labels, int):
            try:
                assert labels in self.labels() or labels == self.background
            except AssertionError:
                raise ValueError(MISS_LABEL.format('', 'is', labels))
            if verbose:
                print "Extracting neighbors for label {}...".format(labels)
            return self._neighbors_with_mask(labels)
        else:  # list case:
            if self.background in labels:
                labels = [self.background] + self.labels(labels)
            else:
                labels = self.labels(labels)
            try:
                assert labels != []
            except AssertionError:
                raise ValueError(MISS_LABEL.format('s', 'are', labels))
            if verbose:
                n_lab = len(labels)
                print "Extracting neighbors for {} labels...".format(n_lab)
            return self._neighborhood_with_mask(labels, verbose=verbose)

    # ##########################################################################
    #
    # Labelled SpatiaImage edition functions:
    #
    # ##########################################################################

    def get_image_with_labels(self, labels, keep_background=True):
        """
        Returns a LabelledImage with only the selected 'labels', the rest are
        replaced by "self._no_label_id".

        Parameters
        ----------
        labels: int|list(int)
            if None (default) returns all labels.
            if an integer, make sure it is in self.labels()
            if a list of integers, make sure they are in self.labels()
            if a string, should be in LABEL_STR to get corresponding
            list of cells (case insensitive)
        keep_background: bool, optional
            indicate if background label should be kept in the returned image

        Returns
        -------
        template_im: SpatialImage
            labelled image with 'labels' only
        """
        try:
            assert self._no_label_id is not None
        except AssertionError:
            raise ValueError(
                "Attribute 'no_label_id' is not defined (None), please set it to an integer value before calling this function!")

        all_labels = self.labels()
        labels = self.labels(labels)
        off_labels = list(set(all_labels) - set(labels))

        back_id = self.background
        if keep_background:
            try:
                assert back_id is not None
            except AssertionError:
                raise ValueError(
                    "You asked to keep the background position, but no backgroun label is defined!")
            else:
                labels.append(back_id)
        else:
            if back_id:
                off_labels.append(back_id)

        if len(off_labels) == 0:
            print "WARNING: you selected ALL label!"
            return self
        if len(labels) == 0:
            print "WARNING: you selected NO label!"
            return

        bbox_dict = self.boundingbox(labels)
        no_value = self.no_label_id
        if len(labels) < len(off_labels):
            template_im = image_with_labels(self, labels, bbox_dict, no_value)
        else:
            template_im = image_without_labels(self, off_labels, bbox_dict,
                                               no_value)

        return template_im

    def get_image_without_labels(self, labels, keep_background=True):
        """
        Returns a SpatialImage without the selected labels.

        Parameters
        ----------
        labels: None|int|list|str
            label or list of labels to keep in the image.
            if None, neighbors for all labels found in self.image will
            be returned.
            strings might be processed trough 'self.labels_checker()'
        keep_background: bool, optional
            indicate if background label should be kept in the returned image
        no_label_value: int
            value use to use in place of discarded labels

        Returns
        -------
        template_im: SpatialImage
            labelled image with 'labels' only
        """
        all_labels = self.labels()
        labels = self.labels(labels)
        off_labels = list(set(all_labels) - set(labels))

        return self.get_image_with_labels(off_labels, keep_background)

    def fuse_labels_in_image(self, labels, value='min', verbose=True):
        """
        Fuse the provided list of labels to its minimal value.

        Parameters
        ----------
        labels: list
            list of labels to fuse
        value: str, optional
            value used to replace the given list of labels, by default use the
            min value of the `labels` list. Can also be the max value.
        verbose: bool, optional
            control verbosity

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)
        """
        if isinstance(labels, np.array):
            labels = labels.tolist()
        elif isinstance(labels, set):
            labels = list(labels)
        else:
            assert isinstance(labels, list) and len(labels) >= 2

        if value != 'min':
            try:
                assert self.background not in labels
            except:
                raise ValueError("Can not replace the background value!")

        if value == "min":
            value = min(labels)
        elif value == "max":
            value = max(labels)
        else:
            raise NotImplementedError(
                "Unknown 'value' definition for '{}'".format(value))

        labels.remove(value)
        nb_labels = len(labels)
        array = self.get_array()
        # - Label "fusion" loop:
        no_bbox = []
        progress = 0
        if verbose:
            print "Fusing the following {} labels: {} to value '{}'.".format(
                nb_labels, labels, value)
        for n, label in enumerate(labels):
            if verbose:
                progress = percent_progress(progress, n, nb_labels)
            # Try to get the label's boundingbox:
            try:
                bbox = self.boundingbox(label)
            except KeyError:
                no_bbox.append(label)
                bbox = None
            # Performs value replacement:
            array = array_replace_label(array, label, self.no_label_id, bbox)

        # - If some boundingbox were missing, print about it:
        if no_bbox:
            n = len(no_bbox)
            print "Could not find boundingbox for {} labels: {}".format(n,
                                                                        no_bbox)

        # - Re-initialise the instance:
        self._re_init_array(array, origin=self.origin,
                            voxelsize=self.voxelsize,
                            metadata_dict=self.metadata,
                            background=self.background,
                            no_label_id=self.no_label_id)

        return

    def remove_labels_from_image(self, labels, verbose=True):
        """
        Remove 'labels' from self.image using 'erase_value'.

        Parameters
        ----------
        labels: list|str
            list of labels to remove from the image
            strings might be processed trough 'self.labels_checker()'
        verbose: bool, optional
            control verbosity

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)
        """
        if isinstance(labels, int):
            labels = [labels]
        has_back_id = False
        if self.background in labels:
            has_back_id = True

        # - Make sure 'labels' is correctly formatted:
        labels = self.labels(labels)

        nb_labels = len(labels)
        # - If no labels and no background to remove, its over:
        if nb_labels == 0 and not has_back_id:
            print 'No labels found in the image to remove!'
            return

        t_start = time.time()  # timer
        array = self.get_array()
        ndim = self.get_dim()
        # - Remove 'background' label:
        if has_back_id:
            if verbose:
                print "Removing 'background' label..."
            array[array == self.background] = self.no_label_id
        # - Remove 'labels' using bounding boxes to speed-up computation:
        if verbose:
            print "Removing {} cell labels.".format(nb_labels)
        progress = 0
        no_bbox = []
        for n, label in enumerate(labels):
            if verbose:
                progress = percent_progress(progress, n, nb_labels)
            # Try to get the label's boundingbox:
            try:
                bbox = self.boundingbox(label)
            except KeyError:
                no_bbox.append(label)
                bbox = None
            # Performs value replacement:
            array = array_replace_label(array, label, self.no_label_id, bbox)

        # - If some boundingbox were missing, print about it:
        if no_bbox:
            n = len(no_bbox)
            print "Could not find boundingbox for {} labels: {}".format(n,
                                                                        no_bbox)

        # - Re-initialise the instance:
        self._re_init_array(array, origin=self.origin,
                            voxelsize=self.voxelsize,
                            metadata_dict=self.metadata,
                            background=self.background,
                            no_label_id=self.no_label_id)
        # - May print about elapsed time:
        if verbose:
            elapsed_time(t_start)

        return array

    def relabel_from_mapping(self, mapping, clear_unmapped=False, **kwargs):
        """
        Relabel the image following a given mapping indicating the original
        label as keys and their new labels as values.
        It is possible to get rid of all other label by setting `clear_unmapped`
        to True.

        Parameters
        ----------
        mapping: dict
            a dictionary indicating the original label as keys and their new
            labels as values
        clear_unmapped: bool, optional
            if True (default False), only the mapped labels are kept in the
            returned image

        **kwargs
        --------
        verbose: bool, optional
            control code verbosity; default = False

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)
        """
        back_id = self.background
        # - **kwargs options:
        verbose = kwargs.get('verbose', False)
        # - Mapping dictionary inspection:
        # -- Check the background is not there:
        if back_id in mapping.keys():
            msg = "The background id has been found in the mapping keys!"
            raise ValueError(msg)
        # -- Check the mapping keys are integers:
        try:
            assert all([isinstance(k, int) for k in mapping.keys()])
        except AssertionError:
            raise TypeError("Mapping dictionary keys should be integers!")
        # -- Check the mapping values are integers:
        try:
            assert all([isinstance(v, int) for v in mapping.values()])
        except AssertionError:
            raise TypeError("Mapping dictionary values should be integers!")
        # -- Check the mapping keys are known labels, and how many are unknown, to the image:
        labels = self.labels()
        in_labels = set(mapping.keys()) & set(labels)
        off_labels = set(mapping.keys()) - in_labels
        n_in = len(in_labels)
        # -- Print a summary of this:
        if verbose:
            n_mapped = len(mapping.keys())
            s = ""
            s += "Got an initial list of {} mapped labels".format(n_mapped)
            if off_labels:
                n_off = len(off_labels)
                pc_in = n_in * 100 / n_mapped
                pc_off = 100 - pc_in
                s += ", {} ({}%) of them are found in the image".format(n_in,
                                                                        pc_in)

                s += " and {} ({}%) of them are not!".format(n_off, pc_off)
            else:
                s += ", all are found in the image!."
            print s

        # - Make a copy of the image to relabel:
        relab_img = self.get_array()
        dtype = self.type
        if clear_unmapped:
            # -- Reset every value to `self._no_label_id` value:
            relab_img.fill(self._no_label_id)
            relab_img = relab_img.astype(dtype)
            # - Get the background from the original image
            if back_id is not None:
                # Detect background location in the image
                mask = self.get_array() == back_id
                # Change the boolean mask to a 0/back_id array:
                mask = mask * back_id
                # Add it to the relabelled image:
                relab_img += mask.astype(dtype)

        t_start = time.time()
        # - Relabelling loop:
        if verbose:
            print "Starting relabelling loop..."
        percent = 0
        increment = max(5, round(100 / n_in, 0))
        for n, old_lab in enumerate(in_labels):
            new_lab = mapping[old_lab]
            if verbose:
                percent = percent_progress(percent, n, n_in, increment)
            bbox = self.boundingbox(old_lab)
            mask = self.get_array()[bbox] == new_lab
            relab_img[bbox] += np.array(mask * new_lab, dtype=dtype)

        # - Re-initialise the instance:
        self._re_init_array(relab_img, origin=self.origin,
                            voxelsize=self.voxelsize,
                            metadata_dict=self.metadata,
                            background=self.background,
                            no_label_id=self.no_label_id)
        # - May print about elapsed time:
        if verbose:
            elapsed_time(t_start)

        return

    def _re_init_array(self, array, **kwargs):
        """
        Hidden function rebuilding the object when labelled image is modified.

        Parameters
        ----------
        array: np.array
            the new labelled array to use
        """

        # - Make a SpatialImage out of the array:
        array = SpatialImage(array, **kwargs)
        # - Re-initialisation of the object with the relabelled LabelledImage:
        self.__init__(array, **kwargs)
        return


class LabelledImage2D(AbstractLabelledImage):
    """
    This class allows to manipulate 3D labelled SpatialImage, eg. segmented image
    of a biological tissue where each cell has a different label or id.
    """

    def __init__(self, image, background=None, no_label_id=None, **kwargs):
        AbstractLabelledImage.__init__(self, image, background=background,
                                       no_label_id=no_label_id, **kwargs)


class LabelledImage3D(AbstractLabelledImage):
    """
    This class allows to manipulate 3D labelled SpatialImage, eg. segmented image
    of a biological tissue where each cell has a different label or id.
    """

    def __init__(self, image, background=None, no_label_id=None, **kwargs):
        AbstractLabelledImage.__init__(self, image, background=background,
                                       no_label_id=no_label_id, **kwargs)


def voxel_n_layers(image, connectivity=18, n=1, **kwargs):
    """
    Extract the n-first layer of non-background voxels, ie. those in contact
    with the background.

    Parameters
    ----------
    image: LabelledImage
        the SpatialImage containing the labelled tissue
    connectivity: int, optional
        connectivity or neighborhood of the structuring element, default is 18,
        should be in [4, 6, 8, 18, 26], where 4 and 8 are 2D structuring
        elements, the rest are 3D structuring elements.
    n: int, optional
        number of layer of voxels to extract, the first one being in contact
        with the background

    Returns
    -------
    LabelledImage
        labelled image made of the selected number of voxel layers
    """
    verbose = kwargs.get('verbose', False)
    if verbose:
        print "Extracting the first layer of voxels...",
    # Get background position (mask)
    mask_img_1 = (image.get_array() == image.background)
    # Dilate it by one voxel using a 18-connexe 3D structuring element:
    struct = structuring_element(connectivity)
    dil_1 = nd.binary_dilation(mask_img_1, structure=struct,
                               iterations=n)
    # Difference with background mask gives the first layer of voxels:
    layer = dil_1 ^ mask_img_1
    if verbose:
        print "Done."

    return image.get_array() * layer, mask_img_1


class AbstractEpidermisLabelledImage(LabelledImage2D, LabelledImage3D):

    def __init__(self, image, background=None, no_label_id=None, **kwargs):
        """
        Parameters
        ----------
        image: SpatialImage
            the SpatialImage containing the labelled tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        no_label_id: int, optional
            if given define the "unknown label" (ie. not a cell)
        """
        if image.is2D():
            LabelledImage2D.__init__(self, image, background=background,
                                     no_label_id=no_label_id)
        else:
            LabelledImage3D.__init__(self, image, background=background,
                                     no_label_id=no_label_id)

    def voxel_n_first_layer(self, n_voxel_layer, connectivity,
                            keep_background=True, **kwargs):
        """
        Extract the n-first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        n_voxel_layer: int
            number of layer of voxel from the background to get
        connectivity: int
            connectivity or neighborhood of the structuring element
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        LabelledImage
            labelled image made of the selected number of voxel layers
        """
        mask_img_1 = None
        if self._voxel_layer1 is None:
            self._voxel_layer1, mask_img_1 = voxel_n_layers(self,
                                                            iter=n_voxel_layer,
                                                            connectivity=connectivity,
                                                            **kwargs)

        if keep_background:
            if mask_img_1 is None:
                mask_img_1 = (self.get_array() == self.background)
            return self._voxel_layer1 + mask_img_1
        else:
            return self._voxel_layer1

    def voxel_first_layer(self, connectivity, keep_background=True,
                          **kwargs):
        """
        Extract the first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        connectivity: int
            connectivity or neighborhood of the structuring element
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        LabelledImage
            labelled image made of the first layer of voxel in contact with the
            background
        """
        mask_img_1 = None
        if self._voxel_layer1 is None:
            self._voxel_layer1, mask_img_1 = voxel_n_layers(self,
                                                            connectivity=connectivity,
                                                            **kwargs)

        if keep_background:
            if mask_img_1 is None:
                mask_img_1 = (self.get_array() == self.background)
            return self._voxel_layer1 + mask_img_1
        else:
            return self._voxel_layer1

    def voxel_first_layer_coordinates(self, **kwargs):
        """
        Returns an (Nxd) array of coordinates indicating voxels first layer
        position.
        """
        vfl = self.voxel_first_layer(keep_background=False, **kwargs)
        return np.array(np.where(vfl != 0)).T

    def cell_first_layer(self, **kwargs):
        """
        List labels corresponding to the first layer of cells (epidermis).
        It is possible to provide an epidermal area threshold (minimum area in
        contact with the background) to consider a label as in the first layer.


        Returns
        -------
        list of L1-cells

        Notes
        ----
        The returned list does not contain 'self._ignoredlabels'.
        However 'self._cell_layer1' is not filtered against 'self._ignoredlabels'
        """
        verbose = kwargs.get('verbose', False)
        if verbose:
            print "Generating list of L1 labels..."

        integers = lambda x: map(int, x)
        bkgd_id = self.background
        # - Create unfiltered list of ALL neighbors to the background:
        # 'self._cell_layer1'will thus contains ALL L1 labels (not filtered by area) even those in 'self._ignoredlabels'.
        if self._cell_layer1 is None:
            background_nei = self.neighbors(bkgd_id, verbose=False)
            self._cell_layer1 = set(integers(background_nei))

        return list(self._cell_layer1)


class EpidermisLabelledImage2D(AbstractEpidermisLabelledImage):

    def __init__(self, image, background=None, no_label_id=None, **kwargs):
        """
        Parameters
        ----------
        image: SpatialImage
            the SpatialImage containing the labelled tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        no_label_id: int, optional
            if given define the "unknown label" (ie. not a cell)
        """
        AbstractEpidermisLabelledImage.__init__(self, image,
                                                background=background,
                                                no_label_id=no_label_id,
                                                **kwargs)

    def voxel_n_first_layer(self, n_voxel_layer, connectivity=4,
                            keep_background=True, **kwargs):
        """
        Extract the n-first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        n_voxel_layer: int
            number of layer of voxel from the background to get
        connectivity: int
            connectivity of the 2D structuring element, default 4
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        LabelledImage
            labelled image made of the selected number of voxel layers
        """
        return AbstractEpidermisLabelledImage.voxel_n_first_layer(self,
                                                                  n_voxel_layer,
                                                                  connectivity=connectivity,
                                                                  keep_background=keep_background,
                                                                  **kwargs)

    def voxel_first_layer(self, connectivity=4, keep_background=True,
                          **kwargs):
        """
        Extract the first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        connectivity: int, optional
            connectivity of the 2D structuring element, default 4
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        LabelledImage
            labelled image made of the first layer of voxel in contact with the
            background
        """
        return AbstractEpidermisLabelledImage.voxel_first_layer(self,
                                                                connectivity=connectivity,
                                                                keep_background=keep_background,
                                                                **kwargs)


class EpidermisLabelledImage3D(AbstractEpidermisLabelledImage):

    def __init__(self, image, background=None, no_label_id=None, **kwargs):
        """
        Parameters
        ----------
        image: SpatialImage
            the SpatialImage containing the labelled tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        no_label_id: int, optional
            if given define the "unknown label" (ie. not a cell)
        """
        AbstractEpidermisLabelledImage.__init__(self, image,
                                                background=background,
                                                no_label_id=no_label_id,
                                                **kwargs)

    def voxel_n_first_layer(self, n_voxel_layer, connectivity=18,
                            keep_background=True, **kwargs):
        """
        Extract the n-first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        n_voxel_layer: int
            number of layer of voxel from the background to get
        connectivity: int
            connectivity of the 3D structuring element, default 18
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        LabelledImage
            labelled image made of the selected number of voxel layers
        """
        return AbstractEpidermisLabelledImage.voxel_n_first_layer(self,
                                                                  n_voxel_layer,
                                                                  connectivity=connectivity,
                                                                  keep_background=keep_background,
                                                                  **kwargs)

    def voxel_first_layer(self, connectivity=18, keep_background=True,
                          **kwargs):
        """
        Extract the first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        connectivity: int, optional
            connectivity of the 3D structuring element, default 18
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        LabelledImage
            labelled image made of the first layer of voxel in contact with the
            background
        """
        return AbstractEpidermisLabelledImage.voxel_first_layer(self,
                                                                connectivity=connectivity,
                                                                keep_background=keep_background,
                                                                **kwargs)


# def LabelledImage(image, **kwargs):
#     """
#     Constructor.
#     """
#     # -- If 'image' is a string, it should relate to the filename and we try to load it using imread:
#     if isinstance(image, str):
#         image = imread(image)
#     elif isinstance(image, np.ndarray):
#         dtype = image.dtype
#         image = SpatialImage(image, dtype=dtype)
#     else:
#         try:
#             assert isinstance(image, SpatialImage)
#         except AssertionError:
#             raise TypeError(
#                 "Input image should be file path, a nupy array or a SpatialImage!")
#
#     background = kwargs.get('background', None)
#     # - If declared value for 'background', return the class Epidermis
#     if background is not None:
#         if image.is2D():
#             return EpidermisLabelledImage2D(image, **kwargs)
#         else:
#             return EpidermisLabelledImage3D(image, **kwargs)
#     else:
#         if image.is2D():
#             return LabelledImage2D(image, **kwargs)
#         else:
#             return LabelledImage3D(image, **kwargs)
#
# class LabelledImage(EpidermisLabelledImage2D, EpidermisLabelledImage3D,
#                     LabelledImage2D, LabelledImage3D):
#
#     def __init__(self, image, **kwargs):
#         """
#         """
#         background = kwargs.get('background', None)
#         is2d = False
#         try:
#             assert image.is2D()
#         except:
#             pass
#         else:
#             is2d = True
#         try:
#             assert image.ndim == 2
#         except:
#             pass
#         else:
#             is2d = True
#         # - If declared value for 'background', return the class Epidermis
#         if background is not None:
#             if is2d:
#                 EpidermisLabelledImage2D.__init__(self, image, **kwargs)
#             else:
#                 EpidermisLabelledImage3D.__init__(self, image, **kwargs)
#         else:
#             if is2d:
#                 LabelledImage2D.__init__(self, image, **kwargs)
#             else:
#                 LabelledImage3D.__init__(self, image, **kwargs)

class LabelledImage(EpidermisLabelledImage2D, EpidermisLabelledImage3D,
                    LabelledImage2D, LabelledImage3D):

    def __init__(self, image, **kwargs):
        """
        """
        # -- If 'image' is a string, it should relate to the filename and we try to load it using imread:
        if isinstance(image, str):
            print "here1"
            image = imread(image)
        elif isinstance(image, np.ndarray):
            dtype = image.dtype
            image = SpatialImage(image, dtype=dtype)
        else:
            try:
                assert isinstance(image, SpatialImage)
            except AssertionError:
                raise TypeError(
                    "Input image should be file path, a nupy array or a SpatialImage!")

        background = kwargs.get('background', None)
        no_label_id = kwargs.get('no_label_id', None)

        # - If declared value for 'background', return the class Epidermis
        if background is not None:
            if image.is2D():
                obj = EpidermisLabelledImage2D(image, background=background,
                                               no_label_id=no_label_id)
                EpidermisLabelledImage2D.__init__(obj, image, **kwargs)
            else:
                obj = EpidermisLabelledImage3D(image, background=background,
                                               no_label_id=no_label_id)
                EpidermisLabelledImage3D.__init__(obj, image, **kwargs)
        else:
            if image.is2D():
                obj = LabelledImage2D(image, background=background,
                                      no_label_id=no_label_id)
                LabelledImage2D.__init__(obj, image, **kwargs)
            else:
                obj = LabelledImage3D(image, background=background,
                                      no_label_id=no_label_id)
                LabelledImage3D.__init__(obj, image, **kwargs)
