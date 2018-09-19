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
from timagetk.components.slices import dilation_by
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


def test_structuring_element(array, struct):
    """
    Test if the array and the strucutring element are compatible, ie. of same
    dimensionality.

    Parameters
    ----------
    array : np.array
        array on wich the structuring element should be applied
    struct : np.array
        array defining the structuring element

    Returns
    -------
    bool
        True if compatible, False otherwise
    """
    return array.ndim == struct.ndim


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
        connectivity order determines which elements of the output array belong
        to the structure, i.e. are considered as neighbors of the central
        element.
        Elements up to a squared distance of connectivity from the center are
        considered neighbors, thus it may range from 1 (no diagonal elements are
        neighbors) to rank (all elements are neighbors), with rank the number of
        dimensions of the image.

    Returns
    -------
    np.array|LabelledImage
        a labelled array with only the inner-wall position as non-null value
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
        connectivity order determines which elements of the output array belong
        to the structure, i.e. are considered as neighbors of the central
        element.
        Elements up to a squared distance of connectivity from the center are
        considered neighbors, thus it may range from 1 (no diagonal elements are
        neighbors) to rank (all elements are neighbors), with rank the number of
        dimensions of the image.

    Returns
    -------
    np.array|LabelledImage
        a labelled array with only the outer-wall position as non-null value
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
    list
        neighbors label of 'label_id'
    """
    if isinstance(label_id, int):
        # Get outer-wall position & return unique list of labels:
        return list(np.unique(label_outer_wall(labelled_img, label_id)))
    elif isinstance(label_id, list):
        return {l: list(np.unique(label_outer_wall(labelled_img, l))) for l in
                label_id}
    else:
        msg = "Parameter 'label_id' should be an integer or a list, got '{}'."
        raise TypeError(msg.format(type(label_id)))


# ------------------------------------------------------------------------------
#
# WHOLE LABELLED IMAGE functions:
#
# ------------------------------------------------------------------------------
def image_with_labels(image, labels):
    """
    Create a new image containing only the given labels.
    Use image as template (shape, origin, voxelsize & metadata).

    Parameters
    ----------
    image : LabelledImage
        labelled spatial image to use as template for labels extraction
    labels : list
        list of labels to keep in the image
    erase_value : int, optional
        value use to use in place of discarded labels

    Returns
    -------
    LabelledImage
        the image containing only the given 'labels'
    """
    erase_value = image.no_label_id
    # - Initialising empty template image
    print "Initialising empty template image..."
    if erase_value == 0:
        template_im = np.zeros_like(image.get_array())
    else:
        template_im = np.ones_like(image.get_array()) * erase_value

    nb_labels = len(labels)
    boundingbox = image.boundingbox(labels)
    # - Add selected 'labels' to the empty image:
    no_bbox = []
    progress = 0
    print "Adding {} labels to the empty template image...".format(nb_labels)
    for n, label in enumerate(labels):
        progress = percent_progress(progress, n, nb_labels)
        try:
            bbox = boundingbox[label]
            xyz = np.array(np.where((image[bbox]) == label)).T
            xyz = tuple([xyz[:, n] + bbox[n].start for n in range(image.ndim)])
            template_im[xyz] = label
        except ValueError:
            no_bbox.append(label)
            template_im[image == label] = label

    # - If some boundingbox were missing, print about it:
    if no_bbox:
        n = len(no_bbox)
        print "Could not find boundingbox for {} labels: {}".format(n, no_bbox)

    return LabelledImage(template_im, voxelsize=image.voxelsize,
                         origin=image.origin, metadata_dict=image.metadata,
                         no_label_id=image.no_label_id)


def image_without_labels(image, labels):
    """
    Create a new image without the given labels.
    Use self.image as template (shape, origin, voxelsize & metadata).

    Parameters
    ----------
    image: LabelledImage
        labelled spatial image to use as template for labels deletion
    labels: list
        list of labels to remove from the image

    Returns
    -------
    LabelledImage
        an image without the given 'labels'
    """
    erase_value = image.no_label_id
    # - Initialising template image:
    print "Initialising template image from self.image..."
    template_im = image.get_array()

    nb_labels = len(labels)
    boundingbox = image.boundingbox(labels)
    # - Remove selected 'labels' from the empty image:
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

    return LabelledImage(template_im, voxelsize=image.voxelsize,
                         origin=image.origin, metadata_dict=image.metadata,
                         no_label_id=image.no_label_id)


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


def hollow_out_labels(image, **kwargs):
    """
    Returns a labelled image containing only the wall, the rest is set to
    'image.no_label_id'.

    Parameters
    ----------
    image : LabelledImage
        labelled image to transform

    Returns
    -------
    LabelledImage
        labelled image containing hollowed out cells (only walls).

    Notes
    -----
    The Laplacian filter is used to detect wall positions, as it highlights
    regions of rapid intensity change.
    """
    verbose = kwargs.get('verbose', True)
    if verbose:
        print '\nHollowing out labelled numpy array... ',

    t_start = time.time()
    # - The laplacian allows to quickly get a mask with all the walls:
    wall_mask = np.array(nd.laplace(image))
    wall_mask = wall_mask != 0  # mask of "wall image"
    # - Get the label values:
    image *= wall_mask

    if verbose:
        elapsed_time(t_start)

    return image


# - GLOBAL VARIABLES:
MISS_LABEL = "The following label{} {} not found in the image: {}"  # ''/'s'; 'is'/'are'; labels


class LabelledImage(SpatialImage):
    """
    Class to manipulate labelled SpatialImage, eg. a segmented image.
    """

    def __init__(self, image, no_label_id=None, **kwargs):
        """
        LabelledImage constructor.

        Parameters
        ----------
        image: np.array|SpatialImage
            a numpy array or a SpatialImage containing a labelled array
        no_label_id: int, optional
            if given define the "unknown label" (ie. not a label)

        kwargs
        ------
        origin: list, optional
            coordinates of the origin in the image, default: [0,0] or [0,0,0]
        voxelsize: list, optional.
            image voxelsize, default: [1.0,1.0] or [1.0,1.0,1.0]
        dtype: str, optional
            image type, default dtype = input_array.dtype
        metadata_dict: dict, optional
            dictionary of image metadata, default is an empty dict
        """
        # - Get variable for SpatialImage initialisation:
        if isinstance(image, SpatialImage):
            input_array = image.get_array()
            origin = image.origin
            voxelsize = image.voxelsize
            dtype = image.dtype
            metadata_dict = image.metadata
        else:
            input_array = image
            origin = kwargs.get('origin', None)
            voxelsize = kwargs.get('voxelsize', None)
            dtype = kwargs.get('voxelsize', image.dtype)
            metadata_dict = kwargs.get('metadata_dict', None)

        # - Inherit SpatialImage class:
        SpatialImage.__init__(self, input_array, origin=origin,
                              voxelsize=voxelsize, dtype=dtype,
                              metadata_dict=metadata_dict)

        # - Initializing EMPTY hidden attributes:
        # -- Property hidden attributes:
        self._no_label_id = None  # id refering to the absence of label
        # -- Useful label lists:
        self._labels = None  # list of image-labels referring to cell-labels
        # -- Useful & recurrent label properties:
        self._bbox = []  # list of bounding boxes (indexed 'self._labels - 1')
        self._bbox_dict = {}  # dict of bounding boxes
        self._neighbors = {}  # unfiltered neighborhood label-dict {vid_i: neighbors(vid_i)}

        # - Initialise object property and most used hidden attributes:
        # -- Define the "no_label_id" value, if any (can be None):
        self.no_label_id = no_label_id
        # -- Get the list of labels found in the image:
        self.labels()
        # -- Get the boundingbox dictionary:
        self.boundingbox()

    @property
    def no_label_id(self):
        """
        Get the value associated to no label.
        This is used as "unknown label" or "erase value".

        Returns
        -------
        no_label_id: int
            the value defined as the "no_label_id"

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        WARNING: no value defined for the 'no label' id!
        >>> im = LabelledImage(a, no_label_id=1)
        >>> im.labels()
        [2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        1
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

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        WARNING: no value defined for the 'no label' id!
        >>> im.no_label_id = 1
        >>> im.labels()
        [2, 3, 4, 5, 6, 7]
        >>> im.no_label_id
        1
        """
        if not isinstance(value, int) and value is not None:
            print "Provided value '{}' is not an integer!".format(value)
            return
        else:
            self._no_label_id = value

    def _defined_no_label_id(self):
        """
        Tests if '_no_label_id' attribute is defined, if not raise a ValueError.
        """
        try:
            assert self._no_label_id is not None
        except AssertionError:
            msg = "Attribute 'no_label_id' is not defined (None)."
            msg += "Please set it (integer) before calling this function!"
            raise ValueError(msg)
        return

    def labels(self, labels=None):
        """
        Get the list of labels found in the image, or make sure provided labels
        exists.

        Parameters
        ----------
        labels: int|list, optional
            if given, used to filter the returned list, else return all labels
            defined in the image by default

        Returns
        -------
        list
            list of label found in the image, except for 'no_label_id'
            (if defined)

        Notes
        -----
        Value defined for 'no_label_id' is removed from the returned list of
        labels since it does not refer to one.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        """
        if isinstance(labels, int):
            labels = [labels]
        # - If the hidden label attribute is None, list all labels in the array:
        if self._labels is None:
            self._labels = np.unique(self.get_array())
        # - Remove value attributed to 'no_label_id':
        unwanted_set = {self._no_label_id}
        label_set = set(self._labels) - unwanted_set

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

        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.nb_labels()
        7
        >>> im = LabelledImage(a, no_label_id=1)
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

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.is_label_in_image(7)
        True
        >>> im.is_label_in_image(10)
        False
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

        >>> from timagetk.components import LabelledImage
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
            # !! remove 'self._no_label_id'
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

    def label_array(self, label, dilation=None):
        """
        Returns an array made from the labelled image cropped around the label
        bounding box.

        Parameters
        ----------
        label: int
            label for which to extract the neighbors
        dilation: int, optional
            if defined (default is None), use this value as a dilation factor
            (in every directions) to be applied to the label boundingbox

        Returns
        -------
        np.array
            labelled array cropped around the label bounding box

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a)
        >>> im.label_array(7)
        array([[7, 7],
               [5, 7],
               [1, 7]])
        >>> im.label_array(7, dilation=1)
        array([[2, 7, 7, 1],
               [6, 5, 7, 3],
               [2, 1, 7, 3],
               [1, 1, 4, 1]])
        """
        # - Get the slice for given label:
        slices = self.boundingbox(label)
        # - Create the cropped image when possible:
        if slices is None:
            crop_img = self.get_array()
        else:
            if dilation:
                slices = dilation_by(slices, dilation)
            crop_img = self.get_array()[slices]

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
        list
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
        dict
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
        >>> from timagetk.components import LabelledImage
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
                assert self.is_label_in_image(labels)
            except AssertionError:
                raise ValueError(MISS_LABEL.format('', 'is', labels))
            if verbose:
                print "Extracting neighbors for label {}...".format(labels)
            return self._neighbors_with_mask(labels)
        else:  # list case:
            try:
                assert self.background in labels
            except:
                labels = self.labels(labels)
            else:
                labels = [self.background] + self.labels(labels)

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

    def get_image_with_labels(self, labels):
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

        Returns
        -------
        LabelledImage
            labelled image with 'labels' only

        Notes
        -----
        Require property 'no_label_id' to be defined!

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a, no_label_id=0)
        >>> im.get_image_with_labels([2, 5])
        LabelledImage([[0, 2, 0, 0, 0, 0],
                       [0, 0, 5, 0, 0, 0],
                       [2, 2, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])
        """
        self._defined_no_label_id()
        all_labels = self.labels()
        labels = self.labels(labels)
        off_labels = list(set(all_labels) - set(labels))

        if len(off_labels) == 0:
            print "WARNING: you selected ALL label!"
            return self
        if len(labels) == 0:
            print "WARNING: you selected NO label!"
            return

        if len(labels) < len(off_labels):
            template_im = image_with_labels(self, labels)
        else:
            template_im = image_without_labels(self, off_labels)

        return template_im

    def get_image_without_labels(self, labels):
        """
        Returns a LabelledImage without the selected labels.

        Parameters
        ----------
        labels: None|int|list|str
            label or list of labels to keep in the image.
            if None, neighbors for all labels found in self.image will
            be returned.
            strings might be processed trough 'self.labels_checker()'

        Returns
        -------
        LabelledImage
            labelled image with 'labels' only

        Notes
        -----
        Require property 'no_label_id' to be defined!

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a, no_label_id=0)
        >>> im.get_image_without_labels([2, 5])
        LabelledImage([[1, 0, 7, 7, 1, 1],
                       [1, 6, 0, 7, 3, 3],
                       [0, 0, 1, 7, 3, 3],
                       [1, 1, 1, 4, 1, 1]])
        """
        all_labels = self.labels()
        labels = self.labels(labels)
        off_labels = list(set(all_labels) - set(labels))
        return self.get_image_with_labels(off_labels)

    def get_wall_image(self, labels=None, **kwargs):
        """
        Get an image made of walls only (hollowed out cells).

        Parameters
        ----------
        labels: list, optional
            list of labels to return in the wall image, by default (None) return
            all labels
        kwargs: dict, optional
            given to 'hollow_out_cells', 'verbose' accepted

        Returns
        -------
        LabelledImage
            the labelled wall image
        """
        if labels is not None:
            image = self.get_image_with_labels(labels)
        else:
            image = self

        return hollow_out_labels(image, **kwargs)

    def fuse_labels_in_image(self, labels, new_value='min', verbose=True):
        """
        Fuse the provided list of labels to a given new_value, or the min or max
        of the list of labels.

        Parameters
        ----------
        labels: list
            list of labels to fuse
        new_value: int|str, optional
            value used to replace the given list of labels, by default use the
            min value of the `labels` list. Can also be the max value.
        verbose: bool, optional
            control verbosity

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a, no_label_id=0)
        >>> im.fuse_labels_in_image([6, 7], new_value=8)
        LabelledImage([[1, 2, 8, 8, 1, 1],
                       [1, 8, 5, 8, 3, 3],
                       [2, 2, 1, 8, 3, 3],
                       [1, 1, 1, 4, 1, 1]])
        >>> im.fuse_labels_in_image([6, 7], new_value='min')
        LabelledImage([[1, 2, 6, 6, 1, 1],
                       [1, 6, 5, 6, 3, 3],
                       [2, 2, 1, 6, 3, 3],
                       [1, 1, 1, 4, 1, 1]])
        >>> im.fuse_labels_in_image([6, 7], new_value='max')
        LabelledImage([[1, 2, 7, 7, 1, 1],
                       [1, 7, 5, 7, 3, 3],
                       [2, 2, 1, 7, 3, 3],
                       [1, 1, 1, 4, 1, 1]])
        """
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        elif isinstance(labels, set):
            labels = list(labels)
        else:
            assert isinstance(labels, list) and len(labels) >= 2

        # - Make sure 'labels' is correctly formatted:
        labels = self.labels(labels)
        nb_labels = len(labels)
        # - If no labels to remove, its over:
        if nb_labels == 0:
            print 'No labels to fuse!'
            return

        # - Define the integer value of 'new_value':
        if new_value == "min":
            new_value = min(labels)
            labels.remove(new_value)
        elif new_value == "max":
            new_value = max(labels)
            labels.remove(new_value)
        elif isinstance(new_value, int):
            if self.is_label_in_image(new_value) and not new_value in labels:
                msg = "Given new_value is in the image and not in the list of labels."
                raise ValueError(msg)
            if new_value in labels:
                labels.remove(new_value)
        else:
            raise NotImplementedError(
                "Unknown 'new_value' definition for '{}'".format(new_value))

        t_start = time.time()  # timer
        array = self.get_array()
        # - Label "fusion" loop:
        no_bbox = []
        progress = 0
        if verbose:
            print "Fusing the following {} labels: {} to new_value '{}'.".format(
                nb_labels, labels, new_value)
        for n, label in enumerate(labels):
            if verbose:
                progress = percent_progress(progress, n, nb_labels)
            # - Try to get the label's boundingbox:
            try:
                bbox = self.boundingbox(label)
            except KeyError:
                no_bbox.append(label)
                bbox = None
            # - Performs value replacement:
            array = array_replace_label(array, label, new_value, bbox)

        # - If some boundingbox were missing, print about it:
        if no_bbox:
            n = len(no_bbox)
            print "Could not find boundingbox for {} labels: {}".format(n,
                                                                        no_bbox)
        # - May print about elapsed time:
        if verbose:
            elapsed_time(t_start)

        # TODO: re-instanciate the object !
        return LabelledImage(array, origin=self.origin,
                             voxelsize=self.voxelsize,
                             metadata_dict=self.metadata,
                             no_label_id=self.no_label_id)

    def remove_labels_from_image(self, labels, verbose=True):
        """
        Remove 'labels' from self.image using 'no_label_id'.

        Parameters
        ----------
        labels: list
            list of labels to remove from the image
        verbose: bool, optional
            control verbosity

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)

        Notes
        -----
        Require property 'no_label_id' to be defined!

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import LabelledImage
        >>> im = LabelledImage(a, no_label_id=0)
        >>> im.remove_labels_from_image([6, 7])
        LabelledImage([[1, 2, 0, 0, 1, 1],
                       [1, 0, 5, 0, 3, 3],
                       [2, 2, 1, 0, 3, 3],
                       [1, 1, 1, 4, 1, 1]])
        """
        if isinstance(labels, int):
            labels = [labels]
        elif isinstance(labels, np.ndarray):
            labels = labels.tolist()
        elif isinstance(labels, set):
            labels = list(labels)
        else:
            assert isinstance(labels, list) and len(labels) >= 2

        # - Make sure 'labels' is correctly formatted:
        labels = self.labels(labels)
        nb_labels = len(labels)
        # - If no labels to remove, its over:
        if nb_labels == 0:
            print 'No labels to remove!'
            return

        t_start = time.time()  # timer
        array = self.get_array()
        # - Remove 'labels' using bounding boxes to speed-up computation:
        no_bbox = []
        progress = 0
        if verbose:
            print "Removing {} cell labels.".format(nb_labels)
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
        # - May print about elapsed time:
        if verbose:
            elapsed_time(t_start)

        # TODO: re-instanciate the object !
        return LabelledImage(array, origin=self.origin,
                             voxelsize=self.voxelsize,
                             metadata_dict=self.metadata,
                             no_label_id=self.no_label_id)

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
        # - **kwargs options:
        verbose = kwargs.get('verbose', False)
        # - Mapping dictionary inspection:
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
        dtype = self.dtype
        if clear_unmapped:
            # -- Reset every value to `self._no_label_id` value:
            relab_img.fill(self._no_label_id)
            relab_img = relab_img.astype(dtype)

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
            mask = self.get_array()[bbox] == old_lab
            relab_img[bbox] += np.array(mask * new_lab, dtype=dtype)

        # - May print about elapsed time:
        if verbose:
            elapsed_time(t_start)

        # TODO: re-instanciate the object !
        return LabelledImage(relab_img, origin=self.origin,
                             voxelsize=self.voxelsize,
                             metadata_dict=self.metadata,
                             no_label_id=self.no_label_id)
