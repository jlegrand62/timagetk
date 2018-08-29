# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
# -----------------------------------------------------------------------------

"""
We here regroup functions related to slices manipulation.
"""

__license__ = "Private/UnderDev"
__revision__ = " $Id$ "

import numpy as np
import scipy.ndimage as nd


def dilation(slices):
    """
    Expand bounding boxes, tuple of 'slice' by one voxel in every direction.
    Hence each slice starts one voxel before and ends one voxel after.

    Parameters
    ----------
    slices: tuple
        a tuple of slices

    Returns
    -------
    tuple of slice
    """
    return [slice(max(0, s.start - 1), s.stop + 1) for s in slices]


def dilation_by(slices, amount=2):
    """
    Expand bounding boxes, tuple of 'slice' by a given number of voxels
    ('amount') in every direction.
    Hence each slice starts 'amount'-voxel before and ends 'amount'-voxel after.

    Parameters
    ----------
    slices: tuple
        a tuple of slices
    amount: int
        the amount of expansion to apply to each slice start and stop

    Returns
    -------
    tuple of slice
    """
    return [slice(max(0, s.start - amount), s.stop + amount) for s in slices]


def change_boundingbox_origins(bboxes, xyz_origin):
    """
    Translate the bounding box to a new origin by removing x, y and z origins
    to the start and stop of their respective dimension.

    Parameters
    ----------
    bboxes: tuple
        length-3 tuple of slices to translate
    xyz_origin: list
        length-3 list of x, y & z values used to translate the bounding box
        origin

    Returns
    -------
    slices
    """
    if isinstance(bboxes, dict):
        return {k: change_boundingbox_origins(bbox, xyz_origin) for k, bbox in
                bboxes.iteritems()}
    elif isinstance(bboxes, tuple):
        ori = xyz_origin
        return [slice(b.start - ori[n], b.stop - ori[n]) for n, b in
                enumerate(bboxes)]
    else:
        raise TypeError("Unknown type for 'bboxes', should be dict or tuple!")


def real_indices(slices, resolutions):
    """
    Transform the discrete (voxels based) coordinates of the bounding box
    (slices) into their real-world size using resolutions.

    Parameters
    ----------
    slices: list
        list of slices or bounding boxes found using scipy.ndimage.find_objects
    resolutions: list
        length-2 (2D) or length-3 (3D) vector of float indicating the size of a
        voxel in real-world units

    Returns
    -------
    tuple of slice
    """
    return [(s.start * r, s.stop * r) for s, r in zip(slices, resolutions)]


def sort_labels_by_bbox(boundingbox, label_1, label_2):
    """
    Determine which provided label as the smallest bounding box.

    Parameters
    ----------
    bounding box: dict
        label-based dictionary of slices tuples
    labels: int
        a label to be found within the image
    label_2: int
        a label to be found within the image

    Returns
    -------
    tuple: contain both labels, first one has the smallest bounding box
    """
    assert isinstance(boundingbox, dict)
    if label_1 not in boundingbox:
        boundingbox[label_1] = None
    if label_2 not in boundingbox:
        boundingbox[label_2] = None

    bbox_1 = boundingbox[label_1]
    bbox_2 = boundingbox[label_2]
    # if bbox only for 'label_1' return 'label_2' first
    if bbox_1 is None and bbox_2:
        return label_2, label_1
    # if bbox only for 'label_2' return 'label_1' first
    if bbox_1 and bbox_2 is None:
        return label_1, label_2
    # if no bbox for 'label_1' and 'label_2' return (None, None)
    if bbox_1 is None and bbox_2 is None:
        return None, None
    # if both found in 'bounding box', compute the smallest one use its volume:
    vol_bbox_1 = (bbox_1[0].stop - bbox_1[0].start) * (
            bbox_1[1].stop - bbox_1[1].start) * (
                         bbox_1[2].stop - bbox_1[2].start)
    vol_bbox_2 = (bbox_2[0].stop - bbox_2[0].start) * (
            bbox_2[1].stop - bbox_2[1].start) * (
                         bbox_2[2].stop - bbox_2[2].start)

    if vol_bbox_1 < vol_bbox_2:
        return label_1, label_2
    else:
        return label_2, label_1


def smallest_boundingbox(image, label_1, label_2):
    """
    Compute bounding box for labels 'label_1' and 'label_2' and return the
    smallest in 'image'.

    Parameters
    ----------
    image: SpatialImage
        labelled image containing labels 'label_1' and 'label_2'
    label_1: int
        a label to be found within the image
    label_2: int
        a label to be found within the image

    Returns
    -------
    tuple: the smallest bounding box object between 'label_1' and 'label_2'
    """
    bbox = nd.find_objects(image, max_label=max([label_1, label_2]))
    # we do 'label_x - 1' since 'nd.find_objects' start at '1' (and not '0') !
    bbox = {label_1: bbox[label_1 - 1], label_2: bbox[label_2 - 1]}
    label_1, label_2 = sort_labels_by_bbox(bbox, label_1, label_2)
    return bbox[label_1]


def region_slice(sliceA, sliceB):
    """
    Returns the slice containing both input slice.

    Parameters
    ----------
    sliceA: slice
        slice for object A
    sliceB: slice
        slice for object B

    Returns
    -------
    a slice with the min start and the max stop (step is set to None)

    Example
    -------
    >>> sliceA = [slice(2, 5), slice(1, 3)]
    >>> sliceB = [slice(2, 4), slice(0, 2)]
    >>> region_slice(sliceA, sliceB)
    [slice(2, 5), slice(0, 3)]
    """
    region = []
    for n in range(len(sliceA)):
        sA, sB = sliceA[n], sliceB[n]
        start = sA.start if sA.start < sB.start else sB.start
        stop = sA.stop if sA.stop > sB.stop else sB.stop
        region.append(slice(start, stop))

    return region


def combine_slices(slices):
    """
    Returns the slice containing all input slices.

    Parameters
    ----------
    slices: list
        list of slice to combine

    Returns
    -------
    a slice with the min start and the max stop (step is set to None)

    Example
    -------
    >>> sliceA = [slice(2, 5), slice(1, 3)]
    >>> sliceB = [slice(2, 4), slice(0, 2)]
    >>> sliceC = [slice(3, 8), slice(1, 6)]
    >>> combine_slices([sliceA, sliceB, sliceC])
    [slice(2, 8), slice(0, 6)]
    """
    region = []
    dim_slice = len(slices[0])
    for dim in range(dim_slice):
        start = min([s[dim].start for s in slices])
        stop = max([s[dim].stop for s in slices])
        region.append(slice(start, stop))

    return region


def overlapping_slices(sliceA, sliceB):
    """
    Test if two tuple or list of slices are overlapping or not.

    Parameters
    ----------
    sliceA: tuple or list of slice
        slice for object A
    sliceB: tuple or list of slice
        slice for object B

    Returns
    -------
    a boolean, True if the slices are overlapping, else False.

    Examples
    --------
    >>> sliceA = [slice(2, 5), slice(1, 3)]
    >>> sliceB = [slice(2, 4), slice(0, 2)]
    >>> overlapping_slices(sliceA, sliceB)
    True

    >>> sliceA = [slice(2, 5), slice(2, 3)]
    >>> sliceB = [slice(2, 4), slice(0, 2)]
    >>> overlapping_slices(sliceA, sliceB)
    False
    """
    nA, nB = len(sliceA), len(sliceB)
    try:
        assert nA == nB
    except AssertionError:
        raise ValueError("Both slices should have the same dimensionality!")

    overlap = [False] * nA
    for n in range(nA):
        sA, sB = sliceA[n], sliceB[n]
        if min([sA.stop, sB.stop]) > max([sA.start, sB.start]):
            overlap[n] = True

    if np.alltrue(overlap):
        return True
    else:
        return False
