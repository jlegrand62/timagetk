# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Sophie Ribes <sophie.ribes@inria.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------
import numpy as np

try:
    from timagetk.components import SpatialImage
    from timagetk.algorithms import sets_metrics
    from timagetk.algorithms import imgtosets
    from timagetk.util import try_spatial_image
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))

__all__ = ['image_overlap']

np_unique = np.unique


def common_elements(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        return list(set(list1).intersection(set(list2)))


def diff_elements(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        return list(
            set(list1).union(set(list2)) - set(list1).intersection(set(list2)))


def image_overlap(sp_img_1, sp_img_2, labels=[], background_id=None,
                  criterion=None):
    """
    Performance evaluation of segmentation algorithms

    Parameters
    ----------
    sp_img_1: ``SpatialImage``
        input ``SpatialImage`` (segmentation)
    sp_img_2: ``SpatialImage``
        input ``SpatialImage`` (segmentation)
    labels: list, optional
        list of labels (default: all)
    background_id: int, optional
        background_id. If specified, removed from the computation
    criterion: str, optional
        criterion. Default is 'Jaccard coefficient'.
                criterion can be either:
                    ['Jaccard coefficient', 'Mean overlap', 'Target overlap',
                     'Volume similarity', 'False negative error',
                     'False positive error', 'Sensitivity', 'Conformity']

    Returns
    -------
    metric: float
        metric value (global)
    ov_dict: dict
        overlap dict (for each label)

    Example
    -------
    >>> jacc_val, jacc_dict = image_overlap(sp_img_1, sp_img_2, criterion='Jaccard coefficient')
    """
    # - Check if both input images are `SpatialImage`:
    try_spatial_image(sp_img_1)
    try_spatial_image(sp_img_2)

    possible_criteria = ['Jaccard coefficient', 'Mean overlap',
                         'Target overlap', 'Volume similarity',
                         'False negative error', 'False positive error',
                         'Sensitivity', 'Conformity']

    if ((criterion is None) or (criterion not in possible_criteria)):
        print('Possible criteria can be either:'), possible_criteria
        criterion = 'Jaccard coefficient'
        print('Setting criterion to:'), criterion
    else:
        criterion = str(criterion)

    labels_max_1 = sorted(np_unique(sp_img_1).tolist())
    labels_max_2 = sorted(np_unique(sp_img_2).tolist())

    comm_labs = common_elements(labels_max_1, labels_max_2)
    diff_labs = diff_elements(labels_max_1, labels_max_2)
    if labels is None or labels == []:
        labels = comm_labs
        if len(diff_labs) > 0:
            print('Diff labs'), diff_labs
    else:
        check_lab = [0 if lab in comm_labs else 1 for ind, lab in
                     enumerate(labels)]
        if 1 in check_lab:
            print('Incorrect label specification')
            return

    if background_id is not None and background_id in labels:
        labels.remove(background_id)

    ov_dict, metric = {}, 0
    sets_dict_1 = imgtosets(sp_img_1, label=labels, background_id=background_id)
    sets_dict_2 = imgtosets(sp_img_2, label=labels, background_id=background_id)
    for ind, lab in enumerate(labels):
        val = sets_metrics(sets_dict_1[lab], sets_dict_2[lab],
                           criterion=criterion)
        metric += val
        ov_dict[lab] = val
    metric = metric / len(labels)
    return metric, ov_dict
