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
#------------------------------------------------------------------------------
import numpy as np
from sets import Set

__all__ = ['sets_metrics']

np_around, np_inf = np.around, np.inf

POSS_CRITERIA = ['Jaccard coefficient', 'Mean overlap', 'Target overlap',
                 'Volume similarity', 'False negative error',
                 'False positive error', 'Sensitivity', 'Conformity']
DEF_CRITERIA = POSS_CRITERIA[0]

def sets_metrics(set_1, set_2, criterion=None):
    # TODO add specificity, compound
    """
    Compute metrics between sets (overlap measurements for individual labeled region).

    Parameters
    ----------
    set_1: Set
        set 1 (source)
    set_2: Set
        set 2 (target)
    criterion: str
        criterion, optional. Default is 'Jaccard coefficient'.
                criterion can be either:
                    ['Jaccard coefficient', 'Mean overlap', 'Target overlap',
                     'Volume similarity', 'False negative error',
                     'False positive error', 'Sensitivity', 'Conformity']

    Returns
    -------
    output: float
        metric value

    Example
    -------
    metric = sets_metrics(set_1, set_2, criterion)

    Reference (implementation): see Chang et al., "Performance measure
    characterization for evaluating neuroimage segmentation algorithms,
    NeuroImage, 2009."
    """
    try:
        assert isinstance(set_1, Set)
    except AssertionError:
        raise TypeError('set_1 must be a Set instance')
    try:
        assert isinstance(set_2, Set)
    except AssertionError:
        raise TypeError('set_2 must be a Set instance')
    
    if ((criterion is None) or (criterion not in POSS_CRITERIA)):
        print('Possible criteria can be either:'), POSS_CRITERIA
        criterion = DEF_CRITERIA
        print('Setting criterion to:'), criterion
    else:
        criterion = str(criterion)

    if (criterion == 'Jaccard coefficient'):
        # Jaccard distance
        jaccard_coefficient = float(len(set_1.intersection(set_2)))/float(len(set_1.union(set_2)))
        metric = jaccard_coefficient

    elif (criterion == 'Mean overlap'):
        # Mean overlap (Dice coefficient)
        mean_overlap = 2.0*(float(len(set_1.intersection(set_2)))/(float(len(set_1)) + float(len(set_2))))
        metric = mean_overlap

    elif (criterion == 'Target overlap'):
        # Target overlap
        target_overlap = float(len(set_1.intersection(set_2)))/float(len(set_2))
        metric = target_overlap

    elif (criterion == 'Volume similarity'):
        # Volume similarity
        volume_similarity = 2.0*((float(len(set_1)) - float(len(set_2)))/(float(len(set_1)) + float(len(set_2))))
        metric = volume_similarity

    elif (criterion == 'False negative error'):
        # False negative error
        false_negative_error = float(len(set_2.difference(set_1)))/float(len(set_2))
        metric = false_negative_error

    elif (criterion == 'False positive error'):
        # False positive error
        false_positive_error = float(len(set_1.difference(set_2)))/float(len(set_1))
        metric = false_positive_error

    elif (criterion == 'Sensitivity'):
        # Sensitivity
        false_negative_error = float(len(set_2.difference(set_1)))/float(len(set_2))
        sensitivity = float(len(set_1.intersection(set_2))) / (float(len(set_1.intersection(set_2))) + false_negative_error)
        metric = sensitivity

    elif (criterion == 'Conformity'):
        # Conformity
        tp = float(len(set_1.intersection(set_2)))
        fpe = float(len(set_1.difference(set_2)))/float(len(set_1))
        fne = float(len(set_2.difference(set_1)))/float(len(set_2))
        if tp != 0:
            metric = 1.0 - ((fpe + fne)/ tp)
        elif tp == 0:
            # failure !
            metric = np_inf

    metric = np_around(metric, decimals=3).tolist()
    return metric
