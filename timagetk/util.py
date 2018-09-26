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

import os
import time

_ROOT = os.path.abspath(os.path.dirname(__file__))


def shared_folder():
    return os.path.join(_ROOT, 'share', 'data')


def data_path(filename):
    tmp = shared_folder()
    return os.path.join(tmp, str(filename))


def clean_warning(message, category, filename, lineno, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)


# TODO: write a `_connecitity_check` function working the same way than `_method_check`
def _method_check(method, valid_methods, default_index=0):
    """
    Check if a given string 'method' is valid and

    Parameters
    ----------
    method: str|None
        name of a method to check, if None returns valid_methods[default_index]
        as th default method
    valid_methods: list
        list of valid methods
    default_index: int, optional
        index of the deafault method to use in the list of 'valid_mathods'

    Returns
    -------
    methods: str
        a valid method
    """
    # - Set the default method if None:
    if method is None:
        method = valid_methods[default_index]
    # - Test the given method is valid:
    else:
        try:
            assert method in valid_methods
        except AssertionError:
            msg = "Unknown method '{}',".format(method)
            msg += "available methods are: {}".format(valid_methods)
            raise NotImplementedError(msg)

    return method


def _general_kwargs(param=False, verbose=False, time=False, debug=False,
                    **kwargs):
    """
    Keyword argument parser for VT general parameters.

    Parameters
    ----------
    param: bool, optional
        if True, default False, print the used parameters
    verbose: bool, optional
        if True, default False, increase code verbosity
    time: bool, optional
        if True, default False, print the CPU & USER elapsed time
    debug: bool, optional
        if True, default False, print the debug parameters

    Returns
    -------
    str_param: str
        VT formatted general parameters
    """
    str_param = ""
    # - Providing 'param=True' will result in printing parameters
    if param:
        str_param += ' -param'
    # - Providing 'verbose=True' will result in increased verbosity of the code
    if verbose:
        str_param += ' -verbose'
    else:
        str_param += ' -noverbose'
    # - Providing 'time=True' will result in printing CPU & User elapsed time
    if time:
        str_param += ' -time'
    else:
        str_param += ' -notime'
    # - Providing 'debug=True' will result in printing debug log
    if debug:
        str_param += ' -debug'
    else:
        str_param += ' -nodebug'

    return str_param


PARALLEL_TYPE = ['openmp', 'omp', 'pthread', 'thread']
DEFAULT_PARALLEL = "default"
OMP_TYPE = ['default', 'static', 'dynamic-one', 'dynamic', 'guided']


def _parallel_kwargs(parallel=True, parallel_type=DEFAULT_PARALLEL, n_job=None,
                     omp_scheduling=None, **kwargs):
    """
    Keyword argument parser for VT parallelism parameters.

    Parameters
    ----------
    parallel: bool, optional
        if True (default), use parallelism
    parallel_type: str, optional
        type of parallelisation to use, can be in ['openmp', 'omp', 'pthread', 'thread']
    n_job: int, optional
        number of core to use in parallel, by default use the maximum number of
        available CPU cores
    omp_scheduling: str, optional
        change 'OpenMP' scheduling option (if available), can be in ['default',
        'static', 'dynamic-one', 'dynamic', 'guided']

    Returns
    -------
    str_param: str
        VT formatted parallelism parameters
    """
    str_param = ""
    # - Check OMP-scheduling:
    if omp_scheduling is not None and parallel_type != 'openmp' or parallel_type != 'omp':
        parallel_type = 'openmp'
    # - Providing 'parallel=True' will result in using parallelism options:
    if parallel:
        str_param += ' -parallel'
        str_param += ' -parallel-type %s' % parallel_type
        if n_job is not None:
            str_param += ' -max-chunks %d' % n_job
        if omp_scheduling is not None:
            str_param += ' -omp-scheduling %s' % omp_scheduling
    else:
        str_param += ' -no-parallel'

    return str_param


def min_percent_step(N, default_step=5):
    """
    Compute the minimu step to apply when printing percentage of progress
    Parameters
    ----------
    N: int
        number of element over which to increment
    default_step: int
        default increment

    Returns
    -------
    minimum_step: int
        the minimum step to use
    """
    return max(default_step, 100 / N)


def percent_progress(progress, n, N, step=None):
    """
    Print a percentage of progress of n over N.

    Parameters
    ----------
    progress: int
        progress state
    n: int
        iteration number
    N: int
        max iteration number
    step: int
        progress step to apply for printing

    Returns
    -------
    percentage of progress print if any
    """
    if step is None:
        step = min_percent_step(N)

    if n == 0:
        print "0%...",
    if n * 100 / float(N) >= progress + step:
        print "{}%...".format(progress + step),
        progress += step
    if n + 1 == N:
        print "100%"

    return progress


def elapsed_time(start, stop=None, round_to=3):
    """
    Return a rounded elapsed time float.

    Parameters
    ----------
    start: float
        start time
    stop: float, optional
        stop time, if None, get it now
    round: int, optional
        number of decimals to returns

    Returns
    -------
    float
        rounded elapsed time
    """
    if stop is None:
        stop = time.time()
    t = round(stop - start, round_to)
    print "done in {}s".format(t)
    return


def get_class_name(obj):
    """
    Returns a string defining the class name.
    No module & package hierarchy returned.

    Parameters
    ----------
    obj : any
        any object for which you want to get the class name

    Returns
    -------
    str
        the name of the class
    """
    return str(type(obj))[:-2].split('.')[-1]


def get_attributes(obj, attr_list):
    """
    Return a dictionary with attributes values from 'obj'.
    By default they are set to None if not defined.

    Parameters
    ----------
    obj : any
        an object from which to try to get attributes
    attr_list : list(str)
        list of attributes to get from the object

    Returns
    -------
    dict
        attr_list as keys and the attribute value as their values.
    """
    return {attr: getattr(obj, attr, None) for attr in attr_list}
