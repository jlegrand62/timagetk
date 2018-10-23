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

try:
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
except ImportError:
    msg = "Could not import Matplotlib package, please install it!"
    msg += "\n"
    msg += "Use `conda install matplotlib` or `pip install matplotlib -U`."
    raise ImportError(msg)

_ROOT = os.path.abspath(os.path.dirname(__file__))


def shared_folder():
    return os.path.join(_ROOT, 'share', 'data')


def data_path(filename):
    tmp = shared_folder()
    return os.path.join(tmp, str(filename))


def clean_warning(message, category, filename, lineno, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)


def stuple(t):
    """
    Sort tuple 't' and return it

    Parameters
    ----------
    t : tuple
        the tuple to sort

    Returns
    -------
    tuple(sorted(t))
    """
    return tuple(sorted(t))


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


def slice_n_hist(image, title="", img_title="", figname="", aspect_ratio=None):
    """
    Display a 2D image with value histogram and cummulative histogram.

    Parameters
    ----------
    image : np.array or SpatialImage
        2D image to represent
    title : str, optional
        If provided (default is empty), add this string of characters as title
    img_title : str, optional
        If provided (default is empty), add this string of characters as title
    fig_name : str, optional
        If provided (default is empty), the image will be saved under this filename.
    aspect_ratio : tuple, optional
        if provided (default, None), change the aspect ratio of the displayed image
    """
    from timagetk.algorithms.exposure import type_to_range
    # TODO: make use of 'skimage.exposure.histogram' and 'skimage.exposure.cumulative_distribution' ?!
    try:
        assert image.ndim == 2
    except:
        raise ValueError("Input `image` should be 2D")

    if aspect_ratio is None:
        aspect_ratio = image.shape[0] / image.shape[1]

    mini, maxi = type_to_range(image)
    # Initialise figure:
    plt.figure()
    plt.suptitle(title)
    gs = gridspec.GridSpec(2, 2, width_ratios=[6, 3], height_ratios=[1, 1])
    # Display 2D image:
    ax = plt.subplot(gs[:, 0])
    plt.imshow(image, 'gray', vmin=mini, vmax=maxi, aspect=aspect_ratio)
    plt.axis('off')
    plt.title(img_title)
    # Plot intensity histogram
    ax = plt.subplot(gs[0, 1])
    plt.title('Intensity histogram')
    plt.hist(image.flatten(), bins=256, range=(mini, maxi + 1), normed=True)
    # exposure.histogram(image)
    # Plot intensity cumulative histogram
    ax = plt.subplot(gs[1, 1])
    plt.title('Cumumative histogram')
    plt.hist(image.flatten(), bins=256, range=(mini, maxi + 1), cumulative=True,
             histtype='step', normed=True)
    # exposure.cumulative_distribution(image)
    if figname != "":
        plt.savefig(figname)

    return


def slice_view(img, x_slice=None, y_slice=None, z_slice=None, title="",
               fig_name="", cmap='gray'):
    """
    Matplotlib representation of an image slice.
    Slice numbering starts at 1, not 0 (like indexing).
    Note that at least one of the '*_slice' parameter should be given, and all
    three can be given for an orthogonal representation of the stack.

    Parameters
    ----------
    img : np.array or SpatialImage
        Image from which to extract the slice
    x_slice : int
        Value defining the slice to represent in x direction.
    y_slice : int
        Value defining the slice to represent in y direction.
    z_slice : int
        Value defining the slice to represent in z direction.
    title : str, optional
        If provided (default is empty), add this string of characters as title
    fig_name : str, optional
        If provided (default is empty), the image will be saved under this filename.
    """
    from timagetk.algorithms.exposure import type_to_range
    try:
        assert x_slice is not None or y_slice is not None or z_slice is not None
    except:
        raise ValueError("Provide at least one x, y or z slice to extract!")
    x_sl, y_sl, z_sl = None, None, None
    if x_slice is not None:
        x_sl = img[x_slice - 1, :, :]
    if y_slice is not None:
        y_sl = img[:, y_slice - 1, :]
    if z_slice is not None:
        z_sl = img[:, :, z_slice - 1]
    # If only one slice is required, display it "alone":
    if sum([sl is not None for sl in [x_sl, y_sl, z_sl]]) == 1:
        sl = [s for s in [x_sl, y_sl, z_sl] if s is not None][0]
        mini, maxi = type_to_range(sl)
        plt.figure()
        plt.imshow(sl, cmap, vmin=mini, vmax=maxi)
        plt.title(title)
        if fig_name != "":
            plt.savefig(fig_name)
    # If three slices are required, display them "orthogonaly":
    elif sum([sl is not None for sl in [x_sl, y_sl, z_sl]]) == 3:
        x_sh, y_sh, z_sh = img.shape
        mini, maxi = type_to_range(img)
        plt.figure()
        gs = gridspec.GridSpec(2, 2, width_ratios=[x_sh, z_sh],
                               height_ratios=[y_sh, z_sh])
        # plot z_slice:
        ax = plt.subplot(gs[0, 0])
        plt.plot([x_slice, x_slice], [0, y_sh], color='yellow')
        plt.plot([0, x_sh], [y_slice, y_slice], color='yellow')
        plt.imshow(z_sl, cmap, vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('z-slice {}/{}'.format(z_slice, z_sh))
        # plot y_slice
        ax = plt.subplot(gs[0, 1])
        plt.plot([z_slice, z_slice], [0, y_sh], color='yellow')
        plt.imshow(y_sl, cmap, vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('y-slice {}/{}'.format(y_slice, y_sh))
        # plot x_slice
        ax = plt.subplot(gs[1, 0])
        plt.plot([0, x_sh], [z_slice, z_slice], color='yellow')
        plt.imshow(x_sl.T, cmap, vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('x-slice {}/{}'.format(x_slice, x_sh))
        # Add suptitle:
        plt.suptitle(title)
        if fig_name != "":
            plt.savefig(fig_name)
    else:
        print "You should not be here !!"
        pass
    return
