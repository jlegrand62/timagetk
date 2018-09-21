# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Jerome Chopard <jerome.chopard@inria.fr>
#           Gregoire Malandain <gregoire.malandain@inria.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       See accompanying file LICENSE.txt
# -----------------------------------------------------------------------------

# ----
# SR: update - 08/2016
# numpy types, tests and conds, 2D and 3D management
# origin, voxelsize, extent, mean, min, max and metadata
# set and get methods
# to_2D(), to_3D() methods
# documentation and unit tests (see test_spatial_image.py)

from __future__ import division

import numpy as np

__all__ = ['SpatialImage']

# - Define default values:
EPS = 1e-9
DEC_VAL = 6
DEFAULT_VXS_2D, DEFAULT_VXS_3D = [1.0, 1.0], [1.0, 1.0, 1.0]
DEFAULT_ORIG_2D, DEFAULT_ORIG_3D = [0, 0], [0, 0, 0]

# - Define possible values for 'dtype':
DICT_TYPES = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
              5: np.int32, 6: np.uint64, 7: np.int64, 8: np.float32,
              9: np.float64, 10: np.float_, 11: np.complex64, 12: np.complex128,
              13: np.complex_, 'uint8': np.uint8, 'uint16': np.uint16,
              'ushort': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
              'uint': np.uint64, 'ulonglong': np.uint64, 'int8': np.int8,
              'int16': np.int16, 'short': np.int16, 'int32': np.int32,
              'int64': np.int64, 'int': np.int64, 'longlong': np.int64,
              'float16': np.float16, 'float32': np.float32,
              'single': np.float32, 'float64': np.float64, 'double': np.float64,
              'float': np.float64, 'float128': np.float_,
              'longdouble': np.float_, 'longfloat': np.float_,
              'complex64': np.complex64, 'singlecomplex': np.complex64,
              'complex128': np.complex128, 'cdouble': np.complex128,
              'cfloat': np.complex128, 'complex': np.complex128,
              'complex256': np.complex_, 'clongdouble': np.complex_,
              'clongfloat': np.complex_, 'longcomplex': np.complex_}
AVAIL_TYPES = sorted([k for k in DICT_TYPES if isinstance(k, str)])
# - Define default type:
DEFAULT_DTYPE = DICT_TYPES[0]
# - List of protected attribute or poperties:
PROTECT_PPTY = ['shape', 'min', 'max', 'mean']
# - Array equality testing methods:
EQ_METHODS = ['max_error', 'cum_error']


def around_list(input_list, dec_val=DEC_VAL):
    if isinstance(input_list, list) and len(input_list) > 0:
        output_list = [np.around(input_list[ind], decimals=dec_val).tolist()
                       for ind, val in enumerate(input_list)]
        return output_list
    else:
        return []


def tuple_to_list(input_tuple):
    if isinstance(input_tuple, tuple):
        output_list = [input_tuple[ind] for ind, val in enumerate(input_tuple)]
        output_list = around_list(output_list)
        return output_list
    else:
        return []


def dimensionality_test(dim, list2test):
    """ Quick testing of dimensionality with print in case of error."""
    d = len(list2test)
    try:
        assert d == dim
    except:
        msg = "Provided values ({}) is not of the same than the array ({})!"
        raise ValueError(msg.format(d, dim))


def tuple_array_to_list(val):
    """
    Returns a list if a tuple or array is provided, else raise a TypeError.
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()
    elif isinstance(val, tuple):
        val = list(val)
    if not isinstance(val, list):
        raise TypeError("Accepted type are tuple, list and np.array!")
    else:
        return val

def check_dimensionality(dim, list2test):
    """
    Tests list dimensionality against array dimensionality.
    """
    try:
        dimensionality_test(dim, list2test)
    except ValueError:
        return None
    else:
        return list2test

def compare_kwargs_metadata(kwd_val, md_val, dim):
    """
    Compare two values, usually keyword or attribute against metadata.
    If they are not similar, return kwd_val.
    
    Parameters
    ----------
    kwd_val : any
        keyword or attribute value
    md_val : any
        metadata value

    Returns
    -------
    any|None
        value
    """
    if kwd_val == md_val:
        return check_dimensionality(dim, kwd_val)
    else:
        return check_dimensionality(dim, kwd_val)


def set_default_origin(input_array):
    """
    Return the default origin depending on the array dimensionality.

    Parameters
    ----------
    input_array : numpy.ndarray
        2D or 3D array defining an image, eg. intensity or labelled image

    Returns
    -------
    list
        default origin coordinates
    """
    print "Warning: incorrect 'origin' specification,",
    if input_array.ndim == 2:
        orig = DEFAULT_ORIG_2D
    else:
        orig = DEFAULT_ORIG_3D
    print "set to default value:", orig

    return orig


def set_default_voxelsize(input_array):
    """
    Return the default voxelsize depending on the array dimensionality.

    Parameters
    ----------
    input_array : numpy.ndarray
        2D or 3D array defining an image, eg. intensity or labelled image

    Returns
    -------
    list
        default voxelsize value
    """
    print "Warning: incorrect 'voxelsize' specification,",
    if input_array.ndim == 2:
        orig = DEFAULT_VXS_2D
    else:
        orig = DEFAULT_VXS_3D
    print "set to default value:", orig

    return orig


def obj_metadata(obj, metadata_dict=None):
    """
    Build the metadata dictionary for basics image array infos.
    Can compare it to a existing one.

    Parameters
    ----------
    obj: SpatialImage
        a SpatialImage to use for metadata definition
    metadata_dict: dict, optional
        a metadata dictionary to compare to the object variables

    Returns
    -------
    metadata_dict: dict
        a verified metadata dictionary
    """
    if not metadata_dict:
        metadata_dict = {}

    # -- Check the most important object values against potentially
    # predefined values in the metadata_dict:
    try:
        assert np.alltrue(metadata_dict['voxelsize'] == obj._voxelsize)
    except KeyError:
        metadata_dict['voxelsize'] = obj._voxelsize
    except AssertionError:
        raise ValueError(
            "Metadata 'voxelsize' does not match the object voxelsize!")
    try:
        assert metadata_dict['shape'] == obj.shape
    except KeyError:
        metadata_dict['shape'] = obj.shape
    except AssertionError:
        print "WARNING: Metadata 'shape' {} do not match the array shape {},".format(
            metadata_dict['shape'], obj.shape),
        metadata_dict['shape'] = obj.shape
        print "it has been updated!"

    try:
        assert metadata_dict['dim'] == obj.ndim
    except KeyError:
        metadata_dict['dim'] = obj.ndim
    except AssertionError:
        raise ValueError("Metadata 'dim' does not match the array dim!")

    try:
        assert np.alltrue(metadata_dict['origin'] == obj._origin)
    except KeyError:
        metadata_dict['origin'] = obj._origin
    except AssertionError:
        raise ValueError(
            "Metadata 'origin' does not match the object origin!")

    try:
        assert np.alltrue(metadata_dict['extent'] == obj._extent)
    except KeyError:
        metadata_dict['extent'] = obj._extent
    except AssertionError:
        raise ValueError(
            "Metadata 'extent' does not match the array extent!")

    try:
        assert metadata_dict['type'] == str(obj.dtype)
    except KeyError:
        metadata_dict['type'] = str(obj.dtype)
    except AssertionError:
        print "WARNING: Metadata 'type' ({}) do not match the array dtype ({}),".format(
            metadata_dict['type'], obj.dtype),
        metadata_dict['dtype'] = str(obj.dtype)
        print "it has been updated!"

    # Next lines compute min, max and mean values of the array every time we
    # call SpatialImage, even when reading the file from disk! This slow down
    # the process!
    # metadata_dict['min'] = obj.min()
    # metadata_dict['max'] = obj.max()
    # metadata_dict['mean'] = obj.mean()
    return metadata_dict


class SpatialImage(np.ndarray):
    """
    This class allows a management of ``SpatialImage`` objects (2D and 3D images).
    A ``SpatialImage`` gathers a numpy array and some metadata (such as voxelsize,
    physical extent, origin, type, etc.).
    Through a ``numpy.ndarray`` inheritance, all usual operations on
    `numpy.ndarray <http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ndarray.html>`_
    objects (sum, product, transposition, etc.) are available.
    All image processing operations are performed on this data structure, that
    is also used to solve inputs (read) and outputs (write).
    """

    def __new__(cls, input_array, origin=None, voxelsize=None, dtype=None,
                metadata_dict=None, **kwargs):
        """
        ``SpatialImage`` constructor (2D and 3D images)

        Parameters
        ----------
        input_array: numpy.ndarray
            2D or 3D array defining an image, eg. intensity or labelled image
        origin: list, optional
            coordinates of the origin in the image, default: [0,0] or [0,0,0]
        voxelsize: list, optional.
            image voxelsize, default: [1.0,1.0] or [1.0,1.0,1.0]
        dtype: str, optional
            image type, default dtype = input_array.dtype
        metadata_dict: dict, optional
            dictionary of image metadata, default is an empty dict

        Returns
        -------
        SpatialImage
            image with metadata

        Notes
        -----
        To use 'input_array' type (from numpy), leave 'dtype' to None.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image_1 = SpatialImage(input_array=test_array)
        >>> isinstance(image_1, SpatialImage)
        True
        >>> image_2 = SpatialImage(input_array=test_array, voxelsize=[0.5,0.5])
        >>> isinstance(image_2, SpatialImage)
        True
        >>> print image_2.voxelsize
        [0.5, 0.5]
        """
        # - Test input array: should be a numpy array of dimension 2 or 3:
        try:
            assert isinstance(input_array, np.ndarray)
        except AssertionError:
            raise TypeError("Input array is not a numpy array!")
        try:
            assert input_array.ndim in [2, 3]
        except AssertionError:
            msg = "Input array must have a dimensionality equal to 2 or 3"
            print ValueError(msg)
            return None  # WEIRD behavior... seems required by some functions
            # Original behavior was to test if the dim was 2D or 3D but was not
            # doing anything otherwise (no else declared!).
        else:
            ndim = input_array.ndim

        # - Initialize or use given metadata dictionary:
        if metadata_dict is None or metadata_dict == []:
            metadata_dict = {}
        else:
            try:
                assert isinstance(metadata_dict, dict)
            except:
                msg = "Parameter 'metadata_dict' should be a dictionary!"
                raise TypeError(msg)

        # ----------------------------------------------------------------------
        # DTYPE:
        # ----------------------------------------------------------------------
        # -- Check or set the type of data from 'dtype':
        if dtype is not None:
            # Check it is a known 'dtype':
            try:
                assert dtype in AVAIL_TYPES
            except AssertionError:
                msg = "Unknown 'dtype' value '{}', available types are: {}"
                raise ValueError(msg.format(dtype, AVAIL_TYPES))
            else:
                if isinstance(dtype, np.dtype):
                    dtype = str(dtype)
                dtype = DICT_TYPES[dtype]
        else:
            # Convert instance 'np.dtype' into a string:
            dtype = input_array.dtype
            dtype = str(dtype)

        # ----------------------------------------------------------------------
        # ARRAY:
        # ----------------------------------------------------------------------
        # - Check input array 'flags' attribute, use 'dtype' value:
        if input_array.flags.f_contiguous:
            obj = np.asarray(input_array, dtype=dtype).view(cls)
        else:
            obj = np.asarray(input_array, dtype=dtype, order='F').view(cls)

        # ----------------------------------------------------------------------
        # ORIGIN:
        # ----------------------------------------------------------------------
        # -- Get 'origin' value in metadata:
        try:
            origin_md = metadata_dict['origin']
        except KeyError:
            origin_md = None
        # -- Check 'origin' value using keyword argument and metadata:
        if origin is None and origin_md is None:
            # Neither keyword nor metadata defined:
            orig = set_default_origin(input_array)
        elif origin is not None and origin_md is not None:
            # Both keyword and metadata defined:
            orig = compare_kwargs_metadata(origin, origin_md, ndim)
        else:
            # Only one of the two defined:
            if origin is not None:
                orig = origin
            else:
                orig = origin_md
            orig = check_dimensionality(input_array.ndim, orig)
        # -- Set 'origin' attribute and set|update metadata:
        obj._origin = orig
        metadata_dict['origin'] = orig

        # ----------------------------------------------------------------------
        # VOXELSIZE:
        # ----------------------------------------------------------------------
        # -- Transform keyword argument 'voxelsize' into a list if possible,
        #  else set to None:
        if isinstance(voxelsize, tuple):  # SR --- BACK COMPAT
            voxelsize = around_list(tuple_to_list(voxelsize))
        if isinstance(voxelsize, np.ndarray):
            voxelsize = around_list(voxelsize.tolist())
        if not isinstance(voxelsize, list):
            voxelsize = None

        # -- Get 'voxelsize' value in metadata:
        try:
            voxelsize_md = metadata_dict['voxelsize']
        except KeyError:
            voxelsize_md = None
        # -- Check 'voxelsize' value using keyword argument and metadata:
        if voxelsize is None and voxelsize_md is None:
            # Neither keyword nor metadata defined:
            vxs = set_default_voxelsize(input_array)
        elif voxelsize is not None and voxelsize_md is not None:
            # Both keyword and metadata defined:
            vxs = compare_kwargs_metadata(voxelsize, voxelsize_md, ndim)
        else:
            # Only one of the two defined:
            if voxelsize is not None:
                vxs = voxelsize
            else:
                vxs = voxelsize_md
                vxs = check_dimensionality(input_array.ndim, vxs)
        # -- Set 'voxelsize' attribute and set|update metadata:
        obj._voxelsize = vxs
        metadata_dict['voxelsize'] = vxs

        # ----------------------------------------------------------------------
        # EXTENT:
        # ----------------------------------------------------------------------
        shape = input_array.shape
        dim = input_array.ndim
        # -- Compute image 'extent' based on shape and voxelsize:
        ext = [vxs[i] * shape[i] for i in xrange(dim)]
        ext = around_list(ext)
        # -- Set 'extent' attribute and set|update metadata:
        obj._extent = ext
        metadata_dict['extent'] = ext

        # ----------------------------------------------------------------------
        # METADATA:
        # ----------------------------------------------------------------------
        # TODO: SpatialImage could have 'filename' attribute or metadata ?
        # -- Set/update the metadata:
        obj._metadata = obj_metadata(obj, metadata_dict)

        return obj

    def __init__(self, input_array, **kwargs):
        """
        """
        pass

    def __array_finalize__(self, obj):
        """
        This is the mechanism that numpy provides to allow subclasses to handle
        the various ways that new instances get created.

        Parameters
        ----------
        obj: the object returned by the __new__ method.
        """
        if obj is not None:
            self._voxelsize = getattr(obj, '_voxelsize', [])
            self._origin = getattr(obj, '_origin', [])
            self._extent = getattr(obj, '_extent', [])
            self._metadata = getattr(obj, '_metadata', {})
            self.min = getattr(obj, 'min', [])
            self.max = getattr(obj, 'max', [])
            self.mean = getattr(obj, 'mean', [])
        else:
            pass

    def __str__(self):
        """
        """
        print "SpatialImage object with following metadata:"
        print  self._metadata
        return

    def astype(self, type, **kwargs):
        """
        Copy of the SpatialImage with updated data type.

        Parameters
        ----------
        type: str
            new type of data to apply
        kwargs: dict
            given to numpy array 'astype()' method

        Returns
        -------
        SpatialImage
            image with the new data type
        """
        # - Convert the numpy array:
        array = self.get_array().astype(type, **kwargs)
        # - Get 'origin', 'voxelsize' & 'metadata':
        origin = self.origin
        voxelsize = self.voxelsize
        md = self.metadata
        # - Update metadata 'type' to new type:
        md['type'] = type
        return SpatialImage(array, origin=origin, voxelsize=voxelsize,
                            metadata_dict=md)

    def is_isometric(self):
        """
        Test if the image is isometric, meaning the voxelsize value is the same
        in every direction.

        Returns
        -------
        is_iso: bool
            True is isometric, else False.
        """
        vxs = self.voxelsize
        is_iso = np.alltrue([vxs_i == vxs[0] for vxs_i in vxs[1:]])
        return is_iso

    def is2D(self):
        """
        Returns True if the SpatialImage is 2D, else False.
        """
        return self.get_dim() == 2

    def is3D(self):
        """
        Returns True if the SpatialImage is 3D, else False.
        """
        return self.get_dim() == 3

    def to_2D(self):
        """
        Convert, if possible, a 3D SpatiamImage with one "flat" dimension (ie.
        with only one slice in this dimension) to a 2D SpatialImage.

        Returns
        -------
        SpatialImage
            the 2D SpatialImage
        """
        if self.is3D() and 1 in self.shape:
            voxelsize, shape, array = self.voxelsize, self.shape, self.get_array()
            ori, md = self.origin, self.metadata
            if shape[0] == 1:
                new_arr = np.squeeze(array, axis=(0,))
                new_vox = [voxelsize[1], voxelsize[2]]
                new_ori = [ori[1], ori[2]]
            elif shape[1] == 1:
                new_arr = np.squeeze(array, axis=(1,))
                new_vox = [voxelsize[0], voxelsize[2]]
                new_ori = [ori[0], ori[2]]
            else:
                new_arr = np.squeeze(array, axis=(2,))
                new_vox = [voxelsize[0], voxelsize[1]]
                new_ori = [ori[0], ori[1]]
            out_sp_img = SpatialImage(new_arr, voxelsize=new_vox,
                                      origin=new_ori, metadata_dict=md)
            return out_sp_img
        else:
            print('This 3D SpatialImage can not be reshaped to 2D.')
            return

    def to_3D(self):
        """
        Convert, a 2D SpatiamImage into a 3D SpatialImage with one "flat"
        dimension (ie. with only one slice in this dimension).

        Returns
        -------
        SpatialImage
            the 3D SpatialImage
        """
        if self.is2D():
            voxelsize, shape, array = self.voxelsize, self.shape, self.get_array()
            ori, md = self.origin, self.metadata
            new_arr = np.reshape(array, (shape[0], shape[1], 1))
            new_vox = [voxelsize[0], voxelsize[1], 1.0]
            new_ori = [ori[0], ori[1], 0]
            out_sp_img = SpatialImage(new_arr, voxelsize=new_vox,
                                      origin=new_ori, metadata_dict=md)
            return out_sp_img
        else:
            print('This SpatialImage is not 2D.')
            return

    def get_available_types(self):
        """
        Print the available bits type dictionary.
        """
        return DICT_TYPES

    def is_available_types(self, type):
        """
        Test if the given type is available.

        Parameters
        ----------
        type: str
            name of the type to find in DICT_TYPES
        """
        return type in DICT_TYPES.keys()

    def equal(self, sp_img, error=EPS, method='max_error'):
        """
        Equality test between two ``SpatialImage``, uses array equality and
        metadata matching.

        Parameters
        ----------
        sp_img: ``SpatialImage``
            another ``SpatialImage`` instance to test for array equality
        error: float, optional
            maximum difference accepted between the two arrays (default=EPS)
        method: str, optional
            type of "error measurement", choose among (default='max_error'):
              - max_error: max difference accepted for a given pixel
              - cum_error: max cumulative (sum) difference for the whole array

        Returns
        -------
        bool
            True/False if (array and metadata) are equal/or not

        Notes
        -----
        Metadata equality test compare defined self.metadata keys to their
        counterpart in 'sp_img'. Hence, a missing key in 'sp_im' or a different
        value will return False.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image_1 = SpatialImage(input_array=test_array)
        >>> image_1.equal(image_1)
        True
        >>> image_2 = SpatialImage(input_array=test_array, voxelsize=[0.5,0.5])
        >>> image_1.equal(image_2)
        SpatialImages metadata are different.
        False
        >>> image_2[1, 1] = 2
        >>> image_1.equal(image_2)
        Max difference between arrays is greater than '1e-09'.
        SpatialImages metadata are different.
        False
        """
        equal = False
        # - Test array equality:
        t_arr = self.equal_array(sp_img, error=error, method=method)
        # - Test metadata equality:
        md_ref = self.metadata
        md = sp_img.metadata
        t_met = all([md.has_key(k) and v == md[k] for k, v in md_ref.items()])
        # - Combine test and print when fail:
        if t_arr and t_met:
            equal = True
        if not t_arr:
            m = 'Max' if method == 'max_error' else 'Cumulative'
            print "{} difference between arrays is greater than '{}'.".format(
                m, error)
        if not t_met:
            print "SpatialImages metadata are different."

        return equal

    def equal_array(self, sp_img, error=EPS, method='max_error'):
        """
        Test array equality between two ``SpatialImage``.

        Parameters
        ----------
        sp_img: ``SpatialImage``
            another ``SpatialImage`` instance to test for array equality
        error: float, optional
            maximum difference accepted between the two arrays, should be
            strictly inferior to this value to return True, default: EPS
        method: str, optional
            type of "error measurement", choose among:
              - max_error: max difference accepted for a given pixel
              - cum_error: max cumulative (sum) difference for the whole array

        Returns
        -------
        bool
            True/False if arrays are equal/or not

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image_1 = SpatialImage(input_array=test_array)
        >>> image_1.equal_array(image_1)
        True
        >>> # - Changing voxelsize does not affect array equality test:
        >>> image_2 = SpatialImage(input_array=test_array, voxelsize=[0.5,0.5])
        >>> image_1.equal_array(image_2)
        True
        >>> # - Changing array value does affect array equality test:
        >>> image_2[0, 0] = 0
        >>> image_1.equal_array(image_2)
        False
        >>> # - Changing accepted max difference affect array equality test:
        >>> image_1.equal_array(image_2, error=2)
        True
        """
        if not isinstance(sp_img, SpatialImage):
            raise TypeError("Parameter 'sp_img' is not a SpatialImage!")
        try:
            assert method in EQ_METHODS
        except AssertionError:
            msg = "Unknown method '{}', should be in {}."
            raise ValueError(msg.format(method, EQ_METHODS))

        # - Starts by testing the shapes are equal:
        if self.shape != sp_img.shape:
            msg = "SpatialImage 'sp_img' has a different shape than this one!"
            print msg
            return False

        # - Test array equality:
        ori_type = self.dtype
        if ori_type.startswith(
                'u'):  # unsigned case is problematic for 'np.subtract'
            tmp_type = DICT_TYPES[ori_type[1:]]
        else:
            tmp_type = ori_type
        # - Compute the difference between the two arrays:
        out_img = np.abs(np.subtract(self, sp_img).astype(tmp_type)).astype(
            ori_type)
        # - Try to find non-null values in this array:
        non_null_idx = np.nonzero(out_img)
        if len(non_null_idx[0]) != 0:
            non_null = out_img[non_null_idx]
            if method == 'max_error':
                equal = np.max(non_null) < error
            else:
                equal = np.sum(non_null) < error
        else:
            equal = True

        return equal

    def get_array(self):
        """
        Get a ``numpy.ndarray`` from a ``SpatialImage``

        Returns
        -------
        numpy.ndarray:
            ``SpatialImage`` array

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image_array = image.get_array()
        >>> isinstance(image_array, SpatialImage)
        False
        >>> isinstance(image_array, np.ndarray)
        True
        """
        return np.array(self)

    def get_dim(self):
        """
        Get ``SpatialImage`` dimension (2D or 3D image)

        Returns
        -------
        int:
            ``SpatialImage`` dimensionality

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image_dim = image.get_dim()
        >>> print image_dim
        2
        """
        return self.ndim

    def get_min(self):
        """
        Get ``SpatialImage`` min value

        Returns
        -------
        *val*
            ``SpatialImage`` min

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image_min = image.get_min()
        >>> print image_min
        1
        """
        try:
            return self._min
        except AttributeError:
            self._min = self.min()
            self.metadata.update({'min': self._min})
            return self._min

    def get_max(self):
        """
        Get ``SpatialImage`` max value

        Returns
        -------
        *val*
            ``SpatialImage`` max

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image_max = image.get_max()
        >>> print image_max
        1
        """
        try:
            return self._max
        except AttributeError:
            self._max = self.max()
            self.metadata.update({'max': self._max})
            return self._max

    def get_mean(self):
        """
        Get ``SpatialImage`` mean

        Returns
        -------
        *val*
            ``SpatialImage`` mean

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image_mean = image.get_mean()
        >>> print image_mean
        1
        """
        try:
            return self._mean
        except AttributeError:
            self._mean = self.mean()
            self.metadata.update({'mean': self._mean})
            return self._mean

    def get_pixel(self, indices):
        """
        Get ``SpatialImage`` pixel value

        Parameters
        ----------
        indices: list
            indices as list of integers

        Returns
        -------
        *self.dtype*
            pixel value

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> indices = [1,1]
        >>> pixel_value = image.get_pixel(indices)
        >>> print pixel_value
        1
        """
        img_dim = self.get_dim()
        if isinstance(indices, list) and len(indices) == img_dim:
            img_shape = self.shape
            if img_dim == 2:
                range_x, range_y = xrange(img_shape[0]), xrange(img_shape[1])
                conds_ind = indices[0] in range_x and indices[1] in range_y
            elif img_dim == 3:
                range_x, range_y, range_z = xrange(img_shape[0]), xrange(
                    img_shape[1]), xrange(img_shape[2])
                conds_ind = indices[0] in range_x and indices[1] in range_y and \
                            indices[2] in range_z
            if conds_ind:
                if img_dim == 2:
                    pix_val = self[indices[0], indices[1]]
                elif img_dim == 3:
                    pix_val = self[indices[0], indices[1], indices[2]]
            return pix_val
        else:
            print('Warning, incorrect specification')
            return

    def get_region(self, indices):
        """
        Extract a region using list of start & stop 'indices'.
        There should be two values per image dimension in 'indices'.
        If the image is 3D and in one dimension, the start and stop are differ
        by one (on layer of voxels), the image is transformed to 2D!

        Parameters
        ----------
        indices: list
            indices as list of integers

        Returns
        -------
        ``SpatialImage``
            output ``SpatialImage``

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> indices = [1,3,1,3]
        >>> out_sp_image = image.get_region(indices)
        >>> isinstance(out_sp_image, SpatialImage)
        True
        """
        # TODO: use slice instead of 'indices' list?
        try:
            assert isinstance(indices, list)
        except AssertionError:
            raise TypeError("Parameters 'indices' must be a list!")

        img_dim = self.get_dim()
        try:
            assert len(indices) == 2 * img_dim
        except AssertionError:
            raise TypeError(
                "Parameter 'indices' must have twice the number of dimension of the image!")

        sh = [(0, s) for s in self.shape]
        conds_ind = all(
            [(indices[i] > sh[i]) & (indices[i + 1] < sh[i + 1] + 1) for i in
             range(0, 2 * img_dim, 2)])
        conds_val = all(
            [indices[i + 1] - indices[i] >= 1 for i in
             range(0, 2 * img_dim, 2)])

        if img_dim < 2 or img_dim > 3:
            raise ValueError(
                "SpatialImage can be 2D or 3D, check dimensionality!")

        if not conds_ind:
            err = "Given 'indices' are not within the image shape!"
            raise ValueError(err)
        elif not conds_val:
            err = "Given 'indices' are wrong in at least one direction!"
            raise ValueError(err)
        else:
            pass

        bbox = (slice(indices[i], indices[i + 1]) for i in
                range(0, 2 * img_dim, 2))
        tmp_arr, tmp_vox = self.get_array(), self.voxelsize
        reg_val = tmp_arr[bbox]
        if img_dim == 3 & 1 in reg_val.shape:  # 3D --> 2D
            if reg_val.shape[0] == 1:
                reg_val = np.squeeze(reg_val, axis=(0,))
                tmp_vox = [tmp_vox[1], tmp_vox[2]]
            elif reg_val.shape[1] == 1:
                reg_val = np.squeeze(reg_val, axis=(1,))
                tmp_vox = [tmp_vox[0], tmp_vox[2]]
            elif reg_val.shape[2] == 1:
                reg_val = np.squeeze(reg_val, axis=(2,))
                tmp_vox = [tmp_vox[0], tmp_vox[1]]

        out_sp_img = SpatialImage(input_array=reg_val, voxelsize=tmp_vox)
        return out_sp_img

    # Commented since 'shape' is already an attribute of numpy.array
    # def get_shape(self):
    #     """
    #     Get ``SpatialImage`` shape
    #
    #     Returns
    #     ----------
    #     :returns: image_shape (*tuple*) -- ``SpatialImage`` shape
    #
    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> from timagetk.components import SpatialImage
    #     >>> test_array = np.ones((5,5), dtype=np.uint8)
    #     >>> image = SpatialImage(input_array=test_array)
    #     >>> image_shape = image.shape
    #     >>> print image_shape
    #     (5, 5)
    #     """
    #     return self.shape

    def set_pixel(self, indices, value):
        """
        Set ``SpatialImage`` pixel value at given array coordinates.

        Parameters
        ----------
        indices: list
            indices as list of integers
        value: array.dtype
            new value for the selected pixel, type of ``SpatialImage`` array

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image.set_pixel([1,1], 2)
        >>> image.get_array()
        array([[1, 1, 1, 1, 1],
               [1, 2, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]], dtype=uint8)
        """
        img_dim = self.get_dim()
        if isinstance(indices, list) and len(indices) == img_dim:
            img_shape = self.shape
            if img_dim == 2:
                range_x, range_y = xrange(img_shape[0]), xrange(img_shape[1])
                conds_ind = indices[0] in range_x and indices[1] in range_y
            elif img_dim == 3:
                range_x, range_y, range_z = xrange(img_shape[0]), xrange(
                    img_shape[1]), xrange(img_shape[2])
                conds_ind = indices[0] in range_x and indices[1] in range_y and \
                            indices[2] in range_z
            if conds_ind:
                if img_dim == 2:
                    self[indices[0], indices[1]] = value
                elif img_dim == 3:
                    self[indices[0], indices[1], indices[2]] = value
            return
        else:
            print('Warning, incorrect specification')
            return

    def set_region(self, indices, val):
        """
        Replace a region

        Parameters
        ----------
        indices: list
            indices as list of integers
        val: array.dtype|np.array
            new value for the selected pixels, type of ``SpatialImage`` array

        Returns
        -------
        ``SpatialImage``
            ``SpatialImage`` instance

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> indices = [1,3,1,3]
        >>> out_sp_image = image.set_region(indices, 3)
        """
        img_dim = self.get_dim()
        conds_type = isinstance(indices, list) and len(indices) == 2 * img_dim
        if conds_type:
            conds_type_2 = isinstance(val, np.ndarray)
            conds_type_3 = isinstance(val, int)
            tmp_arr, tmp_vox = self.get_array(), self.voxelsize
            if conds_type_2:
                if img_dim == 2:
                    conds_shape = (((max(indices[0], indices[1]) - min(
                        indices[0], indices[1])) == val.shape[0])
                                   and ((max(indices[2], indices[3]) - min(
                                indices[2], indices[3])) == val.shape[1]))
                elif img_dim == 3:
                    conds_shape = (((max(indices[0], indices[1]) - min(
                        indices[0], indices[1])) == val.shape[0])
                                   and ((max(indices[2], indices[3]) - min(
                                indices[2], indices[3])) == val.shape[1])
                                   and ((max(indices[4], indices[5]) - min(
                                indices[4], indices[4])) == val.shape[2]))
                if conds_shape:
                    if img_dim == 2:
                        tmp_arr[
                        min(indices[0], indices[1]):max(indices[0], indices[1]),
                        min(indices[2], indices[3]):max(indices[2],
                                                        indices[3])] = val[:, :]
                    elif img_dim == 3:
                        tmp_arr[
                        min(indices[0], indices[1]):max(indices[0], indices[1]),
                        min(indices[2], indices[3]):max(indices[2], indices[3]),
                        min(indices[4], indices[5]):max(indices[4],
                                                        indices[5])] = val[:, :,
                                                                       :]
            elif conds_type_3:
                if img_dim == 2:
                    tmp_arr[
                    min(indices[0], indices[1]):max(indices[0], indices[1]),
                    min(indices[2], indices[3]):max(indices[2],
                                                    indices[3])] = val
                elif img_dim == 3:
                    tmp_arr[
                    min(indices[0], indices[1]):max(indices[0], indices[1]),
                    min(indices[2], indices[3]):max(indices[2], indices[3]),
                    min(indices[4], indices[5]):max(indices[4],
                                                    indices[5])] = val
        else:
            print('Warning, incorrect specification')
        out_sp_img = SpatialImage(input_array=tmp_arr, voxelsize=tmp_vox)
        return out_sp_img

    # ##########################################################################
    #
    # SpatialImage properties:     #
    # ##########################################################################
    @property
    def extent(self):
        """
        Get ``SpatialImage`` physical extent. It is related to the array shape
        and image voxelsize.

        Returns
        -------
        list
            ``SpatialImage`` physical extent

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> print image.extent
        [5.0, 5.0]
        """
        return self._extent

    @extent.setter
    def extent(self, img_extent):
        """
        Set ``SpatialImage`` physical extent, will change voxelsize based on
        array shape.

        Parameters
        ----------
        img_extent: list
            ``SpatialImage`` new physical extent.

        Notes
        -----
        Metadata are updated according to the new physical extent and voxelsize.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> print image.voxelsize
        [1.0, 1.0]
        >>> image.extent = [10.0, 10.0]
        Set extent to '[10.0, 10.0]'
        Changed voxelsize to '[2.0, 2.0]'
        >>> print image.extent
        [10.0, 10.0]
        >>> print image.voxelsize
        [2.0, 2.0]
        """
        dimensionality_test(self.get_dim(), img_extent)
        # - Update 'extent' hidden attribute:
        img_extent = tuple_array_to_list(img_extent)
        img_extent = around_list(img_extent)
        self._extent = img_extent
        # - Update 'voxelsize' hidden attribute:
        vox = [img_extent[i] / float(sh) for i, sh in enumerate(self.shape)]
        vox = around_list(vox)
        self._voxelsize = vox
        # - Update 'extent' & 'voxelsize' metadata:
        self._metadata['extent'] = self.extent
        self._metadata['voxelsize'] = self.voxelsize
        print "Set extent to '{}'".format(self.extent)
        print "Changed voxelsize to '{}'".format(self.voxelsize)
        return

    @property
    def metadata(self):
        """
        Get ``SpatialImage`` metadata

        Returns
        -------
        dict
            ``SpatialImage`` metadata

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> print image.metadata
        {'dim': 2,
         'extent': [5.0, 5.0],
         'origin': [0, 0],
         'shape': (5, 5),
         'type': 'uint8',
         'voxelsize': [1.0, 1.0]
         }
        """
        md = self._metadata
        # - Test whether the metadata dict has been initialized:
        try:
            assert md.has_key('shape')
        except AssertionError:
            md = obj_metadata(self, md)

        # - If attribute and metadata 'shape' are not equal, update metadata
        sh = self.shape
        try:
            assert md['shape'] == sh
        except AssertionError:
            old_shape = md['shape']
            md['shape'] = self.shape
            md['dim'] = self.ndim
            md['type'] = str(self.dtype)
            # --- transposition
            if self.is2D() and old_shape[0] == sh[1] and old_shape[1] == sh[0]:
                vox = [md['voxelsize'][1], md['voxelsize'][0]]
                ext = [md['extent'][1], md['extent'][0]]
                orig = [md['origin'][1], md['origin'][0]]
                md['voxelsize'], md['extent'], md['origin'] = vox, ext, orig
                self._voxelsize, self._extent, self._origin = vox, ext, orig
            elif (self.is3D() and old_shape[0] in sh and old_shape[1] in sh and
                  old_shape[2] in sh):
                print(
                    'Warning: possibly incorrect voxelsize, extent and origin')
                vox, ext, orig = [], [], []
                for ind in range(0, self.ndim):
                    tmp = old_shape.index(sh[ind])
                    vox.append(md['voxelsize'][tmp])
                    ext.append(md['extent'][tmp])
                    orig.append(md['origin'][tmp])
                md['voxelsize'], md['extent'], md['origin'] = vox, ext, orig
                self._voxelsize, self._extent, self._origin = vox, ext, orig
            else:
                print('Warning: incorrect voxelsize, extent and origin')
                vox, ext, orig = [], [], []
                md['voxelsize'], md['extent'], md['origin'] = vox, ext, orig
                self._voxelsize, self._extent, self._origin = vox, ext, orig

        # Update the metadata dictionary
        self._metadata = md
        return self._metadata

    @metadata.setter
    def metadata(self, img_metadata):
        """
        Set ``SpatialImage`` metadata

        Parameters
        ----------
        image_metadata: dict
            ``SpatialImage`` metadata

        Notes
        -----
        Attributes or properties will be updated accordingly.
        Following keys from 'img_metadata' will be ignored:
          - 'shape': depend on the array shape;
          - 'min': depend on the values found within the array;
          - 'max': depend on the values found within the array;
          - 'mean': depend on the values found within the array;

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image.metadata = {'name': 'img_test'}
        >>> image.metadata = {'voxelsize': [1.0, 1.0]}
        >>> image.metadata = {'voxelsize': [1.5, 1.5]}
        Set voxelsize to '[1.5, 1.5]'
        Changed extent to '[7.5, 7.5]'
        >>> image.metadata = {'shape': [5, 5]}
        'shape' is a protected attribute, it will not be updated!
        """
        # - Test provided input is a dictionary
        try:
            assert isinstance(img_metadata, dict)
        except:
            raise TypeError("Input 'img_metadata' is not a dictionary!")
        # - Test '_metadata' attribute is a dictionary
        try:
            assert isinstance(self._metadata, dict)
        except:
            raise TypeError("Attribute 'metadata' is not a dictionary!")

        # - Attribute 'resolution' is deprecated, we do not want to use it here:
        if img_metadata.has_key('resolution'):
            print "Attribute 'resolution' is deprecated, use 'voxelsize' instead!"
            img_metadata.pop('resolution')
        # - Protected attribute or properties:
        msg = "WARNING: '{}' is a protected attribute, it will not be updated!"
        for ppty in PROTECT_PPTY:
            try:
                img_metadata.pop(ppty)
            except KeyError:
                pass
            else:
                print msg.format(ppty)

        # - If both 'extent' and 'voxelsize' are provided, they might not match:
        if img_metadata.has_key('voxelsize') and img_metadata.has_key('extent'):
            print "Both 'extent' and 'voxelsize' were provided...",
            print "Using only 'voxelsize' for safety reasons!"
            img_metadata.pop('extent')

        # - Update the metadata dictionary with new values:
        self._metadata.update(img_metadata)
        # - Update object properties:
        if img_metadata.has_key('origin') and img_metadata[
            'origin'] != self.origin:
            self.origin = img_metadata['origin']
        # Updating 'voxelsize' property also update 'extent' property...
        # if img_metadata.has_key('extent') and img_metadata['extent'] != self.extent:
        # self.extent = around_list(img_metadata['extent'])
        if img_metadata.has_key('voxelsize') and img_metadata[
            'voxelsize'] != self.voxelsize:
            self.voxelsize = around_list(img_metadata['voxelsize'])
        if img_metadata.has_key('type') and img_metadata['type'] != self.dtype:
            self.dtype = img_metadata['type']

        return

    @property
    def origin(self):
        """
        Get ``SpatialImage`` origin.

        Returns
        -------
        list
            ``SpatialImage`` origin coordinates

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> print image.origin
        [0, 0]
        """
        return self._origin

    @origin.setter
    def origin(self, img_origin):
        """
        Set ``SpatialImage`` origin using a list of same length than the image
        dimensionality.

        Parameters
        ----------
        image_origin: list
            ``SpatialImage`` origin coordinates,

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image.origin = [2, 2]
        Set origin to '[2, 2]'
        """
        dimensionality_test(self.get_dim(), img_origin)
        img_origin = tuple_array_to_list(img_origin)
        # - Update hidden attribute 'origin':
        self._origin = img_origin
        # - Update hidden attribute metadata key 'origin':
        self._metadata['origin'] = img_origin
        print "Set origin to '{}'".format(self.origin)
        return

    # Deprecated, use Numpy attribute 'dtype'
    # @property
    # def type(self):
    #     """
    #     Get ``SpatialImage`` type.
    #
    #     Returns
    #     -------
    #     str
    #         ``SpatialImage`` type
    #
    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> from timagetk.components import SpatialImage
    #     >>> test_array = np.ones((5,5), dtype=np.uint8)
    #     >>> image = SpatialImage(input_array=test_array)
    #     >>> print image.dtype
    #     uint8
    #     """
    #     return str(self.dtype)

    # Deprecated, use overloaded NumPy method 'astype()' !!
    # @type.setter
    # def type(self, val):
    #     """
    #     Set ``SpatialImage`` type.
    #
    #     Parameters
    #     ----------
    #     image_type: str
    #         image type (see numpy types).
    #
    #     Returns
    #     -------
    #     ``SpatialImage``
    #         new instance of given type
    #
    #     Example
    #     -------
    #     >>> import numpy as np
    #     >>> from timagetk.components import SpatialImage
    #     >>> test_array = np.ones((5,5), dtype=np.uint8)
    #     >>> image = SpatialImage(input_array=test_array)
    #     >>> image.dtype = np.uint16
    #     >>> print image.dtype
    #     uint16
    #     """
    #     if (val in DICT_TYPES.keys() or val in DICT_TYPES.values()):
    #         for key in DICT_TYPES:
    #             if (val == key or val == DICT_TYPES[key]):
    #                 new_type = DICT_TYPES[key]
    #
    #         self.__init__(self, self.get_array().astype(new_type),
    #                               origin=self.origin, voxelsize=self.voxelsize,
    #                               metadata_dict=self.metadata)
    #     else:
    #         msg = "Unknown type '{}', possible types are: {}"
    #         raise ValueError(msg.format(val, AVAIL_TYPES))
    #     return

    @property
    def voxelsize(self):
        """
        Get ``SpatialImage`` voxelsize.

        Returns
        -------
        list
            ``SpatialImage`` voxelsize

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> print image.voxelsize
        [1.0, 1.0]
        """
        return self._voxelsize

    @voxelsize.setter
    def voxelsize(self, img_vxs):
        """
        Set ``SpatialImage`` voxelsize, will change physical extent based on
        array shape.

        Parameters
        ----------
        image_voxelsize: list
            ``SpatialImage`` new voxelsize

        Notes
        -----
        Metadata are updated according to the new physical extent and voxelsize.

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array)
        >>> image.voxelsize = [0.5, 0.5]
        Set voxelsize to '[0.5, 0.5]'
        Changed extent to '[2.5, 2.5]'
        >>> print image.voxelsize
        [0.5, 0.5]
        >>> print image.extent
        [2.5, 2.5]
        """
        dimensionality_test(self.get_dim(), img_vxs)
        # - Update 'voxelsize' hidden attribute:
        img_vxs = tuple_array_to_list(img_vxs)
        img_vxs = around_list(img_vxs)
        self._voxelsize = img_vxs
        # - Update 'extent' hidden attribute:
        ext = [img_vxs[i] * float(sh) for i, sh in enumerate(self.shape)]
        ext = around_list(ext)
        self._extent = ext
        # - Update 'extent' & 'voxelsize' metadata:
        self._metadata['voxelsize'] = self.voxelsize
        self._metadata['extent'] = self.extent
        print "Set voxelsize to '{}'".format(self.voxelsize)
        print "Changed extent to '{}'".format(self.extent)
        return

    @property
    def resolution(self):
        """
        Ensure backward compatibility with older openalea.image package.
        """
        print DeprecationWarning(
            "Attribute 'resolution' is deprecated, use 'voxelsize' attribute instead!")
        return self._voxelsize

    @resolution.setter
    def resolution(self, voxelsize):
        """
        Ensure backward compatibility with older openalea.image package.
        """
        self.voxelsize = voxelsize

    # ------------------------------------------------------------------------------
    #
    # SpatialImage transformation functions:
    #
    # ------------------------------------------------------------------------------
    def to_8bits(self, unsigned=True):
        """
        Convert the ``SpatialImage`` array to a different type.

        Parameters
        ----------
        unsigned: bool, optional
            if True (default), return as unsigned integer 'uint8', else signed
            integer 'int8'

        Returns
        -------
        ``SpatialImage``
            the converted ``SpatialImage``
        """
        vxs = self.voxelsize
        ori = self.origin
        md = self.metadata
        if unsigned:
            dtype = 'uint8'
        else:
            dtype = 'int8'

        arr = self.get_array()
        if arr.dtype == "uint16" or arr.dtype == "int16":
            factor = 2 ** (16 - 8) - 1
        elif arr.dtype == "uint32" or arr.dtype == "int32":
            factor = 2 ** (32 - 8) - 1
        elif arr.dtype == "uint64" or arr.dtype == "int64":
            factor = 2 ** (64 - 8) - 1
        else:
            raise NotImplementedError(
                'Could not find implementation to do just that!')

        arr = np.divide(arr, factor).astype(DICT_TYPES[dtype])
        return SpatialImage(arr, voxelsize=vxs, origin=ori, metadata_dict=md)

    def revert_axis(self, axis='z'):
        """
        Revert x, y, or z axis

        Parameters
        ----------
        axis: str
            can be either 'x', 'y' or 'z'

        Returns
        -------
        ``SpatialImage``
            ``SpatialImage`` instance

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> test_array = np.ones((5,5,5), dtype=np.uint8)
        >>> image = SpatialImage(input_array=test_array, voxelsize=[0.5, 0.5, 0.5])
        >>> image.revert_axis(axis='y')
        """
        if self.get_dim() == 2:
            self = self.to_3D()

        arr, vox = self.get_array(), self.voxelsize
        if axis == 'x':
            new_arr = arr[::-1, :, :]
        if axis == 'y':
            new_arr = arr[:, ::-1, :]
        elif axis == 'z':
            new_arr = arr[:, :, ::-1]
        out_sp_image = SpatialImage(new_arr, voxelsize=vox)
        if 1 in out_sp_image.shape:
            out_sp_image = out_sp_image.to_2D()
        return out_sp_image
