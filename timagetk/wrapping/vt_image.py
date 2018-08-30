# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Sophie Ribes <sophie.ribes@inria.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------

from ctypes import pointer, c_void_p, c_char_p
import numpy as np

try:
    from timagetk.components import SpatialImage
    from timagetk.wrapping.clib import libvt, c_stdout
    from timagetk.wrapping.type_conversion import np_type_to_vt_type
    from timagetk.wrapping.type_conversion import vt_type_to_c_type
    from timagetk.wrapping.vtImage import _VT_IMAGE, MY_CPU
    from timagetk.wrapping.vt_typedefs import vt_4vsize, vt_fpt, vt_ipt
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))

__all__ = ['vt_image', 'new_vt_image', 'VT_Image']


def sp_img_to_vt_img(sp_img):
    """
    SpatialImage to _VT_IMAGE structure
    """
    try:
        assert isinstance(sp_img, SpatialImage)
    except AssertionError:
        raise TypeError('Input image must be a SpatialImage')

    if sp_img.get_dim() == 2:  # 2D management
        sp_img = sp_img.to_3D()

    _name = '\0'
    _type = np_type_to_vt_type(sp_img.dtype)
    _x, _y, _z = sp_img.shape
    _v = 1
    _dim = vt_4vsize(_v, _x, _y, _z)
    _vx, _vy, _vz = sp_img.voxelsize
    _siz = vt_fpt(_vx, _vy, _vz)
    _off = vt_fpt(0.0, 0.0, 0.0)
    _rot = vt_fpt(0.0, 0.0, 0.0)
    _ctr = vt_ipt(0, 0, 0)
    _cpu = MY_CPU
    # Data pointer cast to a particular c-types object
    _array = c_void_p()
    _buf = sp_img.ctypes.data_as(c_void_p)
    _user = c_char_p()
    _nuser = 0
    # initialization of VT_Image structure
    vt_img = _VT_IMAGE(_name, _type, _dim, _siz, _off, _rot, _ctr, _cpu,
                       _array, _buf, _user, _nuser)
    return vt_img


def vt_img_to_sp_img(vt_image):
    """
    _VT_IMAGE structure to SpatialImage
    """
    dt = vt_type_to_c_type(vt_image.dtype)
    x, y, z, v = vt_image.dim.x, vt_image.dim.y, vt_image.dim.z, vt_image.dim.v
    size = x * y * z * v
    vx, vy, vz = vt_image.siz.x, vt_image.siz.z, vt_image.siz.z
    _ct_array = (dt * size).from_address(vt_image.buf)
    _np_array = np.ctypeslib.as_array(_ct_array)
    # -- This used to be  arr =  np.array(_np_array.reshape(z,x,y).transpose(2,1,0)).
    # but that is wrong. first the shape is x,y,z. Then the transposition
    # doesn't fix the byte ordering which for some reason must be read in
    # Fortran order --
    out_arr = np.array(_np_array.reshape(x, y, z, order="F"))
    out_sp_img = SpatialImage(out_arr, voxelsize=[vx, vy, vz],
                              origin=[0., 0., 0.])
    if 1 in out_sp_img.shape:  # 2D management
        out_sp_img = out_sp_img.to_2D()
    return out_sp_img


def vt_image(sp_img):
    """
    """
    try:
        assert isinstance(sp_img, SpatialImage)
    except AssertionError:
        raise TypeError('Input image must be a SpatialImage')

    vt_image = VT_Image(sp_img)
    return vt_image


def new_vt_image(sp_img_in, dtype=None):
    """
    """
    try:
        assert isinstance(sp_img_in, SpatialImage)
    except AssertionError:
        raise TypeError('Input image must be a SpatialImage')

    if not dtype:
        dtype = sp_img_in.dtype

    ori = sp_img_in.origin
    vxs = sp_img_in.voxelsize
    shape = sp_img_in.shape
    sp_img_out = SpatialImage(np.zeros(shape, dtype=dtype), voxelsize=vxs,
                              origin=ori)
    vt_res = VT_Image(sp_img_out)
    return vt_res


class VT_Image(object):
    """
    Class VT_Image
    """

    def __init__(self, sp_img):
        """
        VT_Image constructor

        Parameters
        ----------
        sp_img: ``SpatialImage``
            SpatialImage instance --- image and metadata
        """
        if not isinstance(sp_img, SpatialImage):
            print('Warning: sp_img is not a SpatialImage instance')
            ndim = sp_img.ndim
            sp_img = SpatialImage(sp_img, voxelsize=[1.] * ndim,
                                  origin=[0.] * ndim)

        if sp_img.get_dim() == 2:  # 2D management
            sp_img = sp_img.to_3D()

        self._data = sp_img
        self.vt_image = sp_img_to_vt_img(sp_img)
        libvt.VT_AllocArrayImage(pointer(self.vt_image))

    def get_vt_image(self):
        return self.vt_image

    def get_spatial_image(self):
        if 1 in self._data.shape:  # 2D management
            tmp_sp_img = self._data.to_2D()
            self._data = tmp_sp_img
        return self._data

    def c_display(self, name=''):
        libvt.VT_PrintImage(c_stdout, pointer(self.vt_image), name)

    def free(self):
        if self.vt_image is not None:
            #libvt.VT_FreeImage(pointer(self.vt_image))
            libvt.VT_Free(self.c_ptr)
            self.vt_image = None

    @property
    def c_ptr(self):
        return pointer(self.get_vt_image())
