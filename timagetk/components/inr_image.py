# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Jerome Chopard <jerome.chopard@inria.fr>
#
#       See accompanying file LICENSE.txt
# -----------------------------------------------------------------------------
"""
Management of the .inr format

Header specifications
---------------------
#INRIMAGE-4#{ // format type
XDIM=150 // x dimension
YDIM=200 // y dimension
ZDIM=100 // z dimension
VDIM=1 // dimension of the data, here 1 for scalar or 3 for thing like RGB values
VX=5 // voxel size in x
VY=5 // voxel size in y
VZ=5 // voxel size in z
// a higher voxel size results in a more precise 3D mesh
TYPE=unsigned fixed // type of data written, can be float, signed fixed or unsigned fixed
SCALE=2**0 // not used by most programs apparently
PIXSIZE=8 bits // size of a char, can be 8, 16, 32 or 64
CPU=pc // the type of cpu for little/big endianness, pc is little endian

// Fill with carriage returns until the header is a multiple of 256 bytes, including the end of the header (4 bytes including the line break)

##} // the line break is included in the header count
"""

import gzip
import os

import numpy as np

try:
    from cStringIO import StringIO  # C version
except ImportError:
    from StringIO import StringIO  # Python version
try:
    from timagetk.components import SpatialImage
    from timagetk.components import try_spatial_image
except ImportError:
    raise ImportError('Unable to import SpatialImage')

__all__ = ["read_inr_image", "write_inr_image"]

# - Accepted file formats:
POSS_EXT = ['.inr', '.inr.gz', '.inr.zip']
# - Accepted types of images, checked against 'TYPE' in header:
POSS_TYPES = ['unsigned fixed', 'signed fixed', 'float']
# - Accepted encoding of images, checked against 'PIXSIZE' in header:
POSS_ENC = [8, 16, 32]


def read_inr_image(inr_file):
    """
    Read an '.inr' file (2D or 3D images)
    The supported formats are: ['.inr', '.inr.gz', '.inr.zip']

    Parameters
    ----------
    inr_file : str
        path to the image

    Returns
    -------
    out_sp_img : ``SpatialImage``
        image and metadata (such as voxelsize, extent, type, etc.)

    Example
    -------
    >>> inr_file = '/home/X/Images/my_inr_file.inr'
    >>> out_sp_img = read_inr_image(inr_file)
    """
    try:
        os.path.exists(inr_file)
    except:
        raise IOError("This file does not exists: {}".format(inr_file))

    (filepath, filename) = os.path.split(inr_file)
    (shortname, ext) = os.path.splitext(filename)
    if (ext == '.gz') or (ext == '.zip'):
        zip_ext = ext
        (shortname, ext) = os.path.splitext(shortname)
        ext += zip_ext
    try:
        assert ext in POSS_EXT
    except AssertionError:
        msg = "Unknown file ext '{}', should be in {}."
        raise NotImplementedError(msg.format(ext, POSS_EXT))

    # - Open file:
    if ext == '.inr.gz' or ext == '.inr.zip':
        with gzip.open(inr_file, 'rb') as fzip:
            f = StringIO(fzip.read())
            fzip.close()
    else:
        f = open(inr_file, 'rb')

    # - Parsing header info to a dictionary:
    # -- Read the first lines of the files by muliples of 256 bytes until the end of the header ('##}\n'):
    header = ""
    while header[-4:] != "##}\n":
        header += f.read(256)
    # -- Define header begining and end:
    head_start, head_end = header.find("{\n") + 1, header.find("##}")
    # -- Split lines:
    infos = [gr for gr in header[head_start:head_end].split("\n") if
             len(gr) > 0]
    # -- Create a dictionary by lines:
    prop = {}
    for prop_def in infos:
        if not prop_def.strip().startswith('#'):
            key, val = prop_def.split("=")
            prop[key] = val

    # - Parsing headers dictionary:
    # -- Image shape:
    x_sh, y_sh, z_sh = int(prop['XDIM']), int(prop['YDIM']), int(prop['ZDIM'])
    # -- Image voxelsize:
    vx, vy, vz = float(prop['VX']), float(prop['VY']), float(prop['VZ'])
    # -- Number of channels in the image:
    vdim = int(prop['VDIM'])
    # -- Image type and encoding:
    img_type, img_enc = prop['TYPE'], int(prop["PIXSIZE"].split()[0])

    # - Convert the encoding found in the header into a numpy encoding:
    conds = img_type in POSS_TYPES and img_enc in POSS_ENC
    if conds:
        if img_type == 'unsigned fixed':
            np_typ = eval("np.dtype(np.uint%d)" % img_enc)
        elif img_type == 'signed fixed':
            np_typ = eval("np.dtype(np.int%d)" % img_enc)
        else:
            np_typ = eval("np.dtype(np.float%d)" % img_enc)
    else:
        if img_type in POSS_TYPES and img_enc not in POSS_ENC:
            print(
                'Warning, unable to detected encoding, might lead to incorrect reading!')
            if img_type == 'unsigned fixed':
                np_typ = np.dtype(np.uint)
            elif img_type == 'signed fixed':
                np_typ = np.dtype(np.int)
            else:
                np_typ = np.dtype(np.float)
        else:
            print('Unable to read this file...')
            return

    # - Get the matrix representing the image: 
    size = np_typ.itemsize * x_sh * y_sh * z_sh * vdim
    mat = np.fromstring(f.read(size), np_typ)
    f.close()
    # - Reshape to single channel array (vdim==1):
    if vdim == 1:
        mat = mat.reshape((x_sh, y_sh, z_sh), order="F")
    else:
        raise TypeError("This inr reader does not accept multi-channel images.")
        # mat = mat.reshape((vdim, x_sh, y_sh, z_sh), order="F")
        # mat = mat.transpose(1, 2, 3, 0)

    out_sp_img = SpatialImage(mat, origin=[0, 0, 0], voxelsize=[vx, vy, vz],
                              metadata_dict=prop)
    # --- 2D images management
    if 1 in out_sp_img.shape:
        out_sp_img = out_sp_img.to_2D()

    return out_sp_img


def write_inr_image(inr_file, sp_img):
    """
    Write an '.inr' file (2D or 3D images).
    The supported formats are: ['.inr', '.inr.gz', '.inr.zip']

    Parameters
    ----------
    inr_file : str
        path to the file.
    sp_img : SpatialImage
        ``SpatialImage`` instance

    Example
    -------
    >>> inr_file = '/home/you/Documents/my_inr_file.inr'
    >>> test_arr = np.ones((5,5), dtype=np.uint8)
    >>> sp_img = SpatialImage(test_arr)
    >>> write_inr_image(inr_file, sp_img)
    """
    # - Assert sp_img is a SpatialImage instance:
    try_spatial_image(sp_img, obj_name='sp_img')

    # - Assert SpatialImage is 2D or 3D:
    try:
        assert sp_img.is2D() or sp_img.is3D()
    except AssertionError:
        raise ValueError("Parameter 'sp_img' should be 2D or 3D.")

    # - Get file extension and check its validity:
    (filepath, filename) = os.path.split(inr_file)
    (shortname, ext) = os.path.splitext(filename)
    if (ext == '.gz') or (ext == '.zip'):
        zip_ext = ext
        (shortname, ext) = os.path.splitext(shortname)
        ext += zip_ext
    try:
        assert ext in POSS_EXT
    except AssertionError:
        raise NotImplementedError(
            "Unknown file ext '{}', should be in {}.".format(
                ext, POSS_EXT))

    if ext == '.inr.gz' or ext == '.inr.zip':
        f = gzip.GzipFile(inr_file, 'wb')
    else:
        f = open(inr_file, 'wb')

    metadata = sp_img.metadata
    info = {'XDIM': metadata['shape'][0], 'YDIM': metadata['shape'][1],
            'VX': metadata['voxelsize'][0], 'VY': metadata['voxelsize'][1]}

    if sp_img.get_dim() == 2:
        info['ZDIM'], info['VZ'] = 1, 1.0
    elif sp_img.get_dim() == 3:
        info['ZDIM'], info['VZ'] = metadata['shape'][2], \
                                   metadata['voxelsize'][2]
    info['#GEOMETRY'] = 'CARTESIAN'
    info['CPU'] = 'decm'
    info['VDIM'] = '1'
    img_typ = str(sp_img.dtype)

    if img_typ[0:4] == 'uint':
        info['TYPE'] = 'unsigned fixed'
    elif img_typ[0:5] == 'float':
        info['TYPE'] = 'float'
    if '8' in img_typ:
        info['PIXSIZE'] = '8 bits'
    elif '16' in img_typ:
        info['PIXSIZE'] = '16 bits'
    elif '32' in img_typ:
        info['PIXSIZE'] = '32 bits'
    elif '64' in img_typ:
        info['PIXSIZE'] = '64 bits'

    # --- header
    head_keys = ['XDIM', 'YDIM', 'ZDIM', 'VDIM', 'TYPE', 'PIXSIZE',
                 'SCALE', 'CPU', 'VX', 'VY', 'VZ', 'TX', 'TY', 'TZ',
                 '#GEOMETRY']
    header = "#INRIMAGE-4#{\n"
    for key in head_keys:
        try:
            header += "%s=%s\n" % (key, info[key])
        except KeyError:
            pass

    for k in set(info) - set(head_keys):
        header += "%s=%s\n" % (k, info[k])

    header_size = len(header) + 4
    if (header_size % 256) > 0:
        header += "\n" * (256 - header_size % 256)
    header += "##}\n"

    f.write(header)
    f.write(sp_img.get_array().tostring("F"))
    #            elif (sp_img._get_dim() == 4):
    #                mat = img.transpose(3,0,1,2)
    #                stream.write(mat.tostring("F") )
    #            else:
    #                raise Exception("Unhandled image dimension %d."%img.ndim)

    return
