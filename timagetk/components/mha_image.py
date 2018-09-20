# -*- python -*-
#       image.serial: read/write spatial nd images
#
#       Copyright 2006 - 2018 INRIA - CIRAD - INRA
#
#       File author(s): Jerome Chopard <jerome.chopard@sophia.inria.fr>
#                       Eric Moscardi <eric.moscardi@sophia.inria.fr>
#                       Daniel Barbeau <daniel.barbeau@sophia.inria.fr>
#                       Gregoire Malandain
#                       Jonathan Legrand
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       Repository: git@gitlab.inria.fr:mosaic/timagetk.git
################################################################################

"""
This module implementes partially the I/O for the metaimage format
(see https://itk.org/Wiki/ITK/MetaIO/Documentation)
It deals with images defined by a single file
"""

__license__ = "Cecill-C"
__revision__ = " $Id$ "

import gzip
from os import path
import numpy as np

try:
    from cStringIO import StringIO  # C version
except ImportError:
    from StringIO import StringIO  # Python version

try:
    from timagetk.components import SpatialImage
except ImportError:
    raise ImportError('Unable to import SpatialImage')

__all__ = ["read_mha_image", "write_mha_image"]

SPECIFIC_HEADER_KEYS = ("ObjectType", "NDims", "DimSize",
                        "ElementNumberOfChannels", "ElementSize",
                        "ElementSpacing", "ElementType", "CompressedData",
                        "BinaryDataByteOrderMSB", "BinaryData",
                        "ElementDataFile")


def _open_mha_imagefile(filename):
    """
    Sub-function dedicated to read mha files.

    Parameters
    ----------
    filename : str
        path to the file to open

    Returns
    -------
    file
        readable file
    """
    program = "open_mha_imagefile"
    if not path.isfile(filename) and path.isfile(filename + ".gz"):
        filename = filename + ".gz"
        print "%s: Warning: path to read image has been changed to %s." % (
            program, filename)
    if not path.isfile(filename) and path.isfile(filename + ".zip"):
        filename = filename + ".zip"
        print "%s: Warning: path to read image has been changed to %s." % (
            program, filename)
    if path.splitext(filename)[1] in (".gz", ".zip"):
        fzip = gzip.open(filename, 'rb')
        f = StringIO(fzip.read())
        fzip.close()
    else:
        f = open(filename, 'rb')

    return f


def _read_header(f):
    """
    Read the file header and parse it in a dictionary.

    Parameters
    ----------
    f : file
        file object for which to read the header

    Returns
    -------
    dict
        dictionary made of the file header
    """

    prop = {}
    while True:
        key, val = f.readline().rstrip('\n\r').split(" = ")
        if key == 'ElementDataFile':
            if val == 'LOCAL':
                break
            else:
                msg = "unable to read that type of data: '" + str(
                    key) + ' = ' + str(val) + "'"
                raise UserWarning(msg)
        else:
            prop[key] = val

    return prop


def read_mha_image(filename):
    """
    Read an image in an '.mha' file, zipped or not according to the extension.
    The supported formats are: ['.mha', '.mha.gz', '.mha.zip']

    Parameters
    ----------
    filename : str
        path to the image

    Returns
    -------
    out_sp_img : ``SpatialImage``
        image and metadata (such as voxelsize, extent, type, etc.)
    """
    f = _open_mha_imagefile(filename)

    # read header
    prop = _read_header(f)

    #
    # Jonathan: 14.05.2012
    #
    prop["filename"] = filename

    #
    # find dimensions
    #
    dim = prop.pop("DimSize").split(' ')
    ndim = len(dim)
    if ndim == 2:
        xdim = int(dim[0])
        ydim = int(dim[1])
        zdim = 1
    elif ndim == 3:
        xdim = int(dim[0])
        ydim = int(dim[1])
        zdim = int(dim[2])
    else:
        msg = "Unable to handle such dimensions: 'DimSize = {}'".format(ndim)
        raise UserWarning(msg)

    vdim = int(prop.pop("ElementNumberOfChannels", 1))

    #
    # find type
    #
    voxeltype = prop.pop("ElementType")
    if voxeltype == 'MET_UCHAR':
        ntyp = np.dtype(np.uint8)
    elif voxeltype == 'MET_USHORT':
        ntyp = np.dtype(np.uint16)
    elif voxeltype == 'MET_UINT':
        ntyp = np.dtype(np.uint32)
    elif voxeltype == 'MET_FLOAT':
        ntyp = np.dtype(np.float32)
    elif voxeltype == 'MET_DOUBLE':
        ntyp = np.dtype(np.float64)
    else:
        msg = "unable to handle such voxel type: 'ElementType = " + str(
            voxeltype) + "'"
        raise UserWarning(msg)

    #
    # find resolution
    #
    resolution = prop.pop("ElementSize").split(' ')
    res = []
    for i in range(0, len(resolution)):
        res.append(float(resolution[i]))

    # read datas
    size = ntyp.itemsize * xdim * ydim * zdim * vdim
    mat = np.fromstring(f.read(size), ntyp)
    if vdim != 1:
        mat = mat.reshape((vdim, xdim, ydim, zdim), order="F")
        mat = mat.transpose(1, 2, 3, 0)
    else:
        mat = mat.reshape((xdim, ydim, zdim), order="F")
        # mat = mat.transpose(2,1,0)

    # create SpatialImage
    if vdim > 1:
        msg = "unable to handle multi-channel images: 'ElementNumberOfChannels = " + str(
            vdim) + "'"
        raise UserWarning(msg)
    # img = SpatialImage(mat, voxelsize=res, vdim, prop)
    img = SpatialImage(mat, voxelsize=res, metadata_dict=prop)

    # return
    f.close()
    return img


def _write_mha_image_to_stream(stream, img):
    """
    Sub-function dedicated to writing the mha file given an open file stream.

    Parameters
    ----------
    stream : file
        open file stream to use
    img : SpatialImage
        image to save as an mha file
    """
    assert img.ndim in (3, 4)

    # metadata
    info = dict(getattr(img, "info", {}))

    info["ObjectType"] = "Image"

    #
    # image dimensions
    # won't support 2D vectorial images
    #
    if img.ndim == 2:
        info["NDims"] = "3"
        info["DimSize"] = str(img.shape[0]) + ' ' + str(img.shape[1]) + ' 1'
        info["ElementNumberOfChannels"] = "1"
    elif img.ndim == 3:
        info["NDims"] = "3"
        info["DimSize"] = str(img.shape[0]) + ' ' + str(
            img.shape[1]) + ' ' + str(img.shape[2])
        info["ElementNumberOfChannels"] = "1"
    elif img.ndim == 4:
        info["NDims"] = "3"
        info["DimSize"] = str(img.shape[0]) + ' ' + str(
            img.shape[1]) + ' ' + str(img.shape[2])
        info["ElementNumberOfChannels"] = str(img.shape[2])
    else:
        msg = "No "
        raise ValueError(msg)

    #
    # image resolutions
    #
    res = getattr(img, "resolution", (1, 1, 1))
    info["ElementSize"] = str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2])
    info["ElementSpacing"] = str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2])

    #
    # data type
    #
    if img.dtype == np.uint8:
        info["ElementType"] = "MET_UCHAR"
    elif img.dtype == np.uint16:
        info["ElementType"] = "MET_USHORT"
    elif img.dtype == np.uint32:
        info["ElementType"] = "MET_UINT"
    elif img.dtype == np.float32:
        info["ElementType"] = "MET_FLOAT"
    elif img.dtype == np.float64:
        info["ElementType"] = "MET_DOUBLE"
    # elif img.dtype == np.float128:
    #   info["TYPE"] = "float"
    #   info["PIXSIZE"] = "128 bits"
    else:
        msg = "unable to write that type of data: %s" % str(img.dtype)
        raise UserWarning(msg)

    info["CompressedData"] = "False"
    info["BinaryDataByteOrderMSB"] = "False"
    info["BinaryData"] = "True"
    info["ElementDataFile"] = "LOCAL"

    #
    # fill header
    #
    header = ''
    for k in SPECIFIC_HEADER_KEYS:
        try:
            header += "%s = %s\n" % (k, info[k])
        except KeyError:
            pass

    #
    # write raw data
    #
    stream.write(header)
    if img.ndim == 2 or img.ndim == 3:
        stream.write(img.tostring("F"))
    # elif img.ndim == 4:
    #    mat = img.transpose(3,0,1,2)
    #    stream.write(mat.tostring("F") )
    else:
        raise Exception("Unhandled image dimension %d." % img.ndim)


def write_mha_image(filename, img):
    """
    Write an image in an '.mha' file, zipped or not according to the extension.

    Parameters
    ----------
    filename : str
        path to the file.
    img : SpatialImage
        ``SpatialImage`` instance

    Notes
    -----
    if 'img' is not a SpatialImage, default values will be used for the
    resolution of the image, see SpatialImage.
    """
    # open stream
    zipped = (path.splitext(filename)[1] in (".gz", ".zip"))

    if zipped:
        f = gzip.GzipFile(filename, "wb")
        # f = StringIO()
    else:
        f = open(filename, 'wb')

    try:
        _write_mha_image_to_stream(f, img)
    except:
        # -- remove probably corrupt file--
        f.close()
        if path.exists(filename) and path.isfile(filename):
            os.remove(filename)
        raise
    else:
        f.close()
