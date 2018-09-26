# -*- python -*-
# -*- coding: utf-8 -*-
#
#       timagetk.algorithms.reconstruction
#
#       Copyright 2006 - 2011 INRIA - CIRAD - INRA
#
#       File author(s): Eric MOSCARDI <eric.moscardi@gmail.com>
#                       Daniel BARBEAU <daniel.barbeau@inria.fr>
#                       Jonathan LEGRAND <jonathan.legrand@ens-lyon.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#       http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite: http://openalea.gforge.inria.fr
# ------------------------------------------------------------------------------

__license__ = "Cecill-C"
__revision__ = " $Id$ "

"""
This is a revised copy of (deprecated?) package 'vplants.mars_alt.mars.reconstruction'
It was using a library that is not available anymore (morpheme).
"""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from timagetk.components import try_spatial_image
from timagetk.algorithms import connexe
from timagetk.components.spatial_image import SpatialImage
from timagetk.components.labelled_image import connectivity_4
from timagetk.components.labelled_image import connectivity_6
from timagetk.components.labelled_image import connectivity_8
from timagetk.plugins import linear_filtering


def end_margin(img, width, axis=None):
    """
    An end margin is an inside black space (0) that can be added into the end of
    array object.

    Parameters
    ----------
    img : np.array
        NxMxP array
    width: int
        size of the margin
    axis : int, optional
        axis along which the margin is added.
        By default, add in all directions (see also stroke).

    Returns
    ------
    np.array
        input array with the end margin

    Examples
    --------
    >>> from numpy import zeros, random
    >>> from timagetk.algorithms.reconstruction import end_margin
    >>> img = random.random((3, 4, 5))
    >>> end_margin(img, 1, 0)
    """
    xdim, ydim, zdim = img.shape
    mat = np.zeros((xdim, ydim, zdim), img.dtype)

    if axis is None:
        mat[:-width, :-width, :-width] = img[:-width, :-width, :-width]
    elif axis == 0:
        mat[:-width, :, :] = img[:-width, :, :]
    elif axis == 1:
        mat[:, :-width, :] = img[:, :-width, :]
    elif axis == 2:
        mat[:, :, :-width] = img[:, :, :-width]
    else:
        raise AttributeError('axis')

    return mat


def stroke(img, width, outside=False):
    """
    A stroke is an outline that can be added to an array object.

    Parameters
    ----------
    img : np.array
        NxMxP array
    width: int
        size of the stroke
    outside : bool, optional
        used to set the position of the stroke.
        By default, the position of the stroke is inside (outside = False)

    Returns
    -------
    np.array
        input array with the stroke

    Examples
    --------
    >>> from numpy import zeros, random
    >>> from timagetk.algorithms.reconstruction import stroke
    >>> img = random.random((3, 4, 5))
    >>> stroke(img, 1)
    """
    xdim, ydim, zdim = img.shape
    w = width
    if outside:
        mat = np.zeros([xdim + 2 * w, ydim + 2 * w, zdim + 2 * w], img.dtype)
        mat[w:xdim + w, w:ydim + w, w:zdim + w] = img
    else:
        mat = np.zeros((xdim, ydim, zdim), img.dtype)
        mat[w:-w, w:-w, w:-w] = img[w:-w, w:-w, w:-w]

    return mat


def max_intensity_projection(image, threshold_value=45):
    """
    This function computes a surfacic view of the meristem, according to a revisited version
    of the method described in Barbier de Reuille and al. in Plant Journal.

    Parameters
    ----------
    image: SpatialImage
        image to be masked
    threshold_value: int, float
        consider intensities superior to threshold.

    Returns
    -------
    SpatialImage
        maximum intensity projection

    Example
    -------
    >>> import matplotlib.pylab as plt
    >>> from timagetk.util import data_path
    >>> from timagetk.components import imread
    >>> from timagetk.algorithms.reconstruction import max_intensity_projection
    >>> img_path = data_path('time_0_cut.inr')
    >>> image = imread(img_path)
    >>> proj = max_intensity_projection(image)
    >>> plt.imshow(proj)
    """
    mip, m_alt = im2surface(image, threshold_value=threshold_value,
                            only_altitude=False, front_half_only=False)
    return mip


def im2surface(image, threshold_value=45, only_altitude=False,
               front_half_only=False):
    """
    This function computes a surfacic view of the meristem, according to a revisited version
    of the method described in Barbier de Reuille and al. in Plant Journal.

    Parameters
    ----------
    image: SpatialImage
        image to be masked
    threshold_value: int, float
        consider intensities superior to threshold.
    only_altitude: bool
        only return altitude map, not maximum intensity projection
    front_half_only: bool
        only consider the first half of all slices in the Z direction.

    Returns
    -------
    mip_img: SpatialImage
        maximum intensity projection. *None* if only_altitude in True
    alt_img: SpatialImage
        altitude of maximum intensity projection

    Example
    -------
    >>> import matplotlib.pylab as plt
    >>> from timagetk.util import data_path
    >>> from timagetk.components import imread
    >>> from timagetk.algorithms.reconstruction import max_intensity_projection
    >>> img_path = data_path('time_0_cut.inr')
    >>> image = imread(img_path)
    >>> proj = max_intensity_projection(image)
    >>> plt.imshow(proj)
    """
    try_spatial_image(image, 'image')
    vxs = image.voxelsize
    ori = image.origin
    md = image.metadata
    dtype = image.dtype

    # - Definition of flat 3D structuring elements:
    connect_4 = connectivity_4().reshape(3, 3, 1)
    connect_6 = connectivity_6()
    connect_8 = connectivity_8().reshape(3, 3, 1)

    # Added gaussian smoothing to reduce
    img_th = linear_filtering(image, 'gaussian_smoothing', std_dev=2.0)
    img_th = img_th >= threshold_value
    img_th = SpatialImage(img_th.astype(dtype), origin=ori, voxelsize=vxs,
                          metadata_dict=md)

    labeling = connexe(img_th, param_str_1='-labels -connectivity 26 -parallel')
    del img_th

    iterations = 15
    dilation1 = binary_dilation(labeling, connect_8, iterations)
    del labeling

    iterations = 10
    erosion1 = binary_erosion(dilation1, connect_8, iterations, border_value=1)
    del dilation1

    iterations = 15
    dilation2 = binary_dilation(erosion1, connect_8, iterations)
    del erosion1

    iterations = 4
    erosion2 = binary_erosion(dilation2, connect_4, iterations, border_value=1)
    del dilation2

    iterations = 15
    erosion3 = binary_erosion(erosion2, connect_8, iterations, border_value=1)
    del erosion2

    iterations = 1
    erosion4 = binary_erosion(erosion3, connect_6, iterations, border_value=1)

    # CONDITIONNER_POUR_ISOSURFACE
    mat1 = stroke(erosion4, 1)

    iterations = 3
    erosion5 = binary_erosion(mat1, connect_6, iterations, border_value=1)
    del mat1

    iterations = 9
    erosion6 = binary_erosion(erosion5, connect_8, iterations, border_value=1)
    del erosion5

    m_xor = np.logical_xor(erosion3, erosion6)
    del erosion3
    del erosion6

    # METTRE_A_ZERO_LES_DERNIERES_COUPES
    mat2 = end_margin(m_xor, 10, 2)
    del m_xor

    mat2 = np.ubyte(mat2)
    m_and = np.where(mat2 == 1, image, 0)
    del mat2

    if front_half_only:
        m_and[:, :, m_and.shape[2] / 2:] = 0

    # ALTITUDE_DU_MIP_DANS_MASQUE_BINAIRE
    x, y, z = m_and.shape
    m_alt = m_and.argmax(2).reshape(x, y, 1)

    m_alt = SpatialImage(m_alt, origin=ori, voxelsize=vxs, metadata_dict=md)
    if only_altitude:
        return m_alt
    else:
        m_mip = m_and.max(2).reshape(x, y, 1)
        m_mip = SpatialImage(m_mip, origin=ori, voxelsize=vxs, metadata_dict=md)
        return m_mip, m_alt


def surface2im(points, altitude):
    """
    This function is used to convert points from maximum intensity projection
    to the real world.

    Parameters
    ----------
    points: list
        list of points from the maximum intensity projection
    altitude: |SpatialImage|
        altitude of maximum intensity projection

    Returns
    -------
    coord: list
        list of points in the real world
    """
    try_spatial_image(altitude, obj_name='altitude')
    coord = list()
    vx, vy, vz = altitude.voxelsize
    for pt in points:
        c = (pt[0] * vx, pt[1] * vy, (altitude[pt[0], pt[1], 0]) * vz)
        coord.append(c)
    return coord


def spatialise_matrix_points(points, image, mip_thresh=45):
    """
    Given a list of points in matrix coordinates (i.e. i,j,k - but k is ignored anyway),
    and a spatial image, returns a list of points in real space (x,y,z) with the Z coordinate
    recomputed from an altitude map extracted from image. This implies that `points` were placed
    on the mip/altitude map result of `max_intensity_projection` applied to `image` with `mip_thresh`.

    Parameters
    ----------
    points: list(tuple(float,float,float)),
        file 2D points to spatialise.
        Can also be a filename pointing to a numpy-compatible list of points.
    image: SpatialImage
        image or path to image to use to spatialise the points.
    mip_thresh: int, float
        threshold used to compute the original altitude map for points.

    Returns
    -------
    points3d: list [of tuple [of float, float, float]]
        3D points in REAL coordinates.
    """
    try_spatial_image(image, obj_name='image')

    if isinstance(points, (str, unicode)):
        points = np.loadtxt(points)

    mip = im2surface(image, threshold_value=mip_thresh, only_altitude=True)
    return surface2im(points, mip)


def surface_landmark_matching(ref_img, ref_pts, flo_img, flo_pts,
                              ref_pts_already_spatialised=False,
                              flo_pts_already_spatialised=False,
                              mip_thresh=45):
    """
    Computes the registration of "flo_img" to "ref_img" by minimizing distances
    between ref_pts and flo_pts.

    Parameters
    ----------
    ref_img: |SpatialImage|, str
        image or path to image to use to reference image.
    ref_pts: list
        ordered sequence of 2D/3D points to use as reference landmarks.
    flo_img: |SpatialImage|, str
        image or path to image to use to floating image
    flo_pts: list
        ordered sequence of 2D/3D points to use as floating landmarks.
    ref_pts_already_spatialised: bool
        If True, consider reference points are already in REAL 3D space.
    flo_pts_already_spatialised: bool
        If True, consider floating points are already in REAL 3D space.
    mip_thresh: int, float
        used to recompute altitude map to project points in 3D if they aren't
        spatialised.

    Returns
    -------
    trs: numpy.ndarray
        The result is a 4x4 **resampling voxel matrix** (*i.e.* from ref_img to flo_img,
         from ref_space to flo_space and NOT from real_space to real_space).

    Notes
    -----
    If `ref_pts_already_spatialised` and `flo_pts_already_spatialised` are True
    and `ref_pts` and `flo_pts` are indeed in real 3D coordinates, then this is
    exactly a landmark matching registration
    """
    try_spatial_image(ref_img, obj_name='ref_img')
    try_spatial_image(flo_img, obj_name='flo_img')

    if isinstance(ref_pts, (str, unicode)):
        ref_pts = np.loadtxt(ref_pts)

    if isinstance(flo_pts, (str, unicode)):
        flo_pts = np.loadtxt(flo_pts)

    if not ref_pts_already_spatialised:
        print "spatialising reference"
        ref_spa_pts = spatialise_matrix_points(ref_pts, ref_img,
                                               mip_thresh=mip_thresh)
    else:
        print "not spatialising reference"
        ref_spa_pts = ref_pts

    if not flo_pts_already_spatialised:
        print "spatialising floating"
        flo_spa_pts = spatialise_matrix_points(flo_pts, flo_img,
                                               mip_thresh=mip_thresh)
    else:
        print "not spatialising floating"
        flo_spa_pts = flo_pts

    trs = pts2transfo(ref_spa_pts, flo_spa_pts)

    # -- trs is from ref_img to flo_img, in other words it is T-1,
    # a resampling matrix to put flo_img into ref_img space. ref_pts and
    # flo_pts are in real coordinates so the matrix is also in
    # real coordinates and must be converted to voxels --
    trs_vox = matrix_real2voxels(trs, flo_img.voxelsize, ref_img.voxelsize)

    return trs_vox


def pts2transfo(x, y):
    """
    Infer rigid transformation from control point pairs using quaternions.

    The quaternion representation is used to register two point sets with known
    correspondences.
    It computes the rigid transformation as a solution to a least squares
    formulation of the problem.

    The rigid transformation, defined by the rotation R and the translation t,
    is optimized by minimizing the following cost function :

        C(R,t) = sum ( |yi - R.xi - t|^2 )

    The optimal translation is given by :

        t_ = y_b - R.x_b

        with x_b and y_b the barycenters of two point sets

    The optimal rotation using quaternions is optimized by minimizing the
    following cost function :

        C(q) = sum ( |yi'*q - q*xi'|^2 )

        with yi' and xi' converted to barycentric coordinates and identified by
        quaternions

    With the matrix representations :

        yi'*q - q*xi' = Ai.q

        C(q) = q^T.|sum(A^T.A)|.q = q^T.B.q

        with A = array([ [       0       ,  (xn_i - yn_i) , (xn_j - yn_j)  ,  (xn_k - yn_k) ],
                         [-(xn_i - yn_i) ,        0       , (-xn_k - yn_k) ,  (xn_j + yn_j) ],
                         [-(xn_j - yn_j) , -(-xn_k - yn_k),      0         ,  (-xn_i - yn_i)],
                         [-(xn_k - yn_k) , -(xn_j + yn_j) , -(-xn_i - yn_i),         0      ] ])

    The unit quaternion representing the best rotation is the unit eigenvector
    corresponding to the smallest eigenvalue of the matrix -B :

        v = a, b.i, c.j, d.k

    The orthogonal matrix corresponding to a rotation by the unit quaternion
    v = a + bi + cj + dk (with |z| = 1) is given by :

        R = array([ [a*a + b*b - c*c - d*d,       2bc - 2ad      ,       2bd + 2ac      ],
                    [      2bc + 2ad      , a*a - b*b + c*c - d*d,       2cd - 2ab      ],
                    [      2bd - 2ac      ,       2cd + 2ab      , a*a - b*b - c*c + d*d] ])


    Parameters
    ----------
    x : list
        list of points
    y : list
        list of points

    Returns
    -------
    T : np.array
        array (R,t) which correspond to the optimal rotation and translation
        T = | R t |
            | 0 1 |

        T.shape(4,4)

    Examples
    --------
    >>> from timagetk.algorithms.reconstruction import pts2transfo
    >>> # x and y, two point sets with 7 known correspondences
    >>> x = [[238.*0.200320, 196.*0.200320, 9.],
             [204.*0.200320, 182.*0.200320, 11.],
             [180.*0.200320, 214.*0.200320, 12.],
             [201.*0.200320, 274.*0.200320, 12.],
             [148.*0.200320, 225.*0.200320, 18.],
             [248.*0.200320, 252.*0.200320, 8.],
             [305.*0.200320, 219.*0.200320, 10.]]
    >>> y = [[173.*0.200320, 151.*0.200320, 17.],
             [147.*0.200320, 179.*0.200320, 16.],
             [165.*0.200320, 208.*0.200320, 12.],
             [226.*0.200320, 204.*0.200320, 9.],
             [170.*0.200320, 254.*0.200320, 10.],
             [223.*0.200320, 155.*0.200320, 13.],
             [218.*0.200320, 109.*0.200320, 23.]]
    >>> pts2transfo(x, y)
    array([[  0.40710149,   0.89363883,   0.18888626, -22.0271968 ],
           [ -0.72459862,   0.19007589,   0.66244094,  51.59203463],
           [  0.55608022,  -0.40654742,   0.72490964,  -0.07837002],
           [  0.        ,   0.        ,   0.        ,   1.        ]])
    """
    # compute barycenters
    # nx vectors of dimension kx
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    nx, kx = x.shape
    x_barycenter = x.sum(0) / float(nx)

    # nx vectors of dimension kx
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    ny, ky = y.shape
    y_barycenter = y.sum(0) / float(ny)

    # Check there are the same number of vectors
    assert nx == ny

    # converting to barycentric coordinates
    x = x - x_barycenter
    y = y - y_barycenter

    # Change of basis (y -> x)
    # ~ y = y - x_barycenter

    # compute of A = yi*q - q*xi
    #             = array([ [       0       ,  (xn_i - yn_i) , (xn_j - yn_j)  ,  (xn_k - yn_k) ],
    #                       [-(xn_i - yn_i) ,        0       , (-xn_k - yn_k) ,  (xn_j + yn_j) ],
    #                       [-(xn_j - yn_j) , -(-xn_k - yn_k),      0         ,  (-xn_i - yn_i)],
    #                       [-(xn_k - yn_k) , -(xn_j + yn_j) , -(-xn_i - yn_i),         0      ] ])
    #

    A = np.zeros([nx, 4, 4])

    A[:, 0, 1] = x[:, 0] - y[:, 0]
    A[:, 0, 2] = x[:, 1] - y[:, 1]
    A[:, 0, 3] = x[:, 2] - y[:, 2]

    A[:, 1, 0] = -A[:, 0, 1]
    A[:, 1, 2] = -x[:, 2] - y[:, 2]
    A[:, 1, 3] = x[:, 1] + y[:, 1]

    A[:, 2, 0] = -A[:, 0, 2]
    A[:, 2, 1] = -A[:, 1, 2]
    A[:, 2, 3] = -x[:, 0] - y[:, 0]

    A[:, 3, 0] = -A[:, 0, 3]
    A[:, 3, 1] = -A[:, 1, 3]
    A[:, 3, 2] = -A[:, 2, 3]

    # compute of B = Sum [A^T.A]
    B = np.zeros([nx, 4, 4])
    At = A.transpose(0, 2, 1)

    # Maybe there is an another way to do not the "FOR" loop
    for i in xrange(nx):
        B[i] = np.dot(At[i], A[i])

    B = B.sum(0)

    # The solution q minimizes the sum of the squares of the errors : C(R) = q^T.B.q is done by
    # the eigenvector corresponding to the biggest eigenvalue of the matrix -B

    W, V = np.linalg.eig(-B)
    max_ind = np.argmax(W)
    # The orthogonal matrix corresponding to a rotation by the unit quaternion q = a + bi + cj + dk (with |q| = 1) is given by
    #   R = array([ [a*a + b*b - c*c - d*d,       2bc - 2ad      ,       2bd + 2ac],
    #               [      2bc + 2ad      , a*a - b*b + c*c - d*d,       2cd - 2ab],
    #               [      2bd - 2ac      ,       2cd + 2ab      , a*a - b*b - c*c + d*d] ])
    #

    # eigenvector corresponding to the biggest eigenvalue
    v = V[:, max_ind]

    R = np.zeros([3, 3])
    R[0, 0] = v[0] * v[0] + v[1] * v[1] - v[2] * v[2] - v[3] * v[3]
    R[0, 1] = 2 * v[1] * v[2] - 2 * v[0] * v[3]
    R[0, 2] = 2 * v[1] * v[3] + 2 * v[0] * v[2]

    R[1, 0] = 2 * v[1] * v[2] + 2 * v[0] * v[3]
    R[1, 1] = v[0] * v[0] - v[1] * v[1] + v[2] * v[2] - v[3] * v[3]
    R[1, 2] = 2 * v[2] * v[3] - 2 * v[0] * v[1]

    R[2, 0] = 2 * v[1] * v[3] - 2 * v[0] * v[2]
    R[2, 1] = 2 * v[2] * v[3] + 2 * v[0] * v[1]
    R[2, 2] = v[0] * v[0] - v[1] * v[1] - v[2] * v[2] + v[3] * v[3]

    # Compute of the matrix (R,t) which correspond to the optimal rotation and translation
    #  M = | R t | = array(4,4)
    #      | 0 1 |
    #

    # compute the optimal translation
    t = y_barycenter - np.dot(R, x_barycenter)

    T = np.zeros([4, 4])
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.

    return T


def matrix_real2voxels(matrix, target_res, source_res):
    """
    Converts a transform matrix (M) expressed in real coordinates
    (a transform from space_r to space_r) into a matrix M' from space_1 to space_2
    where space_s is the voxel space from which M comes from and space_t the
    one where it will end, and space_r is the real common space.

    Parameters
    ----------
    matrix : np.array
        a 4x4 numpy.array.
    target_res : tuple
        a 3-uple of unit vectors for the space_2 (eg: (1.,2.,1)
    source_res : tuple
        a 3-uple of unit vectors for the space_1 (eg: (2.,1.,3)

    Returns
    -------
    np.array
        matrix in "voxel" coordinates (M' mapping space_1 to space_2 , instead of space_r to space_r).
    """
    assert matrix.shape == (4, 4)
    # vx_t, vy_t, vz_t = target_res
    # vx_s, vy_s, vz_s = source_res

    # TODO: this is wrong, no time to check why just now --
    # return np.array( [ [matrix[0,0]* vx_s /vx_t, matrix[0,1]* vx_s /vy_t, matrix[0,2]* vx_s /vz_t, matrix[0,3]/ vx_t],
    #                    [matrix[1,0]* vy_s /vx_t, matrix[1,1]* vy_s /vy_t, matrix[1,2]* vy_s /vz_t, matrix[1,3]/ vy_t],
    #                    [matrix[2,0]* vz_s /vx_t, matrix[2,1]* vz_s /vy_t, matrix[2,2]* vz_s /vz_t, matrix[2,3]/ vz_t],
    #                    [0.,                      0.,                      0.,                                  1.] ] )

    res = matrix.copy()
    h_out = np.diag(source_res)
    res[0:3, 0:3] = np.dot(res[0:3, 0:3], h_out)

    size_in = map(lambda x: 1. / x, target_res)
    h_in = np.diag(size_in)
    res[0:3, :] = np.dot(h_in, res[0:3, :])
    assert (res[3, 0:3] == (0, 0, 0)).all()
    assert res[3, 3] == 1
    return res
