# -*- python -*-
# -*- coding: utf-8 -*-
#
#
#       Copyright 2016 INRIA
#
#       File author(s):
#           Guillaume Baty <guillaume.baty@inria.fr>
#           Sophie Ribes <sophie.ribes@inria.fr>
#           Gregoire Malandain <gregoire.malandain@inria.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       See accompanying file LICENSE.txt
# ------------------------------------------------------------------------------

import numpy as np
import itertools

try:
    from timagetk.wrapping.bal_trsf import BalTransformation
    from timagetk.algorithms import blockmatching
    from timagetk.algorithms import compose_trsf, apply_trsf, mean_trsfs, \
        inv_trsf
    from timagetk.algorithms import mean_images
    from timagetk.components import SpatialImage, imsave
    from timagetk.plugins.registration import rigid_registration
    from timagetk.plugins.registration import affine_registration
    from timagetk.plugins.registration import deformable_registration
except ImportError as e:
    raise ImportError('Import Error: {}'.format(e))
__all__ = ['fusion']


def fusion(list_images, iterations=5, man_trsf_list=None, mean_imgs_prefix="",
           **kwargs):
    """
    Multiview reconstruction (registration).

    Parameters
    ----------
    list_images: list
        list of input ``SpatialImage``
    iterations: int, optional
        number of iterations (default=5)
    man_trsf_list: list, optional
        list of input ``BalTransformation``
        Use it to define initial rigid or affine trsf matrix list obtained
         by manually registering all images (except the first) on the first one.
        They will be used as `init_result_transformation` for blockmaching
         FIRST iteration of rigid step. Should be in real units.
        Successively obtained `rig_trsf` will be used as
         `init_result_transformation` for blockmaching REMAINING
          iterations of rigid step.
    mean_imgs_prefix : str, optional
        if set, filename prefix used to save the intermediary images at each
        iteration, else only the last one is saved

    Returns
    -------
    SpatialImage
         transformed image with its metadata

    Notes
    -----
    kwargs are passed to registration functions.

    Example
    -------
    >>> from timagetk.util import data_path
    >>> from timagetk.components import imread
    >>> from timagetk.algorithms import fusion
    >>> vues = [0, 1, 2]
    >>> list_images = [imread(data_path('fusion_img_' + str(vue) + '.inr'))
                       for vue in vues]
    >>> fus_img = fusion(list_images)
    """
    if iterations is not None:
        iterations = int(abs(iterations))

    # --- check: list of SpatialImage images of min lenght=2:
    try:
        assert isinstance(list_images, list)
    except AssertionError:
        msg = "Paramater 'list_images' should be a list, got '{}'"
        raise TypeError(msg.format(type(list_images)))
    try:
        assert len(list_images) >= 2
    except AssertionError:
        raise ValueError("There should be at least 2 images to fuse.")
    conds_list_img = [isinstance(img, SpatialImage) for img in list_images]
    if False in conds_list_img:
        msg = "These list elements are not 'SpatialImage': {}"
        raise TypeError(msg.format(np.where(conds_list_img == False)))

    # --- check: list of BalTransformation trsf
    if man_trsf_list is not None:
        try:
            assert isinstance(man_trsf_list, list)
        except AssertionError:
            msg = "Paramater 'man_trsf_list' should be a list, got '{}'"
            raise TypeError(msg.format(type(man_trsf_list)))
        try:
            assert len(man_trsf_list) == len(list_images) - 1
        except AssertionError:
            msg = "There should be one less transformation than images to fuse, got {}."
            raise ValueError(msg.format(len(man_trsf_list)))
        conds_list_trsf = [isinstance(trsf, BalTransformation) for trsf in
                           man_trsf_list]
        if False in conds_list_trsf:
            msg = "These list elements are not 'BalTransformation': {}"
            raise TypeError(msg.format(np.where(conds_list_trsf == False)))

    vox_list = [sp_img.voxelsize for sp_img in list_images]
    vox_list = [i for i in
                itertools.chain.from_iterable(vox_list)]  # voxel list
    ext_list = [sp_img.extent for sp_img in list_images]
    ext_list = [i for i in
                itertools.chain.from_iterable(ext_list)]  # extent list

    # -- Use first image in list as reference template:
    if list_images[0].get_dim() == 3:
        min_vox = np.min(vox_list)
        val = int(np.max(ext_list) / np.min(vox_list))
        tmp_arr = np.zeros((val, val, val), dtype=list_images[0].dtype)
        template_img = SpatialImage(tmp_arr,
                                    voxelsize=[min_vox, min_vox, min_vox])
        del tmp_arr
    else:
        raise ValueError("Fusion only work with 3D images.")

    # -- Use first image in list to initialise reference template (ref. image for first round of blockmatching):
    init_ref = apply_trsf(list_images[0], trsf=None, template_img=template_img)

    # -- FIRST iteration of 3-steps registrations: rigid, affine & vectorfield
    # All images in 'list_images' are registered on the first of the list
    init_trsf_list, init_img_list = [], []
    print "\n\n## -- Computing initial mean image..."
    for ind, sp_img in enumerate(list_images[1:]):
        print "# - Performing 3-steps successive registration of image #{} on #{}.".format(
            ind, 0)
        init_trsf = man_trsf_list[ind - 1]
        print "...Rigid registration{}...".format(
            " with 'init-trsf'" if init_trsf is not None else "")
        trsf_rig, res_rig = rigid_registration(sp_img, init_ref,
                                               init_trsf=init_trsf,
                                               pyramid_lowest_level=1, **kwargs)
        # - Update the `man_trsf_list` for the next iteration
        if init_trsf is not None:
            man_trsf_list[ind - 1] = trsf_rig
        print "...Affine registration..."
        trsf_aff, res_aff = affine_registration(sp_img, init_ref,
                                                left_trsf=trsf_rig, **kwargs)

        print "...Composing Rigid & Affine transformations..."
        tmp_trsf = compose_trsf([trsf_rig, trsf_aff])
        print "...Vectorfield registration..."
        trsf_def, res_def = deformable_registration(sp_img, init_ref,
                                                    init_trsf=tmp_trsf,
                                                    **kwargs)

        out_trsf = BalTransformation(c_bal_trsf=trsf_def)
        init_trsf_list.append(out_trsf)
        init_img_list.append(res_def)
    del out_trsf, trsf_def, res_def, trsf_aff, res_aff, trsf_rig, res_rig

    # - Add the reference image to the list of images that will be averaged:
    init_img_list.append(init_ref)
    print "# - Computing the mean image from previous 3-steps registrations..."
    mean_ref = mean_images(init_img_list)
    print "# - Computing the mean transformation from previous 3-steps registrations..."
    mean_trsf = mean_trsfs(init_trsf_list)
    print "# - Computing the inverted mean transformation..."
    mean_trsf_inv = inv_trsf(mean_trsf)
    del mean_trsf
    print "# - Apply it to the mean image..."
    mean_ref_update = apply_trsf(mean_ref, mean_trsf_inv,
                                 template_img=template_img)
    if mean_imgs_prefix != "":
        fname = mean_imgs_prefix + "-mean_img_iter0.inr"
        print "Saving the mean image: {}".format(fname)
        imsave(fname, mean_ref_update)
    del mean_ref, mean_trsf_inv

    # -- REMAINING iterations of 3-steps registrations: rigid, affine & vectorfield
    # For each iteration we use `mean_ref_update` from previous round of registration
    # REMEMBER: the floating images are the ORIGINAL images from `list_images`!!
    # Now even the first image from `list_images` is 3-steps registered !!
    man_trsf_list = [None] + man_trsf_list
    for index in range(1, iterations + 1):
        print "\n\n## -- Interation #{} for mean image computation...".format(
            index)
        # Again, all images in 'list_images' are registered on the first of the list
        init_trsf_list, init_img_list = [], []
        for ind, sp_img in enumerate(list_images):
            print "# - Performing 3-steps successive registration of image #{} on previous mean image.".format(
                ind)
            init_trsf = man_trsf_list[ind]
            print "...Rigid registration{}...".format(
                " with 'init-trsf'" if init_trsf is not None else "")
            trsf_rig, res_rig = rigid_registration(sp_img, mean_ref_update,
                                                   init_trsf=init_trsf,
                                                   pyramid_lowest_level=1,
                                                   **kwargs)
            # - Update the `man_trsf_list` for the next iteration
            if init_trsf is not None:
                man_trsf_list[ind] = trsf_rig
            print "...Affine registration..."
            trsf_aff, res_aff = affine_registration(sp_img, mean_ref_update,
                                                    left_trsf=trsf_rig,
                                                    **kwargs)
            print "...Composing Rigid & Affine transformations..."
            tmp_trsf = compose_trsf([trsf_rig, trsf_aff])
            print "...Vectorfield registration..."
            trsf_def, res_def = deformable_registration(sp_img, mean_ref_update,
                                                        init_trsf=tmp_trsf,
                                                        **kwargs)
            out_trsf = BalTransformation(c_bal_trsf=trsf_def)
            init_trsf_list.append(out_trsf)
            init_img_list.append(res_def)
        del out_trsf, trsf_def, res_def, trsf_aff, res_aff, trsf_rig, res_rig

        # - Add the mean reference image of this round to the list of images that will be averaged:
        init_img_list.append(mean_ref_update)
        # - Compute the mean_image, mean_trsf, mean_trsf_inv & mean_ref_update
        print "# - Computing the mean image from previous 3-steps registrations..."
        mean_ref = mean_images(init_img_list)
        print "# - Computing the mean transformation from previous 3-steps registrations..."
        mean_trsf = mean_trsfs(init_trsf_list)
        print "# - Computing the inverted mean transformation..."
        mean_trsf_inv = inv_trsf(mean_trsf)
        del mean_trsf
        print "# - Apply it to the mean image..."
        # This update the reference image for the next round!
        mean_ref_update = apply_trsf(mean_ref, mean_trsf_inv,
                                     template_img=template_img)
        if mean_imgs_prefix != "":
            fname = mean_imgs_prefix + "-mean_img_iter{}.inr".format(index)
            print "Saving the mean image: {}".format(fname)
            imsave(fname, mean_ref_update)
        del mean_ref, mean_trsf_inv

    return mean_ref_update
