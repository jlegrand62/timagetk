# -*- python -*-
# -*- coding: utf-8 -*-
#
#       Copyright 2018 CNRS - ENS Lyon
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
# -----------------------------------------------------------------------------

__license__ = "Private/UnderDev"
__revision__ = " $Id$ "

import time
import numpy as np
import scipy.ndimage as nd

from timagetk.util import get_class_name
from timagetk.util import get_attributes
from timagetk.util import elapsed_time
from timagetk.util import percent_progress

from timagetk.components import SpatialImage
from timagetk.components import LabelledImage
from timagetk.components.labelled_image import image_with_labels
from timagetk.components.labelled_image import image_without_labels
from timagetk.components.labelled_image import structuring_element
from timagetk.components.labelled_image import test_structuring_element

__all__ = ['TissueImage']

MISS_CELL = "The following cell{} {} not found in the image: {}"  # ''/'s'; 'is'/'are'; labels


def voxel_n_layers(image, background, connectivity=None, n_layers=1, **kwargs):
    """
    Extract the n-first layer of non-background voxels, ie. those in contact
    with the background.

    Parameters
    ----------
    image: np.array
        the TissueImage containing the labelled array with a background
    background: int
        id of the background label
    connectivity: int, optional
        connectivity or neighborhood of the structuring element, default is 18
        in 3D and 4 in 2D
        should be in [4, 6, 8, 18, 26], where 4 and 8 are 2D structuring
        elements, the rest are 3D structuring elements.
    n_layers: int, optional
        number of layer of voxels to extract, the first one being in contact
        with the background

    Returns
    -------
    np.array
        labelled image made of the selected number of voxel layers
    """
    verbose = kwargs.get('verbose', False)
    if verbose:
        print "Extracting the first layer of voxels...",

    t_start = time.time()
    # - Define default connectivity according to dimensionality:
    if connectivity is None:
        if image.ndim == 2:
            connectivity = 4
        else:
            connectivity = 18
    # - Get background position (mask)
    mask_img_1 = (image == background)
    # - Dilate it by one voxel using a 18-connexe 3D structuring element:
    struct = structuring_element(connectivity)
    assert test_structuring_element(image, struct)
    dil_1 = nd.binary_dilation(mask_img_1, structure=struct,
                               iterations=n_layers)
    # - Difference with background mask gives the first layer of voxels:
    layer = dil_1 ^ mask_img_1

    if verbose:
        print elapsed_time(t_start)

    return image * layer, mask_img_1


class TissueImage(LabelledImage):
    """
    Class to manipulate biological tissues, made of cells and potentially a
    background.
    """

    def __new__(cls, image, **kwargs):
        """
        TissueImage construction method.

        Parameters
        ----------
        image : np.array|SpatialImage|LabelledImage
            a numpy array or a SpatialImage containing a labelled array

        kwargs
        ------
        origin: list, optional
            coordinates of the origin in the image, default: [0,0] or [0,0,0]
        voxelsize: list, optional.
            image voxelsize, default: [1.0,1.0] or [1.0,1.0,1.0]
        dtype: str, optional
            image type, default dtype = input_array.dtype
        metadata_dict: dict, optional
            dictionary of image metadata, default is an empty dict
        no_label_id : int, optional
            if given define the "unknown label" (ie. not a label)

        Example
        -------
        >>> import numpy as np
        >>> from timagetk.components import SpatialImage
        >>> from timagetk.components import LabelledImage
        >>> from timagetk.components import TissueImage
        >>> test_array = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
        >>> test_array[0,:] = np.ones((5,), dtype=np.uint8)
        >>> # - Construct from a NumPy array:
        >>> tissue = TissueImage(test_array, voxelsize=[0.5,0.5], no_label_id=0, background=1)
        >>> print tissue.background
        1
        >>> # - Construct from a SpatialImage:
        >>> image = SpatialImage(test_array, voxelsize=[0.5,0.5])
        >>> tissue = TissueImage(image, no_label_id=0, background=1)
        >>> print tissue.background
        1
        >>> # - Construct from a LabelledImage:
        >>> lab_image = LabelledImage(test_array, voxelsize=[0.5,0.5], no_label_id=0)
        >>> tissue = TissueImage(lab_image, background=1)
        >>> print tissue.background
        1
        """
        if isinstance(image, LabelledImage):
            # -- Can be a LabelledImage or any class inheriting from it:
            return super(TissueImage, cls).__new__(cls, image, **kwargs)
        elif isinstance(image, SpatialImage):
            # -- Can be a SpatialImage or any class inheriting from it:
            no_label_id = kwargs.pop('no_label_id', None)
            return super(TissueImage, cls).__new__(cls, image,
                                                   no_label_id=no_label_id,
                                                   **kwargs)
        elif isinstance(image, np.ndarray):
            # -- Case where constructing from a NumPy array:
            origin = kwargs.pop('origin', None)
            voxelsize = kwargs.pop('voxelsize', None)
            dtype = kwargs.pop('dtype', image.dtype)
            metadata = kwargs.pop('metadata_dict', None)
            no_label_id = kwargs.pop('no_label_id', None)
            return super(LabelledImage, cls).__new__(cls, image,
                                                     origin=origin,
                                                     voxelsize=voxelsize,
                                                     dtype=dtype,
                                                     metadata_dict=metadata,
                                                     no_label_id=no_label_id,
                                                     **kwargs)
        else:
            msg = "Undefined construction method for type '{}'!"
            raise NotImplementedError(msg.format(type(image)))

    def __init__(self, image, background=None, **kwargs):
        """
        TissueImage initialisation method.

        Parameters
        ----------
        image: LabelledImage
            the image containing the labelled tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        """
        # - In case a TissueImage is contructed from a TissueImage, get the attributes values:
        if isinstance(image, TissueImage):
            attr_list = ["background"]
            attr_dict = get_attributes(image, attr_list)
            class_name = get_class_name(image)
            msg = "Overriding optional keyword arguments '{}' ({}) with defined attribute ({}) in given '{}'!"
            # -- Check necessity to override 'origin' with attribute value:
            if attr_dict['background'] is not None:
                if background is not None and background != attr_dict[
                    'background']:
                    print msg.format('background', background,
                                     attr_dict['background'], class_name)
                background = attr_dict['background']

        # - Call initialisation method of LabelledImage:
        super(TissueImage, self).__init__(image, **kwargs)

        # - Initializing EMPTY hidden attributes:
        # -- Integer defining the background label:
        self._background_id = None
        # -- List of cells:
        self._cells = None
        # -- List of epidermal cells (L1):
        self._cell_layer1 = None
        # -- Array with only the first layer of voxels in contact with the
        #  background label:
        self._voxel_layer1 = None

        # - Initialise object property and most used hidden attributes:
        # -- Define the background value, if any (can be None):
        self.background = background
        # -- Get the list of cells found in the image:
        self.cells()

    @property
    def background(self):
        """
        Get the background label, can be None.

        Returns
        -------
        int
            the label value for the background
        """
        if self._background_id is None:
            print "WARNING: no value defined for the background id!"
        return self._background_id

    @background.setter
    def background(self, label):
        """
        Set the background label.

        Parameters
        ----------
        int
            integer defining the background id in the image.
        """
        if not isinstance(label, int) and label is not None:
            print "Provided label '{}' is not an integer!".format(label)
            return
        elif label not in self._labels:
            print "Provided label '{}' is not found in the image!".format(label)
            return
        else:
            self._background_id = label
        self.metadata = {'background': self.background}

    def cells(self, cells=None):
        """
        Get the list of cells found in the image, or make sure provided list of
        cells exists.

        Parameters
        ----------
        cells: int|list, optional
            if given, used to filter the returned list, else return all cells
            defined in the image by default

        Returns
        -------
        list
            list of cells found in the image, except for 'background'
            (if defined)

        Notes
        -----
        Value defined for 'background' is removed from the returned list of
        cells since it does not refer to one.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])
        >>> from timagetk.components import TissueImage
        >>> im = TissueImage(a, background=1)
        >>> im.labels()
        [1, 2, 3, 4, 5, 6, 7]
        >>> im.cells()
        [2, 3, 4, 5, 6, 7]
        """
        return list(set(self.labels(cells)) - {self.background})

    def nb_cells(self):
        """
        Return the number of cells.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.array([[1, 2, 7, 7, 1, 1],
                          [1, 6, 5, 7, 3, 3],
                          [2, 2, 1, 7, 3, 3],
                          [1, 1, 1, 4, 1, 1]])

        >>> from timagetk.components import LabelledImage
        >>> im = TissueImage(a, background=None)
        >>> im.nb_cells()
        7
        >>> im = TissueImage(a, background=1)
        >>> im.nb_cells()
        6
        """
        return len(self.cells())

    # ##########################################################################
    #
    # Labelled SpatiaImage edition functions:
    #
    # ##########################################################################

    def get_image_with_cells(self, cells, keep_background=True):
        """
        Returns a LabelledImage with only the selected 'cells', the rest are
        replaced by "self._no_label_id".

        Parameters
        ----------
        cells: int|list(int)
            if None (default) returns all cells.
            if an integer, make sure it is in self.cells()
            if a list of integers, make sure they are in self.cells()
            if a string, should be in LABEL_STR to get corresponding
            list of cells (case insensitive)
        keep_background: bool, optional
            indicate if background label should be kept in the returned image

        Returns
        -------
        template_im: TissueImage
            labelled image with 'cells' only
        """
        try:
            assert self._no_label_id is not None
        except AssertionError:
            msg = "Attribute 'no_label_id' is not defined (None)!"
            raise ValueError(msg)

        all_cells = self.cells()
        cells = self.cells(cells)
        off_cells = list(set(all_cells) - set(cells))

        back_id = self.background
        if keep_background:
            try:
                assert back_id is not None
            except AssertionError:
                msg = "You asked to keep the background position, but no background label is defined!"
                raise ValueError(msg)
            else:
                cells.append(back_id)
        else:
            if back_id:
                off_cells.append(back_id)

        if len(off_cells) == 0:
            print "WARNING: you selected ALL cells!"
            return self
        if len(cells) == 0:
            print "WARNING: you selected NO cell!"
            return

        if len(cells) < len(off_cells):
            template_im = image_with_labels(self, cells)
        else:
            template_im = image_without_labels(self, off_cells)
        return TissueImage(template_im, background=back_id)

    def get_image_without_cells(self, cells, keep_background=True):
        """
        Returns a SpatialImage without the selected cells.

        Parameters
        ----------
        cells: None|int|list|str
            label or list of cells to keep in the image.
            if None, neighbors for all cells found in self.image will
            be returned.
        keep_background: bool, optional
            indicate if background label should be kept in the returned image

        Returns
        -------
        template_im: SpatialImage
            labelled image with 'cells' only
        """
        all_cells = self.cells()
        cells = self.cells(cells)
        off_cells = list(set(all_cells) - set(cells))
        return self.get_image_with_cells(off_cells, keep_background)

    def fuse_cells_in_image(self, cells, value='min', verbose=True):
        """
        Fuse the provided list of cells to its minimal value.

        Parameters
        ----------
        cells: list
            list of cells to fuse
        value: str, optional
            value used to replace the given list of cells, by default use the
            min value of the `cells` list. Can also be the max value.
        verbose: bool, optional
            control verbosity

        Returns
        -------
        Nothing, modify the TissueImage array (re-instantiate the object)
        """
        cells = self.cells(cells)
        return self.fuse_labels_in_image(cells, value, verbose)

    def remove_cells_from_image(self, cells, verbose=True):
        """
        Remove 'cells' from self.image using 'erase_value'.

        Parameters
        ----------
        cells: list|str
            list of cells to remove from the image
        verbose: bool, optional
            control verbosity

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)
        """
        cells = self.cells(cells)
        return self.remove_labels_from_image(cells, verbose)

    def relabelling_cells_from_mapping(self, mapping, clear_unmapped=False,
                                       **kwargs):
        """
        Relabel the image following a given mapping indicating the original
        cell id as keys and their new id as value.
        It is possible to get rid of all unmapped cells by setting
        `clear_unmapped` to True.

        Parameters
        ----------
        mapping: dict
            a dictionary indicating the original cell id as keys and their new
            id as value
        clear_unmapped: bool, optional
            if True (default False), only the mapped cells are kept in the
            returned image (unmapped set to 'no_label_id')

        **kwargs
        --------
        verbose: bool, optional
            control code verbosity; default = False

        Returns
        -------
        Nothing, modify the LabelledImage array (re-instantiate the object)
        """
        try:
            mapping.pop(self.background)
        except KeyError:
            pass
        else:
            print "Removed the background label from the mapping list!"
        return self.relabel_from_mapping(mapping, clear_unmapped, **kwargs)

    def voxel_n_first_layer(self, n_voxel_layer, connectivity,
                            keep_background=True, **kwargs):
        """
        Extract the n-first layer of non-background voxels in contact with the
        background as a TissueImage.

        Parameters
        ----------
        n_voxel_layer: int
            number of layer of voxel from the background to get
        connectivity: int
            connectivity or neighborhood of the structuring element
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        TissueImage
            labelled image made of the selected number of voxel layers
        """
        mask_img_1 = None
        if self._voxel_layer1 is None:
            self._voxel_layer1, mask_img_1 = voxel_n_layers(self,
                                                            iter=n_voxel_layer,
                                                            connectivity=connectivity,
                                                            **kwargs)

        if keep_background:
            if mask_img_1 is None:
                mask_img_1 = (self.get_array() == self.background)
            return self._voxel_layer1 + mask_img_1
        else:
            return self._voxel_layer1

    def voxel_first_layer(self, connectivity, keep_background=True,
                          **kwargs):
        """
        Extract the first layer of non-background voxels in contact with the
        background as a LabelledImage.

        Parameters
        ----------
        connectivity: int
            connectivity or neighborhood of the structuring element
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        TissueImage
            labelled image made of the first layer of voxel in contact with the
            background
        """
        mask_img_1 = None
        if self._voxel_layer1 is None:
            self._voxel_layer1, mask_img_1 = voxel_n_layers(self,
                                                            connectivity=connectivity,
                                                            **kwargs)

        if keep_background:
            if mask_img_1 is None:
                mask_img_1 = (self.get_array() == self.background)
            return self._voxel_layer1 + mask_img_1
        else:
            return self._voxel_layer1

    def voxel_first_layer_coordinates(self, **kwargs):
        """
        Returns an (Nxd) array of coordinates indicating voxels first layer
        position.
        """
        vfl = self.voxel_first_layer(keep_background=False, **kwargs)
        return np.array(np.where(vfl != 0)).T

    def cell_first_layer(self, **kwargs):
        """
        List cells corresponding to the first layer of cells (epidermis).
        It is possible to provide an epidermal area threshold (minimum area in
        contact with the background) to consider a cell as in the first layer.

        Returns
        -------
        list
            list of L1-cells
        """
        verbose = kwargs.get('verbose', False)
        if verbose:
            print "Generating list of L1 cells..."

        integers = lambda x: map(int, x)
        bkgd_id = self.background
        # - Create unfiltered list of ALL neighbors to the background:
        if self._cell_layer1 is None:
            background_nei = self.neighbors(bkgd_id, verbose=False)
            self._cell_layer1 = set(integers(background_nei))

        return list(self._cell_layer1)


class TissueImage2D(TissueImage):
    """
    Class specific to 2D multi-cellular tissues.
    """

    def __init__(self, image, background=None, **kwargs):
        """
        Parameters
        ----------
        image: SpatialImage
            the SpatialImage containing the labelled tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        """
        TissueImage.__init__(self, image, background=background, **kwargs)

    def voxel_n_first_layer(self, n_voxel_layer, connectivity=4,
                            keep_background=True, **kwargs):
        """
        Extract the n-first layer of non-background voxels in contact with the
        background.

        Parameters
        ----------
        n_voxel_layer: int
            number of layer of voxel from the background to get
        connectivity: int
            connectivity of the 2D structuring element, default 4
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        TissueImage2D
            labelled image made of the selected number of voxel layers
        """
        return TissueImage.voxel_n_first_layer(self, n_voxel_layer,
                                               connectivity=connectivity,
                                               keep_background=keep_background,
                                               **kwargs)

    def voxel_first_layer(self, connectivity=4, keep_background=True,
                          **kwargs):
        """
        Extract the first layer of non-background voxels in contact with the
        background.

        Parameters
        ----------
        connectivity: int, optional
            connectivity of the 2D structuring element, default 4
        keep_background: bool, optional
            if true the LabelledImage returned contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        TissueImage2D
            image made of the first layer of voxel in contact with the
            background
        """
        return TissueImage.voxel_first_layer(self, connectivity=connectivity,
                                             keep_background=keep_background,
                                             **kwargs)


class TissueImage3D(TissueImage):
    """
    Class specific to 3D multi-cellular tissues.
    """

    def __init__(self, image, background=None, **kwargs):
        """
        Parameters
        ----------
        image: LabelledImage
            the LabelledImage containing the multi-cellular tissue
        background: int, optional
            if given define the label of the background (ie. space surrounding
            the tissue)
        """
        TissueImage.__init__(self, image, background=background, **kwargs)

    def voxel_n_first_layer(self, n_voxel_layer, connectivity=18,
                            keep_background=True, **kwargs):
        """
        Extract the n-first layer of non-background voxels in contact with the
        background.

        Parameters
        ----------
        n_voxel_layer: int
            number of layer of voxel from the background to get
        connectivity: int
            connectivity of the 3D structuring element, default 18
        keep_background: bool, optional
            if true the returned image contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        TissueImage3D
            image made of the selected number of voxel layers
        """
        return TissueImage.voxel_n_first_layer(self, n_voxel_layer,
                                               connectivity=connectivity,
                                               keep_background=keep_background,
                                               **kwargs)

    def voxel_first_layer(self, connectivity=18, keep_background=True,
                          **kwargs):
        """
        Extract the first layer of non-background voxels in contact with the
        background.

        Parameters
        ----------
        connectivity: int, optional
            connectivity of the 3D structuring element, default 18
        keep_background: bool, optional
            if true the returned image contains the background in addition
            of the first layer of labelled voxels

        Returns
        -------
        TissueImage3D
            image made of the first layer of voxel in contact with the
            background
        """
        return TissueImage.voxel_first_layer(self, connectivity=connectivity,
                                             keep_background=keep_background,
                                             **kwargs)
