.. timagetk documentation master file, created by
   sphinx-quickstart on Mon May 30 19:19:49 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |coord| replace:: J.Legrand, G.Malandain
.. |contrib| replace:: G.Baty, S.Ribes
.. |release| replace:: 1.1.0
.. |licence| replace:: Inria licence
.. |name| replace:: Tissue Image Toolkit

Welcome
*******

**Tissue Image Toolkit** (*timagetk*) is a Python package dedicated to image
processing of multicellular architectures such as plants or animals,
and is intended for biologists, modelers and computer scientists.

.. sidebar:: |name|

   .. image:: _static/logo_timagetk.png
       :width: 70px
       :height: 60px

   Coordination: |coord|

   Contributors: |contrib|

   Active teams:
            * Inria team `Mosaic <https://team.inria.fr/mosaic/>`_
            * Inria team `Morpheme <http://www-sop.inria.fr/morpheme/>`_

   Former teams:
            * Inria-Cirad-Inra `Virtual Plants <https://team.inria.fr/virtualplants/>`_
            * Inria Project Lab `Morphogenetics <https://team.inria.fr/morphogenetics/>`_

   Stable release: |release|

   Written in: C, Python

   Operaring system: Linux, Mac OS

   Licence: |licence|, no commercial use

The package provides the following main functionalities (both in 2D and 3D):

* **Linear filtering:** gaussian, gradient, hessian, laplacian, etc.
* **Grayscale mathematical morphology:** erosion, dilation, opening, closing, hat transform, sequential filters, etc.
* **Segmentation:** h-transform, connected-component labeling, watershed, etc.
* **Registration:** rigid, affine and deformable registration, composition of transformations, sequence registration, multi-view fusion etc.
* **Mathematical morphology and computation of features on labeled images:** erosion, dilation, moments, spatial relationships, etc.
* **Temporal tracking based on graph-theory**
* **Unit tests and examples:** see the :ref:`ref_examples`

Thanks to `Python <https://www.python.org/>`_ language these functionalities can
be combined with many other Python libraries such as for example
`NumPy <http://www.numpy.org/>`_ and `SciPy <http://scipy.org/about.html>`_ for scientific
computing or `matplotlib <http://matplotlib.org/>`_ for curve plotting.


Documentation
*************
This is the :ref:`ref_documentation` for |name| version |release|.

To **build timagetk's dynamic documentation**, you first need to install `sphinx <http://www.sphinx-doc.org/en/stable/>`_.

To install ``sphinx`` **system-wide with pip**, open a shell prompt and type::

$ sudo pip install -U Sphinx

To install ``sphinx`` **inside a conda environment**, `eg.` named ``timagetk``,
open a shell prompt and type::

$ source activate timagetk
$ conda install sphinx

Once sphinx is installed, you can **generate the documentation**.
To do so, go to the ``timagetk/timagetk/doc/`` folder and type::

$ make html

This will build HTML docs in the build directory you chose.
To view the generated documentation, open the file:
``timagetk/timagetk/build/html/index.html``


Installation
************
There are many different ways to install |name|, and the best way depends on how you want to use it and what you already have installed.
In order to help you, the :ref:`ref_installation` instructions have been detailed.


Licence
*******
You can distribute and/or modify |name| under the terms of the |licence|.
Many people have contributed to |name|. Some of the contributors are listed
in the :ref:`ref_credits`.
If |name| contributes to a project that leads to a scientific publication,
please acknowledge this work by :ref:`ref_citing` the project.

.. toctree::
   :maxdepth: 2
