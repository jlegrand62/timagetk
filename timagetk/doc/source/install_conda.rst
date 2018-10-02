.. _ref_conda_installation:

******************
Conda installation
******************

Introducing Conda
*****************

Conda is a tool to install and manage different software libraries, possibly at different versions, into independent system configurations called **environments**.
Each environment has a name and stores its own version of each library (distinct from the one of your general system) so that you can make installations that would be incompatible with the rest of your system without impacting its functioning and breaking everything!

Concretely, miniconda will create a new directory for each environment where it will download libraries, binaries and packages.
The command ``source activate environment_name`` modifies your environment variables (e.g. ``PATH``, ``PYTHONPATH``) so that your system points onto this directory.
To recover your general system installation, you simply need to run the command ``source deactivate``.


Installing miniconda
********************

**Download the latest miniconda installer** from the `official website <https://repo.continuum.io>`_:

* LINUX: https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
* MAC: https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
* Windows: https://repo.continuum.io/miniconda/Miniconda2-latest-Windows-x86_64.exe

With Linux and MacOSX, **make sure to answer YES when asked to add conda to your** ``PATH``.

Linux
-----

**Download** manually or use ``wget`` to perform this download from a shell::

$ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh

**Install miniconda** by running the installer in a shell prompt (where you downloaded the installer)::

$ bash Miniconda2-latest-Linux-x86_64.sh

Once installed, you can remove the installer::

$ rm Miniconda2-latest-Linux-x86_64.sh


MacOSX
------

**Download** manually or use ``wget`` to perform this download from a shell::

$ wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh

**Install miniconda** by running the installer in a shell prompt (where you downloaded the installer)::

$ bash Miniconda2-latest-MacOSX-x86_64.sh

Once installed, you can remove the installer::

$ rm Miniconda2-latest-MacOSX-x86_64.sh

Windows
-------

Download manually, execute the installer and follow the instructions.


Creating a new environment
**************************

Choose between a **manual** or **automatic** environment creation and installation.
We advise to use the automatic version, but for the sake of clarity, we hereafter detail the steps required to perform a manual installation.


Manual creation
---------------

To create a new conda environment, to install timagetk and its dependencies (named ``timagetk`` for instance)::

$ conda create -n timagetk -y python=2.7

By default ``ipython`` is **not installed** in the newly created environment, to install it run::

$ conda install ipython

To **activate the environment**, run::

$ source activate timagetk

Installing Python packages
==========================
To install the python packages inside a conda environment, `eg.` named ``timagetk``, open a shell prompt and type::

$ source activate timagetk
$ conda install numpy networkx scikit-image

Installing optional Python packages
===================================
To install the python packages inside a conda environment, `eg.` named ``timagetk``, open a shell prompt and type::

$ source activate timagetk
$ conda install nose sphinx

You are now set to clone and install the source repository of `timagetk <https://gitlab.inria.fr/mosaic/timagetk>`_ as detailed in :ref:`ref_install_from_source`.


Automatic creation
------------------

After installing miniconda, create a ``YAML`` configuration file, named ``timagetk.yml``, listing requirements and dependencies:

::

    name: timagetk
    channels:
      - defaults
    dependencies:
      - python=2.7
      - ipython
      - numpy
      - scikit-image
      - networkx
      - nose
      - sphinx

To automatically **create the environment** and **install dependencies**, in a new terminal run::

$ conda env create -f timagetk.yml

To **activate the environment**, run::

$ source activate timagetk

You are now set to clone and install the source repository of `timagetk <https://gitlab.inria.fr/mosaic/timagetk>`_ as detailed in :ref:`ref_install_from_source`.

.. toctree::
   :maxdepth: 2
