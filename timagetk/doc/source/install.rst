.. _ref_installation:

*************
Installation
*************

We advise to **create isolated environment** using `conda` (see :ref:`ref_conda_installation`) or `virtualenv`.
This allow to get rid of version requirement incompatibilities and access the latest versions of some of the libraries (packaged version often stay way behind
for stability reasons).


Requirements
************
* python 2.7 (to check the version, open a shell prompt and type: ``python --version``)
* `pip <https://pypi.python.org/pypi/pip>`_ and setuptools (extensions for python package installation)
* `scons <http://scons.org/>`_ (build tool)
* `zlib <http://www.zlib.net/>`_ (compression/decompression)
* `numpy <http://www.numpy.org/>`_ (array support for python)


Installing requirements
***********************
To install `pip`, `setuptools`, `scons` and `zlib`:

* **Ubuntu and Debian:** ``sudo apt-get install python-pip python-setuptools scons libz-dev``
* **Fedora and Redhat:** ``sudo yum install python-pip python-setuptools scons zlib-devel``
* **MacOS** (using `Macports <https://www.macports.org/>`_): ``sudo port install py27-pip py27-setuptools scons zlib-devel``


Python packages
***************
* `numpy <http://www.numpy.org/>`_ (array support for python)
* `networkx <https://networkx.github.io/>`_ (data structures for graphs)
* `scikit-image <https://scikit-image.org/>`_ (intensity values rescaling)


Optional requirements
*********************
* `nose <http://nose.readthedocs.io/en/latest/>`_ (testing, all platforms)
* `sphinx <http://www.sphinx-doc.org/en/stable/>`_ (python documentation generator)


.. _ref_install_from_source:

Installing from source
**********************
First install ``git`` as explaind in :ref:`ref_contributing`.
Then choose among one of the repository above (Github or GitLab) and:

1. Clone the timagetk repository (default is GitLab)::

    $ git clone https://gitlab.inria.fr/mosaic/timagetk

2. Change directory to timagetk::

    $ cd timagetk/

3. Install using python::

    $ python setup.py develop --user

4. Check the ``LD_LIBRARY_PATH`` **environment variable** has been correctly added to your ``.bashrc`` file:

 1. Open your ``.bashrc`` file::

     $ nano ~/.bashrc

 2. Look for the following lines at the end (replacing ``</path/to/timagetk/folder>`` by your installation path), if not there, add them::

     timagetk_path=</path/to/timagetk/folder>
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${timagetk_path}/timagetk/build-scons/lib

5. Open a new shell prompt, or source the bashrc file in the current prompt::

    $ source ~/.bashrc

6. To execute the tests, use ``nose``::

    $ cd timagetk/test/
    $ nosetests -v


.. _ref_contributing:

Contributing
************
If you are interested in contributing to development or running the latest source code, grab the git version.
First, *install the requirements* and the following development tools:

* **Ubuntu & Debian:** ``sudo apt-get install git gitk``
* **Fedora & Redhat:** ``sudo yum install git gitk``
* **MacOS**: ``sudo port install git-core +doc +bash_completion +gitweb``


Github repository
*****************
The `Github timagetk repository <https://github.com/VirtualPlants/timagetk.git>`_ is not maintained anymore, but is kept for historic reasons.


Gitlab repository
*****************
The `GitLab timagetk repository <https://gitlab.inria.fr/mosaic/timagetk>`_ is maintained by members of the Mosaic team.
