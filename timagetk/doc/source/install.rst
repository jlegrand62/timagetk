.. _ref_installation:

*************
Installation
*************

We advise to create isolated environment using `conda` or `virtualenv`.
This allow to get rid of version requirement incompatibilities and access the
latest versions of some of the libraries (packaged version often stay way behind
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

* **Ubuntu and Debian:**
  ``sudo apt-get install python-pip python-setuptools scons libz-dev``

* **Fedora and Redhat:**
  ``sudo yum install python-pip python-setuptools scons zlib-devel``

* **Mac** (using `Macports <https://www.macports.org/>`_):
  ``sudo port install py27-pip py27-setuptools scons zlib-devel``


Optional requirements
*********************
* `nose <http://nose.readthedocs.io/en/latest/>`_ (testing, all platforms)
* `sphinx <http://www.sphinx-doc.org/en/stable/>`_ (python documentation generator)


Installing optional requirements
********************************
To install optional dependencies inside a conda environment, `eg.` named
``timagetk``, open a shell prompt and type::

$ source activate timagetk
$ conda install nose sphinx


Python packages
***************
* `numpy <http://www.numpy.org/>`_ (array support for python)
* `networkx <https://networkx.github.io/>`_ (data structures for graphs)
* `scikit-image <https://scikit-image.org/>`_ (intensity values rescaling)


Installing Python packages
**************************
To install the python packages inside a conda environment, `eg.` named
``timagetk``, open a shell prompt and type::

$ source activate timagetk
$ conda install numpy networkx scikit-image


Contributing
************
If you are interested in contributing to development or running the latest
source code, grab the git version.
First, *install the requirements* and the following development tools:

* **Ubuntu & Debian:** ``sudo apt-get install git gitk``

* **Fedora & Redhat:** ``sudo yum install git gitk``

* **Mac**: ``sudo port install git-core +doc +bash_completion +gitweb``


Github repository
*****************
The `Github timagetk repository <https://github.com/VirtualPlants/timagetk.git>`_
is not maintained anymore, but is kept for historic reasons.


Gitlab repository
*****************
The `GitLab timagetk repository <https://gitlab.inria.fr/mosaic/timagetk>`_
is maintained by members of the Mosaic team.



Installing from source
**********************
First install ``git`` as explaind above.
Then choose among one of the repository above and:

Clone the timagetk repository::

$ git clone https://gitlab.inria.fr/mosaic/timagetk

Change directory to timagetk::

$ cd timagetk/

Install using python::

$ python setup.py develop --user

Check that timagetk has been added to your .bashrc file::

$ nano ~/.bashrc

If it is not the case, add the following lines to the bottom of your .bashrc file::

    timagetk_path=/path/to/timagetk/folder
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${timagetk_path}/timagetk/build-scons/lib

Open a new shell prompt, or source the bashrc file in the current prompt::

    $ source ~/.bashrc

To execute the tests, use ``nose``::

    $ cd timagetk/test/
    $ nosetests -v


