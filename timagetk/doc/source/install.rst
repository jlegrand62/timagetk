.. _ref_installation:

*************
Installation
*************

Requirements
************

* python 2.7 (to check the version, open a shell prompt and type: ``python --version``)
* `pip <https://pypi.python.org/pypi/pip>`_ and setuptools (extensions for python package installation)

* `scons <http://scons.org/>`_ (build tool)
* `zlib <http://www.zlib.net/>`_ (compression/decompression)
* `numpy <http://www.numpy.org/>`_ (array support for python)
* `networkx <https://networkx.github.io/>`_ (data structures for graphs)

To install pip, setuptools, scons, zlib and numpy:

* **Ubuntu and Debian:**
  ``sudo apt-get install python-pip python-setuptools scons libz-dev python-numpy``

* **Fedora and Redhat:**
  ``sudo yum install python-pip python-setuptools scons zlib-devel numpy``

* **Mac** (using `Macports <https://www.macports.org/>`_):
  ``sudo port install py27-pip py27-setuptools scons zlib-devel py27-numpy``

To install NetworkX (all platforms): ``sudo pip install networkx`` or  ``conda install networkx``

To install `nose <http://nose.readthedocs.io/en/latest/>`_ (testing, all platforms): ``sudo pip install nose``

Contributing
************
If you are interested in contributing to development or running the latest source code,
grab the git version. First, install the requirements and the development tools:

* **Ubuntu & Debian:** ``sudo apt-get install git gitk``

* **Fedora & Redhat:** ``sudo yum install git gitk``

* **Mac**: ``sudo port install git-core +doc +bash_completion +gitweb``


Github repository
*****************
Github timagetk repository is not up to date, but is kept for historic reasons.
https://github.com/VirtualPlants/timagetk.git

Gitlab repository
*****************
The GitLab timagetk repository is maintained by members of the Mosaic team.
https://gitlab.inria.fr/mosaic/timagetk

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


*************
Documentation
*************

To build **timagetk**'s dynamic documentation, you first need to install (`sphinx <http://www.sphinx-doc.org/en/stable/>`_).

To install ``sphinx`` **system-wide with pip**, open a shell prompt and type::

$ sudo pip install -U Sphinx

To install ``sphinx`` **inside a conda environment**, named ``<my_env_name>``, open a shell prompt and type::

$ source activate <my_env_name>
$ conda install sphinx

Once sphinx is installed, you can **generate the documentation**.
To do so, go to the ``timagetk/timagetk/doc/`` folder and type::

$ make html

This will build HTML docs in the build directory you chose.
To view the generated documentation, open the file: ``timagetk/timagetk/build/html/index.html``
