.. _installation:


************
Installation
************


Quick Install
=============

DeepGraph can be installed via pip from
`PyPI <https://pypi.python.org/pypi/deepgraph>`_

::

   pip install deepgraph

or if you have `Conda <http://conda.pydata.org/docs/>`_, install with

::

   conda install deepgraph


Installing from Source
======================

Alternatively, you can install DeepGraph from source by downloading a source
archive file (tar.gz or zip). Since DeepGraph is a pure Python package, you
don't need a compiler to build or install it.

Source Archive File
-------------------

  1. Download the source (tar.gz or zip file) from
     https://pypi.python.org/pypi/deepgraph/
     or https://github.com/deepgraph/deepgraph/

  2. Unpack and change directory to the source directory (it should have the
     files README.rst and setup.py).

  3. Run :samp:`python setup.py install` to build and install

  4. (Optional) Run :samp:`py.test` to execute the tests if you have
     `pytest <https://pypi.python.org/pypi/pytest>`_ installed.


GitHub
------

  1. Clone the deepgraph repostitory

       git clone https://github.com/deepgraph/deepgraph.git

  2. Change directory to :samp:`deepgraph`

  3. Run :samp:`python setup.py install` to build and install

  4. (Optional) Run :samp:`py.test` to execute the tests if you have
     `pytest <https://pypi.python.org/pypi/pytest>`_ installed.


If you don't have permission to install software on your system, you can
install into another directory using the :samp:`--user`, :samp:`--prefix`,
or :samp:`--home` flags to setup.py.

For example

::

    python setup.py install --prefix=/home/username/python

or

::

    python setup.py install --home=~

or

::

    python setup.py install --user

If you didn't install in the standard Python site-packages directory
you will need to set your PYTHONPATH variable to the alternate location.
See http://docs.python.org/2/install/index.html#search-path for further details.


Requirements
============

The easiest way to get Python and the required/optional packages is to use
`Conda <http://conda.pydata.org/docs/>`_ (or
`Miniconda <http://conda.pydata.org/miniconda.html>`_), a cross-platform (Linux, Mac
OS X, Windows) Python distribution for data analytics and scientific computing.

Python
------

To use DeepGraph you need `Python <https://www.python.org/>`_ 2.7, 3.3 or
later.


Pandas
------

`Pandas <http://pandas.pydata.org/>`_ is an open source, BSD-licensed library
providing high-performance, easy-to-use data structures and data analysis tools
for the Python programming language.

Pandas is the core dependency of DeepGraph, and it is highly recommended to
install the
`recommended <http://pandas.pydata.org/pandas-docs/stable/install.html#recommended-dependencies>`_
and
`optional <http://pandas.pydata.org/pandas-docs/stable/install.html#optional-dependencies>`_
dependencies of Pandas as well.


NumPy
-----

`NumPy <http://www.numpy.org/>`_ is the fundamental package for scientific
computing with Python.

Needed for internal operations.


Recommended Packages
====================

The following are recommended packages that DeepGraph can use to provide
additional functionality.


Matplotlib
----------

`Matplotlib <http://matplotlib.org/>`_ is a python 2D plotting library which
produces publication quality figures in a variety of hardcopy formats and
interactive environments across platforms.

Allows you to use the :ref:`plotting methods <plotting_methods>` of DeepGraph.

SciPy
-----

`SciPy <http://www.scipy.org/>`_ is a Python-based ecosystem of open-source
software for mathematics, science, and engineering.

Allows you to convert from DeepGraph's network representation to sparse adjacency
matrices (see :py:meth:`return_cs_graph <.return_cs_graph>`).


NetworkX
--------

`NetworkX <https://networkx.github.io/>`_ is a Python language software package
for the creation, manipulation, and study of the structure, dynamics, and
functions of complex networks.

Allows you to convert from DeepGraph's network representation to NetworkX's network
representation (see :py:meth:`return_nx_graph <.return_nx_graph>`).

Graph-Tool
----------

`graph\_tool <https://graph-tool.skewed.de/>`_ is an efficient Python module for
manipulation and statistical analysis of graphs (a.k.a. networks).

Allows you to convert from DeepGraph's network representation to Graph-Tool's
network representation (see :py:meth:`return_gt_graph <.return_gt_graph>`).


Optional Packages
=================

The following are recommended packages that DeepGraph can use to provide
additional functionality.

Scikit-Learn
------------
`sklearn <http://scikit-learn.org/stable/>`_ is a Python module integrating
classical machine learning algorithms in the tightly-knit world of scientific
Python packages (numpy, scipy, matplotlib).
