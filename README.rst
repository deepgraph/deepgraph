
|PyPi Version| |PyPi Downloads| |Conda Version| |Conda Downloads| |Documentation|

DeepGraph
=========

DeepGraph is a scalable, general-purpose data analysis package. It implements a
`network representation <https://en.wikipedia.org/wiki/Network_theory>`_ based
on `pandas <http://pandas.pydata.org/>`_
`DataFrames <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
and provides methods to construct, partition and plot networks, to interface
with popular network packages and more.

It is based on the network representation introduced
`here <http://arxiv.org/abs/1604.00971>`_. DeepGraph is also capable of
representing
`multilayer networks <http://deepgraph.readthedocs.io/en/latest/tutorials/terrorists.html>`_.


Main Features
-------------

Utilizing one of Pandas' primary data structures, the
`DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_,
DeepGraph represents the (super)nodes of a graph by one (set of) table(s), and their
pairwise relations (i.e. the (super)edges of a graph) by another (set of) table(s).
DeepGraph's main features are

- `Create edges <https://deepgraph.readthedocs.io/en/latest/api_reference.html#creating-edges>`_:
  Methods that enable an iterative, yet
  vectorized computation of pairwise relations (edges) between nodes using
  arbitrary, user-defined functions on the nodes' properties. The methods
  provide arguments to parallelize the computation and control memory consumption,
  making them suitable for very large data-sets and adjustable to whatever
  hardware you have at hand (from netbooks to cluster architectures).

  Note: the documentation provides a
  `tutorial <https://deepgraph.readthedocs.io/en/latest/tutorials/pairwise_correlations.html>`_
  on how to compute large correlation matrices in parallel using DeepGraph.

- `Partition nodes, edges or a graph <https://deepgraph.readthedocs.io/en/latest/api_reference.html#graph-partitioning>`_:
  Methods to partition nodes,
  edges or a graph by the graph’s properties and labels, enabling the
  aggregation, computation and allocation of information on and between
  arbitrary *groups* of nodes. These methods also let you express
  elaborate queries on the information contained in a deep graph.

- `Interfaces to other packages <https://deepgraph.readthedocs.io/en/latest/api_reference.html#graph-interfaces>`_:
  Methods to convert to common
  network representations and graph objects of popular Python network packages
  (e.g., SciPy sparse matrices, NetworkX graphs, graph-tool graphs).

- `Plotting <https://deepgraph.readthedocs.io/en/latest/api_reference.html#plotting-methods>`_:
  A number of useful plotting methods for networks,
  including drawings on geographical map projections using `basemap <https://github.com/matplotlib/basemap>`__.


Quick Start
-----------

The source code is hosted on GitHub at: https://github.com/deepgraph/deepgraph.

Binary installers are available at the
`Python Package Index (PyPI) <https://pypi.python.org/pypi/deepgraph>`_
and on
`conda-forge <https://anaconda.org/conda-forge/deepgraph>`_.

DeepGraph can be installed via pip::

   $ pip install deepgraph

or if you're using `Conda <http://conda.pydata.org/docs/>`_,
install with::

   $ conda install -c conda-forge deepgraph

Then, import and get started with::

   >>> import deepgraph as dg
   >>> help(dg)

Dependencies
------------

**Required dependencies**

+---------------------------------------+---------------------------+
| Package                               | Minimum supported version |
+=======================================+===========================+
| `Python <https://www.python.org/>`_   | 3.9                       |
+---------------------------------------+---------------------------+
| `NumPy <http://www.numpy.org/>`_      | 1.21.6                    |
+---------------------------------------+---------------------------+
| `Pandas <http://pandas.pydata.org/>`_ | 1.2                       |
+---------------------------------------+---------------------------+

**Optional dependencies ("extras")**

+-----------------------------------------------------+-----------------+-----------+
| Dependency                                          | Minimum Version | pip extra |
+=====================================================+=================+===========+
| `Matplotlib <http://matplotlib.org/>`_              | 3.1             | plot      |
+-----------------------------------------------------+-----------------+-----------+
| `basemap <https://matplotlib.org/basemap/stable/>`_ | 2.0             | basemap   |
+-----------------------------------------------------+-----------------+-----------+
| `PyTables <http://www.pytables.org/>`_              | 3.7             | tables    |
+-----------------------------------------------------+-----------------+-----------+
| `SciPy <http://www.scipy.org/>`_                    | 1.5.4           | scipy     |
+-----------------------------------------------------+-----------------+-----------+
| `NetworkX <https://networkx.github.io/>`_           | 2.4             | networkx  |
+-----------------------------------------------------+-----------------+-----------+
| `graph\_tool <https://graph-tool.skewed.de/>`_      | 2.27            | N/A       |
+-----------------------------------------------------+-----------------+-----------+

See the `full installation instructions <https://deepgraph.readthedocs.io/en/latest/installation.html>`_
for further details.


Documentation
-------------

The official documentation is hosted here:
http://deepgraph.readthedocs.io

The documentation provides a good starting point for learning how
to use the library.

The `API Reference <https://deepgraph.readthedocs.io/en/latest/api_reference.html>`_
lists all available methods of the core
`DeepGraph <https://deepgraph.readthedocs.io/en/latest/generated/deepgraph.deepgraph.DeepGraph.html>`_
class, including links to their respective source code and docstrings. These docstrings
provide detailed information, usage examples and notes for each method.


Development
-----------

All forms of contributions to this project are welcome, whether it's bug reports, bug fixes,
documentation enhancements, feature requests, or new ideas.

How to Contribute

- Report Issues: If you encounter any bugs or issues, please
  `create an issue <https://github.com/deepgraph/deepgraph/issues>`_ detailing the problem.
- Submit Pull Requests: For bug fixes, enhancements, or new features, fork the repository and
  submit a pull request with your changes.
- Documentation Improvements: Help us improve our documentation by suggesting edits or additions.
- Share Ideas: Have an idea to improve the project? Feel free to
  `open a discussion <https://github.com/deepgraph/deepgraph/discussions>`_.

For additional inquiries or direct communication, you can reach me via email: dominik.traxl@posteo.org.


How to Get Started as a Developer
---------------------------------

See the `Installation from Source & Environment Setup
<https://deepgraph.readthedocs.io/en/latest/installation.html#installation-from-source-environment-setup>`_
section in the documentation for complete instructions on building from the git source tree.


Citing DeepGraph
----------------

Please acknowledge the authors and cite the use of this software when results
are used in publications or published elsewhere. Various citation formats are
available here:
https://dx.doi.org/10.1063/1.4952963
For your convenience, you can find the BibTex entry below:

::

   @Article{traxl-2016-deep,
       author      = {Dominik Traxl AND Niklas Boers AND J\"urgen Kurths},
       title       = {Deep Graphs - A general framework to represent and analyze
                      heterogeneous complex systems across scales},
       journal     = {Chaos},
       year        = {2016},
       volume      = {26},
       number      = {6},
       eid         = {065303},
       doi         = {http://dx.doi.org/10.1063/1.4952963},
       eprinttype  = {arxiv},
       eprintclass = {physics.data-an, cs.SI, physics.ao-ph, physics.soc-ph},
       eprint      = {http://arxiv.org/abs/1604.00971v1},
       version     = {1},
       date        = {2016-04-04},
       url         = {http://arxiv.org/abs/1604.00971v1}
   }


Licence
-------

Distributed with a `BSD-3-Clause License. <https://github.com/deepgraph/deepgraph/blob/master/LICENSE>`_::

    Copyright (C) 2017-2025 DeepGraph Developers
    Dominik Traxl <dominik.traxl@posteo.org>


.. |PyPi Version| image:: https://badge.fury.io/py/DeepGraph.svg
    :target: https://pypi.org/project/DeepGraph/

.. |PyPi Downloads| image:: https://img.shields.io/pypi/dm/deepgraph.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/DeepGraph/

.. |Conda Version| image:: https://anaconda.org/conda-forge/deepgraph/badges/version.svg
   :target: https://anaconda.org/conda-forge/deepgraph

.. |Conda Downloads| image:: https://img.shields.io/conda/dn/conda-forge/deepgraph.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/deepgraph

.. |Documentation| image:: https://readthedocs.org/projects/deepgraph/badge/?version=latest
    :target: http://deepgraph.readthedocs.io/en/latest/?badge=latest
