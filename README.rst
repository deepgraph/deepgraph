
  .. image:: https://anaconda.org/deepgraph/deepgraph/badges/version.svg
     :target: https://anaconda.org/deepgraph/deepgraph

  .. image:: https://anaconda.org/deepgraph/deepgraph/badges/installer/conda.svg
     :target: https://conda.anaconda.org/deepgraph

  .. image:: https://readthedocs.org/projects/deepgraph/badge/?version=latest
     :target: http://deepgraph.readthedocs.org/en/latest/?badge=latest
     :alt: Documentation Status

  .. image:: https://badge.fury.io/py/deepgraph.svg
     :target: https://badge.fury.io/py/deepgraph


DeepGraph
=========

DeepGraph is a scalable, general-purpose data analysis package. It implements a
`network representation <https://en.wikipedia.org/wiki/Network_theory>`_ based
on `pandas <http://pandas.pydata.org/>`_
`DataFrames <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
and provides methods to construct, partition and plot networks, to interface
with popular network packages and more.

It is based on a new network representation introduced
`here <http://arxiv.org/abs/1604.00971>`_. DeepGraph is also capable of
representing
`multilayer networks <http://deepgraph.readthedocs.io/en/latest/tutorials/terrorists.html>`_.


Main Features
-------------

This network package is targeted specifically towards
`Pandas <http://pandas.pydata.org/>`_ users. Utilizing one of Pandas' primary
data structures, the
`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_,
we represent the (super)nodes of a graph by one set of tables, and their
pairwise relations (i.e. the (super)edges of a graph) by another set of tables.
DeepGraph's main features are

- `Create edges <https://deepgraph.readthedocs.io/en/latest/api_reference.html#creating-edges>`_:
  Methods that enable an iterative, yet
  vectorized computation of pairwise relations (edges) between nodes using
  arbitrary, user-defined functions on the nodes' properties. The methods
  provide arguments to parallelize the computation and control memory consumption,
  making them suitable for very large data-sets and adjustable to whatever
  hardware you have at hand (from netbooks to cluster architectures).

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
  including drawings on geographical map projections.


Quick Start
-----------

DeepGraph can be installed via pip from
`PyPI <https://pypi.python.org/pypi/deepgraph>`_

::

   $ pip install deepgraph

or if you're using `Conda <http://conda.pydata.org/docs/>`_,
install with

::

   $ conda install -c https://conda.anaconda.org/deepgraph deepgraph

Then, import and get started with::

   >>> import deepgraph as dg
   >>> help(dg)


Documentation
-------------

The official documentation is hosted here:
http://deepgraph.readthedocs.io

The documentation provides a good starting point for learning how
to use the library. Expect the docs to continue to expand as time goes on.


Development
-----------

Since this project is fairly new, it's not unlikely you might encounter some
bugs here and there. Although the core functionalities are covered pretty well
by test scripts, particularly the plotting methods could use some more testing.

Furthermore, at this point, you can expect rather frequent updates to the
package as well as the documentation. So please make sure to check for updates
every once in a while.

So far the package has only been developed by me, a fact that I would like
to change very much. So if you feel like contributing in any way, shape or
form, please feel free to contact me, report bugs, create pull requestes,
milestones, etc. You can contact me via email: dominik.traxl@posteo.org


Bug Reports
-----------

To search for bugs or report them, please use the bug tracker:
https://github.com/deepgraph/deepgraph/issues


Citing DeepGraph
----------------

Please acknowledge and cite the use of this software and its authors when
results are used in publications or published elsewhere. You can use the
following BibTex entry

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

Distributed with a `BSD license <LICENSE.txt>`_::

    Copyright (C) 2017 DeepGraph Developers
    Dominik Traxl <dominik.traxl@posteo.org>
