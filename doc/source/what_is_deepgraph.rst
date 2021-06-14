.. _what_is_deepgraph:


*****************
What is DeepGraph
*****************

DeepGraph is an open source `Python <https://www.python.org/>`_ implementation
of a new network representation introduced
`here <http://arxiv.org/abs/1604.00971>`_. Its purpose is to facilitate
`data analysis <https://en.wikipedia.org/wiki/Data_analysis>`_ by
interpreting data in terms of
`network theory <https://en.wikipedia.org/wiki/Network_theory>`_.

The basis of this software package is `Pandas <http://pandas.pydata.org/>`_, a
fast and flexible data analysis tool for the Python programming language.
Utilizing one of its primary data structures, the
`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_,
we represent objects (i.e. the nodes of a network) by one DataFrame, and their
pairwise relations (i.e. the edges of a network) by another DataFrame.

One of the main features of DeepGraph is an efficient and scalable creation of
edges. Given a set of nodes in the form of a DataFrame (or an on disc
`HDFStore <http://pandas.pydata.org/pandas-docs/stable/io.html#hdf5-pytables>`_),
DeepGraph's :py:meth:`core class <deepgraph.deepgraph.DeepGraph>` provides
:ref:`methods <creating_edges>` to iteratively compute pairwise relations
between the nodes (e.g. similarity/distance measures) using arbitrary, user-defined
functions on the nodes' features. These methods provide arguments to
parallelize the computation and control memory consumption, making them
suitable for very large data-sets and adjustable to whatever hardware you have
at hand (from netbooks to cluster architectures).

Furthermore, once a graph is constructed, DeepGraph allows you to partition its
:py:meth:`nodes <deepgraph.deepgraph.DeepGraph.partition_nodes>`,
:py:meth:`edges <deepgraph.deepgraph.DeepGraph.partition_edges>` or the entire
:py:meth:`graph <deepgraph.deepgraph.DeepGraph.partition_graph>` by the
graph's properties and labels, enabling the aggregation, computation and
allocation of information on and between arbitrary *groups* of nodes. These
methods also let you express elaborate queries on the information contained in
a deep graph.

DeepGraph is not meant to replace or compete with already existing Python
network libraries, such as `NetworkX <https://networkx.github.io/>`_ or
`graph\_tool <https://graph-tool.skewed.de/>`_, but rather to combine and
extend their capabilities with the merits of Pandas. For that matter, the core
class of DeepGraph provides :ref:`interfacing methods <interfacing_methods>` to
convert to common network representations and graph objects of popular Python
network packages.

Deepgraph also implements a number of useful
`plotting methods <https://deepgraph.readthedocs.io/en/latest/api_reference.html#plotting-methods>`_,
including drawings on geographical map projections.

It's also possible to represent :ref:`multilayer networks <tutorial_terror>` by
deep graphs. We're thinking of implementing an interface to a suitable package
dedicated to the analysis of multilayer networks.

.. note::
    Please acknowledge the authors and cite the use of this software when results
    are used in publications or published elsewhere. Various citation formats are
    available here:
    https://aip.scitation.org/action/showCitFormats?type=show&doi=10.1063%2F1.4952963

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

**To get started, have a look at**

  - :ref:`Installation of DeepGraph <installation>`
  - :ref:`DeepGraph's Tutorials <tutorials>`
  - :ref:`API Reference <api_reference>`

**Want to share feedback, or contribute?**

So far the package has only been developed by :ref:`me <contact>`, a fact that
I would like to change very much. So if you feel like contributing in any way,
shape or form, please feel free to contact me, report bugs, create pull
requestes, milestones, etc. You can contact me via email:
dominik.traxl@posteo.org


.. note::

    This documentation assumes general familiarity with
    `NumPy <http://www.numpy.org/>`_ and `Pandas <http://pandas.pydata.org/>`_.
    If you haven’t used these packages, do invest some time in learning about
    them first.

.. note::

    DeepGraph is free software; you can redistribute it and/or modify it under
    the terms of the :doc:`BSD License </reference/legal>`. We highly welcome
    contributions from the community.
