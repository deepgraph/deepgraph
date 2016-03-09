.. _what_is_deepgraph:


*****************
What is DeepGraph
*****************

DeepGraph is an open source `Python <https://www.python.org/>`_ implementation of a new network
representation introduced in (coming soon..). Its purpose is to facilitate any kind of
`data analysis <https://en.wikipedia.org/wiki/Data_analysis>`_ by
interpreting data in terms of
`network theory <https://en.wikipedia.org/wiki/Network_theory>`_.

The basis of this software package is `Pandas <http://pandas.pydata.org/>`_, a
fast and flexible data analysis tool for the Python programming language. Utilizing one of
its primary data structures, the
`DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_,
we represent objects (i.e. the nodes of a network) by one DataFrame, and their
pairwise relations (i.e. the edges of a network) by another DataFrame.

.. This
 enables the representation and analysis of diverse properties of objects and
 their relations in a computationally highly efficient and scalable manner.

One of the main features of DeepGraph is an efficient and scalable creation of edges. Given a set of
nodes in the form of a DataFrame (or an on disc `HDFStore <http://pandas.pydata.org/pandas-docs/stable/io.html#hdf5-pytables>`_), DeepGraph's
:py:meth:`core class <deepgraph.deepgraph.DeepGraph>` provides :ref:`methods <creating_edges>` to
iteratively compute pairwise relations between the nodes using arbitrary,
user-defined functions on the nodes' features. These methods provide arguments
to parallelize the computation and control memory consumption, making them
suitable for very large data-sets and adjustable to whatever hardware you have
at hand.

Furthermore, once a graph is
constructed, DeepGraph allows you to partition its
:py:meth:`nodes <deepgraph.deepgraph.DeepGraph.partition_nodes>`,
:py:meth:`edges <deepgraph.deepgraph.DeepGraph.partition_edges>` or the
:py:meth:`entire graph <deepgraph.deepgraph.DeepGraph.partition_graph>` by the
graph's properties and labels, enabling the aggregation, computation and
allocation of information on and between arbitrary *groups* of nodes.

DeepGraph is not meant to replace or compete with already existing Python
network libraries, such as `NetworkX <https://networkx.github.io/>`_ or
`graph\_tool <https://graph-tool.skewed.de/>`_, but rather to combine and extend
their capabilities with the merits of Pandas. For that matter, the core class
of DeepGraph provides :ref:`interfacing methods <interfacing_methods>` to convert to common network
representations and graph objects of popular Python network packages.

**To get started, have a look at**

  - :ref:`Installation of DeepGraph <installation>`
  - :ref:`DeepGraph's Tutorials <tutorials>`
  - :ref:`API Reference <api_reference>`

.. note::

    This documentation assumes general familiarity with
    `NumPy <http://www.numpy.org/>`_ and `Pandas <http://pandas.pydata.org/>`_.
    If you haven’t used these packages, do invest some time in learning about
    them first.

.. note::

    DeepGraph is free software; you can redistribute it and/or modify it under
    the terms of the :doc:`BSD License </reference/legal>`. We highly welcome
    contributions from the community.
