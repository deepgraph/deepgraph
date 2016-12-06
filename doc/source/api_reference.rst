.. _api_reference:


*************
API Reference
*************

The API reference summarizes DeepGraph's core class, its methods and the
functions subpackage.

.. Table of Contents
.. =================

.. .. contents::
..    :depth: 2

The DeepGraph class
===================

.. currentmodule:: deepgraph.deepgraph

.. autosummary::
   :toctree: generated/

   DeepGraph

.. _creating_edges:

Creating Edges
--------------

.. autosummary::
   :toctree: generated/

   DeepGraph.create_edges
   DeepGraph.create_edges_ft

Graph Partitioning
------------------

.. autosummary::
   :toctree: generated/

   DeepGraph.partition_nodes
   DeepGraph.partition_edges
   DeepGraph.partition_graph

.. _interfacing_methods:

Graph Interfaces
----------------

.. autosummary::
   :toctree: generated/

   DeepGraph.return_cs_graph
   DeepGraph.return_nx_graph
   DeepGraph.return_nx_multigraph
   DeepGraph.return_gt_graph

.. _plotting_methods:

Plotting Methods
----------------

.. autosummary::
   :toctree: generated/

    DeepGraph.plot_2d
    DeepGraph.plot_2d_generator
    DeepGraph.plot_map
    DeepGraph.plot_map_generator
    DeepGraph.plot_hist
    DeepGraph.plot_logfile

Other Methods
-------------

.. autosummary::
   :toctree: generated/

   DeepGraph.append_binning_labels_v
   DeepGraph.append_cp
   DeepGraph.filter_by_values_v
   DeepGraph.filter_by_values_e
   DeepGraph.filter_by_interval_v
   DeepGraph.filter_by_interval_e
   DeepGraph.update_edges

The Functions Module
====================

.. currentmodule:: deepgraph.functions

.. autosummary::
   deepgraph.functions

Connector Functions
-------------------

.. autosummary::
   :toctree: generated/

     great_circle_dist
     cp_node_intersection
     cp_intersection_strength
     hypergeometric_p_value

Selector Functions
------------------

.. autosummary::
   :toctree: generated/


