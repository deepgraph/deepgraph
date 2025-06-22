"""
DeepGraph - Representation & Analysis of Complex Systems
========================================================

DeepGraph is a scalable, general-purpose data analysis package. It
implements a network representation based on pandas DataFrames and
provides methods to construct, partition and plot graphs, to interface
with popular network packages and more.

It is based on the network representation introduced here:
http://arxiv.org/abs/1604.00971

This module provides the ``deepgraph.DeepGraph`` class for graph representation,
construction and partitioning, with interfacing methods to common
network representations and popular Python network packages. This
class also provides plotting methods to visualize graphs and
their properties and to benchmark the graph construction
parameters.


Documentation
-------------

See https://deepgraph.readthedocs.io for the full documentation, and
https://arxiv.org/abs/1604.00971 for the paper describing the theoretical
framework. Otherwise, see the docstrings of the objects in the deepgraph
namespace.

>>> import deepgraph as dg
>>> help(dg.DeepGraph)

The docstrings assume that ``deepgraph`` has been imported as ``dg``,
``numpy`` as ``np``, and ``pandas`` as ``pd``.

Citing DeepGraph
----------------

Please acknowledge and cite the use of this software and its authors
when results are used in publications or published elsewhere. You can
use the following BibTex entry

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

"""

# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

from deepgraph.deepgraph import DeepGraph

__version__ = "1.1.0"
__author__ = "Dominik Traxl <dominik.traxl@posteo.org>"
__copyright__ = "Copyright 2017-2025 Dominik Traxl"
__license__ = "BSD-3-Clause"
__URL__ = "https://github.com/deepgraph/deepgraph/"
__bibtex__ = """@Article{traxl-2016-deep,
  author      = {Dominik Traxl AND Niklas Boers AND J\"urgen Kurths},
  title       = {Deep Graphs - a general framework to represent and analyze
                 heterogeneous complex systems across scales},
  version     = {1},
  date        = {2016-04-04},
  eprinttype  = {arxiv},
  eprintclass = {physics.data-an, cs.SI, physics.ao-ph, physics.soc-ph},
  eprint      = {https://arxiv.org/abs/1604.00971v1},
  url         = {https://arxiv.org/abs/1604.00971v1}
}"""
