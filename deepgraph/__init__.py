"""
DeepGraph - Representation & Analysis of Complex Systems
========================================================

DeepGraph (dg) is an efficient, general-purpose data analysis Python package.
It provides the means to construct, manipulate, partition and study graphs
(a.k.a. networks). It is based on the paper [coming soon..].

This module provides:

   1. A ``deepgraph.DeepGraph`` class for graph representation,
       construction and partitioning, with interfacing methods to common
       network representations and popular Python network packages. This
       class also provides plotting methods to visualize graphs and
       their properties and to benchmark the graph construction
       parameters.

   2. A ``deepgraph.functions`` module, providing auxiliary
       **connector** and **selector** functions to create edges between
       nodes.

Documentation
-------------

See http://deepgraph.readthedocs.org for a full documentation, and
[coming soon..] for the paper describing the theoretical framework. Otherwise,
see the docstrings of the objects in the deepgraph namespace.

The docstrings assume that ``deepgraph`` has been imported as ``dg``,
``numpy`` as ``np`` and ``pandas`` as ``pd``.

>>> import deepgraph as dg
>>> help(dg.DeepGraph)
>>> help(dg.functions)

Citing DeepGraph
----------------

If deepgraph contributes to a project that leads to a scientific
publication, please acknowledge this fact by citing the project. You
can use this BibTeX entry: [coming soon..]

"""

from __future__ import print_function, division, absolute_import

# Copyright (C) 2016 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD license.

from .deepgraph import DeepGraph
from . import functions

__all__ = ['DeepGraph', 'functions']
__version__ = '0.0.2'
__author__ = "Dominik Traxl <dominik.traxl@posteo.org>"
__copyright__ = "Copyright 2014-2016 Dominik Traxl"
__license__ = "BSD"
__URL__ = "https://github.com/deepgraph/deepgraph/"
__bibtex__ = """coming soon.."""
