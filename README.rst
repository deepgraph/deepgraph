
  .. image:: https://anaconda.org/deepgraph/deepgraph/badges/build.svg
     :target: https://anaconda.org/deepgraph/deepgraph/builds

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
and provides methods to construct, partition and plot graphs, to interface with
popular network packages and more.

It is based on a new network representation introduced
`here <http://arxiv.org/abs/1604.00971>`_. DeepGraph is also capable of
representing
`multilayer networks <http://deepgraph.readthedocs.io/en/latest/tutorials/terrorists.html>`_.


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

    Copyright (C) 2016 DeepGraph Developers
    Dominik Traxl <dominik.traxl@posteo.org>
