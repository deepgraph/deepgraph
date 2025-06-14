.. _installation:

************
Installation
************


Installation Using `pip` or `conda`
===================================

DeepGraph can be installed via pip from
`PyPI <https://pypi.python.org/pypi/deepgraph>`_::

  $ pip install deepgraph

To install with optional dependencies (i.e. "extras", extending the base functionality of DeepGraph), use::

  $ pip install deepgraph[plot,basemap,tables,scipy,networkx]

or simply use::

  $ pip install deepgraph[all]

to install all optional dependencies. Extras and their added
functionality are described :ref:`below <optional_dependencies>`.

.. note::
   There is one optional dependency that can not (yet) be install via pip: `graph-tool <https://graph-tool.skewed.de/>`_.
   Please refer to the linked project homepage for recommended installation.

Alternatively, if you're using `Conda <http://conda.pydata.org/docs/>`_,
install with::

  $ conda install -c conda-forge deepgraph

It is not possible to specify optional dependencies ("extras") directly when installing a Python package using conda.
To enable the extra features of deepgraph, you have to manually add the corresponding packages to your conda
environment.

For example, if you want to use features that require "scipy", you would need to install scipy separately using::

   $ conda install scipy

To install deepgraph including all optional dependencies, you could also use an `environment.yml` file, e.g.

.. code-block:: yaml

   name: deepgraph-env
   channels:
     - conda-forge
   dependencies:
     - python>=3.9
     - numpy>=1.21.6
     - pandas>=1.2
     - matplotlib>=3.1
     - basemap>=2.0
     - pytables>=3.7
     - scipy>=1.5.4
     - networkx>=2.4
     - graph-tool>=2.27
     - deepgraph

Then create the environment with::

   $ conda env create -f environment.yml

and activate it via::

   $ conda activate deepgraph-env


Installation from Source & Environment Setup
============================================

DeepGraph uses `uv <https://docs.astral.sh/uv/>`_ to manage the package. To get started as a developer,
follow the instructions below

  1. Install `uv`
      If you haven't already, install uv by following the
      `official instructions <https://docs.astral.sh/uv/getting-started/installation/>`_::

        $ curl -Ls https://astral.sh/uv/install.sh | sh

  2. Clone the repository
      Use git to clone the repository and cd into it::

        $ git clone https://github.com/deepgraph/deepgraph.git
        $ cd deepgraph

  3. Set up the development environment
      Use uv to create and manage your virtual environment::

        $ uv sync

      This will create a virtual environment, install the required and the developer dependencies,
      build the DeepGraph package and install it in editable mode.

  4. Installing with optional dependencies ("extras")
      To install extras (i.e. optional dependencies, extending the core functionality of DeepGraph), use
      the following command::

        $ uv sync --extra <EXTRA>

      where <EXTRA> can be one of
        - tables
        - scipy
        - networkx
        - plot
        - basemap

      To install all optional dependencies, simply use::

        $ uv sync --all-extras

  5. Run tests
      To run tests, use the following command::

        $ uv run pytest .

  6. Build the documentation
      To build the documentation, run the following commands::

        $ cd doc
        $ make html


Dependencies
============

+---------------------------------------+---------------------------+
| Package                               | Minimum supported version |
+=======================================+===========================+
| `Python <https://www.python.org/>`_   | 3.9                       |
+---------------------------------------+---------------------------+
| `NumPy <http://www.numpy.org/>`_      | 1.21.6                    |
+---------------------------------------+---------------------------+
| `Pandas <http://pandas.pydata.org/>`_ | 1.2                       |
+---------------------------------------+---------------------------+

It is highly recommended to install Pandas'
`optional dependencies <https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html#optional-dependencies>`_.
In particular, it is recommended to install the
`performance dependencies <https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html#performance-dependencies-recommended>`_.


.. _optional_dependencies:

Optional Dependencies / Extras
==============================

The following are recommended packages that DeepGraph can use to provide
additional functionality.

+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Dependency                                          | Minimum Version | pip extra | Notes                                                                                                                                                                                                                                                                                                                      |
+=====================================================+=================+===========+============================================================================================================================================================================================================================================================================================================================+
| `Matplotlib <http://matplotlib.org/>`_              | 3.1             | plot      | Used by the :ref:`plotting methods <plotting_methods>` of DeepGraph.                                                                                                                                                                                                                                                       |
+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `basemap <https://matplotlib.org/basemap/stable/>`_ | 2.0             | basemap   | Used by :py:meth:`plot_map <.plot_map>` and :py:meth:`plot_map_generator <.plot_map_generator>` to plot networks on map projections.                                                                                                                                                                                       |
+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `PyTables <http://www.pytables.org/>`_              | 3.7             | tables    | Necessary for HDF5-based storage of pandas DataFrames. A :py:class:`DeepGraph <.DeepGraph>` may be instantiated with an HDFStore containing a node table in order to iteratively create edges directly from disc (see :py:meth:`create_edges <.create_edges>` and :py:meth:`create_edges_ft <.create_edges_ft>`).          |
+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `SciPy <http://www.scipy.org/>`_                    | 1.5.4           | scipy     | Used by :py:meth:`return_cs_graph <.return_cs_graph>` to convert from DeepGraph's network representation to sparse adjacency matrices, and by :py:meth:`append_cp <.append_cp>` to append a `connected components <https://en.wikipedia.org/wiki/Component_(graph_theory)>`_ column to the node table.                     |
+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `NetworkX <https://networkx.github.io/>`_           | 2.4             | networkx  | Used by :py:meth:`return_nx_graph <.return_nx_graph>` and :py:meth:`return_nx_multigraph <.return_nx_multigraph>` to convert from DeepGraph's network representation to NetworkX's network representations.                                                                                                                |
+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `graph\_tool <https://graph-tool.skewed.de/>`_      | 2.27            | N/A       | Used by :py:meth:`return_gt_graph <.return_gt_graph>` to convert from DeepGraph's network representation to Graph-Tool's network representation. See `here <https://graph-tool.skewed.de/installation.html>`_ for installation instructions.                                                                               |
+-----------------------------------------------------+-----------------+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
