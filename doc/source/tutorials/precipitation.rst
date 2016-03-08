
Building a DeepGraph of Precipitation Events
============================================

THIS IS WORK IN PROGRESS!!

This is a short introduction to DeepGraph. In the following, we build a
deep graph of extreme precipitation events, as described in [ref].

First of all, we need to import some packages

.. code:: python

    %matplotlib inline

    import numpy as np
    import pandas as pd

    import deepgraph as dg

The Nodes of a Graph
--------------------

Then, we need a pd.DataFrame object ``v`` representing the node set
:math:`V` of the graph :math:`G = (V, E)`. Each row of ``v`` represents
a **node** :math:`V_i \in V`, each column name of ``v`` corresponds to a
**type of feature** of the nodes, and each cell in ``v`` represents a
**feature**.

.. code:: python

    v = pd.read_pickle('./data/precipitation/v.pickle')

The nodes look like this

.. code:: python

    print(v.head())


.. parsed-literal::

       time    r  area   vol     lat     lon      dtime
    0     0  148   744  3304  14.875 -68.375 1998-01-01
    1     0  161   744  3594  14.875 -68.125 1998-01-01
    2     0  127   744  2835  14.875 -50.125 1998-01-01
    3     0   34   745   760  14.375 -52.125 1998-01-01
    4     0   76   746  1702  14.125 -52.625 1998-01-01


In order to create edges between these nodes, we now create a
``dg.DeepGraph`` instance

.. code:: python

    g = dg.DeepGraph(v, supernode_labels_by={'x': 'lon', 'y': 'lat', 'g_id': ['lon', 'lat']})

where we use the optional argument ``supernode_labels_by`` to create
**supernode labels** (which are categorical **features**) enumerating
latitude/longitude coordinates and geographical locations. The new node
table, which now is an attribute of ``g``, therefore looks like

.. code:: python

    print(g.v.head())


.. parsed-literal::

       time    r  area   vol     lat     lon      dtime    x    y   g_id
    0     0  148   744  3304  14.875 -68.375 1998-01-01   66  219   9701
    1     0  161   744  3594  14.875 -68.125 1998-01-01   67  219   9887
    2     0  127   744  2835  14.875 -50.125 1998-01-01  139  219  25048
    3     0   34   745   760  14.375 -52.125 1998-01-01  131  217  23312
    4     0   76   746  1702  14.125 -52.625 1998-01-01  129  216  22872


The Edges of a Graph
--------------------

The edge set :math:`E` of :math:`G = (V, E)` is also represented by a
``pd.DataFrame`` object, which we denote by ``e``. Its index is a
``pd.core.index.MultiIndex``, whose first level contains the indices of
the **source nodes**, and the second level contains the indices of the
**target nodes**. Each row of ``e`` corresponds to an **edge**
:math:`E_{ij} \in E`, each column name of ``e`` corresponds to a **type
of relation**, and each cell in ``e`` represents a **relation**.

We now use DeepGraph to create edges between the nodes given by ``v``.
For that matter, we need a **connector** function, which takes



