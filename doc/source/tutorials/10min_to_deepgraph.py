
# coding: utf-8

# # 10 Minutes to DeepGraph

# [:download:`ipython notebook <10min_to_deepgraph.ipynb>`] [:download:`python script <10min_to_deepgraph.py>`]

# This is a short introduction to DeepGraph. In the following, we demonstrate DeepGraph's core functionalities by a toy data-set, "flying balls".
#
# First of all, we need to import some packages

# In[1]:

# for plots
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8, 6

# the usual
import numpy as np
import pandas as pd

import deepgraph as dg


# **Loading Toy Data**

# Then, we need data in the form of a pandas `DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_, representing the nodes of our graph

# In[2]:

v = pd.read_pickle('flying_balls.pickle')
print(v.head())


# The data consists of

# In[3]:

print(len(v))


# space-time measurements of 50 different toy balls in two-dimensional space. Each space-time measurement (i.e. row of ``v``) represents a **node**.
#
# Let's plot the data such that each ball has it's own color

# In[4]:

plt.scatter(v.x, v.y, s=v.time, c=v.ball_id)


# ## Creating Edges

# In order to create edges between these nodes, we now initiate a :py:class:`dg.DeepGraph <.DeepGraph>` instance

# In[5]:

g = dg.DeepGraph(v)
g


# and use it to create edges between the nodes given by :py:attr:`g.v <.DeepGraph.v>`. For that matter, we may define a **connector** function

# In[6]:

def x_dist(x_s, x_t):
    dx = x_t - x_s
    return dx


# and pass it to :py:meth:`g.create_edges <.create_edges>` in order to compute the distance in the x-coordinate of each pair of nodes

# In[7]:

g.create_edges(connectors=x_dist)
g


# In[8]:

print(g.e.head())


# Let's say we're only interested in creating edges between nodes with a x-distance smaller than 1000. Then we may additionally define a **selector**

# In[9]:

def x_dist_selector(dx, sources, targets):
    dxa = np.abs(dx)
    sources = sources[dxa <= 1000]
    targets = targets[dxa <= 1000]
    return sources, targets


# and pass both the **connector** and **selector** to :py:meth:`g.create_edges <.create_edges>`

# In[10]:

g.create_edges(connectors=x_dist, selectors=x_dist_selector)
g


# In[11]:

print(g.e.head())


# There is, however, a much more efficient way of creating edges that involve a simple distance threshold such as the one above

# ## Creating Edges on a FastTrack

# In order to efficiently create edges including a selection of edges via a simple distance threshold as above, one should use the :py:meth:`create_edges_ft <.create_edges_ft>` method. It relies on a sorted DataFrame, so we need to sort :py:attr:`g.v <.DeepGraph.v>` first

# In[12]:

g.v.sort_values('x', inplace=True)


# In[13]:

g.create_edges_ft(ft_feature=('x', 1000))
g


# Let's compare the efficiency

# In[14]:

get_ipython().magic('timeit -n3 -r3 g.create_edges(connectors=x_dist, selectors=x_dist_selector)')


# In[15]:

get_ipython().magic("timeit -n3 -r3 g.create_edges_ft(ft_feature=('x', 1000))")


# The :py:meth:`create_edges_ft <.create_edges_ft>` method also accepts **connectors** and **selectors** as input. Let's connect only those measurements that are close in space and time

# In[16]:

def y_dist(y_s, y_t):
    dy = y_t - y_s
    return dy

def time_dist(time_t, time_s):
    dt = time_t - time_s
    return dt

def y_dist_selector(dy, sources, targets):
    dya = np.abs(dy)
    sources = sources[dya <= 100]
    targets = targets[dya <= 100]
    return sources, targets

def time_dist_selector(dt, sources, targets):
    dta = np.abs(dt)
    sources = sources[dta <= 1]
    targets = targets[dta <= 1]
    return sources, targets


# In[17]:

g.create_edges_ft(ft_feature=('x', 100),
                  connectors=[y_dist, time_dist],
                  selectors=[y_dist_selector, time_dist_selector])
g


# In[18]:

print(g.e.head())


# We can now plot the flying balls and the edges we just created with the :py:meth:`plot_2d <.plot_2d>` method

# In[19]:

obj = g.plot_2d('x', 'y', edges=True, kwds_scatter={'c': g.v.ball_id, 's': g.v.time})
obj['ax'].set_xlim(1000,3000)


# ## Graph Partitioning

# The :py:class:`DeepGraph <.DeepGraph>` class also offers methods to partition :py:meth:`nodes <.partition_nodes>`, :py:meth:`edges <.partition_edges>` and an entire :py:meth:`graph <.partition_graph>`. See the docstrings and the :ref:`other tutorial <tutorial_pcp>` for details and examples.

# ## Graph Interfaces

# Furthermore, you may inspect the docstrings of :py:meth:`return_cs_graph <.return_cs_graph>`, :py:meth:`return_nx_graph <.return_nx_graph>` and :py:meth:`return_gt_graph <.return_gt_graph>` to see how to convert from DeepGraph's DataFrame representation of a network to sparse adjacency matrices, NetworkX's network representation and graph_tool's network representation.

# ## Plotting Methods

# DeepGraph also offers a number of useful Plotting methods. See :ref:`plotting methods <plotting_methods>` for details and inspect the corresponding docstrings for examples.
