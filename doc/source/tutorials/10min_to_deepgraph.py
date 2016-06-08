
# coding: utf-8

# # 10 Minutes to DeepGraph

# This is a short introduction to DeepGraph. In the following, we demonstrate DeepGraph's core functionalities by a toy data-set, "flying balls".
#
# First of all, we need to import some packages

# In[19]:

# for plots
import matplotlib.pyplot as plt

# the usual
import numpy as np
import pandas as pd

import deepgraph as dg

# notebook display
# get_ipython().magic('matplotlib inline')
# plt.rcParams['figure.figsize'] = 8, 6
# pd.options.display.max_rows = 10
# pd.set_option('expand_frame_repr', False)


# **Loading Toy Data**

# In[2]:

v = pd.read_csv('flying_balls.csv', index_col=0)
print(v)


# The data consists of 1168 space-time measurements of 50 different toy balls in two-dimensional space. Each space-time measurement (i.e. row of ``v``) represents a **node**.
#
# Let's plot the data such that each ball has it's own color

# In[3]:

plt.scatter(v.x, v.y, s=v.time, c=v.ball_id)


# ## Creating Edges

# In[4]:

g = dg.DeepGraph(v)
g


# In[5]:

def x_dist(x_s, x_t):
    dx = x_t - x_s
    return dx


# In[6]:

g.create_edges(connectors=x_dist)
g


# In[7]:

print(g.e)


# Let's say we're only interested in creating edges between nodes with a x-distance smaller than 1000. Then we may additionally define a **selector**

# In[8]:

def x_dist_selector(dx, sources, targets):
    dxa = np.abs(dx)
    sources = sources[dxa <= 1000]
    targets = targets[dxa <= 1000]
    return sources, targets


# In[9]:

g.create_edges(connectors=x_dist, selectors=x_dist_selector)
g


# In[10]:

print(g.e)


# There is, however, a much more efficient way of creating edges that involve a simple distance threshold such as the one above

# ## Creating Edges on a FastTrack

# In[11]:

g.v.sort_values('x', inplace=True)


# In[12]:

g.create_edges_ft(ft_feature=('x', 1000))
g


# Let's compare the efficiency

# In[13]:

# get_ipython().magic('timeit -n3 -r3 g.create_edges(connectors=x_dist, selectors=x_dist_selector)')


# In[14]:

# get_ipython().magic("timeit -n3 -r3 g.create_edges_ft(ft_feature=('x', 1000))")


# In[15]:

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


# In[16]:

g.create_edges_ft(ft_feature=('x', 100),
                  connectors=[y_dist, time_dist],
                  selectors=[y_dist_selector, time_dist_selector])
g


# In[17]:

print(g.e)


# In[18]:

obj = g.plot_2d('x', 'y', edges=True,
                kwds_scatter={'c': g.v.ball_id, 's': g.v.time})
obj['ax'].set_xlim(1000,3000)


# ## Graph Partitioning

# ## Graph Interfaces

# ## Plotting Methods
