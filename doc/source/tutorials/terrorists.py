
# coding: utf-8

# # From Multilayer Networks to Deep Graphs

# ## The Noordin Top Terrorist Data

# ### Preprocessing

# In[1]:

# data i/o
import os
import subprocess
import zipfile

# for plots
import matplotlib.pyplot as plt

# the usual
import numpy as np
import pandas as pd

import deepgraph as dg

# notebook display
# get_ipython().magic('matplotlib inline')
# pd.options.display.max_rows = 10
# pd.set_option('expand_frame_repr', False)


# ### Preprocessing the Nodes

# In[2]:

# zip file containing node attributes
os.makedirs("tmp", exists_ok=True)
get_nodes_zip = ("wget -O tmp/terrorist_nodes.zip "
                 "https://sites.google.com/site/sfeverton18/"
                 "research/appendix-1/Noordin%20Subset%20%28ORA%29.zip?"
                 "attredirects=0&d=1")
subprocess.call(get_nodes_zip.split())

# unzip
zf = zipfile.ZipFile('tmp/terrorist_nodes.zip')
zf.extract('Attributes.csv', path='tmp/')
zf.close()

# create node table
v = pd.read_csv('tmp/Attributes.csv')
v.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)

# create a copy of all nodes for each layer (i.e., create "node-layers")
# there are 10 layers and 79 nodes on each layer
v = pd.concat(10*[v])

# add "aspect" as column to v
layer_names = ['Business', 'Communication', 'O Logistics', 'O Meetings',
               'O Operations', 'O Training', 'T Classmates', 'T Friendship',
               'T Kinship', 'T Soulmates']
layers = [[name]*79 for name in layer_names]
layers = [item for sublist in layers for item in sublist]
v['layer'] = layers

# set unique node index
v.reset_index(inplace=True)
v.rename(columns={'index': 'V_N'}, inplace=True)

# swap columns
cols = list(v)
cols[1], cols[10] = cols[10], cols[1]
v = v[cols]

# get rid of the attribute columns for demonstrational purposes,
# will be inserted again later
v, vinfo = v.iloc[:, :2], v.iloc[:, 2:]


# ### Preprocessing the Edges

# In[3]:

# paj file containing edges for different layers
get_paj = ("wget -O tmp/terrorists.paj "
           "https://sites.google.com/site/sfeverton18/"
           "research/appendix-1/Noordin%20Subset%20%28Pajek%29.paj?"
           "attredirects=0&d=1")
subprocess.call(get_paj.split())

# get data blocks from paj file
with open('tmp/terrorists.paj') as txtfile:
    comments = []
    data = []
    part = []
    for line in txtfile:
        if line.startswith('*'):
            # comment lines
            comment = line
            comments.append(comment)
            if part:
                data.append(part)
                part = []
        else:
            # vertices
            if comment.startswith('*Vertices') and len(line.split()) > 1:
                sublist = line.split('"')
                sublist = sublist[:2] + sublist[-1].split()
                part.append(sublist)
            # edges or partitions
            elif not line.isspace():
                part.append(line.split())
    # append last block
    data.append(part)

# extract edge tables from data blocks
ecomments = []
eparts = []
for i, c in enumerate(comments):
    if c.startswith('*Network'):
        del data[0]
    elif c.startswith('*Partition'):
        del data[0]
    elif c.startswith('*Vector'):
        del data[0]
    elif c.startswith('*Arcs') or c.startswith('*Edges'):
        ecomments.append(c)
        eparts.append(data.pop(0))

# layer data parts (indices found manually via comments)
inds = [11, 10, 5, 6, 7, 8, 0, 1, 2, 3]
eparts = [eparts[ind] for ind in inds]

# convert to DataFrames
layer_frames = []
for name, epart in zip(layer_names, eparts):
    frame = pd.DataFrame(epart, dtype=np.int16)
    # get rid of self-loops, bidirectional edges
    frame = frame[frame[0] < frame[1]]
    # rename columns
    frame.rename(columns={0: 's', 1: 't', 2: name}, inplace=True)
    frame['s'] -= 1
    frame['t'] -= 1
    layer_frames.append(frame)

# set indices
for i, e in enumerate(layer_frames):
    e['s'] += i*79
    e['t'] += i*79
    e.set_index(['s', 't'], inplace=True)

# concat the layers
e = pd.concat(layer_frames)

# edge table as described in the paper
e_paper = e.copy()


# In[4]:

# alternative representation of e
e['type'] = 0
e['weight'] = 0
for layer in layer_names:
    where = e[layer].notnull()
    e.loc[where, 'type'] = layer
    e.loc[where, 'weight'] = e.loc[where, layer]
e = e[['type', 'weight']]


# ## DeepGraph's Supra-Graph Representation of a MLN, $G = (V, E)$

# Above, we have processed the downloaded data into a node table ``v`` and an edge table ``e``, that correspond to the supra-graph representation of a multilayer network. This is the preferred representation of a MLN by a deep graph, since all other representations are entailed in the supra-graph's partition lattice, as we will demonstrate below.

# In[5]:

g = dg.DeepGraph(v, e)
print(g)


# Let's have a look at the node table first

# In[6]:

print(g.v)


# As you can see, there are 790 nodes in total. Each of the 10 layers,

# In[7]:

print(g.v.layer.unique())


# is comprised of 79 nodes. Every node has a feature of type ``V_N``, indicating the individual the node belongs to, and a feature of type ``layer``, corresponding to the layer the node belongs to. Each of the 790 nodes corresponds to a node-layer of the MLN representation of this data.
#
# The edge table,

# In[8]:

print(g.e)


# In[9]:

g.e['type'].unique()


# which - in the case of this data set - correspond to the layers of the nodes. This is due to the fact that there are no inter-layer connections in the Noordin Top Terrorist Network (such as, e.g., an edge from layer ``Business`` to layer ``Communication`` would be). The edges here are all (undirected) intra-layer edges (e.g., Business $\rightarrow$ Business, Operations $\rightarrow$ Operations).

# To see how the edges are distributed among the different types, you can simply type

# In[10]:

g.e['type'].value_counts()


# Let's have a look at how many "actors" (nodes with at least one connection) there are within each layer

# In[11]:

# append degree
gtg = g.return_gt_graph()
g.v['deg'] = gtg.degree_property_map('total').a

# how many "actors" are there per layer?
g.v[g.v.deg != 0].groupby('layer').size()


# In[12]:

# create graph_tool graph for layout
import graph_tool.draw as gtd
gtg = g.return_gt_graph()
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg, gamma=.5)
pos = pos.get_2d_array([0, 1])
g.v['x'] = pos[0]
g.v['y'] = pos[1]

# configure nodes
kwds_scatter = {'s': 1,
                'c': 'k'}

# configure edges
kwds_quiver = {'headwidth': 1,
               'alpha': .3,
               'cmap': 'prism'}
# color by type
C = g.e.groupby('type').grouper.group_info[0]

# plot
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
g.plot_2d('x', 'y', edges=True, C=C,
          kwds_scatter=kwds_scatter,
          kwds_quiver=kwds_quiver, ax=ax[0])

# turn axis off, set x/y-lim
ax[0].axis('off')
ax[0].set_xlim((g.v.x.min() - 1, g.v.x.max() + 1))
ax[0].set_ylim((g.v.y.min() - 1, g.v.y.max() + 1))

# plot adjacency matrix
adj = g.return_cs_graph().todense()
adj = adj + adj.T
inds = np.where(adj != 0)
ax[1].scatter(inds[0], inds[1], c='k', marker='.')
ax[1].grid()
ax[1].set_xlim(-1, 791)
ax[1].set_ylim(-1,791)


# ## Redistributing Information on the Partition Lattice of the MLN

# Based on the types of features ``V_N`` and ``layer``, we can now redistribute the information contained in the supra-graph ``g``. This redistribution allows for several representations of the graph, which we will demonstrate in the following.

# ### The SuperGraph $G^L = (V^L, E^L)$

# Partitioning by the type of feature ``layer`` leads to the supergraph $G^L = (V^L,E^L)$, where every supernode $V^{L}_{i^L} \in V^{L}$ corresponds to a distinct layer, encompassing all its respective nodes. Superedges $E^{L}_{i^L, j^L} \in E^{L}$ with either $i^L = j^L$ or $i^L \neq j^L$ correspond to collections of intra- and
# inter-layer edges of the MLN, respectively.

# In[13]:

# partition the graph
lv, le = g.partition_graph('layer',
                           relation_funcs={'weight': ['sum', 'mean', 'std']})
lg = dg.DeepGraph(lv, le)
print(lg)


# In[14]:

print(lg.v)


# In[15]:

print(lg.e)


# Let's plot the graph ``g`` grouped by its layers.

# In[16]:

# append layer_id to group nodes by layers
g.v['layer_id'] = g.v.groupby('layer').grouper.group_info[0].astype(np.int32)

# create graph_tool graph object
gtg = g.return_gt_graph(features=['layer_id'])
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg, groups=gtg.vp['layer_id'], mu=.15)
pos = pos.get_2d_array([0, 1])
g.v['x'] = pos[0]
g.v['y'] = pos[1]

# configure nodes
kwds_scatter = {'s': 10,
                'c': 'k'}

# configure edges
kwds_quiver = {'headwidth': 1,
               'alpha': .4,
               'cmap': 'viridis'}
# color by weight
C = g.e.weight.values

# plot
fig, ax = plt.subplots(figsize=(12, 12))
obj = g.plot_2d('x', 'y', edges=True, C=C,
          kwds_scatter=kwds_scatter,
          kwds_quiver=kwds_quiver, ax=ax)

# turn axis off, set x/y-lim and name layers
ax.axis('off')
margin = 10
ax.set_xlim((g.v.x.min() - margin, g.v.x.max() + margin))
ax.set_ylim((g.v.y.min() - margin, g.v.y.max() + margin))
for layer in layer_names:
    plt.text(g.v[g.v['layer'] == layer].x.mean() - margin * 3,
             g.v[g.v['layer'] == layer].y.max() + margin,
             layer, fontsize=15)


# We can also plot the supergraph $G^L = (V^L, E^L)$

# In[17]:

# create graph_tool graph of lg
gtg = lg.return_gt_graph(relations=True, node_indices=True, edge_indices=True)

# create plot
gtd.graph_draw(gtg,
               vertex_text=gtg.vp['i'], vertex_text_position=-2,
               vertex_fill_color='w',
               vertex_text_color='k',
               edge_text=gtg.ep['n_edges'],
               inline=True, fit_view=.8,
               output_size=(400,400))


# ### The SuperGraph $G^N = (V^N, E^N)$

# Partitioning by the type of feature ``V_N`` leads to the supergraph $G^{N} = (V^{N}, E^{N})$, where each supernode $V^{N}_{i^N} \in V^{N}$ corresponds to a node of the MLN. Superedges $E^{N}_{i^N j^N} \in E^{N}$ with $i^N = j^N$ correspond to the coupling edges of a MLN.

# In[18]:

# partition by MLN's node indices
nv, ne, gv, ge = g.partition_graph('V_N', return_gve=True)

# for each superedge, get types of edges and their weights
def type_weights(group):
    index = group['type'].values
    data = group['weight'].values
    return pd.Series(data=data, index=index)
ne_weights = ge.apply(type_weights).unstack()
ne = pd.concat((ne, ne_weights), axis=1)

# create graph
ng = dg.DeepGraph(nv, ne)
ng


# In[19]:

print(ng.v)


# In[20]:

print(ng.e)


# Let's plot the graph ``g`` grouped by ``V_N``.

# In[21]:

# create graph_tool graph object
g.v['V_N'] = g.v['V_N'].astype(np.int32)  # sfpd only takes int32
g_tmp = dg.DeepGraph(v)
gtg = g_tmp.return_gt_graph(features='V_N')
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg, groups=gtg.vp['V_N'], mu=.3, gamma=.01)
pos = pos.get_2d_array([0, 1])
g.v['x'] = pos[0]
g.v['y'] = pos[1]

# configure nodes
kwds_scatter = {'c': 'k'}

# configure edges
kwds_quiver = {'headwidth': 1,
               'alpha': .2,
               'cmap': 'viridis_r'}
# color by type
C = g.e.groupby('type').grouper.group_info[0]

# plot
fig, ax = plt.subplots(figsize=(15,15))
g.plot_2d('x', 'y', edges=True,
          kwds_scatter=kwds_scatter, C=C,
          kwds_quiver=kwds_quiver, ax=ax)

# turn axis off, set x/y-lim and name nodes
name_dic = {i: name for i, name in enumerate(vinfo.iloc[:79].Name)}
ax.axis('off')
ax.set_xlim((g.v.x.min() - 1, g.v.x.max() + 1))
ax.set_ylim((g.v.y.min() - 1, g.v.y.max() + 1))
for node in g.v['V_N'].unique():
    plt.text(g.v[g.v['V_N'] == node].x.mean() - 1,
             g.v[g.v['V_N'] == node].y.max() + 1,
             name_dic[node], fontsize=12)


# Let's also plot the supergraph $G^N = (V^N, E^N)$, where the color of the superedges corresponds to the number of edges within the respective superedge.

# In[22]:

# get rid of isolated node for nicer layout
ng.v.drop(57, inplace=True, errors='ignore')

# create graph_tool graph object
gtg = ng.return_gt_graph(features=True, relations='n_edges')
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg)
pos = pos.get_2d_array([0, 1])
ng.v['x'] = pos[0]
ng.v['y'] = pos[1]

# configure nodes
kwds_scatter = {'s': 100,
                'c': 'k'}

# configure edges
# split edges with only one type of connection
C_split_0 = ng.e['n_edges'].values.copy()
C_split_0[C_split_0 == 1] = 0

# edges with one type of connection
kwds_quiver_0 = {'alpha': .3,
                 'width': .001}

# edges with more than one type
kwds_quiver = {'headwidth': 1,
               'width': .003,
               'alpha': .7,
               'cmap': 'Blues',
               'clim': (1, ng.e.n_edges.max())}

# create plot
fig, ax = plt.subplots(figsize=(15,15))
ng.plot_2d('x', 'y', edges=True, C_split_0=C_split_0,
           kwds_scatter=kwds_scatter, kwds_quiver_0=kwds_quiver_0,
           kwds_quiver=kwds_quiver, ax=ax)

# turn axis off, set x/y-lim and name nodes
ax.axis('off')
ax.set_xlim(ng.v.x.min() - 1, ng.v.x.max() + 1)
ax.set_ylim(ng.v.y.min() - 1, ng.v.y.max() + 1)
for i in ng.v.index:
    plt.text(ng.v.at[i, 'x'], ng.v.at[i, 'y'] + .3, i, fontsize=12)


# ### The Tensor-Like Representation  $G^{NL} = (V^{NL}, E^{NL})$

# In[23]:

# partition the graph
relation_funcs = {'type': 'sum', 'weight': 'sum'}  # just to transfer relations
nlv, nle = g.partition_graph(['V_N', 'layer'], relation_funcs=relation_funcs)
nlg = dg.DeepGraph(nlv, nle)
nlg


# In[24]:

print(nlg.v)


# In[25]:

print(nlg.e)


# In[26]:

print(nlg.e.loc[2, 'Communication', :, 'Communication'])


# ## The "Hidden Layers" of a MLN

# In[27]:

print(vinfo)


# As you can see, there are 9 different attributes associated with each individual, such as their military training, nationality, education level, etc. Let's append this information to the node table, and plot the nodes grouped by their education level.

# In[28]:

# append node information to g
v = pd.concat((v, vinfo), axis=1)
g = dg.DeepGraph(v, e)


# In[29]:

# create graph_tool graph object
g.v['Education Level'] = g.v['Education Level'].astype(np.int32)
g_tmp = dg.DeepGraph(g.v)
gtg = g_tmp.return_gt_graph(features=['Education Level'])
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg, groups=gtg.vp['Education Level'], mu=.3, gamma=.1)
pos = pos.get_2d_array([0, 1])
g.v['x'] = pos[0]
g.v['y'] = pos[1]

# configure nodes
kwds_scatter = {'s': 10,
                'c': 'k'}

# configure edges
kwds_quiver = {'width': 0.002,
               'headwidth': 1,
               'alpha': .2,
               'cmap': 'prism'}
# color by type
C = g.e.groupby('type').grouper.group_info[0]

# plot
fig, ax = plt.subplots(figsize=(13,12))
obj = g.plot_2d('x', 'y', edges=True,
          kwds_scatter=kwds_scatter, C=C,
          kwds_quiver=kwds_quiver, ax=ax)

# turn axis off, set x/y-lim and name layers
ax.axis('off')
ax.set_xlim((g.v.x.min() - 1, g.v.x.max() + 1))
ax.set_ylim((g.v.y.min() - 1, g.v.y.max() + 1))
for el in g.v['Education Level'].unique():
    plt.text(g.v[g.v['Education Level'] == el].x.mean() - 1,
             g.v[g.v['Education Level'] == el].y.max() + 1,
             'EL {}'.format(el), fontsize=20)


# Let's also append the information to the supergraph $G^N$, and plot this supergraph grouped by education level.

# In[30]:

# append info to ng.v
ng.v = pd.concat((ng.v, vinfo[:79]), axis=1)


# In[31]:

# create graph_tool graph object
ng.v['Education Level'] = ng.v['Education Level'].astype(np.int32)
g_tmp = dg.DeepGraph(ng.v)
gtg = g_tmp.return_gt_graph(features=['Education Level'])
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg, groups=gtg.vp['Education Level'], mu=.3, gamma=.01)
pos = pos.get_2d_array([0, 1])
ng.v['x'] = pos[0]
ng.v['y'] = pos[1]

# configure nodes
kwds_scatter = {'s': 50,
                'c': 'k'}

# configure edges
# split edges with only one type of connection
C_split_0 = ng.e['n_edges'].values.copy()
C_split_0[C_split_0 == 1] = 0

# edges with one type of connection
kwds_quiver_0 = {'alpha': .3,
                 'width': .001}

# edges with more than one type
kwds_quiver = {'headwidth': 1,
               'width': .002,
               'alpha': .7,
               'cmap': 'Blues',
               'clim': (1, ng.e.n_edges.max())}

# create plot
fig, ax = plt.subplots(figsize=(15,15))
obj = ng.plot_2d('x', 'y', edges=True, C_split_0=C_split_0,
                 kwds_scatter=kwds_scatter, kwds_quiver_0=kwds_quiver_0,
                 kwds_quiver=kwds_quiver, ax=ax)

# turn axis off, set x/y-lim and name nodes
ax.axis('off')
ax.set_xlim(ng.v.x.min() - 1, ng.v.x.max() + 1)
ax.set_ylim(ng.v.y.min() - 1, ng.v.y.max() + 1)
for i in ng.v.index:
    plt.text(ng.v.at[i, 'x'],
             ng.v.at[i, 'y'] + .2,
             i, fontsize=8)

for el in ng.v['Education Level'].unique():
    plt.text(ng.v[ng.v['Education Level'] == el].x.mean() - .5,
             ng.v[ng.v['Education Level'] == el].y.max() + 1,
             'EL {}'.format(el), fontsize=20)


# We can now further partition the supergraph $G^N$ into groups with the same education level.

# In[32]:

# partition ng by "Education Level"
relation_funcs = {l: lambda x: x.notnull().sum() for l in layer_names}
relation_funcs['n_edges'] = 'sum'
ELnv, ELne = ng.partition_graph('Education Level',
                                relation_funcs=relation_funcs,
                                n_edges=False)

# compute "undirected" weights
s = ELne.index.get_level_values(0)
t = ELne.index.get_level_values(1)
df1 = ELne[s <= t]
df2 = ELne[s > t].swaplevel(0,1)
df2.index.names = df2.index.names[::-1]
ELne = df1.add(df2, fill_value=0)

# set dtypes
for col in ELne.columns:
    ELne[col] = ELne[col].astype(int)

# find the type of connection most dominant between supernodes
ELne['dominant_type'] = ELne[layer_names].idxmax(axis=1)

# change column order
ELne = ELne[['n_edges'] + ['dominant_type'] + layer_names]

# create graph
ELng = dg.DeepGraph(ELnv, ELne)
ELng


# In[33]:

print(ELng.v)


# In[34]:

print(ELng.e)


# Let's plot the supergraph of education levels, where the node size relates to the number of individuals, edge colors correspond to the number of edges, and edge labels correspond to the most dominant type of connection between nodes.

# In[35]:

# create graph_tool graph object
gtg = ELng.return_gt_graph(features=True, relations=True, node_indices=True)
gtg.set_directed(False)

# get sfdp layout postitions
pos = gtd.sfdp_layout(gtg,
                      vweight=gtg.vp['n_nodes'],
                      eweight=gtg.ep['n_edges'])
pos = pos.get_2d_array([0, 1])

# create plot
gtg.vp['n_nodes'].a *= 3
gtd.graph_draw(gtg,
               vertex_text=gtg.vp['i'],
               vertex_text_color='k', vertex_size=gtg.vp['n_nodes'],
               edge_text=gtg.ep['dominant_type'],
               edge_color=gtg.ep['n_edges'],
               inline=True, output_size=(900,900), fit_view=True)


# ## Partitioning Edges Based on Node Properties

# Here, we demonstrate very briefly how to use the additional information of the nodes to perform queries on the edges.

# In[36]:

# create "undirected" edge table (swap-copy all edges)
g.e = pd.concat((e, e.swaplevel(0,1)))
g.e.sort_index(inplace=True)


# In[37]:

print(g.partition_edges(source_features=['Nationality']))


# In[38]:

print(g.partition_edges(source_features=['Nationality'],
                        target_features=['Military Training']))


# In[39]:

print(g.partition_edges(source_features=['Nationality'],
                        target_features=['Military Training'],
                        relations='type'))


# ## Alternative Representation of the MLN Edges

# The edges of the supra-graph representation as presented in the paper look like this

# In[40]:

print(e_paper)


# As you can see, the edge table is also comprised of 1014 edges between the nodes in ``v``. However, every type of connection get's its own column, where a "nan" value means that an edge does not have a relation of the corresponding type.
