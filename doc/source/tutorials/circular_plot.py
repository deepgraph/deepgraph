import subprocess
import zipfile

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import graph_tool as gt
import graph_tool.draw as gtd

import deepgraph as dg

# ## The Noordin Top Terrorist Data

# **Preprocessing the Nodes**

# zip file containing node attributes
get_nodes_zip = ("wget -O terrorist_nodes.zip "
                 "https://sites.google.com/site/sfeverton18/"
                 "research/appendix-1/Noordin%20Subset%20%28ORA%29.zip?"
                 "attredirects=0&d=1")
subprocess.call(get_nodes_zip.split())

# unzip
zf = zipfile.ZipFile('terrorist_nodes.zip')
zf.extract('Attributes.csv')
zf.close()

# create node table
v = pd.read_csv('Attributes.csv')
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
v = v.ix[:,cols]

# get rid of the data columns for demonstrational purposes,
# will be inserted again later
# v, vinfo = v.iloc[:, :2], v.iloc[:, 2:]

# **Preprocessing the Edges**

# paj file containing edges for different layers
get_paj = ("wget -O terrorists.paj "
           "https://sites.google.com/site/sfeverton18/"
           "research/appendix-1/Noordin%20Subset%20%28Pajek%29.paj?"
           "attredirects=0&d=1")
subprocess.call(get_paj.split())

# get data blocks from paj file
with open('terrorists.paj') as txtfile:
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
            elif len(line) > 1:
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

# with corresponding data parts (indices found manually via comments)
inds = [11, 10, 5, 6, 7, 8, 0, 1, 2, 3]
eparts = [eparts[ind] for ind in inds]

# convert to DataFrames
layer_frames = []
for name, epart in zip(layer_names, eparts):
    frame = pd.DataFrame(epart, dtype=np.int16)
    # get rid of self-loops, bidirectional edges
    frame = frame[frame[0] < frame[1]]
    # and rename columns
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

# alternative representation of e (uses less ram)
e['type'] = 0
e['weight'] = 0
for layer in layer_names:
    where = e[layer].notnull()
    e.loc[where, 'type'] = layer
    e.loc[where, 'weight'] = e.loc[where, layer]
e = e[['type', 'weight']]

# ## DeepGraph's Supra-Graph Representation of a MLN, $G = (V, E)$

# Above, we have preprocessed the downloaded data into a node table v
# and an edge table e, that correspond to the supra-graph representation
# of a multilayer network. This is the preferred representation of a MLN
# by a deep graph, since - as we will demonstrate below - all other
# representations are entailed in the supra-graph's partition lattice.

g = dg.DeepGraph(v, e)

# ======================================================================
# plot circular
# scale factor (resolution of the plot)
scale_factor = 1

# create graph_tool graph object
gtg = g.return_gt_graph(features=True, relations=True)
gtg.set_directed(False)

# build tree
t = gt.Graph()

# add nodes
nodes_by_layer = {}
for layer in layer_names:
    nodes_by_layer[layer] = t.add_vertex(n=79)

layer_nodes = {}
for layer in layer_names:
    layer_nodes[layer] = t.add_vertex()

root = t.add_vertex()

# add edges
for layer in layer_names:
    t.add_edge(root, layer_nodes[layer])
    for node in nodes_by_layer[layer]:
        t.add_edge(layer_nodes[layer], node)

# radial tree layout
tpos = gtd.radial_tree_layout(t, root)
cts = gtd.get_hierarchy_control_points(gtg, t, tpos)

# add layer nodes to gtg and adjust plot sizes
vsize = gtg.new_vertex_property('double', [5*scale_factor]*790)
names = gtg.vp['Name']
for layer in layer_names:
    node = gtg.add_vertex()
    names[node] = layer
    vsize[node] = 0

pos = gtg.own_property(tpos)

# labels
text_rot = gtg.new_vertex_property('double')
for node in gtg.vertices():
    if pos[node][0] > 0:
        text_rot[node] = np.arctan(pos[node][1] / pos[node][0])
    else:
        text_rot[node] = np.pi + np.arctan(pos[node][1] / pos[node][0])

# font size of node texts
fs = np.ones(800) * 3.125 * scale_factor
fs[790:] = 15 * scale_factor
fs = gtg.new_vertex_property('double', fs)

# plot
gtd.graph_draw(gtg, pos=pos,
               edge_control_points=cts,
               vertex_fill_color=gtg.vp['V_N'],
               vcmap=cm.viridis_r,
               vertex_size=vsize,
               vertex_font_size=fs,
               vertex_text=names,
               vertex_text_color='k',
               vertex_text_rotation=text_rot,
               vertex_text_position=1,
               # edge_color=gtg.ep['weight'],
               # ecmap=cm.viridis,
               vertex_anchor=0,
               edge_pen_width=.375*scale_factor,
               bg_color=[1,1,1,0],
               output_size=[1024*scale_factor,1024*scale_factor],
               output='radial_g_{}.png'.format(scale_factor),
               )
