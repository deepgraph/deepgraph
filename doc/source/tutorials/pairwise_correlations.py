
# coding: utf-8

# # Computing Very Large Correlation Matrices in Parallel

# In[1]:


# data i/o
import os

# compute in parallel
from multiprocessing import Pool

# the usual
import numpy as np
import pandas as pd

import deepgraph as dg


# Let's create a set of variables and store it as a 2d-matrix ``X`` (``shape=(n_features, n_samples)``) on disc. To speed up the computation of the correlation coefficients later on, we whiten each variable.

# In[2]:


# create observations
from numpy.random import RandomState
prng = RandomState(0)
n_features = int(5e3)
n_samples = int(1e2)
X = prng.randint(100, size=(n_features, n_samples)).astype(np.float64)

# uncomment the next line to compute ranked variables for Spearman's correlation coefficients
# X = X.argsort(axis=1).argsort(axis=1)

# whiten variables for fast parallel computation later on
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# save in binary format
np.save('samples', X)


# In[3]:


# parameters (change these to control RAM usage)
step_size = 1e5
n_processes = 100

# load samples as memory-map
X = np.load('samples.npy', mmap_mode='r')

# create node table that stores references to the mem-mapped samples
v = pd.DataFrame({'index': range(X.shape[0])})

# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
    features_s = X[index_s]
    features_t = X[index_t]
    corr = np.einsum('ij,ij->i', features_s, features_t) / n_samples
    return corr

# index array for parallelization
pos_array = np.array(np.linspace(0, n_features*(n_features-1)//2, n_processes), dtype=int)

# parallel computation
def create_ei(i):

    from_pos = pos_array[i]
    to_pos = pos_array[i+1]

    # initiate DeepGraph
    g = dg.DeepGraph(v)

    # create edges
    g.create_edges(connectors=corr, step_size=step_size,
                   from_pos=from_pos, to_pos=to_pos)

    # store edge table
    g.e.to_pickle('tmp/correlations/{}.pickle'.format(str(i).zfill(3)))

# computation
if __name__ == '__main__':
    os.makedirs("tmp/correlations", exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = Pool()
    for _ in p.imap_unordered(create_ei, indices):
        pass



# Let's collect the computed correlation values and store them in an hdf file.

# In[4]:


# store correlation values
files = os.listdir('tmp/correlations/')
files.sort()
store = pd.HDFStore('e.h5', mode='w')
for f in files:
    et = pd.read_pickle('tmp/correlations/{}'.format(f))
    store.append('e', et, format='t', data_columns=True, index=False)
store.close()


# Let's have a quick look at the correlations.

# In[5]:


# load correlation table
e = pd.read_hdf('e.h5')
print(e)


# And finally, let's see where most of the computation time is spent.

# In[6]:


g = dg.DeepGraph(v)
p = get_ipython().run_line_magic('prun', '-r g.create_edges(connectors=corr, step_size=step_size)')


# In[7]:


p.print_stats(20)


# As you can see, most of the time is spent by getting the requested features in the corr-function, followed by computing the correlation values themselves.
