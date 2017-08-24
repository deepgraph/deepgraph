
.. _tutorial_pairwise_correlations:

Computing Very Large Correlation Matrices in Parallel
=====================================================

[:download:`ipython notebook <pairwise_correlations.ipynb>`] [:download:`python script <pairwise_correlations.py>`]

In this short tutorial, we'll demonstrate how DeepGraph can be used to efficiently compute very large correlation matrices in parallel, with full control over RAM usage.

Assume you have a set of ``n_samples`` samples, each comprised of ``n_features`` features and you want to compute the `Pearson correlation coefficients <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ between all pairs of samples. If your data is small enough, you may use `scipy.stats.pearsonr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#scipy.stats.pearsonr>`_ or `numpy.corrcoef <https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html>`_, but for large data, neither of these methods is feasible. Scipy's pearsonr  would be very slow, since you'd have to compute pair-wise correlations in a double loop, and numpy's corrcoef would most likely blow your RAM.

Using DeepGraph's :py:meth:`create_edges <.create_edges>` method, you can compute all pair-wise correlations efficiently. In this tutorial, the samples are stored on disc and only the relevant subset of samples for each iteration will be loaded into memory by the computing nodes. Parallelization is achieved by using python's standard library `multiprocessing <https://docs.python.org/3.6/library/multiprocessing.html>`_, but it should be straight-forward to modify the code to accommodate other parallelization libraries. It should also be straight-forward to modify the code in order to compute other correlation/distance/similarity-measures between a set of samples.

First of all, we need to import some packages

.. code:: python

    # data i/o
    import os

    # compute in parallel
    from multiprocessing import Pool

    # the usual
    import numpy as np
    from numpy.random import RandomState
    import pandas as pd

    import deepgraph as dg


Let's create a set variables and store it as a 2d-matrix ``X``
(``shape=(n_samples, n_features)``) on disc. To speed up the computation
of the Pearson correlation coefficients later on, we whiten each
variable.

.. code:: python

    # create observations
    prng = RandomState(0)
    n_samples = int(5e3)
    n_features = int(1e2)

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in range(X.shape[0]):
        X[i] = prng.randint(0, 100, size=n_features)

    # whiten variables for fast parallel computation later on
    X = ((X.T - X.mean(axis=1)) / X.std(axis=1)).T

    # save in binary format
    np.save('samples', X)


Now we can compute the pair-wise correlations using DeepGraph's :py:meth:`create_edges <.create_edges>` method. Note that the node table :py:attr:`v <.DeepGraph.v>` only stores references to the mem-mapped array containing the samples.

.. code:: python

    # parameters (change these to control RAM usage)
    step_size = 1e5
    n_processes = 100

    # load samples as memory-map
    X = np.load('samples.npy', mmap_mode='r')

    # create node table that stores references to the mem-mapped samples
    v = pd.DataFrame({'index': range(X.shape[0])})

    # connector function to compute pairwise pearson correlations
    def corr(index_s, index_t):
        samples_s = X[index_s]
        samples_t = X[index_t]
        corr = np.einsum('ij,ij->i', samples_s, samples_t) / n_features
        return corr

    # index array for parallelization
    pos_array = np.array(np.linspace(0, n_samples*(n_samples-1)//2, n_processes), dtype=int)

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
        indices = np.arange(0, n_processes - 1)
        p = Pool()
        for _ in p.imap_unordered(create_ei, indices):
            pass


Let's collect the computed correlation values and store them in an hdf
file.

.. code:: python

    # store correlation values
    files = os.listdir('tmp/correlations/')
    files.sort()
    store = pd.HDFStore('e.h5', mode='w')
    for f in files:
        et = pd.read_pickle('tmp/correlations/{}'.format(f))
        store.append('e', et, format='t', data_columns=True, index=False)
    store.close()


Let's have a quick look at the correlations.

.. code:: python

    # load correlation table
    e = pd.read_hdf('e.h5')
    print(e)


.. parsed-literal::

                   corr
    s    t
    0    1    -0.006066
         2     0.094063
         3    -0.025529
         4     0.074080
         5     0.035490
         6     0.005221
         7     0.032064
         8     0.000378
         9    -0.049318
         10   -0.084853
         11    0.026407
         12    0.028543
         13   -0.013347
         14   -0.180113
         15    0.151164
         16   -0.094398
         17   -0.124582
         18   -0.000781
         19   -0.044138
         20   -0.193609
         21    0.003877
         22    0.048305
         23    0.006477
         24   -0.021291
         25   -0.070756
         26   -0.014906
         27   -0.197605
         28   -0.103509
         29    0.071503
         30    0.120718
    ...             ...
    4991 4998 -0.012007
         4999 -0.252836
    4992 4993  0.202024
         4994 -0.046088
         4995 -0.028314
         4996 -0.052319
         4997 -0.010797
         4998 -0.025321
         4999 -0.093721
    4993 4994 -0.027568
         4995  0.045602
         4996 -0.102075
         4997  0.035370
         4998 -0.069946
         4999 -0.031208
    4994 4995  0.108063
         4996  0.144441
         4997  0.078353
         4998 -0.024799
         4999 -0.026432
    4995 4996 -0.019991
         4997 -0.178458
         4998 -0.162406
         4999  0.102835
    4996 4997  0.115812
         4998 -0.061167
         4999  0.018606
    4997 4998 -0.151932
         4999 -0.271358
    4998 4999  0.106453

    [12497500 rows x 1 columns]


And finally, let's see where most of the computation time is spent.

.. code:: python

    g = dg.DeepGraph(v)
    p = %prun -r g.create_edges(connectors=corr, step_size=step_size)


.. code:: python

    p.print_stats(20)


.. parsed-literal::

             252629 function calls (247853 primitive calls) in 6.007 seconds

       Ordered by: internal time
       List reduced from 526 to 20 due to restriction <20>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          250    3.196    0.013    3.202    0.013 memmap.py:334(__getitem__)
          125    1.195    0.010    1.195    0.010 {built-in method numpy.core.multiarray.c_einsum}
            2    0.361    0.181    0.361    0.181 {method 'get_labels' of 'pandas._libs.hashtable.Int64HashTable' objects}
          125    0.148    0.001    4.584    0.037 deepgraph.py:4553(map)
          250    0.113    0.000    0.122    0.000 internals.py:4473(_stack_arrays)
            4    0.102    0.025    0.102    0.025 {built-in method numpy.core.multiarray.concatenate}
          129    0.085    0.001    0.085    0.001 {method 'take' of 'numpy.ndarray' objects}
          125    0.084    0.001    4.996    0.040 deepgraph.py:5289(_select_and_return)
            2    0.074    0.037    0.190    0.095 algorithms.py:429(safe_sort)
          125    0.042    0.000    0.042    0.000 {deepgraph._triu_indices._reduce_triu_indices}
            2    0.040    0.020    0.040    0.020 function_base.py:4684(delete)
          125    0.040    0.000    0.040    0.000 {built-in method deepgraph._triu_indices._triu_indices}
          126    0.039    0.000    0.039    0.000 api.py:93(_sanitize_and_check)
            2    0.032    0.016    0.032    0.016 {built-in method numpy.core.multiarray.putmask}
            4    0.029    0.007    0.029    0.007 {built-in method pandas._libs.algos.ensure_int16}
          125    0.020    0.000    4.417    0.035 <ipython-input-3-ddd5575c35f5>:12(corr)
    49804/49196    0.015    0.000    0.043    0.000 {built-in method builtins.isinstance}
            1    0.014    0.014    6.007    6.007 deepgraph.py:178(create_edges)
            1    0.014    0.014    5.965    5.965 deepgraph.py:4783(_matrix_iterator)
            2    0.011    0.006    0.563    0.281 algorithms.py:527(factorize)






.. parsed-literal::

    <pstats.Stats at 0x7fc79c237ba8>



As you can see, most of the time is spent by getting the requested
samples in the corr-function, followed by computing the correlation
values themselves.
