
.. _tutorial_pairwise_correlations:

Computing Very Large Correlation Matrices in Parallel
=====================================================

[:download:`ipython notebook <pairwise_correlations.ipynb>`] [:download:`python script <pairwise_correlations.py>`]

In this short tutorial, we'll demonstrate how DeepGraph can be used to efficiently compute very large correlation matrices in parallel, with full control over RAM usage.

Assume you have a set of ``n_samples`` samples, each comprised of ``n_features`` features and you want to compute the `Pearson correlation coefficients <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ between all pairs of features (for the `Spearman's rank correlation coefficients <https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient>`_, see the *Note*-box below). If your data is small enough, you may use `scipy.stats.pearsonr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html#scipy.stats.pearsonr>`_ or `numpy.corrcoef <https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html>`_, but for large data, neither of these methods is feasible. Scipy's pearsonr  would be very slow, since you'd have to compute pair-wise correlations in a double loop, and numpy's corrcoef would most likely blow your RAM.

Using DeepGraph's :py:meth:`create_edges <.create_edges>` method, you can compute all pair-wise correlations efficiently. In this tutorial, the data is stored on disc and only the relevant subset of features for each iteration will be loaded into memory by the computing nodes. Parallelization is achieved by using python's standard library `multiprocessing <https://docs.python.org/3.6/library/multiprocessing.html>`_, but it should be straight-forward to modify the code to accommodate other parallelization libraries. It should also be straight-forward to modify the code in order to compute other correlation/distance/similarity-measures between a set of features.

First of all, we need to import some packages

.. code:: python

    # data i/o
    import os

    # compute in parallel
    from multiprocessing import Pool

    # the usual
    import numpy as np
    import pandas as pd

    import deepgraph as dg


Let's create a set of variables and store it as a 2d-matrix ``X``
(``shape=(n_features, n_samples)``) on disc. To speed up the computation
of the correlation coefficients later on, we whiten each variable.

.. code:: python

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


.. note::
    On the computation of the `Spearman's rank correlation coefficients <https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient>`_: Since the Spearman correlation coefficient is defined as the Pearson correlation coefficient between the ranked variables, it suffices to uncomment the indicated line in the above code-block in order to compute the Spearman's rank correlation coefficients in the following.

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

             244867 function calls (239629 primitive calls) in 14.193 seconds

       Ordered by: internal time
       List reduced from 541 to 20 due to restriction <20>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          250    9.355    0.037    9.361    0.037 memmap.py:334(__getitem__)
          125    1.584    0.013    1.584    0.013 {built-in method numpy.core.multiarray.c_einsum}
          125    1.012    0.008   12.013    0.096 deepgraph.py:4558(map)
            2    0.581    0.290    0.581    0.290 {method 'get_labels' of 'pandas._libs.hashtable.Int64HashTable' objects}
            1    0.301    0.301    0.414    0.414 multi.py:795(_engine)
            5    0.157    0.031    0.157    0.031 {built-in method numpy.core.multiarray.concatenate}
          250    0.157    0.001    0.170    0.001 internals.py:5017(_stack_arrays)
            2    0.105    0.053    0.105    0.053 {pandas._libs.algos.take_1d_int64_int64}
          889    0.094    0.000    0.094    0.000 {method 'reduce' of 'numpy.ufunc' objects}
          125    0.089    0.001   12.489    0.100 deepgraph.py:5294(_select_and_return)
          125    0.074    0.001    0.074    0.001 {deepgraph._triu_indices._reduce_triu_indices}
          125    0.066    0.001    0.066    0.001 {built-in method deepgraph._triu_indices._triu_indices}
            4    0.038    0.009    0.038    0.009 {built-in method pandas._libs.algos.ensure_int16}
          125    0.033    0.000   10.979    0.088 <ipython-input-3-26c4f59cd911>:12(corr)
            2    0.028    0.014    0.028    0.014 function_base.py:4703(delete)
            1    0.027    0.027   14.163   14.163 deepgraph.py:4788(_matrix_iterator)
            1    0.027    0.027    0.113    0.113 multi.py:56(_codes_to_ints)
    45771/45222    0.020    0.000    0.043    0.000 {built-in method builtins.isinstance}
            1    0.019    0.019   14.193   14.193 deepgraph.py:183(create_edges)
            2    0.012    0.006    0.700    0.350 algorithms.py:576(factorize)


As you can see, most of the time is spent by getting the requested
features in the corr-function, followed by computing the correlation
values themselves.
