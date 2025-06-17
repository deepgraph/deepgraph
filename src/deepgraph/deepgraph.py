"""The core module of DeepGraph (dg).

This module contains the core class ``dg.DeepGraph`` providing the means
to construct, manipulate and partition graphs, and offering interfacing
methods to common network representations and popular Python network
packages. This class also provides plotting methods to visualize graphs
and their properties and to benchmark the graph construction parameters.

For further information type

>>> help(dg.DeepGraph)

"""
import inspect

# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License

import os
from datetime import datetime
from itertools import chain

from deepgraph.iterators_and_indexers import (
    _matrix_iterator,
    _ft_iterator,
    _iter_edges,
    _initiate_create_edges,
    _aggregate_super_table,
)
from deepgraph.utils import _is_array_like, _dic_translator, _create_bin_edges, _flatten

try:
    import matplotlib as mpl

    display = "DISPLAY" in os.environ
    if not display:
        mpl.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    mpl = None
    plt = None

import numpy as np
import pandas as pd

# get rid of false positive SettingWithCopyWarnings, see
# http://stackoverflow.com/questions/20625582/how-to-deal-with-this-pandas-warning
pd.options.mode.chained_assignment = None


class DeepGraph:
    """The core class of DeepGraph (dg).

    This class encapsulates the graph representation as ``pandas.DataFrame``
    objects in its attributes ``v`` and ``e``. It can be initialized with a
    node table ``v``, whose rows represent the nodes of the graph, as well
    as an edge table ``e``, whose rows represent edges between the nodes.

    Given a node table ``v``, it provides methods to iteratively compute
    pairwise relations between the nodes using arbitrary, user-defined
    functions. These methods provide arguments to parallelize the
    computation and control memory consumption (see ``create_edges`` and
    ``create_edges_ft``).

    Also provides methods to partition nodes, edges or an entire graph by
    the graph's properties and labels, and to create common network
    representations and graph objects of popular Python network packages.

    Furthermore, it provides methods to visualize graphs and their
    properties and to benchmark the graph construction parameters.

    Optionally, the convenience parameter ``supernode_labels_by`` can be
    passed, creating supernode labels by enumerating all distinct (tuples
    of) values of a (multiple) column(s) of ``v`` . Superedge labels can be
    created analogously, by passing the parameter ``superedge_labels_by``.

    Parameters
    ----------
    v : pandas.DataFrame or pandas.HDFStore, optional (default=None)
        The node table, a table representation of the nodes of a graph. The
        index of ``v`` must be unique and represents the node indices. The
        column names of ``v`` represent the types of features of the nodes,
        and each cell represents a feature of a node. Only a reference to
        the input DataFrame is created, not a copy. May also be a
        ``pandas.HDFStore``, but only ``create_edges`` and
        ``create_edges_ft`` may then be used (so far).

    e : pandas.DataFrame, optional (default=None)
        The edge table, a table representation of the edges between the
        nodes given by ``v``. Its index has to be a
        ``pandas.core.index.MultiIndex``, whose first level contains the
        indices of the source nodes, and the second level contains the
        indices of the target nodes. Each row of ``e`` represents an edge,
        column names of ``e`` represent the types of relations of the edges,
        and each cell in ``e`` represents a relation of an edge. Only a
        reference to the input DataFrame is created, not a copy.

    supernode_labels_by : dict, optional (default=None)
        A dictionary whose keys are strings and their values are (lists of)
        column names of ``v``. Appends a column to ``v`` for each key, whose
        values correspond to supernode labels, enumerating all distinct
        (tuples of) values of the column(s) given by the dict's value.

    superedge_labels_by : dict, optional (default=None)
        A dictionary whose keys are strings and their values are (lists of)
        column names of ``e``. Appends a column to ``e`` for each key, whose
        values correspond to superedge labels enumerating all distinct
        (tuples of) values of the column(s) given by the dict's value.

    Attributes
    ----------
    v : pandas.DataFrame
        See Parameters.

    e : pandas.DataFrame
        See Parameters.

    n : int
        Property: Number of nodes.

    m : int
        Property: Number of edges.

    f : pd.DataFrame
        Property: types of features and number of features of corresponding
        type.

    r : pd.DataFrame
        Property: types of relations and number of relations of
        corresponding type.

    """

    def __init__(self, v=None, e=None, supernode_labels_by=None, superedge_labels_by=None):
        # create supernode labels by common features
        if supernode_labels_by is not None:
            for key, value in supernode_labels_by.items():
                v[key] = v.groupby(value).grouper.group_info[0]

        # create superedge labels by common relations
        if superedge_labels_by is not None:
            for key, value in superedge_labels_by.items():
                e[key] = e.groupby(value).grouper.group_info[0]

        # assert v input, set as class attribute
        if v is not None:
            # assert (isinstance(v, pd.DataFrame) or
            #         isinstance(v, pd.HDFStore)), (
            #     "v has to be <type 'pd.DataFrame'> "
            #     "or <type 'pd.HDFStore'>, not {}".format(type(v)))
            self.v = v

        # assert e input, set as class attribute
        if e is not None:
            # assert isinstance(e, pd.DataFrame), (
            #     "e has to be <type 'pd.DataFrame'>, not {}".format(type(e)))
            self.e = e

    def __repr__(self):
        msg = "<{} object, with n={} node(s) and m={} edge(s) at 0x{:02x}>"
        return msg.format(type(self).__name__, self.n, self.m, id(self))

    def __str__(self):
        msg = "<{} object, with n={} node(s) and m={} edge(s) at 0x{:02x}>"
        return msg.format(type(self).__name__, self.n, self.m, id(self))

    def create_edges(
        self,
        connectors=None,
        selectors=None,
        transfer_features=None,
        r_dtype_dic=None,
        no_transfer_rs=None,
        step_size=int(1e7),
        from_pos=0,
        to_pos=None,
        hdf_key=None,
        verbose=False,
        logfile=None,
    ):
        """Create an edge table ``e`` linking the nodes in ``v``.

        This method enables an iterative computation of pairwise relations
        (edges) between the nodes represented by ``v``. It does so in a
        flexible, efficient and vectorized fashion, easily parallelizable and
        with full control over RAM usage.

        1. Connectors

        The simplest use-case is to define a single connector function
        acting on a single column of the node table ``v``. For instance, given
        a node table ``v``

        >>> import pandas as pd
        >>> import deepgraph as dg
        >>> v = pd.DataFrame({'time': [0.,2.,9.], 'x': [3.,1.,12.]})
        >>> g = dg.DeepGraph(v)

        >>> g.v
           time   x
        0     0   3
        1     2   1
        2     9  12

        one may define a function

        >>> def time_difference(time_s, time_t):
        ...     dt = time_t - time_s
        ...     return dt

        and pass it to ``create_edges``, in order to compute the time
        difference of each pair of nodes

        >>> g.create_edges(connectors=time_difference)

        >>> g.e
             dt
        s t
        0 1   2
          2   9
        1 2   7

        As one can see, the connector function takes column names of ``v`` with
        additional '_s' and '_t' endings (indicating source node values and
        target node values, respectively) as input, and returns a variable with
        the computed values. The resulting edge table ``g.e`` is indexed by the
        node indices ('s' and 't', representing source and target node indices,
        respectively), and has one column ('dt', the name of the returned
        variable) with the computed values of the given connector. Note that
        only the upper triangle adjacency matrix is computed, which is always
        the case. See Notes for further information.

        One may also pass a list of functions to ``connectors``, which are then
        computed in the list's order. Generally, a connector function can take
        multiple column names of ``v`` (with '_s' and/or '_t' appended) as
        input, as well as already computed relations of former connectors.
        Also, any connector function may have multiple output variables. Every
        output variable has to be a 1-dimensional ``np.ndarray`` (with
        arbitrary dtype, including ``object``). The return statement may not
        contain any operators, only references to each computed relation.

        For instance, considering the above example, one may define an
        additional connector

        >>> def velocity(dt, x_s, x_t):
        ...     dx = x_t - x_s
        ...     v = dx / dt
        ...     return v, dx

        and then apply both connectors on ``v``, resulting in

        >>> g.create_edges(connectors=[time_difference, velocity])

        >>> g.e
             dt  dx         v
        s t
        0 1   2  -2 -1.000000
          2   9   9  1.000000
        1 2   7  11  1.571429

        2. Selectors

        However, one is often only interested in a subset of all possible
        edges. In order to select edges during the iteration process - based on
        some conditions on the node's features and their computed relations -
        one may pass a (list of) selector function(s) to ``create_edges``. For
        instance, given the above example, one may define a selector

        >>> def dt_thresh(dt, sources, targets):
        ...     sources = sources[dt > 5]
        ...     targets = targets[dt > 5]
        ...     return sources, targets

        and apply it in conjunction with the ``time_difference`` connector

        >>> g.create_edges(connectors=time_difference, selectors=dt_thresh)

        >>> g.e
             dt
        s t
        0 2   9
        1 2   7

        leaving only edges with a time difference larger than 5.

        Every selector function must have ``sources`` and ``targets`` as input
        arguments as well as in the return statement. Most generally, they may
        depend on column names of ``v`` (with '_s' and/or '_t' appended) and/or
        computed relations of connector functions, and/or computed relations of
        former selector functions. Apart from ``sources`` and ``targets``, they
        may additionally return computed relations. Given this input/output
        flexibility of selectors, one could in fact compute all required
        relations, and select any desired subset of edges, with a single
        selector function. The purpose of splitting connectors and/or
        selectors, however, is to control the iteration's performance by
        consecutively computing relations and selecting edges: **hierarchical
        selection**.

        3. Hierarchical Selection

        As the algorithm iterates through the chunks of all possible source and
        target node indices ([0, g.n*(g.n-1)/2]), it goes through the list of
        ``selectors`` at each step. If a selector has a relation as input, it
        must have either been computed by a former selector, or the selector
        requests its computation by the corresponding connector function in
        ``connectors`` (this connector may not depend on any other not yet
        computed relations). Once the input relations are computed (if
        requested), the selector is applied and returns updated indices, which
        are then passed to the next selector. Hence, with each selector, the
        indices are reduced and consecutive computation of relations only
        consider the remaining indices. After all selectors have been applied,
        the connector functions that have not been requested by any selector
        are computed (on the final, reduced chunk of node and target indices).

        4. Transferring Features

        The argument ``transfer_features``, which takes a (list of) column
        name(s) of ``v``, makes it possible to transfer features of ``v`` to
        the created edge table ``e``

        >>> g.create_edges(connectors=time_difference,
        ...                transfer_features=['x', 'time'])

        >>> g.e
             dt  time_s  time_t  x_s  x_t
        s t
        0 1   2       0       2    3    1
          2   9       0       9    3   12
        1 2   7       2       9    1   12

        If computation time and memory consumption are of no concern, one might
        skip the remaing paragraphs.

        5. Logging

        Clearly, the order of the hierarchical selection as described in 3.
        influences the computation's efficiency. The complexity of a relation's
        computation and the (expected average) number of deleted edges of a
        selector should be considered primarily. In order to track and
        benchmark the iteration process, the progress and time measurements are
        printed for each iteration step, if ``verbose`` is set to True.
        Furthermore, one may create a logfile (which can also be plot by
        ``dg.DeepGraph.plot_logfile``) by setting the argument ``logfile`` to a
        string, indicating the file name of the created logfile.

        6. Parallelization and Memory Control

        The arguments ``from_pos``, ``to_pos`` and ``step_size`` control the
        range of processed pairs of nodes and the number of pairs of nodes to
        process at each iteration step. They may be used for parallel
        computation and to control RAM usage. See Parameters for details.

        It is also possible to initiate ``dg.DeepGraph`` with a
        ``pandas.HDFStore`` containing the DataFrame representing the node
        table. Only the data requested by ``transfer_features`` and the user-
        defined ``connectors`` and ``selectors`` at each iteration step is then
        pulled from the store, which is particularly useful for large node
        tables and parallel computation. The only requirement is that the node
        table contained in the store is in table(t) format, not fixed(f)
        format. For instance, considering the above created node table, one may
        store it in a hdf file

        >>> vstore = pd.HDFStore('vstore.h5')
        >>> vstore.put('node_table', v, format='t', index=False)

        initiate a DeepGraph instance with the store

        >>> g = dg.DeepGraph(vstore)

        >>> g.v
        <class 'pandas.io.pytables.HDFStore'>
        File path: vstore.h5
        /node_table            frame_table  (typ->appendable,nrows->3,ncols->2,
        indexers->[index])

        and then create edges the same way as if ``g.v`` were a DataFrame

        >>> g.create_edges(connectors=time_difference)

        >>> g.e
             dt
        s t
        0 1   2
          2   9
        1 2   7

        In case the store has multiple nodes, ``hdf_key`` has to be set to the
        node corresponding to the node table of the graph.

        Also, one may pass a (list of) name(s) of computed relations,
        ``no_transfer_rs``, which should not be transferred to the created edge
        table ``e``. This can be advantageous, for instance, if a selector
        depends on computed relations that are of no further interest.

        Furthermore, it is possible to force the dtype of computed relations
        with the argument ``r_dtype_dic``. The dtype of a relation is then set
        at each iteration step, but **after** all selectors and connectors were
        processed.

        7. Creating Edges on a Fast Track

        If the selection of edges includes a simple distance threshold, i.e. a
        selector function defined as follows:

        >>> def ft_selector(x_s, x_t, threshold, sources, targets):
        ...     dx = x_t - x_s
        ...     sources = sources[dx <= threshold]
        ...     targets = targets[dx <= threshold]
        ...     return sources, targets, dx

        the method ``create_edges_ft`` should be considered, since it provides
        a much faster iteration algorithm.

        Parameters
        ----------
        connectors : function or array_like, optional (default=None)
            User defined connector function(s) that compute pairwise relations
            between the nodes in ``v``. A connector accepts multiple column
            names of ``v`` (with '_s' and/or '_t' appended, indicating source
            node values and target node values, respectively) as input, as well
            as already computed relations of former connectors. A connector
            function may have multiple output variables. Every output variable
            has to be a 1-dimensional ``np.ndarray`` (with arbitrary dtype,
            including ``object``). See above and ``dg.functions`` for examplary
            connector functions.

        selectors : function or array_like, optional (default=None)
            User defined selector function(s) that select edges during the
            iteration process, based on some conditions on the node's features
            and their computed relations. Every selector function must have
            ``sources`` and ``targets`` as input arguments as well as in the
            return statement. A selector may depend on column names of ``v``
            (with '_s' and/or '_t' appended) and/or computed relations of
            connector functions, and/or computed relations of former selector
            functions. Apart from ``sources`` and ``targets``, they may also
            return computed relations (see connectors). See above, and
            ``dg.functions`` for exemplary selector functions.

        transfer_features : str, int or array_like, optional (default=None)
            A (list of) column name(s) of ``v``, indicating which features of
            ``v`` to transfer to ``e`` (appending '_s' and '_t' to the column
            names of ``e``, indicating source and target node features,
            respectively).

        r_dtype_dic : dict, optional (default=None)
            A dictionary with names of computed relations of connectors and/or
            selectors as keys and dtypes as values. Forces the data types of
            the computed relations in ``e`` during the iteration (but **after**
            all selectors and connectors were processed), otherwise infers
            them.

        no_transfer_rs : str or array_like, optional (default=None)
            Name(s) of computed relations that are not to be transferred to the
            created edge table ``e``. Can be used to save memory, e.g., if a
            selector depends on computed relations that are of no interest
            otherwise.

        step_size : int, optional (default=1e6)
            The number of pairs of nodes to process at each iteration step.
            Must be in [ 1, g.n*(g.n-1)/2 ]. Its value determines computation
            speed and memory consumption.

        from_pos : int, optional (default=0)
            Determines from which pair of nodes to start the iteration process.
            Must be in [ 0, g.n*(g.n-1)/2 [. May be used in conjuction with
            ``to_pos`` for parallel computation.

        to_pos : positive integer, optional (default=None)
            Determines at which pair of nodes to stop the iteration process
            (the endpoint is excluded). Must be in [ 1, g.n*(g.n-1)/2 ] and
            larger than ``from_pos``. Defaults to None, which translates to the
            last pair of nodes, g.n*(g.n-1)/2. May be used in conjunction with
            ``from_pos`` for parallel computation.

        hdf_key : str, optional (default=None)
            If you initialized ``dg.DeepGraph`` with a ``pandas.HDFStore`` and
            the store has multiple nodes, you must pass the key to the node in
            the store that corresponds to the node table.

        verbose : bool, optional (default=False)
            Whether to print information at each step of the iteration process.

        logfile : str, optional (default=None)
            Create a log-file named by ``logfile``. Contains the time and date
            of the method's call, the input arguments and time mesaurements for
            each iteration step. A plot of ``logfile`` can be created by
            ``dg.DeepGraph.plot_logfile``.

        Returns
        -------
        e : pd.DataFrame
            Set the created edge table ``e`` as attribute of ``dg.DeepGraph``.

        See also
        --------
        create_edges_ft

        Notes
        -----
        1. Input and output data types

        Since connectors (and selectors) take columns of a pandas DataFrame as
        input, there are no restrictions on the data types of which pairwise
        relations are computed. In the most general case, a DataFrame's column
        has ``object`` as dtype, and its values may then be arbitrary Python
        objects. The same goes for the output variables of connectors (and
        selectors). The only requirement is that each ouput variable is
        1-dimensional.

        However, it is also possible to use the values of a column of ``v`` as
        references to arbitrary objects, which may sometimes be more
        convenient. In case a connector (or selector) needs the node's original
        indices as input, one may simply copy them to a column, e.g.

        >>> v['indices'] = v.index

        and then define the connector's (or selector's) input arguments
        accordingly.

        2. Connectors and selectors

        The only requirement on connectors and selectors is that their input
        arguments and return statements are consistent with the column names of
        ``v`` and the passing of computed relations (see above, 3. Hierarchical
        Selection).

        Whatever happens inside the functions is entirely up to the user. This
        means, for instance, that one may wrap arbitrary functions within a
        connector (selector), such as optimized C functions or existing
        functions whose input/output is not consistent with the
        ``create_edges`` method (see, e.g., the methods provided in
        ``dg.functions``, ``scipy`` or scikit learn's ``sklearn.metrics`` and
        ``sklearn.neighbors.DistanceMetric``). One could also store a
        connector's (selector's) computations directly within the function, or
        let the function print out any desired information during iteration.

        3. Why not compute the full adjacency matrix?

        This is due to efficiency. For any asymmetric function (i.e., f(s, t)
        != f(t, s)), one can always create an additional connector (or output
        variable) that computes the mirrored values of that function.

        """

        # logging
        if logfile:
            _, _, _, argvalues = inspect.getargvalues(inspect.currentframe())
            with open(logfile, "w") as log:
                print("# LOG FILE", file=log)
                print("# function call on: {}".format(datetime.now()), file=log)
                print("#", file=log)
                print("# Parameters", file=log)
                print("# ----------", file=log)
                for arg, value in argvalues.items():
                    print("# ", (arg, value), end="", file=log)
                    print("", file=log)
                print("#", file=log)
                print("# Iterations", file=log)
                print("# ----------", file=log)
                print("# max_pairs exceeded(1) | nr.of pairs | nr.of edges | " "comp.time(s)\n", file=log)

        # measure performance
        start_generation = datetime.now()

        # v shortcut
        v = self.v

        # adjust keywords
        min_chunk_size = step_size
        ft_feature = None

        # create empty transfer features list if not given
        if transfer_features is None:
            transfer_features = []
        elif not _is_array_like(transfer_features):
            transfer_features = [transfer_features]

        # hdf_key
        if isinstance(v, pd.HDFStore) and hdf_key is None:
            assert len(v.keys()) == 1, (
                "hdf store has multiple nodes, hdf_key corresponding to the " " node table has to be passed."
            )
            hdf_key = self.v.keys()[0]

        # initialize
        coldtypedic, verboseprint = _initiate_create_edges(
            verbose, v, ft_feature, connectors, selectors, r_dtype_dic, transfer_features, no_transfer_rs, hdf_key
        )

        # iteratively create link data frame (matrix iterator)
        self.e = _matrix_iterator(
            v, min_chunk_size, from_pos, to_pos, coldtypedic, transfer_features, verboseprint, logfile, hdf_key
        )

        # performance
        deltat = datetime.now() - start_generation
        verboseprint("")
        verboseprint(
            "computation time of function call:",
            "\ts =",
            int(deltat.total_seconds()),
            "\tms =",
            str(deltat.microseconds / 1000.0)[:6],
            "\n",
        )

    def create_edges_ft(
        self,
        ft_feature,
        connectors=None,
        selectors=None,
        transfer_features=None,
        r_dtype_dic=None,
        no_transfer_rs=None,
        min_chunk_size=1000,
        max_pairs=int(1e7),
        from_pos=0,
        to_pos=None,
        hdf_key=None,
        verbose=False,
        logfile=None,
    ):
        """Create (ft) an edge table ``e`` linking the nodes in ``v``.

        This method implements the same functionalities as ``create_edges``,
        with the difference of providing a much quicker iteration algorithm
        based on a so-called fast-track feature. It is advised to read the
        docstring of ``create_edges`` before this one, since only the
        differences are explained in the following.

        Apart from the hierarchical selection through ``connectors`` and
        ``selectors`` as described in the method ``create_edges`` (see 1.-3.),
        this method necessarily includes the (internal) selector function

        >>> def ft_selector(ftf_s, ftf_t, ftt, sources, targets):
        ...     ft_r = ftf_t - ftf_s
        ...     sources = sources[ft_r <= ftt]
        ...     targets = targets[ft_r <= ftt]
        ...     return sources, targets, ft_r

        where ``ftf`` is the fast-track feature (a column name of ``v``),
        ``ftt`` the fast-track threshold (a positive number), and ft_r the
        computed fast-track relation. The argument ``ft_feature``, which has
        to be a tuple (``ftf``, ``ftt``), determines these variables.

        1. The Fast-Track Feature

        The simplest use-case, therefore, is to only pass ``ft_feature``. For
        instance, given a node table

        >>> import pandas as pd
        >>> import deepgraph as dg
        >>> v = pd.DataFrame({'time': [-3.6,-1.1,1.4,4., 6.3],
        ...                   'x': [-3.,3.,1.,12.,7.]})
        >>> g = dg.DeepGraph(v)

        >>> g.v
           time   x
        0  -3.6  -3
        1  -1.1   3
        2   1.4   1
        3   4.0  12
        4   6.3   7

        one may create and select edges by

        >>> g.create_edges_ft(ft_feature=('time', 5))

        >>> g.e
             ft_r
        s t
        0 1   2.5
          2   5.0
        1 2   2.5
        2 3   2.6
          4   4.9
        3 4   2.3

        leaving only edges with a time difference smaller than (or equal to)
        ``ftt`` = 5. Note that the node table always has to be sorted by the
        fast-track feature. This is due to the fact that the algorithm only
        processes pairs of nodes whose fast-track relation is smaller than (or
        equal to) the fast-track threshold, and the (pre)determination of these
        pairs relies on a sorted DataFrame.

        2. Hierarchical Selection

        Additionally, one may define ``connectors`` and ``selectors`` as
        described in ``create_edges`` (see 1.-3.). Per default, the (internal)
        fast-track selector is applied first. It's order of application,
        however, may be determined by inserting the string 'ft_selector' in the
        desired position of the list ``selectors``.

        The remaining arguments are as described in ``create_edges``, apart
        from ``min_chunk_size``, ``max_pairs``, ``from_pos`` and ``to_pos``. If
        computation time and/or memory consumption are a concern, one may
        therefore read the remaining paragraph.

        3. Parallelization and Memory Control on a FastTrack

        At each iteration step, the algorithm takes a number of nodes (n =
        ``min_chunk_size``, per default n=1000) and computes the fast track
        relation (distance) between the last node and the first node, d_ftf =
        ftf_last - ftf_first. In case d_ftf > ``ftt``, all nodes with a fast-
        track feature < ftf_last - ``ftt`` are considered source nodes, and
        their relations with all n nodes are computed (hierarchical selection).
        In case d_ftf <= ``ftt``, n is increased, s.t. d_ftf > ``ftt``. This
        might lead to a large number of pairs of nodes to process at a given
        iteration step. To control memory consumption, one might
        therefore set ``max_pairs`` to a suitable value, triggering a
        subiteration if this value is exceeded.

        To parallelize the iterative computation, one may pass the
        arguments ``from_pos`` and ``to_pos``. They determine the range of
        **source nodes** to process (endpoint excluded). Hence, ``from_pos``
        has to be in [0, g.n[, and ``to_pos`` in [1,g.n]. For instance, given
        the node table above

        >>> g.v
           time   x
        0  -3.6  -3
        1  -1.1   3
        2   1.4   1
        3   4.0  12
        4   6.3   7

        we can compute all relations of the source nodes in [1,3[ by

        >>> g.create_edges_ft(ft_feature=('time', 5), from_pos=1, to_pos=3)

        >>> g.e
             ft_r
        s t
        1 2   2.5
        2 3   2.6
          4   4.9

        Like ``create_edges``, this method also works with a ``pd.HDFStore``
        containing the DataFrame representing the node table. Only the data
        requested by ``ft_feature``, ``transfer_features`` and the user-defined
        ``connectors`` and ``selectors`` at each iteration step is then pulled
        from the store. The node table in the store has to be in table(t)
        format, and additionally, the fast_track feature has to be a data
        column. For instance, storing the above node table

        >>> vstore = pd.HDFStore('vstore.h5')
        >>> vstore.put('node_table', v, format='t', data_columns=True,
        ...            index=False)

        one may initiate a DeepGraph instance with the store

        >>> g = dg.DeepGraph(vstore)

        >>> g.v
        <class 'pandas.io.pytables.HDFStore'>
        File path: vstore.h5
        /node_table            frame_table  (typ->appendable,nrows->5,ncols->2,
        indexers->[index],dc->[time,x])

        and then create edges the same way as if ``g.v`` were a DataFrame

        >>> g.create_edges_ft(ft_feature=('time', 5), from_pos=1, to_pos=3)

        >>> g.e
             ft_r
        s t
        1 2   2.5
        2 3   2.6
          4   4.9

        .. warning:: There is no assertion whether the node table in a store is
                     sorted by the fast-track feature! The result of an
                     unsorted table is unpredictable, and generally not
                     correct.

        Parameters
        ----------
        ft_feature : tuple
            A tuple (ftf, ftt), where ftf is a column name of ``v`` (the fast-
            track feature) and ftt a positive number (the fast-track
            threshold). The fast-track feature may contain integers or floats,
            but datetime-like values are also accepted. In that case,
            ``ft_feature`` has to be a tuple of length 3, (ftf, ftt, dt_unit),
            where dt_unit is on of {'D','h','m','s','ms','us','ns'}:

             - `D`: days
             - `h`: hours
             - `m`: minutes
             - `s`: seconds
             - `ms`: milliseconds
             - `us`: microseconds
             - `ns`: nanoseconds

            determining the unit in which the temporal distance is measured.
            The variable name of the fast-track relation transferred to ``e``
            is ``ft_r``.

        connectors : function or array_like, optional (default=None)
            User defined connector function(s) that compute pairwise relations
            between the nodes in ``v``. A connector accepts multiple column
            names of ``v`` (with '_s' and/or '_t' appended, indicating source
            node values and target node values, respectively) as input, as well
            as already computed relations of former connectors. A connector
            function may have multiple output variables. Every output variable
            has to be a 1-dimensional ``np.ndarray`` (with arbitrary dtype,
            including ``object``). A connector may also depend on the fast-
            track relations ('ft_r'). See ``dg.functions`` for examplary
            connector functions.

        selectors : function or array_like, optional (default=None)
            User defined selector function(s) that select edges during the
            iteration process, based on some conditions on the node's features
            and their computed relations. Every selector function must have
            ``sources`` and ``targets`` as input arguments as well as in the
            return statement. A selector may depend on column names of ``v``
            (with '_s' and/or '_t' appended) and/or computed relations of
            connector functions, and/or computed relations of former selector
            functions. Apart from ``sources`` and ``targets``, they may also
            return computed relations (see connectors). A selector may also
            depend on the fast-track relations ('ft_r'). See ``dg.functions``
            for exemplary selector functions.

            Note: To specify the hierarchical order of the selection by the
            fast-track selector, insert the string 'ft_selector' in the
            corresponding position of the ``selectors`` list. Otherwise,
            computation of ft_r and selection by the fast-track selector is
            carried out first.

        transfer_features : str, int or array_like, optional (default=None)
            A (list of) column name(s) of ``v``, indicating which features of
            ``v`` to transfer to ``e`` (appending '_s' and '_t' to the column
            names of ``e``, indicating source and target node features,
            respectively).

        r_dtype_dic : dict, optional (default=None)
            A dictionary with names of computed relations of connectors and/or
            selectors as keys and dtypes as values. Forces the data types of
            the computed relations in ``e`` during the iteration (but **after**
            all selectors and connectors were processed), otherwise infers
            them.

        no_transfer_rs : str or array_like, optional (default=None)
            Name(s) of computed relations that are not to be transferred to the
            created edge table ``e``. Can be used to save memory, e.g., if a
            selector depends on computed relations that are of no interest
            otherwise.

        min_chunk_size : int, optional (default=1000)
            The minimum number of nodes to form pairs of at each iteration
            step. See above for details.

        max_pairs : positive integer, optional (default=1e6)
            The maximum number of pairs of nodes to process at any given
            iteration step. If the number is exceeded, a memory saving
            subiteration is applied.

        from_pos : int, optional (default=0)
            The locational index (.iloc) of ``v`` to start the iteration.
            Determines the range of **source nodes** to process, in conjuction
            with ``to_pos``. Has to be in [0, g.n[, and smaller than
            ``to_pos``. See above for details and an example.

        to_pos : int, optional (default=None)
            The locational index (.iloc) of ``v`` to end the iteration
            (excluded). Determines the range of **source nodes** to process, in
            conjuction with ``from_pos``. Has to be in [1, g.n], and larger
            than ``from_pos``. Defaults to None, which translates to the last
            node of ``v``, to_pos=g.n. See above for details and an example.

        hdf_key : str, optional (default=None)
            If you initialized ``dg.DeepGraph`` with a ``pandas.HDFStore`` and
            the store has multiple nodes, you must pass the key to the node in
            the store that corresponds to the node table.

        verbose : bool, optional (default=False)
            Whether to print information at each step of the iteration process.

        logfile : str, optional (default=None)
            Create a log-file named by ``logfile``. Contains the time and date
            of the method's call, the input arguments and time mesaurements for
            each iteration step. A plot of ``logfile`` can be created by
            ``dg.DeepGraph.plot_logfile``.

        Returns
        -------
        e : pd.DataFrame
            Set the created edge table ``e`` as attribute of ``dg.DeepGraph``.

        See also
        --------
        create_edges

        Notes
        -----
        The parameter ``min_chunk_size`` enforces a vectorized iteration and
        changing its value can both accelerate or slow down computation time.
        This depends mostly on the distribution of values of the fast track
        feature, and the complexity of the given ``connectors`` and
        ``selectors``. Use the logging capabilites to determine a good value.

        When using a ``pd.HDFStore`` for the computation, the following advice
        might be considered. Recall that the only requirements on the node in
        the store are: the format is table(t), not fixed(t); the node is sorted
        by the fast-track feature; and the fast-track feature is a data column.

        The recommended procedure of storing a given node table ``v`` in a
        store is the following (using the above node table):

        >>> vstore = pd.HDFStore('vstore.h5')
        >>> vstore.put('node_table', v, format='t', data_columns=True,
        ...            index=False)

        Setting index=False significantly decreases the time to construct the
        node in the store, and also reduces the resulting file size. It has no
        impact, however, on the capability of querying the store (with the
        pd.HDFStore.select* methods).

        However, there are two reasons one might want to create a pytables
        index of the fast-track feature:

        1. The node table might be too large to be sorted in memory. To sort it
        on disc, one may proceed as follows. Assuming an unsorted (large) node
        table

        >>> v = pd.DataFrame({'time': [6.3,-3.6,4.,-1.1,1.4],
        ...                   'x': [-3.,3.,1.,12.,7.]})

        >>> v
           time   x
        0   6.3  -3
        1  -3.6   3
        2   4.0   1
        3  -1.1  12
        4   1.4   7

        one stores it as recommended

        >>> vstore = pd.HDFStore('vstore.h5')
        >>> vstore.put('node_table', v, format='t', data_columns=True,
        ...            index=False)
        >>> vstore.get_storer('node_table').group.table
        /node_table/table (Table(5,)) ''
          description := {
          "index": Int64Col(shape=(), dflt=0, pos=0),
          "time": Float64Col(shape=(), dflt=0.0, pos=1),
          "x": Float64Col(shape=(), dflt=0.0, pos=2)}
          byteorder := 'little'
          chunkshape := (2730,)

        creates a (full) pytables index of the fast-track feature

        >>> vstore.create_table_index('node_table', columns=['time'],
        ...                           kind='full')
        >>> vstore.get_storer('node_table').group.table
        /node_table/table (Table(5,)) ''
          description := {
          "index": Int64Col(shape=(), dflt=0, pos=0),
          "time": Float64Col(shape=(), dflt=0.0, pos=1),
          "x": Float64Col(shape=(), dflt=0.0, pos=2)}
          byteorder := 'little'
          chunkshape := (2730,)
          autoindex := True
          colindexes := {
            "time": Index(6, full, shuffle, zlib(1)).is_csi=True}

        and then sorts it on disc with

        >>> vstore.close()
        >>> !ptrepack --chunkshape=auto --sortby=time vstore.h5 s_vstore.h5
        >>> s_vstore = pd.HDFStore('s_vstore.h5')

        >>> s_vstore.node_table
           time   x
        1  -3.6   3
        3  -1.1  12
        4   1.4   7
        2   4.0   1
        0   6.3  -3

        2. To speed up the internal queries on the fast-track feature

        >>> s_vstore.create_table_index('node_table', columns=['time'],
        ...                             kind='full')

        See
        http://stackoverflow.com/questions/17893370/ptrepack-sortby-needs-full-index
        and
        https://gist.github.com/michaelaye/810bd0720bb1732067ff
        for details, benchmarks, and the effects of compressing the store.

        """

        # logging
        if logfile:
            _, _, _, argvalues = inspect.getargvalues(inspect.currentframe())
            with open(logfile, "w") as log:
                print("# LOG FILE", file=log)
                print("# function call on: {}".format(datetime.now()), file=log)
                print("#", file=log)
                print("# Parameters", file=log)
                print("# ----------", file=log)
                for arg, value in argvalues.items():
                    print("# ", (arg, value), end="", file=log)
                    print("", file=log)
                print("#", file=log)
                print("# Iterations", file=log)
                print("# ----------", file=log)
                print("# max_pairs exceeded(1) | nr.of pairs | nr.of edges | " "comp.time(s)\n", file=log)

        # measure performance
        start_generation = datetime.now()

        # v shortcut
        v = self.v

        # hdf key
        if isinstance(v, pd.HDFStore) and hdf_key is None:
            assert len(v.keys()) == 1, (
                "hdf store has multiple nodes, hdf_key corresponding to the " "node table has to be passed."
            )
            hdf_key = self.v.keys()[0]

        # datetime?
        if isinstance(v, pd.HDFStore):
            is_datetime = isinstance(pd.Index(v.select_column(hdf_key, ft_feature[0], stop=0)), pd.DatetimeIndex)
        else:
            is_datetime = isinstance(pd.Index(v.iloc[0:0][ft_feature[0]]), pd.DatetimeIndex)

        # for datetime fast track features, split ft_feature
        if is_datetime:
            assert len(ft_feature) == 3, "for a datetime-like fast track feature, " "the unit has to specified"
            dt_unit = ft_feature[-1]
            ft_feature = ft_feature[:2]
        else:
            dt_unit = None

        # create empty transfer features list if not given
        if transfer_features is None:
            transfer_features = []
        elif not _is_array_like(transfer_features):
            transfer_features = [transfer_features]

        # assert that v is sorted by the fast track feature
        if isinstance(v, pd.DataFrame):
            assert pd.Index(
                v[ft_feature[0]]
            ).is_monotonic_increasing, "The node table is not sorted by the fast track feature."

        # initialize
        coldtypedic, verboseprint = _initiate_create_edges(
            verbose, v, ft_feature, connectors, selectors, r_dtype_dic, transfer_features, no_transfer_rs, hdf_key
        )

        # iteratively create link data frame (fast track iterator)
        self.e = _ft_iterator(
            self,
            v,
            min_chunk_size,
            from_pos,
            to_pos,
            dt_unit,
            ft_feature,
            coldtypedic,
            transfer_features,
            max_pairs,
            verboseprint,
            logfile,
            hdf_key,
        )

        # performance
        deltat = datetime.now() - start_generation
        verboseprint("")
        verboseprint(
            "computation time of function call:",
            "\ts =",
            int(deltat.total_seconds()),
            "\tms =",
            str(deltat.microseconds / 1000.0)[:6],
            "\n",
        )

    def partition_nodes(self, features, feature_funcs=None, n_nodes=True, return_gv=False):
        """Return a supernode DataFrame ``sv``.

        This is essentially a wrapper around the pandas groupby method: ``sv``
        = ``v``.groupby(``features``).agg(``feature_funcs``). It creates a
        (intersection) partition of the nodes in ``v`` by the type(s) of
        feature(s) ``features``, resulting in a supernode DataFrame ``sv``. By
        passing a dictionary of functions on the features of ``v``,
        ``feature_funcs``, one may aggregate user-defined values of the
        partition's elements, the supernodes' features. If ``n_nodes`` is True,
        create a column with the number of each supernode's constituent nodes.
        If ``return_gv`` is True, return the created groupby object to
        facilitate additional operations, such as ``gv``.apply(func, *args,
        **kwargs).

        For details, type help(``v``.groupby), and/or inspect the available
        methods of ``gv``.

        For examples, see below. For an in-depth description and mathematical
        details of graph partitioning, see
        https://arxiv.org/pdf/1604.00971v1.pdf, in particular Sec. III A, E
        and F.

        Parameters
        ----------
        features : str, int or array_like
            Column name(s) of ``v``, indicating the type(s) of feature(s) used
            to induce a (intersection) partition. Creates a pandas groupby
            object, ``gv`` = ``v``.groupby(``features``).

        feature_funcs : dict, optional (default=None)
            Each key must be a column name of ``v``, each value either a
            function, or a list of functions, working when passed a
            ``pandas.DataFrame`` or when passed to ``pandas.DataFrame.apply``.
            See the docstring of ``gv``.agg for details: help(``gv``.agg).

        n_nodes : bool, optional (default=True)
            Whether to create a ``n_nodes`` column in ``sv``, indicating the
            number of nodes in each supernode.

        return_gv : bool, optional (default=False)
            If True, also return the ``v``.groupby(``features``) object,
            ``gv``.

        Returns
        -------
        sv : pd.DataFrame
            The aggreated DataFrame of supernodes, ``sv``.

        gv : pandas.core.groupby.DataFrameGroupBy
            The pandas groupby object, ``v``.groupby(``features``).

        See also
        --------
        partition_edges
        partition_graph

        Notes
        -----
        Currently, NA groups in GroupBy are automatically excluded (silently).
        One workaround is to use a placeholder (e.g., -1, 'none') for NA values
        before doing the groupby (calling this method). See
        http://stackoverflow.com/questions/18429491/groupby-columns-with-nan-missing-values
        and https://github.com/pydata/pandas/issues/3729.

        Examples
        --------
        First, we need a node table, in order to demonstrate its partitioning:

        >>> import pandas as pd
        >>> import deepgraph as dg
        >>> v = pd.DataFrame({'x': [-3.4,2.1,-1.1,0.9,2.3],
        ...                   'time': [0,0,2,2,9],
        ...                   'color': ['g','g','b','g','r'],
        ...                   'size': [1,3,2,3,1]})
        >>> g = dg.DeepGraph(v)
        >>> g.v
          color  size  time    x
        0     g     1     0 -3.4
        1     g     3     0  2.1
        2     b     2     2 -1.1
        3     g     3     2  0.9
        4     r     1     9  2.3

        Create a partition by the type of feature 'color':

        >>> g.partition_nodes('color')
               n_nodes
        color
        b            1
        g            3
        r            1

        Create an intersection partition by the types of features 'color' and
        'size' (which is a further refinement of the last partition):

        >>> g.partition_nodes(['color', 'size'])
                    n_nodes
        color size
        b     2           1
        g     1           1
              3           2
        r     1           1

        Partition by 'color' and collect x values:

        >>> g.partition_nodes('color', {'time': lambda x: list(x)})
               n_nodes       time
        color
        b            1        [2]
        g            3  [0, 0, 2]
        r            1        [9]

        Partition by 'color' and aggregate with different functions:

        >>> g.partition_nodes('color', {'time': [lambda x: list(x), np.max],
        ...                             'x': [np.mean, np.sum, np.std]})
               n_nodes    x_mean  x_sum     x_std time_<lambda>  time_amax
        color
        b            1 -1.100000   -1.1       NaN           [2]          2
        g            3 -0.133333   -0.4  2.891943     [0, 0, 2]          2
        r            1  2.300000    2.3       NaN           [9]          9

        """

        # groupby and aggregate
        gv = self.v.groupby(features)
        sv = _aggregate_super_table(funcs=feature_funcs, size=n_nodes, gt=gv)
        if n_nodes:
            try:
                sv.rename(columns={"size": "n_nodes"}, inplace=True)
            except TypeError:
                sv = sv.rename(columns={"size": "n_nodes"})
        if return_gv:
            return sv, gv
        else:
            return sv

    def partition_edges(
        self,
        relations=None,
        source_features=None,
        target_features=None,
        relation_funcs=None,
        n_edges=True,
        return_ge=False,
    ):
        """Return a superedge DataFrame ``se``.

        This method allows you to partition the edges in ``e`` by their types
        of relations, but also by the types of features of their incident
        source and target nodes, and any combination of the three.

        Essentially, this method is a wrapper around the pandas groupby method:
        ``se`` = ``e``.groupby(``relations`` + features_s +
        features_t).agg(``relation_funcs``), where ``relations`` are column
        names of ``e``, and in order to group ``e`` by features_s and/or
        features_t, the features of type ``source_features`` and/or
        ``target_features`` (column names of ``v``) are transferred to ``e``,
        appending '_s' and/or '_t' to the corresponding column names of ``e``
        (if they are not already present). The only requirement on the
        combination of ``relations``, ``source_features`` and
        ``target_features`` is that at least on of the lists has to be of
        length >= 1.

        By passing a dictionary of functions on the relations of ``e``,
        ``relation_funcs``, one may aggregate user-defined values of the
        partition's elements, the superedges' relations. If ``n_edges`` is
        True, create a column with the number of each superedge's constituent
        edges. If ``return_ge`` is True, return the created groupby object to
        facilitate additional operations, such as ``ge``.apply(func, *args,
        **kwargs).

        For details, type help(``g.e``.groupby), and/or inspect the available
        methods of ``ge``.

        For examples, see below. For an in-depth description and mathematical
        details of graph partitioning, see
        https://arxiv.org/pdf/1604.00971v1.pdf, in particular Sec. III B, E
        and F.

        Parameters
        ----------
        relations : str, int or array_like, optional (default=None)
            Column name(s) of ``e``, indicating the type(s) of relation(s) used
            to induce a (intersection) partition of ``e`` (in conjunction with
            ``source_features`` and ``target_features``).

        source_features : str, int or array_like, optional (default=None)
            Column name(s) of ``v``, indicating the type(s) of feature(s) of
            the edges' incident source nodes used to induce a (intersection)
            partition of ``e`` (in conjunction with ``relations`` and
            ``target_features``).

        target_features : str, int or array_like, optional (default=None)
            Column name(s) of ``v``, indicating the type(s) of feature(s) of
            the edges' incident target nodes used to induce a (intersection)
            partition of ``e`` (in conjunction with ``relations`` and
            ``source_features``).

        relation_funcs : dict, optional (default=None)
            Each key must be a column name of ``e``, each value a (list of)
            function(s), working when passed a ``pandas.DataFrame`` or when
            passed to ``pandas.DataFrame.apply``. See the docstring of
            ``ge``.agg for details: help(``ge``.agg).

        n_edges : bool, optional (default=True)
            Whether to create a ``n_edges`` column in ``se``, indicating the
            number of edges in each superedge.

        return_ge : bool, optional (default=False)
            If True, also return the pandas groupby object, ``ge``.

        Returns
        -------
        se : pd.DataFrame
            The aggreated DataFrame of superedges, ``se``.

        ge : pandas.core.groupby.DataFrameGroupBy
            The pandas groupby object, ``ge``.

        See also
        --------
        partition_nodes
        partition_graph

        Notes
        -----
        Currently, NA groups in GroupBy are automatically excluded (silently).
        One workaround is to use a placeholder (e.g., -1, 'none') for NA values
        before doing the groupby (calling this method). See
        http://stackoverflow.com/questions/18429491/groupby-columns-with-nan-missing-values
        and https://github.com/pydata/pandas/issues/3729.

        Examples
        --------
        First, we need to create a graph in order to demonstrate how to
        partition its edge set.

        Create a node table:

        >>> import pandas as pd
        >>> import deepgraph as dg
        >>> v = pd.DataFrame({'x': [-3.4,2.1,-1.1,0.9,2.3],
        ...                   'time': [0,1,2,5,9],
        ...                   'color': ['g','g','b','g','r'],
        ...                   'size': [1,3,2,3,1]})
        >>> g = dg.DeepGraph(v)

        >>> g.v
          color  size  time    x
        0     g     1     0 -3.4
        1     g     3     1  2.1
        2     b     2     2 -1.1
        3     g     3     5  0.9
        4     r     1     9  2.3

        Create an edge table:

        >>> def some_relations(ft_r, x_s,x_t,color_s,color_t,size_s,size_t):
        ...     dx = x_t - x_s
        ...     v = dx / ft_r
        ...     same_color = color_s == color_t
        ...     larger_than = size_s > size_t
        ...     return dx, v, same_color, larger_than
        >>> g.create_edges_ft(('time', 5), connectors=some_relations)
        >>> g.e.rename(columns={'ft_r': 'dt'}, inplace=True)
        >>> g.e['inds'] = g.e.index.values  # to ease the eyes

        >>> g.e
              dx  dt larger_than same_color         v    inds
        s t
        0 1  5.5   1       False       True  5.500000  (0, 1)
          2  2.3   2       False      False  1.150000  (0, 2)
          3  4.3   5       False       True  0.860000  (0, 3)
        1 2 -3.2   1        True      False -3.200000  (1, 2)
          3 -1.2   4       False       True -0.300000  (1, 3)
        2 3  2.0   3       False      False  0.666667  (2, 3)
        3 4  1.4   4        True      False  0.350000  (3, 4)

        Partitioning by the type of relation 'larger_than':

        >>> g.partition_edges(relations='larger_than',
        ...                   relation_funcs={'dx': ['mean', 'std'],
        ...                                   'same_color': 'sum'})
                     n_edges  same_color_sum  dx_mean    dx_std
        larger_than
        False              5               3     2.58  2.558711
        True               2               0    -0.90  3.252691

        A refinement of the last partition by the type of relation
        'same_color':

        >>> g.partition_edges(relations=['larger_than', 'same_color'],
        ...                   relation_funcs={'dx': ['mean', 'std'],
        ...                                   'dt': lambda x: tuple(x)})
                                n_edges dt_<lambda>   dx_mean    dx_std
        larger_than same_color
        False       False             2      (2, 3)  2.150000  0.212132
                    True              3   (1, 5, 4)  2.866667  3.572581
        True        False             2      (1, 4) -0.900000  3.252691

        Partitioning by the type of source feature 'color':

        >>> g.partition_edges(source_features='color',
        ...                   relation_funcs={'same_color': 'sum'})
                 n_edges  same_color
        color_s
        b              1           0
        g              6           3

        As one can see, the type of feature 'color' of the source nodes has
        been transferred to ``e``:

        >>> g.e
              dx  dt larger_than same_color         v    inds color_s
        s t
        0 1  5.5   1       False       True  5.500000  (0, 1)       g
          2  2.3   2       False      False  1.150000  (0, 2)       g
          3  4.3   5       False       True  0.860000  (0, 3)       g
        1 2 -3.2   1        True      False -3.200000  (1, 2)       g
          3 -1.2   4       False       True -0.300000  (1, 3)       g
        2 3  2.0   3       False      False  0.666667  (2, 3)       b
        3 4  1.4   4        True      False  0.350000  (3, 4)       g

        A further refinement of the last partition by the type of source
        feature 'size':

        >>> g.partition_edges(source_features=['color', 'size'],
        ...                   relation_funcs={'same_color': 'sum',
        ...                                   'inds': lambda x: tuple(x)})
                        n_edges  same_color                      inds
        color_s size_s
        b       2             1           0                 ((2, 3),)
        g       1             3           2  ((0, 1), (0, 2), (0, 3))
                3             3           1  ((1, 2), (1, 3), (3, 4))

        Partitioning by the types of target features ('color', 'size'):

        >>> g.partition_edges(target_features=['color', 'size'],
        ...                   relation_funcs={'same_color': 'sum',
        ...                                   'inds': lambda x: tuple(x)})
                        n_edges  same_color                              inds
        color_t size_t
        b       2             2           0                  ((0, 2), (1, 2))
        g       3             4           3  ((0, 1), (0, 3), (1, 3), (2, 3))
        r       1             1           0                         ((3, 4),)

        Partitioning by the type of source feature 'color' and the type of
        target feature 'size':

        >>> g.partition_edges(source_features='color', target_features='size',
        ...                   relation_funcs={'same_color': 'sum',
        ...                                   'inds': lambda x: tuple(x)})
                        n_edges  same_color                      inds
        color_s size_t
        b       3             1           0                 ((2, 3),)
        g       1             1           0                 ((3, 4),)
                2             2           0          ((0, 2), (1, 2))
                3             3           3  ((0, 1), (0, 3), (1, 3))

        A further refinement of the last partition by the type of relation
        'larger_than':

        >>> g.partition_edges(relations='larger_than',
        ...                   source_features='color', target_features='size',
        ...                   relation_funcs={'inds': lambda x: tuple(x)})
                                    n_edges                      inds
        larger_than color_s size_t
        False       b       3             1                 ((2, 3),)
                    g       2             1                 ((0, 2),)
                            3             3  ((0, 1), (0, 3), (1, 3))
        True        g       1             1                 ((3, 4),)
                            2             1                 ((1, 2),)

        """

        if not relations:
            relations = []

        if not _is_array_like(relations):
            relations = [relations]

        # transfer feature columns to g.e, for fast groupby
        if source_features:
            if not _is_array_like(source_features):
                source_features = [source_features]
            cols_s = []
            for sf_col in source_features:
                cols_s.append(sf_col + "_s")
                if sf_col + "_s" not in self.e.columns:
                    s = self.e.index.get_level_values(0)
                    self.e.loc[:, sf_col + "_s"] = self.v.loc[s, sf_col].values
        else:
            cols_s = []

        if target_features:
            if not _is_array_like(target_features):
                target_features = [target_features]
            cols_t = []
            for tf_col in target_features:
                cols_t.append(tf_col + "_t")
                if tf_col + "_t" not in self.e.columns:
                    s = self.e.index.get_level_values(1)
                    self.e.loc[:, tf_col + "_t"] = self.v.loc[s, tf_col].values
        else:
            cols_t = []

        cols = relations + cols_s + cols_t

        ge = self.e.groupby(cols)
        se = _aggregate_super_table(funcs=relation_funcs, size=n_edges, gt=ge)

        if n_edges:
            se = se.rename(columns={"size": "n_edges"})

        if return_ge:
            return se, ge
        else:
            return se

    def partition_graph(
        self, features, feature_funcs=None, relation_funcs=None, n_nodes=True, n_edges=True, return_gve=False
    ):
        """Return supergraph DataFrames ``sv`` and ``se``.

        This method allows partitioning of the  graph represented by ``v`` and
        ``e`` into a supergraph, ``sv`` and ``se``. It creates a (intersection)
        partition of the nodes in ``v`` by the type(s) of feature(s)
        ``features``, together with the (intersection) partition's
        **corresponding** partition of the edges in ``e``.

        Essentially, this method is a wrapper around pandas groupby methods:
        ``sv`` = ``v``.groupby(``features``).agg(``feature_funcs``) and
        ``se`` = ``e``.groupby(features_s+features_t).agg(``relation_funcs``).
        In order to group ``e`` by features_s and features_t, the features of
        type ``features`` are transferred to ``e``, appending '_s' and '_t' to
        the corresponding column names of ``e``, indicating source and target
        features, respectively (if they are not already present).

        By passing a dictionary of functions on the features (relations) of
        ``v`` (``e``), ``feature_funcs`` (``relation_funcs``), one may
        aggregate user-defined values of the partition's elements, the
        supernodes' (superedges') features (relations). If ``n_nodes``
        (``n_edges``) is True, create a column with the number of each
        supernode's (superedge's) constituent nodes (edges).

        If ``return_gve`` is True, return the created groupby objects to
        facilitate additional operations, such as ``gv``.apply(func, *args,
        **kwargs) or ``ge``.apply(func, *args, **kwargs).

        For details, type help(``g.v``.groupby), and/or inspect the available
        methods of ``gv``.

        For examples, see below. For an in-depth description and mathematical
        details of graph partitioning, see
        https://arxiv.org/pdf/1604.00971v1.pdf, in particular Sec. III C, E
        and F.

        Parameters
        ----------
        features : str, int or array_like
            Column name(s) of ``v``, indicating the type(s) of feature(s) used
            to induce a (intersection) partition of ``v``, and its
            **corresponding** partition of the edges in ``e``. Creates pandas
            groupby objects, ``gv`` and ``ge``.

        feature_funcs : dict, optional (default=None)
            Each key must be a column name of ``v``, each value either a
            function, or a list of functions, working when passed a
            ``pandas.DataFrame`` or when passed to ``pandas.DataFrame.apply``.
            See the docstring of ``gv``.agg for details: help(``gv``.agg).

        relation_funcs : dict, optional (default=None)
            Each key must be a column name of ``e``, each value either a
            function, or a list of functions, working when passed a
            ``pandas.DataFrame`` or when passed to ``pandas.DataFrame.apply``.
            See the docstring of ``ge``.agg for details: help(``ge``.agg).

        n_nodes : bool, optional (default=True)
            Whether to create a ``n_nodes`` column in ``sv``, indicating the
            number of nodes in each supernode.

        n_edges : bool, optional (default=True)
            Whether to create a ``n_edges`` column in ``se``, indicating the
            number of edges in each superedge.

        return_gve : bool, optional (default=False)
            If True, also return the pandas groupby objects, ``gv`` and ``ge``.

        Returns
        -------
        sv : pd.DataFrame
            The aggreated DataFrame of supernodes, ``sv``.

        se : pd.DataFrame
            The aggregated DataFrame of superedges, ``se``.

        gv : pandas.core.groupby.DataFrameGroupBy
            The pandas groupby object, ``v``.groupby(``features``).

        ge : pandas.core.groupby.DataFrameGroupBy
            The pandas groupby object, ``e``.groupby(features_i+feaures_j).

        See also
        --------
        partition_nodes
        partition_edges

        Notes
        -----
        Currently, NA groups in GroupBy are automatically excluded (silently).
        One workaround is to use a placeholder (e.g., -1, 'none') for NA values
        before doing the groupby (calling this method). See
        http://stackoverflow.com/questions/18429491/groupby-columns-with-nan-missing-values
        and https://github.com/pydata/pandas/issues/3729.

        Examples
        --------
        First, we need to create a graph in order to demonstrate its
        partitioning into a supergraph.

        Create a node table:

        >>> import pandas as pd
        >>> import deepgraph as dg
        >>> v = pd.DataFrame({'x': [-3.4,2.1,-1.1,0.9,2.3],
        ...                   'time': [0,1,2,5,9],
        ...                   'color': ['g','g','b','g','r'],
        ...                   'size': [1,3,2,3,1]})
        >>> g = dg.DeepGraph(v)

        >>> g.v
          color  size  time    x
        0     g     1     0 -3.4
        1     g     3     1  2.1
        2     b     2     2 -1.1
        3     g     3     5  0.9
        4     r     1     9  2.3

        Create an edge table:

        >>> def some_relations(ft_r, x_s,x_t,color_s,color_t,size_s,size_t):
        ...     dx = x_t - x_s
        ...     v = dx / ft_r
        ...     same_color = color_s == color_t
        ...     larger_than = size_s > size_t
        ...     return dx, v, same_color, larger_than
        >>> g.create_edges_ft(('time', 5), connectors=some_relations)
        >>> g.e.rename(columns={'ft_r': 'dt'}, inplace=True)
        >>> g.e['inds'] = g.e.index.values  # to ease the eyes

        >>> g.e
              dx  dt larger_than same_color         v    inds
        s t
        0 1  5.5   1       False       True  5.500000  (0, 1)
          2  2.3   2       False      False  1.150000  (0, 2)
          3  4.3   5       False       True  0.860000  (0, 3)
        1 2 -3.2   1        True      False -3.200000  (1, 2)
          3 -1.2   4       False       True -0.300000  (1, 3)
        2 3  2.0   3       False      False  0.666667  (2, 3)
        3 4  1.4   4        True      False  0.350000  (3, 4)

        Create a supergraph by partitioning by the type of feature 'color':

        >>> sv, se = g.partition_graph('color')

        >>> sv
               n_nodes
        color
        b            1
        g            3
        r            1

        >>> se
                         n_edges
        color_s color_t
        b       g              1
        g       b              2
                g              3
                r              1

        Create intersection partitions by the types of features 'color' and
        'size' (which are further refinements of the last partitions):

        >>> sv, se = g.partition_graph(
        ...     ['color', 'size'],
        ...     relation_funcs={'inds': lambda x: tuple(x)})

        >>> sv
                    n_nodes
        color size
        b     2           1
        g     1           1
              3           2
        r     1           1

        >>> se
                                       n_edges              inds
        color_s size_s color_t size_t
        b       2      g       3             1         ((2, 3),)
        g       1      b       2             1         ((0, 2),)
                       g       3             2  ((0, 1), (0, 3))
                3      b       2             1         ((1, 2),)
                       g       3             1         ((1, 3),)
                       r       1             1         ((3, 4),)

        Partition by 'color' and aggregate some properties:

        >>> sv, se = g.partition_graph('color',
        ...     feature_funcs={'time': lambda x: list(x)},
        ...     relation_funcs={'larger_than': 'sum', 'same_color': 'sum'})

        >>> sv
               n_nodes       time
        color
        b            1        [2]
        g            3  [0, 1, 5]
        r            1        [9]

        >>> se
                         n_edges larger_than  same_color
        color_s color_t
        b       g              1       False           0
        g       b              2        True           0
                g              3       False           3
                r              1        True           0

        """

        gv = self.v.groupby(features)
        sv = _aggregate_super_table(funcs=feature_funcs, size=n_nodes, gt=gv)
        if n_nodes:
            sv.rename(columns={"size": "n_nodes"}, inplace=True)

        # transfer feature columns to g.e, for fast groupby
        cols_s = []
        cols_t = []
        if not _is_array_like(features):
            features = [features]
        for col in features:
            cols_s.append(col + "_s")
            cols_t.append(col + "_t")
            if col + "_s" not in self.e.columns:
                s = self.e.index.get_level_values(0)
                self.e.loc[:, col + "_s"] = self.v.loc[s, col].values
            if col + "_t" not in self.e.columns:
                t = self.e.index.get_level_values(1)
                self.e.loc[:, col + "_t"] = self.v.loc[t, col].values

        ge = self.e.groupby(cols_s + cols_t)
        se = _aggregate_super_table(funcs=relation_funcs, size=n_edges, gt=ge)
        if n_edges:
            se = se.rename(columns={"size": "n_edges"})

        if return_gve:
            return sv, se, gv, ge
        else:
            return sv, se

    def return_cs_graph(self, relations=False, dropna=True):
        """Return ``scipy.sparse.coo_matrix`` representation(s).

        Create a compressed sparse graph representation for each type of
        relation given by ``relations``. ``relations`` can either be False,
        True, or a (list of) column name(s) of ``e``. If ``relations`` is False
        (default), return a single csgraph entailing all edges in ``e.index``,
        each with a weight of 1 (in that case, ``dropna`` is discarded). If
        ``relations`` is True, create one csgraph for each column of ``e``,
        where the weights are given by the columns' values. If only a subset of
        columns is to be mapped to csgraphs, ``relations`` has to be a (list
        of) column name(s) of ``e``.

        The argument ``dropna`` indicates whether to discard edges with NA
        values or not. If ``dropna`` is True or False, it applies to all types
        of relations given by ``relations``. However, ``dropna`` can also be
        array_like with the same shape as ``relations`` (or with the same shape
        as ``e.columns``, if ``relations`` is True).

        Parameters
        ----------
        relations : bool, str or array_like, optional (default=False)
            The types of relations to be mapped to scipy csgraphs. Can be
            False, True, or a (list of) column name(s) of ``e``.

        dropna : bool or array_like, optional (default=True)
            Whether to drop edges with NA values. If True or False, applies to
            all relations given by ``relations``. Otherwise, must be the same
            shape as ``relations``. If ``relations`` is False, ``dropna`` is
            discarded.

        Returns
        -------
        csgraph : scipy.sparse.coo_matrix or dict
            A dictionary, where keys are column names of ``e``, and values are
            the corresponding ``scipy.sparse.coo_matrix`` instance(s). If only
            one csgraph is created, return it directly.

        See also
        --------
        return_nx_graph
        return_nx_multigraph
        return_gt_graph

        """

        from scipy.sparse import coo_matrix

        # get indices
        index = self.v.index
        indices = index.values
        n = len(indices)

        # enumerate indices if necessary
        if type(index) is pd.RangeIndex:
            if index.start == 0 and index.stop == n:
                inddic = None
            else:
                inddic = {j: i for i, j in enumerate(indices)}
        else:
            inddic = {j: i for i, j in enumerate(indices)}

        # for default arguments
        if relations is False:
            s = self.e.index.get_level_values(0).values
            t = self.e.index.get_level_values(1).values
            if inddic:
                s = _dic_translator(s, inddic)
                t = _dic_translator(t, inddic)
            else:
                pass

            # create cs graph
            cs_g = coo_matrix((np.ones(len(s), dtype=bool), (s, t)), shape=(n, n), dtype=bool)

        else:
            if relations is True:
                relations = self.e.columns.values

            # check that relations and dropna have the same shape
            if _is_array_like(relations) and _is_array_like(dropna):
                assert len(relations) == len(dropna), "dropna and relations have different shapes!"

            if not _is_array_like(relations):
                relations = [relations]
            if not _is_array_like(dropna):
                dropna = [dropna] * len(relations)

            # create coo_matrices
            cs_g = {}
            for r, drop in zip(relations, dropna):
                if drop:
                    data = self.e[r].dropna()
                else:
                    data = self.e[r]
                s = data.index.get_level_values(0).values
                t = data.index.get_level_values(1).values
                if inddic:
                    s = _dic_translator(s, inddic)
                    t = _dic_translator(t, inddic)
                else:
                    pass

                # create cs graph
                cs_g[r] = coo_matrix((data.values, (s, t)), shape=(n, n), dtype=data.dtype)

            # if there is only one csgraph
            if len(cs_g) == 1:
                cs_g = cs_g[r]

        return cs_g

    def return_nx_graph(self, features=False, relations=False, dropna="none"):
        """Return a ``networkx.DiGraph`` representation.

        Create a ``networkx.DiGraph`` representation of the graph given by
        ``v`` and ``e``. Node and edge properties to transfer can be indicated
        by the ``features`` and ``relations`` input arguments. Whether to drop
        edges with NA values in the subset of types of relations given by
        ``relations`` can be controlled by ``dropna``.

        Needs pandas >= 0.17.0.

        Parameters
        ----------
        features : bool, str, or array_like, optional (default=False)
            Indicates which types of features to transfer as node attributes.
            Can be column name(s) of ``v``, False or True. If False, create no
            node attributes. If True, create node attributes for every column
            in ``v``. If str or array_like, must be column name(s) of ``v``
            indicating which types of features to transfer.

        relations : bool, str, or array_like, optional (default=False)
            Indicates which types of relations to transfer as edge attributes.
            Can be column name(s) of ``e``, False or True. If False, create no
            edge attributes (all edges in ``e.index`` are transferred,
            regardless of ``dropna``). If True, create edge attributes for
            every column in ``e`` (all edges in ``e.index`` are transferred,
            regardless of ``dropna``). If str or array_like, must be column
            name(s) of ``e`` indicating which types of relations to transfer
            (which edges are transferred can be controlled by ``dropna``).

        dropna : str, optional (default='none')
            One of {'none','any','all'}. If 'none', all edges in ``e.index``
            are transferred. If 'any', drop all edges (rows) in
            ``e[relations]`` where any NA values are present. If 'all', drop
            all edges (rows) in ``e[relations]`` where all values are NA. Only
            has an effect if ``relations`` is str or array_like.

        Returns
        -------
        nx_g : networkx.DiGraph

        See also
        --------
        return_nx_multigraph
        return_cs_graph
        return_gt_graph

        """

        import networkx as nx

        # create empty DiGraph
        nx_g = nx.DiGraph()

        # select features
        if features is False:
            vt = pd.DataFrame(index=self.v.index)
        elif features is True:
            vt = self.v
        elif _is_array_like(features):
            vt = self.v[features]
        else:
            vt = self.v[features].to_frame()

        # create nx compatible tuple, (index, weight_dict)
        vt = vt.to_dict("index")
        vt = ((key, value) for key, value in vt.items())

        # add nodes
        nx_g.add_nodes_from(vt)

        # select relations
        if hasattr(self, "e"):
            if relations is False:
                et = pd.DataFrame(index=self.e.index)

            elif relations is True:
                et = self.e

            elif _is_array_like(relations):
                if dropna != "none":
                    et = self.e[relations].dropna(how=dropna)
                else:
                    et = self.e[relations]

            else:
                if dropna != "none":
                    et = self.e[relations].to_frame().dropna(how=dropna)
                else:
                    et = self.e[relations].to_frame()

            # create nx compatible tuple, (index, index, weight_dict)
            et = et.to_dict("index")
            et = ((key[0], key[1], value) for key, value in et.items())

            # add edges
            nx_g.add_edges_from(et)

        return nx_g

    def return_nx_multigraph(self, features=False, relations=False, dropna=True):
        """Return a ``networkx.MultiDiGraph`` representation.

        Create a ``networkx.MultiDiGraph`` representation of the graph given by
        ``v`` and ``e``. As opposed to ``return_nx_graph``, where every row of
        ``e`` is treated as one edge, this method treats every cell of
        ``e`` as one edge. The input argument ``features`` indicates which node
        properties to transfer. ``relations`` indicates which edges to
        transfer. Whether to drop edges with NA values can be controlled by
        ``dropna``.

        Needs pandas >= 0.17.0.

        Parameters
        ----------
        features : bool, str, or array_like, optional (default=False)
            Indicates which types of features to transfer as node attributes.
            Can be column name(s) of ``v``, False or True. If False, create no
            node attributes. If True, create node attributes for every column
            in ``v``. If str or array_like, must be column name(s) of ``v``
            indicating which types of features to transfer.

        relations : bool, str, or array_like, optional (default=False)
            Indicates which cells of ``e`` to transfer as edges. Can be False,
            True, or a (list of) column name(s) of ``e``. If False (default),
            all cells of ``e`` are translated to edges, but their values are
            not transferred as edge attributes. If True, all cells of ``e`` are
            translated, and their values are transferred as edge attributes. If
            str or array_like, must be column name(s) of ``e``, restricting the
            translation of cells to edges to ``e[relations]`` (values are
            transferred as edge attributes).

        dropna : bool, optional (default=True)
            Whether to drop edges with NA values. Cells in ``e`` with NA values
            are not translated to edges.

        Returns
        -------
        nx_g : networkx.MultiDiGraph

        See also
        --------
        return_nx_graph
        return_cs_graph
        return_gt_graph

        """

        import networkx as nx

        # create empty MultiDiGraph
        nx_g = nx.MultiDiGraph()

        # select features
        if features is False:
            vt = pd.DataFrame(index=self.v.index)
        elif features is True:
            vt = self.v
        elif _is_array_like(features):
            vt = self.v[features]
        else:
            vt = self.v[features].to_frame()

        # create nx compatible tuple, (index, weight_dict)
        vt = vt.to_dict("index")
        vt = ((key, value) for key, value in vt.items())

        # add nodes
        nx_g.add_nodes_from(vt)

        # select relations
        if hasattr(self, "e"):
            if relations is False:
                if dropna:
                    et = self.e.count(axis=1).to_dict()
                    et = ((key,) * value for key, value in et.items())
                    et = chain(*et)
                else:
                    ncol = len(self.e.columns)
                    et = self.e.index
                    et = ((key,) * ncol for key in et)
                    et = chain(*et)

            elif relations is True:
                et = _iter_edges(self.e, dropna)

            elif _is_array_like(relations):
                et = self.e[relations]
                et = _iter_edges(et, dropna)

            else:
                et = self.e[relations].to_frame()
                et = _iter_edges(et, dropna)

            # add edges
            nx_g.add_edges_from(et)

        return nx_g

    def return_gt_graph(self, features=False, relations=False, dropna="none", node_indices=False, edge_indices=False):
        """Return a ``graph_tool.Graph`` representation.

        Create a ``graph_tool.Graph`` (directed) representation of the graph
        given by ``v`` and ``e``. Node and edge properties to transfer can be
        indicated by the ``features`` and ``relations`` input arguments.
        Whether to drop edges with NA values in the subset of types of
        relations given by ``relations`` can be controlled by ``dropna``. If
        the nodes in ``v`` are not indexed by consecutive integers starting
        from 0, one may internalize the original node and edge indices as
        propertymaps by setting ``node_indices`` and/or ``edge_indices`` to
        True.

        Parameters
        ----------
        features : bool, str, or array_like, optional (default=False)
            Indicates which types of features to internalize as
            ``graph_tool.PropertyMap``. Can be column name(s) of ``v``, False
            or True. If False, create no propertymaps. If True, create
            propertymaps for every column in ``v``. If str or array_like, must
            be column name(s) of ``v`` indicating which types of features to
            internalize.

        relations : bool, str, or array_like, optional (default=False)
            Indicates which types of relations to internalize as
            ``graph_tool.PropertyMap``. Can be column name(s) of ``e``, False
            or True. If False, create no propertymaps (all edges in ``e.index``
            are transferred, regardless of ``dropna``). If True, create
            propertymaps for every column in ``e`` (all edges in ``e.index``
            are transferred, regardless of ``dropna``). If str or array_like,
            must be column name(s) of ``e`` indicating which types of relations
            to internalize (which edges are transferred can be controlled by
            ``dropna``).

        dropna : str, optional (default='none')
            One of {'none','any','all'}. If 'none', all edges in ``e.index``
            are transferred. If 'any', drop all edges (rows) in
            ``e[relations]`` where any NA values are present. If 'all', drop
            all edges (rows) in ``e[relations]`` where all values are NA. Only
            has an effect if ``relations`` is str or array_like.

        node_indices : bool, optional (default=False)
            If True, internalize a vertex propertymap ``i`` with the original
            node indices.

        edge_indices : bool, optional (default=False)
            If True, internalize edge propertymaps ``s`` and ``t`` with the
            original source and target node indices of the edges, respectively.

        Returns
        -------
        gt_g : graph_tool.Graph

        See also
        --------
        return_cs_graph
        return_nx_graph
        return_nx_multigraph

        Notes
        -----
        If the index of ``v`` is not pd.RangeIndex(start=0,stop=len(``v``),
        step=1), the indices will be enumerated, which is expensive for large
        graphs.

        """

        import graph_tool as gt

        # propertymap dtypes
        dtdic = {
            "bool": "bool",
            # int16_t: 'short',
            "uint8": "int16_t",
            "int8": "int16_t",
            "int16": "int16_t",
            # int32_t: 'int',
            "uint16": "int32_t",
            "int32": "int32_t",
            # int64_t: 'long',
            "uint32": "int64_t",
            "int64": "int64_t",
            "uint64": "int64_t",
            # double: 'float',
            "float16": "double",
            "float32": "double",
            "float64": "double",
            "float128": "double",
        }

        # get indices
        index = self.v.index
        indices = index.values
        n = len(indices)

        # enumerate indices if necessary
        if type(index) is pd.RangeIndex:
            if index.start == 0 and index.stop == n:
                inddic = None
            else:
                inddic = {j: i for i, j in enumerate(indices)}
        else:
            inddic = {j: i for i, j in enumerate(indices)}

        # create empty Graph
        gt_g = gt.Graph(directed=True)

        # select features
        if features is False:
            vt = pd.DataFrame(index=index)
        elif features is True:
            vt = self.v
        elif _is_array_like(features):
            vt = self.v[features]
        else:
            vt = self.v[features].to_frame()

        # add nodes
        gt_g.add_vertex(n)

        # add vertex propertymaps
        if node_indices:
            try:
                pm = gt_g.new_vertex_property(dtdic[str(index.dtype)], indices)
            except KeyError:
                pm = gt_g.new_vertex_property("object", indices)
            # internalize
            gt_g.vertex_properties["i"] = pm

        for col in vt.columns:
            try:
                pm = gt_g.new_vertex_property(dtdic[str(vt[col].dtype)], vt[col].values)
            except KeyError:
                pm = gt_g.new_vertex_property("object", vt[col].values)
            # internalize
            gt_g.vertex_properties[str(col)] = pm

        # select relations
        if hasattr(self, "e"):
            if relations is False:
                et = pd.DataFrame(index=self.e.index)
            elif relations is True:
                et = self.e
            elif _is_array_like(relations):
                if dropna != "none":
                    et = self.e[relations].dropna(how=dropna)
                else:
                    et = self.e[relations]
            else:
                if dropna != "none":
                    et = self.e[relations].to_frame().dropna(how=dropna)
                else:
                    et = self.e[relations].to_frame()

            # add edges
            s = et.index.get_level_values(level=0).values
            t = et.index.get_level_values(level=1).values
            if inddic:
                ns = _dic_translator(s, inddic).astype(int)
                nt = _dic_translator(t, inddic).astype(int)
                gt_g.add_edge_list(np.column_stack((ns, nt)))
                del ns, nt
            else:
                gt_g.add_edge_list(np.column_stack((s, t)))

            # add edge propertymaps
            if edge_indices:
                try:
                    s = gt_g.new_edge_property(dtdic[str(s.dtype)], s)
                    t = gt_g.new_edge_property(dtdic[str(t.dtype)], t)
                except KeyError:
                    s = gt_g.new_edge_property("object", s)
                    t = gt_g.new_edge_property("object", t)
                # internalize
                gt_g.edge_properties["s"] = s
                gt_g.edge_properties["t"] = t

            for col in et.columns:
                try:
                    pm = gt_g.new_edge_property(dtdic[str(et[col].dtype)], et[col].values)
                except KeyError:
                    pm = gt_g.new_edge_property("object", et[col].values)
                # internalize
                gt_g.edge_properties[str(col)] = pm

        return gt_g

    def append_cp(
        self, directed=False, connection="weak", col_name="cp", label_by_size=True, consolidate_singles=False
    ):
        """Append a component membership column to ``v``.

        Append a column to ``v`` indicating the component membership of each
        node. Requires scipy.

        Parameters
        ----------
        directed : bool, optional (default=False)
            If True , then operate on a directed graph: only move from point i
            to point j along paths csgraph[i, j]. If False, then find the
            shortest path on an undirected graph: the algorithm can progress
            from point i to j along csgraph[i, j] or csgraph[j, i].

        connection : str, optional (default='weak')
            One of {'weak','strong'}. For directed graphs, the type of
            connection to use.  Nodes i and j are strongly connected if a path
            exists both from i to j and from j to i.  Nodes i and j are weakly
            connected if only one of these paths exists. Only has an effect if
            ``directed`` is True

        col_name : str, optional (default='cp')
            The name of the appended column of component labels.

        label_by_size : bool, optional (default=True)
            Whether to rename component membership labels to reflect component
            sizes. If True, the smallest component corresponds to the largest
            label, and the largest component corresponds to the label 0 (or 1
            if ``consolidate_singles`` is True). If False, pass on labels given
            by scipy's connected_components method directly (faster and uses
            less memory).

        consolidate_singles: bool, optional (default=False)
            If True, all singular components (components comprised of one node
            only) are consolidated under the label 0. Also, all other labels
            are renamed to reflect component sizes, see ``label_by_size``.

        Returns
        -------
        v : pd.DataFrame
            appends an extra column to ``v`` indicating component membership.

        """

        from scipy.sparse.csgraph import connected_components

        # create cs graph
        cs_g = self.return_cs_graph()

        # find components
        labels = connected_components(cs_g, directed=directed, connection=connection)[1]

        # append cp column to v
        self.v[col_name] = labels

        # if indicated, consolidate singular components and label by size
        if consolidate_singles:
            cp_counts = self.v[col_name].value_counts()

            # if there are singular components
            f1cp = len(cp_counts) - np.searchsorted(cp_counts.values[::-1], 2)
            rndic = {j: i + 1 for i, j in enumerate(cp_counts.index[:f1cp])}
            rndic.update({i: 0 for i in cp_counts.index[f1cp:]})

            # relabel cp column
            self.v[col_name] = self.v[col_name].apply(lambda x: rndic[x])

        # if indicated, label by size
        elif label_by_size:
            cp_counts = self.v[col_name].value_counts()
            rndic = {j: i for i, j in enumerate(cp_counts.index)}

            # relabel cp column
            self.v[col_name] = self.v[col_name].apply(lambda x: rndic[x])

    def append_binning_labels_v(self, col, col_name, bins=10, log_bins=False, floor=False, return_bin_edges=False):
        """Append a column with binning labels of the values in ``v[col]``.

        Append a column ``col_name`` to ``v`` with the indices of the bins to
        which each value in ``v[col]`` belongs to.

        If ``bins`` is an int, it determines the number of bins to create. If
        ``log_bins`` is True, this number determines the (approximate) number
        of bins to create for each magnitude. For linear bins, it is the number
        of bins for the whole range of values. If ``floor`` is set True, the
        bin edges are floored to the closest integer. If ``return_bin_edges``
        is set True, the created bin edges are returned.

        If ``bins`` is a sequence, it defines the bin edges, including the
        rightmost edge, allowing for non-uniform bin widths.

        See ``np.digitize`` for details.

        Parameters
        ----------
        col : int or str
            A column name of ``v``, whose corresponding values are binned and
            labelled.

        col_name : str
            The column name for the created labels.

        bins : int or array_lke, optional (default=10)
            If ``bins`` is an int, it determines the number of bins to create.
            If ``log_bins`` is True, this number determines the (approximate)
            number of bins to create for each magnitude. For linear bins, it is
            the number of bins for the whole range of values. If ``bins`` is a
            sequence, it defines the bin edges, including the rightmost edge,
            allowing for non-uniform bin widths.

        log_bins : bool, optional (default=False)
            Whether to use logarithmically or linearly spaced bins.

        floor : bool, optional (default=False)
            Whether to floor the bin edges to the closest integers.

        return_bin_edges : bool, optional (default=False)
            Whether to return the bin edges.

        Returns
        -------
        v : pd.DataFrame
            Appends an extra column ``col_name`` to ``v`` with the binning
            labels.

        bin_edges : np.ndarray
            Optionally, return the created bin edges.

        Examples
        --------
        First, we need a node table:

        >>> import pandas as pd
        >>> import deepgraph as dg
        >>> v = pd.DataFrame({'time': [1,2,12,105,899]})
        >>> g = dg.DeepGraph(v)

        >>> g.v
           time
        0     1
        1     2
        2    12
        3   105
        4   899

        Binning time values with default arguments:

        >>> bin_edges = g.append_binning_labels_v('time', 'time_l',
        ...                                       return_bin_edges=True)

        >>> bin_edges
        array([   1.        ,  100.77777778,  200.55555556,  300.33333333,
                400.11111111,  499.88888889,  599.66666667,  699.44444444,
                799.22222222,  899.        ])

        >>> g.v
           time  time_l
        0     1       1
        1     2       1
        2    12       1
        3   105       2
        4   899      10

        Binning time values with logarithmically spaced bins:

        >>> bin_edges = g.append_binning_labels_v('time', 'time_l', bins=5,
        ...                                       log_bins=True,
        ...                                       return_bin_edges=True)

        >>> bin_edges
        array([   1.        ,    1.62548451,    2.64219989,    4.29485499,
                  6.98122026,   11.34786539,   18.44577941,   29.9833287 ,
                 48.73743635,   79.22194781,  128.77404899,  209.32022185,
                340.24677814,  553.06586728,  899.        ])

        >>> g.v
           time  time_l
        0     1       1
        1     2       2
        2    12       6
        3   105      10
        4   899      15

        Binning time values with logarithmically spaced bins (floored):

        >>> bin_edges = g.append_binning_labels_v('time', 'time_l', bins=5,
        ...                                       log_bins=True, floor=True,
        ...                                       return_bin_edges=True)

        >>> bin_edges
        array([   1.,    2.,    4.,    6.,   11.,   18.,   29.,   48.,   79.,
                128.,  209.,  340.,  553.,  899.])

        >>> g.v
           time  time_l
        0     1       1
        1     2       2
        2    12       5
        3   105       9
        4   899      14

        """

        x = self.v[col]

        # create bins
        if _is_array_like(bins):
            bin_edges = bins
        else:
            bin_edges = _create_bin_edges(x, bins, log_bins, floor)

        self.v[col_name] = np.digitize(x, bin_edges)

        if return_bin_edges:
            return bin_edges

    def append_datetime_categories_v(self, col="time", timeofday=None, met_season=None):
        """Append datetime categories to ``v``.

        Appends a "time of the day" and/or a meteorological season to ``v``,
        based on a given datetime column ``col``.

        Parameters
        ----------
        col : str, optional (default='time')
            A column of ``v`` comprised of datetimes.

        timeofday : str, optional (default=None)
            If given, the time of the day is appended as a column with the
            label ``timeofday`` to ``v``. The time of the day is defined
            as::

                [00:06[ = 0 (night)
                [06:12[ = 1 (forenoon)
                [12:18[ = 2 (afternoon)
                [18:24] = 3 (evening)

        met_season : str, optional (default=None)
            If given, the modern mid-latitude meteorological season, see
            http://en.wikipedia.org/wiki/Season#Modern_mid-latitude_meteorological
            is appended as a column with the label
            ``met_season`` to ``v``. The season is defined as:

                [12:03[ = 0
                [03:06[ = 1
                [06:09[ = 2
                [09:12[ = 3

        Returns
        -------
        v : pd.DataFrame
            appends extra column(s) to ``v`` with datetime properties.

        """

        def _timeofday(datetimes):
            def categorize(hour):
                if hour < 6:
                    return 0
                elif hour >= 6 and hour < 12:
                    return 1
                elif hour >= 12 and hour < 18:
                    return 2
                elif hour >= 18 and hour <= 24:
                    return 3

            hour = datetimes.apply(lambda x: x.hour)
            timeofday = hour.apply(categorize).values
            return timeofday

        def _met_season(datetimes):
            def season(month):
                if month >= 12 or month < 3:
                    return 0
                elif month >= 3 and month < 6:
                    return 1
                elif month >= 6 and month < 9:
                    return 2
                elif month >= 9 and month < 12:
                    return 3

            month = datetimes.apply(lambda x: x.month)
            season = month.apply(season).values
            return season

        if timeofday:
            self.v[timeofday] = _timeofday(self.v[col])
            self.v[timeofday] = self.v[timeofday].astype("uint8")

        if met_season:
            self.v[met_season] = _met_season(self.v[col])
            self.v[met_season] = self.v[met_season].astype("uint8")

    def update_edges(self):
        """After removing nodes in ``v``, update ``e``.

        If you deleted rows from ``v``, you can remove all edges associated
        with the deleted nodes in ``e`` by calling this method.

        Returns
        -------
        e : pd.DataFrame
            update ``e``

        """

        # reduce edge table
        if hasattr(self, "e"):
            s = self.e.index.get_level_values(0)
            t = self.e.index.get_level_values(1)
            self.e = self.e.loc[(s.isin(self.v.index)) & (t.isin(self.v.index))]

    def filter_by_interval_v(self, col, interval, endpoint=True):
        """Keep only nodes in ``v`` with features of type ``col`` in
        ``interval``.

        Remove all nodes from ``v`` (and their corresponding edges in ``e``)
        with features of type ``col`` outside the interval given by a tuple of
        values. The endpoint is included, if ``endpoint`` is not set to False.

        Parameters
        ----------
        col : str or int
            A column name of ``v``, indicating the type of feature used in the
            filtering.

        interval : tuple
            A tuple of two values, (value, larger_value). All nodes outside the
            interval are removed.

        endpoint : bool, optional (default=True)
            False excludes the endpoint.

        Returns
        -------
        v : pd.DataFrame
            update ``v``

        e : pd.DataFrame
            update ``e``

        """

        # reduce node table
        if endpoint:
            self.v = self.v[(self.v[col] >= interval[0]) & (self.v[col] <= interval[1])]
        else:
            self.v = self.v[(self.v[col] >= interval[0]) & (self.v[col] < interval[1])]

        # reduce edge table
        if hasattr(self, "e"):
            self.update_edges()

    def filter_by_interval_e(self, col, interval, endpoint=True):
        """Keep only edges in ``e`` with relations of type ``col`` in
        ``interval``.

        Remove all edges from ``e`` with relations of type ``col`` outside the
        interval given by a tuple of values. The endpoint is included, if
        ``endpoint`` is not set to False.

        Parameters
        ----------
        col : str or int
            A column name of ``e``, indicating the type of relation used in the
            filtering.

        interval : tuple
            A tuple of two values, (value, larger_value). All edges outside the
            interval are removed.

        endpoint : bool, optional (default=True)
            False excludes the endpoint.

        Returns
        -------
        e : pd.DataFrame
            update ``e``

        """

        # reduce edge table
        if endpoint:
            self.e = self.e[(self.e[col] >= interval[0]) & (self.e[col] <= interval[1])]
        else:
            self.e = self.e[(self.e[col] >= interval[0]) & (self.e[col] < interval[1])]

    def filter_by_values_v(self, col, values):
        """Keep only nodes in ``v`` with features of type ``col`` in
        ``values``.

        Remove all nodes from ``v`` (and their corresponding edges in
        ``e``) with feature(s) of type ``col`` not in the list of features
        given by ``values``.

        Parameters
        ----------
        col : str or int
            A column name of ``v``, indicating the type of feature used in the
            filtering.

        values : object or array_like
            The value(s) indicating which nodes to keep.

        Returns
        -------
        v : pd.DataFrame
            update ``v``

        e : pd.DataFrame
            update ``e``

        """

        # reduce node table
        if not _is_array_like(values):
            values = [values]
        self.v = self.v[(self.v[col].isin(values))]

        # reduce edge table
        if hasattr(self, "e"):
            self.update_edges()

    def filter_by_values_e(self, col, values):
        """Keep only edges in ``e`` with relations of type ``col`` in
        ``values``.

        Remove all edges from ``e`` with relation(s) of type ``col`` not in the
        list of relations given by ``values``.

        Parameters
        ----------
        col : str or int
            A column name of ``e``, indicating the type of relation used in the
            filtering.

        values : object or array_like
            The value(s) indicating which edges to keep.

        Returns
        -------
        e : pd.DataFrame
            update ``e``

        """

        # reduce node table
        if not _is_array_like(values):
            values = [values]
        self.e = self.e[(self.e[col].isin(values))]

    def plot_2d(
        self,
        x,
        y,
        edges=False,
        C=None,
        C_split_0=None,
        kwds_scatter=None,
        kwds_quiver=None,
        kwds_quiver_0=None,
        ax=None,
    ):
        """Plot nodes and corresponding edges in 2 dimensions.

        Create a scatter plot of the nodes in ``v``, and optionally a quiver
        plot of the corresponding edges in ``e``.

        The xy-coordinates of the scatter plot are determined by the values of
        ``v[x]`` and ``v[y]``, where ``x`` and ``y`` are column names of ``v``
        (the arrow's coordinates are determined automatically).

        In order to map colors to the arrows, either ``C`` or ``C_split_0``
        can be be passed, an array of the same length as ``e``. Passing ``C``
        creates a single quiver plot (qu). Passing ``C_split_0`` creates two
        separate quiver plots, one for all edges where ``C_split_0`` == 0
        (qu_0), and one for all other edges (qu). By default, the arrows of
        qu_0 have no head, indicating "undirected" edges. This can be useful,
        for instance, when ``C_split_0`` represents an array of temporal
        distances.

        In order to control the plotting parameters of the scatter, quiver
        and/or quiver_0 plots, one may pass keyword arguments by setting
        ``kwds_scatter``, ``kwds_quiver`` and/or ``kwds_quiver_0``.

        Can be used iteratively by passing ``ax``.

        Parameters
        ----------
        x : int or str
            A column name of ``v``, determining the x-coordinates of the
            scatter plot of nodes.

        y : int or str
            A column name of ``v``, determining the y-coordinates of the
            scatter plot of nodes.

        edges : bool, optional (default=True)
            Whether to create a quiver plot (2-D field of arrows) of the edges
            between the nodes.

        C : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. Has no effect if ``C_split_0`` is passed as
            an argument.

        C_split_0 : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. If this parameter is passed, ``C`` has no
            effect, and two separate quiver plots are created (qu and qu_0).

        kwds_scatter : dict, optional (default=None)
            kwargs to be passed to scatter.

        kwds_quiver : dict, optional (default=None)
            kwargs to be passed to quiver (qu).

        kwds_quiver_0 : dict, optional (default=None)
            kwargs to be passed to quiver (qu_0). Only has an effect if
            ``C_split_0`` has been set.

        ax : matplotlib axes object, optional (default=None)
            An axes instance to use.

        Returns
        -------
        obj : dict
            If ``C_split_0`` has been passed, return a dict of matplotlib
            objects with the following keys: ['fig', 'ax', 'pc', 'qu', 'qu_0'].
            Otherwise, return a dict with keys: ['fig', 'ax', 'pc', 'qu'].

        Notes
        -----
        When passing ``C_split_0``, the color of the arrows in qu_0 can be set
        by passing the keyword argument `color` to ``kwds_quiver_0``. The color
        of the arrows in qu, however, are determined by ``C_split_0``.

        The default drawing order is set to:
        1. quiver_0 (zorder=1)
        2. quiver (zorder=2)
        3. scatter (zorder=3)
        This order can be changed by setting the ``zorder`` in
        ``kwds_quiver_0``, ``kwds_quiver`` and/or ``kwds_scatter``.
        See also http://matplotlib.org/examples/pylab_examples/zorder_demo.html

        See also
        --------
        plot_2d_generator
        plot_3d
        plot_map
        plot_map_generator

        """

        return self._plot_2d(
            is_map=False,
            x=x,
            y=y,
            edges=edges,
            C=C,
            C_split_0=C_split_0,
            kwds_scatter=kwds_scatter,
            kwds_quiver=kwds_quiver,
            kwds_quiver_0=kwds_quiver_0,
            kwds_basemap=None,
            ax=ax,
            m=None,
        )

    def plot_2d_generator(
        self,
        x,
        y,
        by,
        edges=False,
        C=None,
        C_split_0=None,
        kwds_scatter=None,
        kwds_quiver=None,
        kwds_quiver_0=None,
        passable_ax=False,
    ):
        """Plot nodes and corresponding edges by groups.

        Create a generator of scatter plots of the nodes in ``v``, split in
        groups by ``v``.groupby(``by``). If edges is set True, also create a
        quiver plot of each group's corresponding edges.

        The xy-coordinates of the scatter plots are determined by the values of
        ``v[x]`` and ``v[y]``, where ``x`` and ``y`` are column names of ``v``
        (the arrow's coordinates are determined automatically).

        In order to map colors to the arrows, either ``C`` or ``C_split_0``
        can be be passed, an array of the same length as ``e``. Passing ``C``
        creates a single quiver plot (qu). Passing ``C_split_0`` creates two
        separate quiver plots, one for all edges where ``C_split_0`` == 0
        (qu_0), and one for all other edges (qu). By default, the arrows of
        qu_0 have no head, indicating "undirected" edges. This can be useful,
        for instance, when ``C_split_0`` represents an array of temporal
        distances.

        When mapping colors to arrows by setting ``C`` (or ``C_split_0``),
        `clim` is automatically set to the min and max values of the entire
        array. In case one wants clim to be set to min and max values for each
        group's colors, one may explicitly pass `clim` = None to
        ``kwds_quiver``.

        The same behaviour occurs when passing a sequence of ``g.n`` Numbers as
        colors `c` to ``kwds_scatter``. In that case, `vmin` and `vmax` are
        automatically set to `c`.min() and `c`.max() of all nodes. Explicitly
        setting `vmin` and `vmax` to `None`, the min and max values of the
        groups' color arrays are used.

        In order to control the plotting parameters of the scatter, quiver
        and/or quiver_0 plots, one may pass keyword arguments by setting
        ``kwds_scatter``, ``kwds_quiver`` and/or ``kwds_quiver_0``.

        If ``passable_ax`` is True, create a generator of functions. Each
        function takes a matplotlib axes object as input, and returns a
        scatter/quiver plot.

        Parameters
        ----------
        x : int or str
            A column name of ``v``, determining the x-coordinates of the
            scatter plot of nodes.

        y : int or str
            A column name of ``v``, determining the y-coordinates of the
            scatter plot of nodes.

        by : array_like
            Column name(s) of ``v``, determining the groups to create plots of.

        edges : bool, optional (default=True)
            Whether to create a quiver plot (2-D field of arrows) of the edges
            between the nodes.

        C : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. Has no effect if ``C_split_0`` is passed as
            an argument.

        C_split_0 : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. If this parameter is passed, ``C`` has no
            effect, and two separate quiver plots are created (qu and qu_0).

        kwds_scatter : dict, optional (default=None)
            kwargs to be passed to scatter.

        kwds_quiver : dict, optional (default=None)
            kwargs to be passed to quiver (qu).

        kwds_quiver_0 : dict, optional (default=None)
            kwargs to be passed to quiver (qu_0). Only has an effect if
            ``C_split_0`` has been set.

        passable_ax : bool, optional (default=False)
            If True, return a generator of functions. Each function takes a
            matplotlib axes object as input, and returns a dict of matplotlib
            objects.

        Returns
        -------
        obj : generator
            If ``C_split_0`` has been passed, return a generator of dicts of
            matplotlib objects with the following keys: ['fig', 'ax', 'pc',
            'qu', 'qu_0', 'group']. Otherwise, return a generator of dicts
            with keys: ['fig', 'ax', 'pc', 'qu', 'group'].
            If ``passable_ax`` is True, return a generator of functions. Each
            function takes a matplotlib axes object as input, and returns a
            dict as described above.

        Notes
        -----
        When passing ``C_split_0``, the color of the arrows in qu_0 can be set
        by passing the keyword argument `color` to ``kwds_quiver_0``. The color
        of the arrows in qu, however, are determined by ``C_split_0``.

        The default drawing order is set to:
        1. quiver_0 (zorder=1)
        2. quiver (zorder=2)
        3. scatter (zorder=3)
        This order can be changed by setting the ``zorder`` in
        ``kwds_quiver_0``, ``kwds_quiver`` and/or ``kwds_scatter``.
        See also http://matplotlib.org/examples/pylab_examples/zorder_demo.html

        See also
        --------
        append_binning_labels_v
        plot_2d
        plot_3d
        plot_map
        plot_map_generator

        """

        return self._plot_2d_generator(
            is_map=False,
            x=x,
            y=y,
            by=by,
            edges=edges,
            C=C,
            C_split_0=C_split_0,
            kwds_basemap=None,
            kwds_scatter=kwds_scatter,
            kwds_quiver=kwds_quiver,
            kwds_quiver_0=kwds_quiver_0,
            passable_ax=passable_ax,
        )

    def plot_map(
        self,
        lon,
        lat,
        edges=False,
        C=None,
        C_split_0=None,
        kwds_basemap=None,
        kwds_scatter=None,
        kwds_quiver=None,
        kwds_quiver_0=None,
        ax=None,
        m=None,
    ):
        """Plot nodes and corresponding edges on a basemap.

        Create a scatter plot of the nodes in ``v`` and optionally a quiver
        plot of the corresponding edges in ``e`` on a
        ``mpl_toolkits.basemap.Basemap`` instance.

        The coordinates of the scatter plot are determined by the node's
        longitudes and latitudes (in degrees): ``v[lon]`` and ``v[lat]``, where
        ``lon`` and ``lat`` are column names of ``v`` (the arrow's coordinates
        are determined automatically).

        In order to map colors to the arrows, either ``C`` or ``C_split_0``
        can be be passed, an array of the same length as ``e``. Passing ``C``
        creates a single quiver plot (qu). Passing ``C_split_0`` creates two
        separate quiver plots, one for all edges where ``C_split_0`` == 0
        (qu_0), and one for all other edges (qu). By default, the arrows of
        qu_0 have no head, indicating "undirected" edges. This can be useful,
        for instance, when ``C_split_0`` represents an array of temporal
        distances.

        In order to control the parameters of the basemap, scatter, quiver
        and/or quiver_0 plots, one may pass keyword arguments by setting
        ``kwds_basemap``, ``kwds_scatter``, ``kwds_quiver`` and/or
        ``kwds_quiver_0``.

        Can be used iteratively by passing ``ax`` and/or ``m``.

        Parameters
        ----------
        lon : int or str
            A column name of ``v``. The corresponding values must be longitudes
            in degrees.

        lat : int or str
            A column name of ``v``. The corresponding values must be latitudes
            in degrees.

        edges : bool, optional (default=True)
            Whether to create a quiver plot (2-D field of arrows) of the edges
            between the nodes.

        C : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. Has no effect if ``C_split_0`` is passed as
            an argument.

        C_split_0 : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. If this parameter is passed, ``C`` has no
            effect, and two separate quiver plots are created (qu and qu_0).

        kwds_basemap : dict, optional (default=None)
            kwargs passed to basemap.

        kwds_scatter : dict, optional (default=None)
            kwargs to be passed to scatter.

        kwds_quiver : dict, optional (default=None)
            kwargs to be passed to quiver (qu).

        kwds_quiver_0 : dict, optional (default=None)
            kwargs to be passed to quiver (qu_0). Only has an effect if
            ``C_split_0`` has been set.

        ax : matplotlib axes object, optional (default=None)
            An axes instance to use.

        m : Basemap object, optional (default=None)
            A mpl_toolkits.basemap.Basemap instance to use.

        Returns
        -------
        obj : dict
            If ``C_split_0`` has been passed, return a dict of matplotlib
            objects with the following keys: ['fig', 'ax', 'm', 'pc', 'qu',
            'qu_0']. Otherwise, return a dict with keys: ['fig', 'ax', 'm',
            'pc', 'qu'].

        Notes
        -----
        When passing ``C_split_0``, the color of the arrows in qu_0 can be set
        by passing the keyword argument `color` to ``kwds_quiver_0``. The color
        of the arrows in qu, however, are determined by ``C_split_0``.

        The default drawing order is set to:
        1. quiver_0 (zorder=1)
        2. quiver (zorder=2)
        3. scatter (zorder=3)
        This order can be changed by setting the ``zorder`` in
        ``kwds_quiver_0``, ``kwds_quiver`` and/or ``kwds_scatter``.
        See also http://matplotlib.org/examples/pylab_examples/zorder_demo.html

        See also
        --------
        plot_map_generator
        plot_2d
        plot_2d_generator
        plot_3d

        """

        return self._plot_2d(
            is_map=True,
            x=lon,
            y=lat,
            edges=edges,
            C=C,
            C_split_0=C_split_0,
            kwds_basemap=kwds_basemap,
            kwds_scatter=kwds_scatter,
            kwds_quiver=kwds_quiver,
            kwds_quiver_0=kwds_quiver_0,
            ax=ax,
            m=m,
        )

    def plot_map_generator(
        self,
        lon,
        lat,
        by,
        edges=False,
        C=None,
        C_split_0=None,
        kwds_basemap=None,
        kwds_scatter=None,
        kwds_quiver=None,
        kwds_quiver_0=None,
        passable_ax=False,
    ):
        """Plot nodes and corresponding edges by groups, on basemaps.

        Create a generator of scatter plots of the nodes in ``v``, split in
        groups by ``v``.groupby(``by``), on a ``mpl_toolkits.basemap.Basemap``
        instance. If edges is set True, also create a quiver plot of each
        group's corresponding edges.

        The coordinates of the scatter plots are determined by the node's
        longitudes and latitudes (in degrees): ``v[lon]`` and ``v[lat]``, where
        ``lon`` and ``lat`` are column names of ``v`` (the arrow's coordinates
        are determined automatically).

        In order to map colors to the arrows, either ``C`` or ``C_split_0``
        can be be passed, an array of the same length as ``e``. Passing ``C``
        creates a single quiver plot (qu). Passing ``C_split_0`` creates two
        separate quiver plots, one for all edges where ``C_split_0`` == 0
        (qu_0), and one for all other edges (qu). By default, the arrows of
        qu_0 have no head, indicating "undirected" edges. This can be useful,
        for instance, when ``C_split_0`` represents an array of temporal
        distances.

        When mapping colors to arrows by setting ``C`` (or ``C_split_0``),
        `clim` is automatically set to the min and max values of the entire
        array. In case one wants clim to be set to min and max values for each
        group's colors, one may explicitly pass `clim` = None to
        ``kwds_quiver``.

        The same behaviour occurs when passing a sequence of ``g.n`` Numbers as
        colors `c` to ``kwds_scatter``. In that case, `vmin` and `vmax` are
        automatically set to `c`.min() and `c`.max() of all nodes. Explicitly
        setting `vmin` and `vmax` to `None`, the min and max values of the
        groups' color arrays are used.

        In order to control the parameters of the basemap, scatter, quiver
        and/or quiver_0 plots, one may pass keyword arguments by setting
        ``kwds_basemap``, ``kwds_scatter``, ``kwds_quiver`` and/or
        ``kwds_quiver_0``.

        If ``passable_ax`` is True, create a generator of functions. Each
        function takes a matplotlib axes object (and/or a Basemap object) as
        input, and returns a scatter/quiver plot.

        Parameters
        ----------
        lon : int or str
            A column name of ``v``. The corresponding values must be longitudes
            in degrees.

        lat : int or str
            A column name of ``v``. The corresponding values must be latitudes
            in degrees.

        by : array_like
            Column name(s) of ``v``, determining the groups to create plots of.

        edges : bool, optional (default=True)
            Whether to create a quiver plot (2-D field of arrows) of the edges
            between the nodes.

        C : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. Has no effect if ``C_split_0`` is passed as
            an argument.

        C_split_0 : array_like, optional (default=None)
            An optional array used to map colors to the arrows. Must have the
            same length es ``e``. If this parameter is passed, ``C`` has no
            effect, and two separate quiver plots are created (qu and qu_0).

        kwds_basemap : dict, optional (default=None)
            kwargs passed to basemap.

        kwds_scatter : dict, optional (default=None)
            kwargs to be passed to scatter.

        kwds_quiver : dict, optional (default=None)
            kwargs to be passed to quiver (qu).

        kwds_quiver_0 : dict, optional (default=None)
            kwargs to be passed to quiver (qu_0). Only has an effect if
            ``C_split_0`` has been set.

        passable_ax : bool, optional (default=False)
            If True, return a generator of functions. Each function takes a
            matplotlib axes object (and/or a Basemap object) as input, and
            returns a dict of matplotlib objects.

        Returns
        -------
        obj : generator
            If ``C_split_0`` has been passed, return a generator of dicts of
            matplotlib objects with the following keys: ['fig', 'ax', 'm',
            'pc', 'qu', 'qu_0', 'group']. Otherwise, return a generator of
            dicts with keys: ['fig', 'ax', 'm', 'pc', 'qu', 'group'].
            If ``passable_ax`` is True, return a generator of functions. Each
            function takes a matplotlib axes object (and/or a Basemap object)
            as input, and returns a dict as described above.

        Notes
        -----
        When passing ``C_split_0``, the color of the arrows in qu_0 can be set
        by passing the keyword argument `color` to ``kwds_quiver_0``. The color
        of the arrows in qu, however, are determined by ``C_split_0``.

        The default drawing order is set to:
        1. quiver_0 (zorder=1)
        2. quiver (zorder=2)
        3. scatter (zorder=3)
        This order can be changed by setting the ``zorder`` in
        ``kwds_quiver_0``, ``kwds_quiver`` and/or ``kwds_scatter``.
        See also http://matplotlib.org/examples/pylab_examples/zorder_demo.html

        See also
        --------
        append_binning_labels_v
        plot_map
        plot_2d
        plot_2d_generator
        plot_3d

        """

        return self._plot_2d_generator(
            is_map=True,
            x=lon,
            y=lat,
            by=by,
            edges=edges,
            C=C,
            C_split_0=C_split_0,
            kwds_basemap=kwds_basemap,
            kwds_scatter=kwds_scatter,
            kwds_quiver=kwds_quiver,
            kwds_quiver_0=kwds_quiver_0,
            passable_ax=passable_ax,
        )

    def plot_3d(self, x, y, z, edges=False, kwds_scatter=None, kwds_quiver=None, ax=None):
        """Work in progress!

        experimental, quiver3D scaling?

        See also
        --------
        plot_2d
        plot_2d_generator
        plot_map
        plot_map_generator

        """

        # set kwds
        if kwds_scatter is None:
            kwds_scatter = {}
        if kwds_quiver is None:
            kwds_quiver = {}

        from mpl_toolkits.mplot3d.axes3d import Axes3D  # @UnusedImport

        # return dict of matplotlib objects
        obj = {}

        # create figure and axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        # create PathCollection by scatter
        x, y, z = (self.v[x], self.v[y], self.v[z])
        pc = plt.scatter(x, y, zs=z, zdir="z", **kwds_scatter)

        obj["pc"] = pc

        # draw edges as arrows
        if edges is True:
            # get unique indices of edgeed nodes
            s = self.e.index.get_level_values(level=0).values
            t = self.e.index.get_level_values(level=1).values

            # xy position of sources, delta xy
            xs, ys, zs = (x.loc[s].values, y.loc[s].values, z.loc[s].values)
            xt, yt, zt = (x.loc[t].values, y.loc[t].values, z.loc[t].values)

            # upcast dtypes
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            zs = np.array(zs, dtype=float)
            xt = np.array(xt, dtype=float)
            yt = np.array(yt, dtype=float)
            zt = np.array(zs, dtype=float)

            dx = xt - xs
            dy = yt - ys
            dz = zt - zs

            qu = ax.quiver(xs, ys, zs, dx, dy, dz, **kwds_quiver)

            obj["qu"] = qu

        return obj

    def plot_rects_label_numeric(self, label, xl, xr, colors=None, ax=None, **kwargs):
        """Work in progress!

        Plot rectangles given by `label_xl_xr_df`.

        Parameters
        ----------
        label_xl_xr_df : pd.DataFrame
            A pandas.DataFrame object with three columns, the first column
            containing the categorical variable (labels),
            the second column containing the left x values, the
            third column the right x values of the boxes.
        kwargs : keywords
            kwargs to pass to matplotlib.pyplot.vlines

        Returns
        -------
        obj : dict of matplotlib objects
            Keys are ['fig', 'ax', 'vlines']

        See also
        --------
        plot_rects_numeric_numeric

        """

        from matplotlib.collections import PolyCollection

        v = self.v[[label, xl, xr]]

        # return dict of matplotlib objects
        obj = {}

        # create figure and axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        obj["fig"] = fig
        obj["ax"] = ax

        # include colors in dataframe for sorting
        if colors is not None:
            v["color"] = colors
        else:
            v["color"] = 1

        # rectangle coordinates
        xl, xr = (v[xl].values, v[xr].values)
        widths = xr - xl
        yb = v[label] - 0.6
        heights = np.ones(len(xr)) * 1.2

        recs = []
        for x, y, width, height in zip(xl, yb, widths, heights):
            recs.append(((x, y), (x, y + height), (x + width, y + height), (x + width, y)))

        # create poly collection of rectangles
        c = PolyCollection(recs, **kwargs)

        # set colors
        c.set_array(v["color"])
        obj["c"] = c

        # add PolyCollection
        ax.add_collection(c)

        # set yticklabels
        positions = np.arange(v[label].max() + 1)
        ax.set_yticks(positions)

        # set x/y lims
        dx = 0.05 * (xr.max() - xl.min())
        dy = 0.05 * (yb.max() + 1.2 - yb.min())
        ax.set_xlim((xl.min() - dx, xr.max() + dx))
        ax.set_ylim((yb.min() - dy, yb.max() + 1.2 + dy))

        return obj

    def plot_rects_numeric_numeric(self, yb, yt, xl, xr, colors=None, ax=None, **kwargs):
        """Work in progress!

        Create a raster plot of all components given by `yb_yt_xl_xr_df`.

        Parameters
        ----------
        yb_yt_xl_xr_df : pd.DataFrame
            A pandas.DataFrame object with four columns
        kwargs : keywords
            kwargs to pass to matplotlib.pyplot.vlines

        Returns
        -------
        obj : dict of matplotlib objects
            Keys are ['fig', 'ax', 'vlines']

        See also
        --------
        box_label_numeric

        """

        from matplotlib.collections import PolyCollection

        v = self.v[[yb, yt, xl, xr]]

        # return dict of matplotlib objects
        obj = {}

        # create figure and axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        obj["fig"] = fig
        obj["ax"] = ax

        # include colors in dataframe for sorting
        if colors is not None:
            v["color"] = colors
        else:
            v["color"] = 1

        # rectangle coordinates
        xl, xr = (v[xl].values, v[xr].values)
        widths = xr - xl
        yb, yt = (v[yb], v[yt])
        heights = yt - yb

        recs = []
        for x, y, width, height in zip(xl, yb, widths, heights):
            recs.append(((x, y), (x, y + height), (x + width, y + height), (x + width, y)))

        # create poly collection of rectangles
        c = PolyCollection(recs, **kwargs)

        # set colors
        c.set_array(v["color"])

        obj["c"] = c

        # add PolyCollection
        ax.add_collection(c)

        # set x/y lims
        dx = 0.05 * (xr.max() - xl.min())
        dy = 0.05 * (yt.max() - yb.min())
        ax.set_xlim((xl.min() - dx, xr.max() + dx))
        ax.set_ylim((yb.min() - dy, yt.max() + dy))

        return obj

    def plot_raster(self, label, time="time", ax=None, **kwargs):
        """Work in progress!

        Create a raster plot of all nodes given by `supernode_id_time_df`.

        Parameters
        ----------
        supernode_id_time_df : pd.DataFrame
            A pandas.DataFrame object with two columns, the first column
            containing the labels, the second column containing the
            times of the nodes.
        kwargs : keywords
            kwargs to pass to matplotlib.pyplot.vlines

        Returns
        -------
        obj : dict of matplotlib objects
            Keys are ['fig', 'ax', 'vlines']

        """

        # return dict of matplotlib objects
        obj = {}

        # create figure and axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        obj["fig"] = fig
        obj["ax"] = ax

        # sort by labels
        v = self.v[[label, time]].sort_values(label)

        # unique labels
        labels = v[label].unique()

        # create raster plot
        vlines = []
        for i, l in enumerate(labels):
            vlines.append(ax.vlines(v[v[label] == l][time].values, i + 0.5, i + 1.5, **kwargs))

        obj["vlines"] = vlines

        # set labels as yticklabels
        positions = np.arange(1, len(labels) + 1)
        labels = labels

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)

        # set x/y lims
        dx = 0.05 * (v[time].max() - v[time].min())
        dy = 0.05 * (positions.max() - positions.min())
        ax.set_xlim((v[time].min() - dx, v[time].max() + dx))
        ax.set_ylim((positions.min() - dy, positions.max() + dy))

        # set x/y label
        ax.set_xlabel("time")
        ax.set_ylabel(label)

        return obj

    @staticmethod
    def plot_hist(x, bins=10, log_bins=False, density=False, floor=False, ax=None, **kwargs):
        """Plot a histogram (or pdf) of x.

        Compute and plot the histogram (or probability density) of x. Keyword
        arguments are passed to plt.plot. See parameters and ``np.histogram``
        for details.

        Parameters
        ----------
        x : array_like
            The data from which a frequency distribution is plot.

        bins : int or array_like, optional (default=10)
            If ``bins`` is an int, it determines the number of bins to create.
            If ``log_bins`` is True, this number determines the (approximate)
            number of bins to create for each magnitude. For linear bins, it is
            the number of bins for the whole range of values. If ``bins`` is a
            sequence, it defines the bin edges, including the rightmost edge,
            allowing for non-uniform bin widths.

        log_bins : bool, optional (default=False)
            Whether to use logarithmically or linearly spaced bins.

        density : bool, optional (default=False)
            If False, the result will contain the number of samples in each
            bin.  If True, the result is the value of the probability *density*
            function at the bin, normalized such that the *integral* over the
            range is 1. Note that the sum of the histogram values will not be
            equal to 1 unless bins of unity width are chosen; it is not a
            probability *mass* function.

        floor : bool, optional (default=False)
            Whether to floor the bin edges to the closest integers. Only has an
            effect if ``bins`` is an int.

        ax : matplotlib axes object, optional (default=None)
            An axes instance to use.

        Returns
        -------
        ax : matplotlib axes object
            A matplotlib axes instance.

        hist : np.ndarray
            The values of the histogram. See ``density``.

        bin_edges : np.ndarray
            The edges of the bins.

        """

        # create bins
        if _is_array_like(bins):
            bin_edges = bins
        else:
            bin_edges = _create_bin_edges(x, bins, log_bins, floor)

        # counts and bin_centers
        hist, _ = np.histogram(x, bin_edges, density=density)
        hist = hist.astype(float)
        hist[hist == 0] = np.nan
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

        # plot
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(bin_centers, hist, **kwargs)

        # set scales
        if log_bins:
            ax.set_xscale("log")

        return ax, hist, bin_edges

    @staticmethod
    def plot_logfile(logfile):
        """Plot a logfile.

        Plot a benchmark logfile created by ``create_edges`` or
        ``create_edges_ft``.

        Parameters
        ----------
        logfile : str
            The filename of the logfile.

        Returns
        -------
        obj : dict
            Depending on the logfile, return a dict of matplotlib objects with
            a subset of the following keys: ['fig', 'ax', 'pc_n', 'pc_e',
            'cb_n', 'cb_e']

        """

        # load data from log file
        logfile = np.loadtxt(logfile)

        # return dict of matplotlib objects
        obj = {}

        #     0            1            2            3
        # exceeded | nr.of pairs | nr.of edges | comp.time

        # partition by non-/exceeded max_pairs
        log_n = logfile[logfile[:, 0] == 0]
        log_e = logfile[logfile[:, 0] == 1]

        fig, ax = plt.subplots()

        obj["fig"] = fig
        obj["ax"] = ax

        # scatter normal iterations
        pc_n = ax.scatter(
            log_n[:, 1], log_n[:, 3], s=20, c=np.log10(log_n[:, 2] + 1), marker="o", label="normal", edgecolors="none"
        )

        obj["pc_n"] = pc_n

        # scatter max_pair exceeded iterations
        pc_e = ax.scatter(
            log_e[:, 1],
            log_e[:, 3],
            s=30,
            c=np.log10(log_e[:, 2] + 1),
            cmap="gist_earth",
            marker="D",
            label="max_pairs exceeded",
        )

        obj["pc_e"] = pc_e

        msg = "iterations: {:d} | total time: {:.2f}s | total edges: {:d}"
        ax.set_title(msg.format(len(logfile), logfile[:, 3].sum(), int(logfile[:, 2].sum())))
        ax.set_xlabel("nr.of pairs")
        ax.set_ylabel("comp.time (s)")
        ax.set_xscale("log")
        ax.legend(loc=2)
        ax.grid()

        if len(log_e) == 0:
            cb_n = fig.colorbar(pc_n, fraction=0.03)
            cb_n.set_label("log10(n_edges) (normal)")

            fig.tight_layout()

            obj["cb_n"] = cb_n

        elif len(log_n) == 0:
            cb_e = fig.colorbar(pc_e, fraction=0.03)
            cb_e.set_label("log10(n_edges) (exceeded)")

            fig.tight_layout()

            obj["cb_e"] = cb_e

        else:
            cb_e = fig.colorbar(pc_e, fraction=0.03)
            cb_n = fig.colorbar(pc_n, fraction=0.03)
            cb_n.set_label("log10(n_edges) (normal)")
            cb_e.set_label("log10(n_edges) (exceeded)")

            fig.tight_layout()

            obj["cb_n"] = cb_n
            obj["cb_e"] = cb_e

        return obj

    @property
    def n(self):
        """The number of nodes"""
        if hasattr(self, "v"):
            if isinstance(self.v, pd.HDFStore):
                if len(self.v.keys()) == 1:
                    n = self.v.get_storer(self.v.keys()[0]).nrows
                else:
                    n = "NA"
            else:
                n = len(self.v)
        else:
            n = 0
        return n

    @property
    def m(self):
        """The number of edges"""
        if hasattr(self, "e"):
            m = len(self.e)
        else:
            m = 0
        return m

    @property
    def f(self):
        """Types of features and number of features of corresponding type."""
        if hasattr(self, "v"):
            if isinstance(self.v, pd.HDFStore):
                f = "NA"
            else:
                f = self.v.count()
        else:
            f = "there are no nodes"
        return f

    @property
    def r(self):
        """Types of relations and number of relations of corresponding type."""
        if hasattr(self, "e"):
            r = self.e.count()
        else:
            r = "there are no edges"
        return r

    def _plot_2d(
        self, is_map, x, y, edges, C, C_split_0, kwds_scatter, kwds_quiver, kwds_quiver_0, kwds_basemap, ax, m
    ):
        if is_map:
            from mpl_toolkits.basemap import Basemap

        # set kwds
        if kwds_basemap is None:
            kwds_basemap = {}
        else:
            kwds_basemap = kwds_basemap.copy()
        if kwds_scatter is None:
            kwds_scatter = {}
        else:
            kwds_scatter = kwds_scatter.copy()
        if kwds_quiver is None:
            kwds_quiver = {}
        else:
            kwds_quiver = kwds_quiver.copy()
        if kwds_quiver_0 is None:
            kwds_quiver_0 = {}
        else:
            kwds_quiver_0 = kwds_quiver_0.copy()

        # set draw order
        try:
            zorder_qu0 = kwds_quiver_0.pop("zorder")
        except KeyError:
            zorder_qu0 = 1
        try:
            zorder_qu = kwds_quiver.pop("zorder")
        except KeyError:
            zorder_qu = 2
        try:
            zorder_pc = kwds_scatter.pop("zorder")
        except KeyError:
            zorder_pc = 3

        # create dict for matplotlib objects
        obj = {}

        # create figure, axes (and basemap)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        obj["fig"] = fig
        obj["ax"] = ax

        if is_map and m is None:
            m = Basemap(ax=ax, **kwds_basemap)
            obj["m"] = m
        elif is_map and m is not None:
            obj["m"] = m

        # create PathCollection by scatter
        x_str = x
        y_str = y
        x, y = (self.v[x_str].values, self.v[y_str].values)
        if is_map:
            axm = m
            x, y = m(x, y)
            # bug in basemap, it changed dtypes
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
        else:
            axm = ax

        pc = axm.scatter(x, y, zorder=zorder_pc, **kwds_scatter)
        obj["pc"] = pc

        # draw edges as arrows
        if edges is True:
            # source- and target-indices
            s = self.e.index.get_level_values(level=0).values
            t = self.e.index.get_level_values(level=1).values

            # latlon position of sources and targets, vector components
            x, y = (self.v[x_str], self.v[y_str])
            if is_map:
                xs, ys = m(x.loc[s].values, y.loc[s].values)
                xt, yt = m(x.loc[t].values, y.loc[t].values)
            else:
                xs, ys = (x.loc[s].values, y.loc[s].values)
                xt, yt = (x.loc[t].values, y.loc[t].values)

            # upcast dtypes
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            xt = np.array(xt, dtype=float)
            yt = np.array(yt, dtype=float)

            dx = xt - xs
            dy = yt - ys

            # bug in basemap, changed dtypes
            if is_map:
                dx = np.array(dx, dtype=float)
                dy = np.array(dy, dtype=float)

            # create quiver plot
            if C_split_0 is not None:
                try:
                    color = kwds_quiver_0.pop("color")
                except KeyError:
                    color = "k"
                try:
                    headwidth = kwds_quiver_0.pop("headwidth")
                except KeyError:
                    headwidth = 1

                C = C_split_0

                qu_0 = axm.quiver(
                    xs[C == 0],
                    ys[C == 0],
                    dx[C == 0],
                    dy[C == 0],
                    color=color,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    headwidth=headwidth,
                    zorder=zorder_qu0,
                    **kwds_quiver_0,
                )

                qu = axm.quiver(
                    xs[C != 0],
                    ys[C != 0],
                    dx[C != 0],
                    dy[C != 0],
                    C[C != 0],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    zorder=zorder_qu,
                    **kwds_quiver,
                )

                obj["qu_0"] = qu_0
                obj["qu"] = qu

            elif C is not None:
                qu = axm.quiver(
                    xs, ys, dx, dy, C, angles="xy", scale_units="xy", scale=1, zorder=zorder_qu, **kwds_quiver
                )
                obj["qu"] = qu

            else:
                qu = axm.quiver(xs, ys, dx, dy, angles="xy", scale_units="xy", scale=1, zorder=zorder_qu, **kwds_quiver)
                obj["qu"] = qu

        return obj

    def _plot_2d_generator(
        self, is_map, x, y, by, edges, C, C_split_0, kwds_basemap, kwds_scatter, kwds_quiver, kwds_quiver_0, passable_ax
    ):
        if is_map:
            from mpl_toolkits.basemap import Basemap

        # set kwargs
        if kwds_basemap is None:
            kwds_basemap = {}
        else:
            kwds_basemap = kwds_basemap.copy()
        if kwds_scatter is None:
            kwds_scatter = {}
        else:
            kwds_scatter = kwds_scatter.copy()
        if kwds_quiver is None:
            kwds_quiver = {}
        else:
            kwds_quiver = kwds_quiver.copy()
        if kwds_quiver_0 is None:
            kwds_quiver_0 = {}
        else:
            kwds_quiver_0 = kwds_quiver_0.copy()

        # set draw order
        try:
            zorder_qu0 = kwds_quiver_0.pop("zorder")
        except KeyError:
            zorder_qu0 = 1
        try:
            zorder_qu = kwds_quiver.pop("zorder")
        except KeyError:
            zorder_qu = 2
        try:
            zorder_pc = kwds_scatter.pop("zorder")
        except KeyError:
            zorder_pc = 3

        # assert there's no color given in quiver kwds
        if kwds_quiver is not None:
            assert "color" not in kwds_quiver.keys(), "use 'C' or 'C_split_0' for setting the color of quiver!"

        # select v
        v = self.v[_flatten([x, y, by])]

        # store array_like kwargs in dataframe for filtering
        #     and change standard kwargs

        # set xlim/ylim for non map plots
        if not is_map:
            dx = (v[x].max() - v[x].min()) * 0.05
            dy = (v[y].max() - v[y].min()) * 0.05
            xlim = (v[x].min() - dx, v[x].max() + dx)
            ylim = (v[y].min() - dy, v[y].max() + dy)

        # scatter size
        try:
            pc_s = kwds_scatter.pop("s")
            v["pc_s"] = pc_s
        except KeyError:
            v["pc_s"] = 20

        # scatter color
        try:
            pc_c = kwds_scatter.pop("c")
            v["pc_c"] = pc_c
        except KeyError:
            pc_c = None
            v["pc_c"] = 1

        # scatter vmin/vmax -> entire min/max
        try:
            pc_vmin = kwds_scatter.pop("vmin")
        except KeyError:
            if pc_c is not None:
                try:
                    pc_vmin = pc_c.min()
                except AttributeError:
                    pc_vmin = None
            else:
                pc_vmin = None
        try:
            pc_vmax = kwds_scatter.pop("vmax")
        except KeyError:
            if pc_c is not None:
                try:
                    pc_vmax = pc_c.max()
                except AttributeError:
                    pc_vmax = None
            else:
                pc_vmax = None

        # quiver colors, and quiver clim -> entire min/max
        if edges is True:
            if C_split_0 is not None:
                e = pd.DataFrame(data={"Cqu0": C_split_0}, index=self.e.index)
                try:
                    qu_clim = kwds_quiver.pop("clim")
                except KeyError:
                    qu_clim = [C_split_0.min(), C_split_0.max()]

            elif C is not None:
                e = pd.DataFrame(data={"C": C}, index=self.e.index)
                try:
                    qu_clim = kwds_quiver.pop("clim")
                except KeyError:
                    qu_clim = [C.min(), C.max()]

            else:
                e = pd.DataFrame(index=self.e.index)
                qu_clim = None

            # change standard kwargs for quiver_0 at [C_split_0 == 0]
            try:
                color = kwds_quiver_0.pop("color")
            except KeyError:
                color = "k"
            try:
                qu_0_headwidth = kwds_quiver_0.pop("headwidth")
            except KeyError:
                qu_0_headwidth = 1

        else:
            e = None

        # generator loop
        x_str = x
        y_str = y
        gv = v.groupby(by)
        for labels, group in gv:

            def obj(ax=None, m=None):
                """Plot nodes and corresponding edges.

                See ``plot_2d_generator`` or ``plot_map_generator`` for
                details.

                Parameters
                ----------
                ax : matplotlib axes object, optional (default=None)
                    An axes instance to use.

                m : Basemap object, optional (default=None)
                    A mpl_toolkits.basemap.Basemap instance to use.

                Returns
                -------
                obj : dict
                    Return a dict of matplotlib objects.

                """

                # store group labels in obj
                obj = {"group": labels}

                # filter edges by group
                g = DeepGraph(group, e)
                g.update_edges()

                # create figure, axes (and basemap)
                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = ax.get_figure()

                obj["fig"] = fig
                obj["ax"] = ax

                if is_map and m is None:
                    m = Basemap(ax=ax, **kwds_basemap.copy())
                    obj["m"] = m
                elif is_map and m is not None:
                    obj["m"] = m
                else:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                # create PathCollection by scatter
                x, y = (g.v[x_str].values, g.v[y_str].values)
                if is_map:
                    axm = m
                    x, y = m(x, y)
                else:
                    axm = ax

                # need to change colors to list, in case they're not numbers
                pc = axm.scatter(
                    x,
                    y,
                    c=g.v.pc_c.values.tolist(),
                    s=g.v.pc_s.values,
                    vmin=pc_vmin,
                    vmax=pc_vmax,
                    zorder=zorder_pc,
                    **kwds_scatter,
                )
                obj["pc"] = pc

                # draw edges as arrows
                if edges is True:
                    # source- and target-indices
                    s = g.e.index.get_level_values(level=0).values
                    t = g.e.index.get_level_values(level=1).values

                    # xy position of sources and targets, vector components
                    x, y = (g.v[x_str], g.v[y_str])
                    if is_map:
                        xs, ys = m(x.loc[s].values, y.loc[s].values)
                        xt, yt = m(x.loc[t].values, y.loc[t].values)
                    else:
                        xs, ys = (x.loc[s].values, y.loc[s].values)
                        xt, yt = (x.loc[t].values, y.loc[t].values)

                    # upcast dtypes
                    xs = np.array(xs, dtype=float)
                    ys = np.array(ys, dtype=float)
                    xt = np.array(xt, dtype=float)
                    yt = np.array(yt, dtype=float)

                    dx = xt - xs
                    dy = yt - ys

                    # bug in basemap, changes dtypes
                    if is_map:
                        dx = np.array(dx, dtype=float)
                        dy = np.array(dy, dtype=float)

                    if C_split_0 is not None:
                        C = g.e.Cqu0.values

                        qu_0 = axm.quiver(
                            xs[C == 0],
                            ys[C == 0],
                            dx[C == 0],
                            dy[C == 0],
                            color=color,
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            headwidth=qu_0_headwidth,
                            zorder=zorder_qu0,
                            **kwds_quiver_0,
                        )

                        qu = axm.quiver(
                            xs[C != 0],
                            ys[C != 0],
                            dx[C != 0],
                            dy[C != 0],
                            C[C != 0],
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            clim=qu_clim,
                            zorder=zorder_qu,
                            **kwds_quiver,
                        )

                        obj["qu_0"] = qu_0
                        obj["qu"] = qu

                    elif C is not None:
                        C = g.e.C.values

                        qu = axm.quiver(
                            xs,
                            ys,
                            dx,
                            dy,
                            C,
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            clim=qu_clim,
                            zorder=zorder_qu,
                            **kwds_quiver,
                        )

                        obj["qu"] = qu

                    else:
                        qu = axm.quiver(
                            xs, ys, dx, dy, angles="xy", scale_units="xy", scale=1, zorder=zorder_qu, **kwds_quiver
                        )

                        obj["qu"] = qu

                return obj

            if passable_ax:
                yield obj
            else:
                yield obj()
