# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd

from deepgraph._triu_indices import _triu_indices, _union_of_indices, _reduce_triu_indices
from deepgraph.connector_selector_implementations import _ft_connector, _ft_selector
from deepgraph.connectors_and_selectors import CreatorFunction, Selector, Connector
from deepgraph.utils import _flatten, _is_array_like, _merge_dicts

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


def _initiate_create_edges(
    verbose, v, ft_feature, connectors, selectors, r_dtype_dic, transfer_features, no_transfer_rs, hdf_key
):
    # verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # reset all class attributes, necessary for consecutive calls
    # of create_edges
    CreatorFunction.reset("all")

    # ====================================================================
    #  INITIALIZATION OF CLASSES
    # ====================================================================

    # for single connectors/selectors, create lists
    if not connectors:
        connectors = []
    elif not _is_array_like(connectors):
        connectors = [connectors]
    if not selectors:
        selectors = []
    elif not _is_array_like(selectors):
        selectors = [selectors]

    # fast track
    if ft_feature is not None:
        if _ft_connector not in connectors:
            # if clause necesarry for consecutive calls
            connectors.append(_ft_connector)
        if "ft_selector" in selectors:
            selectors = [_ft_selector if s == "ft_selector" else s for s in selectors]
        else:
            selectors.insert(0, _ft_selector)

    # adjust dtype dict for relations
    if r_dtype_dic is None:
        r_dtype_dic = {}

    # initialize connectors
    for connector in connectors:
        Connector(connector)

    # initialize selectors
    for selector in selectors:
        Selector(selector)

    # flatten all attributes of CreatorFunction
    CreatorFunction.flatten_variables()

    # check for consistency of given functions
    CreatorFunction.assertions(v, r_dtype_dic)

    # ====================================================================
    #  EDGE COLUMNS AND DTYPES
    # ====================================================================

    # 1) dtype of node indices
    if isinstance(v, pd.HDFStore):
        assert hasattr(v.get_storer(hdf_key).group, "table"), "{} must be in table(t) format, not fixed(f).".format(
            hdf_key
        )
        v = v.select(hdf_key, start=0, stop=0)
    ndic = {"s": v.index.dtype, "t": v.index.dtype}

    # 2) output rs of connectors and selectors
    rs = _flatten(CreatorFunction.c_output_rs + CreatorFunction.s_output_rs)
    rdic = {}
    for r in rs:
        try:
            rdic[r] = r_dtype_dic[r]
        except KeyError:
            rdic[r] = None

    # 3) transfer features
    tfdic = {}
    for tf in transfer_features:
        tfdic[tf + "_s"] = v[tf].dtype
        tfdic[tf + "_t"] = v[tf].dtype

    # put all together
    coldtypedic = _merge_dicts(ndic, rdic, tfdic)

    # get rid of no_transfer relations
    if no_transfer_rs:
        if not _is_array_like(no_transfer_rs):
            no_transfer_rs = [no_transfer_rs]
        for key in no_transfer_rs:
            if key in coldtypedic:
                del coldtypedic[key]

    return coldtypedic, verboseprint


def _count_edges(ei):
    if type(ei) is list:
        c = 0
        for eik in ei:
            c += eik.shape[0]
    else:
        c = ei.shape[0]
    return c


def _copied_rs(ei):
    if type(ei) is list:
        return ei[0].columns.values
    else:
        return ei.columns.values


def _aggregate_super_table(funcs=None, size=False, gt=None):
    # aggregate values
    t = []
    if size:
        tt = gt.size()
        tt.name = "size"
        t.append(tt)

    if funcs is not None:
        tt = gt.agg(funcs)

        # flatten hierarchical index
        cols = []
        for col in tt.columns:
            if isinstance(col, tuple):
                cols.append("_".join(col).strip())
            else:
                cols.append(col)
        tt.columns = cols

        t.append(tt)

    try:
        t = pd.concat(t, axis=1)
    except TypeError:
        t = dd.concat(t, axis=1)

    return t


def _matrix_iterator(
    v, min_chunk_size, from_pos, to_pos, coldtypedic, transfer_features, verboseprint, logfile, hdf_key
):
    ft_feature = None
    dt_unit = None

    # if hdf, find requested features
    if isinstance(v, pd.HDFStore):
        v_is_hdf = True
        rf = [transfer_features, CreatorFunction.c_input_features, CreatorFunction.s_input_features]
        rf = set(
            [feature[:-2] if feature.endswith("_s") or feature.endswith("_t") else feature for feature in _flatten(rf)]
        )
    else:
        v_is_hdf = False

    if v_is_hdf:
        N = v.get_storer(hdf_key).nrows
    else:
        N = len(v)

    if to_pos is None:
        to_pos = N * (N - 1) / 2

    # assertions
    assert to_pos <= N * (N - 1) / 2, "the given to_pos parameter is too large, " "{} > g.n*(g.n-1)/2={}".format(
        to_pos, N * (N - 1) / 2
    )

    assert from_pos < N * (N - 1) / 2, "the given from_pos argument is too large, " "{} (given) > {} (max)".format(
        from_pos, int(N * (N - 1) / 2) - 1
    )

    assert from_pos < to_pos, "to_pos must be larger than from_pos"

    # cumulatively count the generated edges
    cum_edges = 0

    # iterate through matrix
    c = 0
    ei_list = []
    pos_array, n_steps = _pos_array(from_pos, to_pos, min_chunk_size)
    for from_pos, to_pos in pos_array:
        # measure time per iteration
        starttime = datetime.now()

        # print
        verboseprint("# =====================================================")
        verboseprint("Iteration {} of {} ({:.2f}%)".format(c + 1, n_steps, float(c + 1) / n_steps * 100))
        c += 1

        # construct node indices
        sources_k, targets_k = _triu_indices(N, from_pos, to_pos)

        # unique indices of sources' & targets' union
        indices = _union_of_indices(N, sources_k, targets_k)

        # create triu_indices for subset of v
        sources_k, targets_k = _reduce_triu_indices(sources_k, targets_k, indices)

        # select subset of v
        if v_is_hdf:
            vi = v.select(hdf_key, where=indices, columns=rf)
        else:
            vi = v.iloc[indices]

        # return i'th selection
        ei = _select_and_return(vi, sources_k, targets_k, ft_feature, dt_unit, transfer_features, coldtypedic)
        ei_list.append(ei)

        # print
        cum_edges += _count_edges(ei)
        timediff = datetime.now() - starttime
        verboseprint(" nr of edges:", [_count_edges(ei)], ", cum nr of edges:", [cum_edges])
        verboseprint(" pos_interval:", [from_pos, to_pos])
        verboseprint(" nr of pairs (total):", [int(N * (N - 1) / 2)])
        verboseprint(" copied rs: {}".format(ei.columns.values))
        verboseprint(
            " computation time:",
            "\ts =",
            int(timediff.total_seconds()),
            "\tms =",
            str(timediff.microseconds / 1000.0)[:6],
            "\n",
        )

        # logging
        if logfile:
            with open(logfile, "a") as log:
                print("0\t{}\t".format(len(sources_k)), end="", file=log)
                print("{}\t{:.3f}".format(_count_edges(ei), timediff.total_seconds()), file=log)

    # concat eik_list
    e = pd.concat(ei_list, ignore_index=True, copy=False)

    # set indices
    e.set_index(["s", "t"], inplace=True)

    return e


def _ft_iterator(
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
):
    # fast track feature references
    ftf = ft_feature[0]
    ftt = ft_feature[1]

    # if hdf, find requested features
    if isinstance(v, pd.HDFStore):
        v_is_hdf = True
        rf = [transfer_features, CreatorFunction.c_input_features, CreatorFunction.s_input_features]
        rf = set(
            [feature[:-2] if feature.endswith("_s") or feature.endswith("_t") else feature for feature in _flatten(rf)]
        )
        rf.remove("ft_feature")
        rf.add(ftf)
    else:
        v_is_hdf = False

    # number of nodes
    if v_is_hdf:
        N = v.get_storer(hdf_key).nrows
    else:
        N = len(v)

    # iteration parameters
    n = min_chunk_size
    i = from_pos
    if to_pos is None:
        to_pos = N
        to_pos_default = True
    else:
        assert to_pos <= N, "the given to_pos parameter is too large, " "{} > len(v)={}".format(to_pos, N)
        to_pos_default = False

    assert from_pos < N, "the given from_pos argument is too large, " "{} >= len(v)".format(from_pos)
    assert from_pos < to_pos, "to_pos must be larger than from_pos"

    # cumulatively count the generated edges
    cum_edges = 0

    # for testing / logging
    self._triggered = {
        "large_enough": 0,
        "increase_p1d": 0,
        "increased_leq_N": 0,
        "increased_end_of_table": 0,
        "gap_cases": 0,
    }

    # create ei list to append to
    ei_list = []
    # start iteration
    while i < to_pos:
        # measure time per iteration
        starttime = datetime.now()

        # select partial fast track feature
        if v_is_hdf:
            vi = v.select(hdf_key, start=i, stop=i + n, columns=[ftf])[ftf].values
        else:
            vi = v.iloc[i : i + n][ftf].values

        # compute max p1 difference of vi, if p1d <= ftt, increase chunk size
        ftf_first = vi[0]
        ftf_last = vi[-1]
        if dt_unit is None:
            p1d = ftf_last - ftf_first
        else:
            p1d = (ftf_last - ftf_first) / np.timedelta64(1, dt_unit)

        # ================================================================
        # case 1) min_chunk_size large enough
        if p1d > ftt:
            # for testing
            self._triggered["large_enough"] += 1

            # take one more node here, since 'trapez' always takes away the
            # last node
            if v_is_hdf:
                vi = v.select(hdf_key, start=i, stop=i + n + 1, columns=rf)
            else:
                vi = v.iloc[i : i + n + 1]

            ei, ns = _ft_create_ei(
                self,
                vi,
                ft_feature,
                dt_unit,
                coldtypedic,
                transfer_features,
                max_pairs,
                verboseprint,
                logfile,
                symmetry="trapez",
            )

            ei_list.extend(ei)

            cum_edges += _count_edges(ei)
            verboseprint(" processed sources:", [ns])
            verboseprint(" mapped with targets:", [len(vi) - 1])
            verboseprint(" pos interval:", [i, i + ns])
            verboseprint(" nr of nodes (total):", [N])
            verboseprint(" ft_feature of last source:", [vi.at[vi.iloc[ns - 1].name, ftf]])
            verboseprint(" nr of edges:", [_count_edges(ei)], ", cum nr of edges:", [cum_edges])
            verboseprint(" copied rs: {}".format(_copied_rs(ei)))

            i += ns

        # ================================================================
        # case 2) increase vi, s.t. p1d > ftt
        # when increasing, form only pairs with p1d <= ftt, no excessive ones
        else:
            # for testing
            self._triggered["increase_p1d"] += 1

            verboseprint("min_chunk_size too small, increasing partial v..")

            # again, include the first node with ftf > ftf_first + ftt to
            # pass to _ft_create_ei, which will use it and then get rid of it.
            if dt_unit is None:
                if v_is_hdf:
                    where = "{} <= {}".format(ftf, ftf_first + ftt)
                    upto = v.select_as_coordinates(hdf_key, where=where, start=i)[-1] + 2
                else:
                    upto = i + np.searchsorted(v[ftf].values[i:], ftf_first + ftt, side="right") + 1
            else:
                if v_is_hdf:
                    # is there a better way then converting to timestamp?
                    ts = pd.Timestamp(ftf_first + np.timedelta64(ftt, dt_unit))
                    where = "{} <= {!r}".format(ftf, ts)
                    upto = v.select_as_coordinates(hdf_key, where=where, start=i)[-1] + 2
                else:
                    ts = ftf_first + np.timedelta64(ftt, dt_unit)
                    upto = i + np.searchsorted(v[ftf].values[i:], ts, side="right") + 1

            if upto <= N:
                # for testing
                self._triggered["increased_leq_N"] += 1

                if v_is_hdf:
                    vi = v.select(hdf_key, start=i, stop=upto, columns=rf)
                else:
                    vi = v.iloc[i:upto]

                ei, ns = _ft_create_ei(
                    self,
                    vi,
                    ft_feature,
                    dt_unit,
                    coldtypedic,
                    transfer_features,
                    max_pairs,
                    verboseprint,
                    logfile,
                    symmetry="trapez",
                )

                ei_list.extend(ei)

                cum_edges += _count_edges(ei)
                verboseprint(" processed sources:", [ns])
                verboseprint(" mapped with targets:", [upto - i - 1])
                verboseprint(" pos interval:", [i, i + ns])
                verboseprint(" nr of nodes (total):", [N])
                verboseprint(" ft_feature of last source:", [vi.at[vi.iloc[ns - 1].name, ftf]])
                verboseprint(" nr of edges:", [_count_edges(ei)], ", cum nr of edges:", [cum_edges])
                verboseprint(" copied rs: {}".format(_copied_rs(ei)))

                i += ns

            # ============================================================
            # case 3) end of table, compute upper triangle matrix
            else:
                # for testing
                self._triggered["increased_end_of_table"] += 1

                if v_is_hdf:
                    vi = v.select(hdf_key, start=i, columns=rf)
                else:
                    vi = v.iloc[i:]

                ei, ns = _ft_create_ei(
                    self,
                    vi,
                    ft_feature,
                    dt_unit,
                    coldtypedic,
                    transfer_features,
                    max_pairs,
                    verboseprint,
                    logfile,
                    symmetry="triangle",
                )

                ei_list.extend(ei)

                cum_edges += _count_edges(ei)
                verboseprint("# LAST", [len(vi)], "EVENTS PROCESSED (END OF TABLE)")
                verboseprint("# =====================================================")
                verboseprint(" processed sources:", [ns])
                verboseprint(" mapped with targets:", [ns])
                verboseprint(" pos interval:", [i, i + len(vi)])
                verboseprint(" nr of nodes (total):", [N])
                verboseprint(" ft_feature of last source:", [vi.at[vi.iloc[-1].name, ftf]])
                verboseprint(" nr of edges:", [_count_edges(ei)], ", cum nr of edges:", [cum_edges])
                verboseprint(" copied rs: {}".format(_copied_rs(ei)))

                i += ns

        timediff = datetime.now() - starttime
        verboseprint(
            " computation time:",
            "\ts =",
            int(timediff.total_seconds()),
            "\tms =",
            str(timediff.microseconds / 1000.0)[:6],
            "\n",
        )
        # logging
        if logfile:
            with open(logfile, "a") as log:
                print("{}\t{:.3f}".format(_count_edges(ei), timediff.total_seconds()), file=log)

    ei_list_without_emtpy_dataframes = [ei_list_i for ei_list_i in ei_list if not ei_list_i.empty]
    if ei_list_without_emtpy_dataframes:
        e = pd.concat(ei_list_without_emtpy_dataframes, ignore_index=True, copy=False)
    else:
        e = ei_list[0].iloc[0:0]

    # set indices
    e.set_index(["s", "t"], inplace=True)

    # delete excessive sources (only return sources up to to_pos)
    # PERFORMANCE (look for better solution, not 'isin'...)
    if to_pos_default is False:
        if v_is_hdf:
            indices = v.select_column(hdf_key, "index", start=from_pos, stop=to_pos).values
        else:
            indices = v.iloc[from_pos:to_pos].index.values

        s = e.index.get_level_values(level=0)
        e = e.loc[s.isin(indices)]

    return e


def _ft_subiterator(nl, vi, ft_feature, dt_unit, coldtypedic, transfer_features, pairs, max_pairs, verboseprint):
    # iterate through node indices
    c = 0
    eik_list = []
    pos_array, n_steps = _pos_array(0, pairs, max_pairs)
    for from_pos, to_pos in pos_array:
        verboseprint("subiteration {} of {}".format(c + 1, n_steps))
        c += 1

        # # construct node indices
        sources_k, targets_k = _triu_indices(nl, from_pos, to_pos)

        # unique indices of sources' & targets' union
        indices = _union_of_indices(nl, sources_k, targets_k)

        # create triu_indices for subset of v
        sources_k, targets_k = _reduce_triu_indices(sources_k, targets_k, indices)

        # select subset of vi
        vik = vi.iloc[indices]

        # return k'th selection
        eik = _select_and_return(vik, sources_k, targets_k, ft_feature, dt_unit, transfer_features, coldtypedic)

        eik_list.append(eik)

    return eik_list


def _ft_create_ei(
    self, vi, ft_feature, dt_unit, coldtypedic, transfer_features, max_pairs, verboseprint, logfile, symmetry
):
    ftf = ft_feature[0]
    ftt = ft_feature[1]

    if symmetry == "trapez":
        # dimensions of the trapez
        # nl: number of targets
        # ns: number of sources
        # nd: nl - ns
        if dt_unit is None:
            ns = (vi[ftf] < vi.at[vi.iloc[-1].name, ftf] - ftt).sum()
        else:
            ns = (vi[ftf] < vi.at[vi.iloc[-1].name, ftf] - np.timedelta64(ftt, dt_unit)).sum()

        vi = vi.iloc[:-1]
        nl = len(vi)

        nd = nl - ns
        # number of pairs
        pairs = (ns * (ns - 1)) // 2 + nd * ns

        if nl == 1:
            # for testing
            self._triggered["gap_cases"] += 1

            # only happens for "gap" cases
            verboseprint("# =====================================================")
            verboseprint(" nr of pairs: [{}]".format(pairs))
            ei = pd.DataFrame({col: pd.Series(data=[], dtype=dtype) for col, dtype in coldtypedic.items()})

            return [ei], ns

        else:
            if pairs > max_pairs:
                verboseprint("# =====================================================")
                verboseprint("maximum number of pairs exceeded")
                verboseprint(" nr of pairs: [{}]".format(pairs))
                ei = _ft_subiterator(
                    nl, vi, ft_feature, dt_unit, coldtypedic, transfer_features, pairs, max_pairs, verboseprint
                )
                # logging
                if logfile:
                    with open(logfile, "a") as log:
                        print("1\t{}\t".format(pairs), end="", file=log)

                return ei, ns

            else:
                # construct node indices
                sources, targets = _triu_indices(nl, 0, pairs)

                verboseprint("# =====================================================")
                verboseprint(" nr of pairs: [{}]".format(pairs))
                ei = _select_and_return(vi, sources, targets, ft_feature, dt_unit, transfer_features, coldtypedic)
                # logging
                if logfile:
                    with open(logfile, "a") as log:
                        print("0\t{}\t".format(len(sources)), end="", file=log)

                return [ei], ns

    elif symmetry == "triangle":
        # dimensions of the square
        # nl: number of targets
        # ns: number of sources
        # nd: nl - ns = 0
        nl = len(vi)
        ns = len(vi)
        nd = 0

        # number of pairs
        pairs = (nl * (nl - 1)) // 2

        if pairs > max_pairs:
            verboseprint("# =====================================================")
            verboseprint("maximum number of pairs exceeded")
            verboseprint(" nr of pairs: [{}]".format(pairs))
            ei = _ft_subiterator(
                nl, vi, ft_feature, dt_unit, coldtypedic, transfer_features, pairs, max_pairs, verboseprint
            )
            # logging
            if logfile:
                with open(logfile, "a") as log:
                    print("1\t{}\t".format(pairs), end="", file=log)

            return ei, ns

        else:
            # construct node indices
            sources, targets = np.triu_indices(nl, k=1)
            sources = sources.astype(np.uint64)
            targets = targets.astype(np.uint64)

            verboseprint("# =====================================================")
            verboseprint(" nr of pairs: [{}]".format(pairs))
            ei = _select_and_return(vi, sources, targets, ft_feature, dt_unit, transfer_features, coldtypedic)
            # logging
            if logfile:
                with open(logfile, "a") as log:
                    print("0\t{}\t".format(len(sources)), end="", file=log)

            return [ei], ns


def _select_and_return(vi, sources, targets, ft_feature, dt_unit, transfer_features, coldtypedic):
    for selector in CreatorFunction.s_instances:
        sources, targets = selector.select_and_store(vi, sources, targets, ft_feature, dt_unit)

    # output relations that have not been requested by selectors
    for r in CreatorFunction.c_output_rs:
        if r not in CreatorFunction.stored_relations:
            Selector.request_r(r, vi, sources, targets, dt_unit, ft_feature)

    # ========================================================================
    #  COLLECT DATA (indices, rs and transfer features)
    # ========================================================================

    data = CreatorFunction.stored_relations

    # go back in original v base
    sources = vi.index.values[sources]
    targets = vi.index.values[targets]

    # node indices
    data["s"] = sources
    data["t"] = targets

    # transfer features
    for tf in transfer_features:
        data[tf + "_s"] = vi.loc[sources, tf].values
        data[tf + "_t"] = vi.loc[targets, tf].values

    # create ei
    ei = pd.DataFrame({col: data[col] for col in coldtypedic})

    # dtypes
    cdd = {key: value for key, value in coldtypedic.items() if value is not None}
    ei = ei.astype(cdd)

    # reset stored_relations
    CreatorFunction.reset("stored_relations")

    return ei


def _pos_array(from_pos, to_pos, step_size):
    # make sure all arguments are type(int)
    from_pos = int(from_pos)
    to_pos = int(to_pos)
    step_size = int(step_size)
    # create range generators
    a = range(from_pos, to_pos, step_size)
    b = range(from_pos + step_size, to_pos, step_size)
    b = chain(b, [to_pos])
    # number of steps
    n_steps = len(a)
    return zip(a, b), n_steps


def _iter_edges(e, dropna):
    """To use as ebunch for networkx.MultiDiGraph.add_edges_from(ebunch)."""
    for col in e:
        et = e[col]
        if dropna:
            et = et.dropna()
        for ind, val in et.items():
            yield (ind[0], ind[1], col, {col: val})
