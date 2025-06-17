"""Test DeepGraph methods"""

# Copyright (C) 2017-2020 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from deepgraph import DeepGraph

# test node table
v = pd.DataFrame(
    {
        "x": [-3.4, 2.1, -1.1, 0.9, 2.3],
        "time": [0, 1, 2, 5, 9],
        "color": ["g", "g", "b", "g", "r"],
        "size": [1, 3, 2, 3, 1],
    }
)
g = DeepGraph(v)


# create test edge table
def some_relations(ft_r, x_s, x_t, color_s, color_t, size_s, size_t):
    dx = x_t - x_s
    v = dx / ft_r
    same_color = color_s == color_t
    larger_than = size_s > size_t
    return dx, v, same_color, larger_than


g.create_edges_ft(("time", 5), connectors=some_relations)
g.e.rename(columns={"ft_r": "dt"}, inplace=True)


class TestPartitionNodes:
    sv_full_true = pd.DataFrame({"color": ["b", "g", "r"], "n_nodes": [1, 3, 1], "time": [[2], [0, 1, 5], [9]]})
    sv_full_true.set_index("color", inplace=True)

    def test_feature(self):
        sv_true = self.sv_full_true.drop("time", axis=1)

        sv_test = g.partition_nodes("color")

        assert_frame_equal(sv_test, sv_true)

    def test_feature_funcs(self):
        sv_true = self.sv_full_true

        sv_test = g.partition_nodes("color", {"time": lambda x: list(x)})

        assert_frame_equal(sv_test, sv_true)

    def test_n_nodes(self):
        sv_true = self.sv_full_true.drop("n_nodes", axis=1)

        sv_test = g.partition_nodes("color", {"time": lambda x: list(x)}, n_nodes=False)

        assert_frame_equal(sv_test, sv_true)

    def test_return_gv(self):
        sv_true = self.sv_full_true.drop("time", axis=1)

        sv_test, _ = g.partition_nodes("color", return_gv=True)

        assert_frame_equal(sv_test, sv_true)

    def test_features(self):
        sv_true = pd.DataFrame(
            {
                "color": ["b", "g", "g", "r"],
                "size": [2, 1, 3, 1],
                "n_nodes": [1, 1, 2, 1],
            }
        )
        sv_true.set_index(["color", "size"], inplace=True)

        sv_test = g.partition_nodes(["color", "size"])

        assert_frame_equal(sv_test, sv_true)


class TestPartitionEdges:
    def test_relation(self):
        se_true = pd.DataFrame({"larger_than": [False, True], "n_edges": [5, 2]})
        se_true.set_index("larger_than", inplace=True)

        se_test = g.partition_edges("larger_than")

        assert_frame_equal(se_test, se_true)

    def test_relations(self):
        se_true = pd.DataFrame(
            {
                "larger_than": [False, False, True],
                "same_color": [False, True, False],
                "n_edges": [2, 3, 2],
            }
        )
        se_true.set_index(["larger_than", "same_color"], inplace=True)

        se_test = g.partition_edges(relations=["larger_than", "same_color"])

        assert_frame_equal(se_test, se_true)

    def test_relation_func(self):
        se_true = pd.DataFrame({"larger_than": [False, True], "n_edges": [5, 2], "same_color": [3, 0]})
        se_true.set_index("larger_than", inplace=True)

        se_test = g.partition_edges(relations="larger_than", relation_funcs={"same_color": "sum"})

        assert_frame_equal(se_test.sort_index(axis=1), se_true.sort_index(axis=1))

    def test_combine_groups(self):
        se_true = pd.DataFrame(
            {
                "larger_than": [False, False, False, True, True],
                "color_s": ["b", "g", "g", "g", "g"],
                "size_t": [3, 2, 3, 1, 2],
                "n_edges": [1, 1, 3, 1, 1],
            }
        )
        se_true.set_index(["larger_than", "color_s", "size_t"], inplace=True)

        se_test = g.partition_edges(relations="larger_than", source_features="color", target_features="size")

        assert_frame_equal(se_test, se_true)

    def test_n_edges(self):
        se_true = pd.DataFrame({"larger_than": [False, True], "same_color": [3, 0]})
        se_true.set_index("larger_than", inplace=True)

        se_test = g.partition_edges(relations="larger_than", relation_funcs={"same_color": "sum"}, n_edges=False)

        assert_frame_equal(se_test, se_true)

    def test_return_ge(self):
        se_true = pd.DataFrame({"larger_than": [False, True], "n_edges": [5, 2]})
        se_true.set_index("larger_than", inplace=True)

        se_test, _ = g.partition_edges("larger_than", return_ge=True)

        assert_frame_equal(se_test, se_true)


class TestPartitionGraph:
    def test_feature(self):
        sv_true = pd.DataFrame({"color": ["b", "g", "r"], "n_nodes": [1, 3, 1]})
        sv_true.set_index("color", inplace=True)

        se_true = pd.DataFrame(
            {
                "color_s": ["b", "g", "g", "g"],
                "color_t": ["g", "b", "g", "r"],
                "n_edges": [1, 2, 3, 1],
            }
        )
        se_true.set_index(["color_s", "color_t"], inplace=True)

        sv_test, se_test = g.partition_graph("color")

        assert_frame_equal(sv_test, sv_true)
        assert_frame_equal(se_test, se_true)

    def test_features_agg(self):
        sv_true = pd.DataFrame(
            {
                "color": ["b", "g", "g", "r"],
                "size": [2, 1, 3, 1],
                "n_nodes": [1, 1, 2, 1],
                "time": [2, 0, 5, 9],
            }
        )
        sv_true.set_index(["color", "size"], inplace=True)

        se_true = pd.DataFrame(
            {
                "color_s": ["b", "g", "g", "g", "g", "g"],
                "color_t": ["g", "b", "g", "b", "g", "r"],
                "size_s": [2, 1, 1, 3, 3, 3],
                "size_t": [3, 2, 3, 2, 3, 1],
                "n_edges": [1, 1, 2, 1, 1, 1],
                "dt": [3.0, 2.0, 3.0, 1.0, 4.0, 4.0],
            }
        )
        se_true.set_index(["color_s", "size_s", "color_t", "size_t"], inplace=True)

        sv_test, se_test = g.partition_graph(
            ["color", "size"],
            feature_funcs={"time": "max"},
            relation_funcs={"dt": "mean"},
        )

        assert_frame_equal(sv_test.sort_index(axis=1), sv_true.sort_index(axis=1))
        assert_frame_equal(se_test.sort_index(axis=1), se_true.sort_index(axis=1), check_dtype=False)

    def test_n_nodes_n_edges(self):
        sv_true = pd.DataFrame({"color": ["b", "g", "g", "r"], "size": [2, 1, 3, 1], "time": [2, 0, 5, 9]})
        sv_true.set_index(["color", "size"], inplace=True)

        se_true = pd.DataFrame(
            {
                "color_s": ["b", "g", "g", "g", "g", "g"],
                "color_t": ["g", "b", "g", "b", "g", "r"],
                "size_s": [2, 1, 1, 3, 3, 3],
                "size_t": [3, 2, 3, 2, 3, 1],
                "dt": [3.0, 2.0, 3.0, 1.0, 4.0, 4.0],
            }
        )
        se_true.set_index(["color_s", "size_s", "color_t", "size_t"], inplace=True)

        sv_test, se_test = g.partition_graph(
            ["color", "size"],
            feature_funcs={"time": "max"},
            relation_funcs={"dt": "mean"},
            n_nodes=False,
            n_edges=False,
        )

        assert_frame_equal(sv_test.sort_index(axis=1), sv_true.sort_index(axis=1))
        assert_frame_equal(se_test.sort_index(axis=1), se_true.sort_index(axis=1), check_dtype=False)

    def test_return_gve(self):
        sv_true = pd.DataFrame({"color": ["b", "g", "g", "r"], "size": [2, 1, 3, 1], "time": [2, 0, 5, 9]})
        sv_true.set_index(["color", "size"], inplace=True)

        se_true = pd.DataFrame(
            {
                "color_s": ["b", "g", "g", "g", "g", "g"],
                "color_t": ["g", "b", "g", "b", "g", "r"],
                "size_s": [2, 1, 1, 3, 3, 3],
                "size_t": [3, 2, 3, 2, 3, 1],
                "n_edges": [1, 1, 2, 1, 1, 1],
                "dt": [3.0, 2.0, 3.0, 1.0, 4.0, 4.0],
            }
        )
        se_true.set_index(["color_s", "size_s", "color_t", "size_t"], inplace=True)

        sv_test, se_test, _, _ = g.partition_graph(
            ["color", "size"],
            feature_funcs={"time": "max"},
            relation_funcs={"dt": "mean"},
            n_nodes=False,
            return_gve=True,
        )

        assert_frame_equal(sv_test.sort_index(axis=1), sv_true.sort_index(axis=1))
        assert_frame_equal(se_test.sort_index(axis=1), se_true.sort_index(axis=1), check_dtype=False)


class TestInterfaces:
    e = g.e[["dx", "dt", "larger_than", "same_color", "v"]]
    e = e.astype({"same_color": object, "larger_than": object, "dt": np.float64})
    e.iloc[0:5, 0] = np.nan
    e.iloc[1, 1] = np.nan
    e.iloc[2, 2] = np.nan
    e.iloc[3, 3] = np.nan
    e.iloc[4, 4] = np.nan

    g = DeepGraph(v, e)

    v_shift_ind = g.v.reset_index()
    v_shift_ind["index"] += 2
    v_shift_ind.set_index("index", inplace=True)
    e_shift_ind = e.reset_index()
    e_shift_ind["s"] += 2
    e_shift_ind["t"] += 2
    e_shift_ind.set_index(["s", "t"], inplace=True)

    g_si = DeepGraph(v_shift_ind, e_shift_ind)

    def test_return_cs_graph(self):
        pytest.importorskip("scipy")

        # relations = False
        csgraph_true = np.array(
            [
                [False, True, True, True, False],
                [False, False, True, True, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
                [False, False, False, False, False],
            ]
        )

        csgraph_test = self.g.return_cs_graph(relations=False).toarray()
        csgraph_test_si = self.g_si.return_cs_graph(relations=False).toarray()

        assert_allclose(csgraph_true, csgraph_test)
        assert_allclose(csgraph_true, csgraph_test_si)

        # relations = 'dt', dropna = True
        csgraph_true = np.array(
            [
                [0.0, 1.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 1.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        csgraph_test = self.g.return_cs_graph("dt", dropna=True).toarray()
        csgraph_test_si = self.g_si.return_cs_graph("dt", dropna=True).toarray()

        assert_allclose(csgraph_true, csgraph_test)
        assert_allclose(csgraph_true, csgraph_test_si)

        # relations = 'dt', dropna = False
        csgraph_true = np.array(
            [
                [0.0, 1.0, np.nan, 5.0, 0.0],
                [0.0, 0.0, 1.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        csgraph_test = self.g.return_cs_graph("dt", dropna=False).toarray()
        csgraph_test_si = self.g_si.return_cs_graph("dt", dropna=False).toarray()

        assert_allclose(csgraph_true, csgraph_test)
        assert_allclose(csgraph_true, csgraph_test_si)

        # relations = ['dx', 'dt]
        csgraph_test = self.g.return_cs_graph(["dx", "dt"], [True, False])
        assert csgraph_test["dx"].nnz == 2
        assert csgraph_test["dt"].nnz == 7

    def test_return_nx_graph(self):
        # only testing that return_nx_graph produces no errors
        #     work in progress!

        pytest.importorskip("networkx")
        pytest.importorskip("pandas", minversion="0.17.0")

        self.g.return_nx_graph(False, False)
        self.g.return_nx_graph(False, True)
        self.g.return_nx_graph(True, False)
        self.g.return_nx_graph("color", "dx")
        self.g.return_nx_graph(["color"], ["dx", "dt"])
        self.g_si.return_nx_graph(["color"], ["dx", "dt"])

        # test dropna
        nxg = self.g.return_nx_graph(["color"], ["dx", "dt"], dropna="none")
        assert nxg.number_of_edges() == 7

        nxg = self.g.return_nx_graph(["color"], ["dx", "dt"], dropna="all")
        assert nxg.number_of_edges() == 6

        nxg = self.g.return_nx_graph(["color"], ["dx", "dt"], dropna="any")
        assert nxg.number_of_edges() == 2

    def test_return_nx_multigraph(self):
        # only testing that return_nx_multigraph produces no errors
        #     work in progress!

        pytest.importorskip("networkx")
        pytest.importorskip("pandas", minversion="0.17.0")

        # input: features, relations, dropna
        self.g.return_nx_multigraph(False, False, True)
        self.g.return_nx_multigraph(False, True, False)
        self.g.return_nx_multigraph(False, True, True)
        self.g.return_nx_multigraph(True, False, True)
        self.g.return_nx_multigraph("color", "dx", True)
        self.g.return_nx_multigraph(["color"], ["dx", "dt"], True)
        self.g_si.return_nx_multigraph(["color"], ["dx", "dt"], True)

        # test dropna
        nxg = self.g.return_nx_multigraph(["color"], ["dx", "dt"], False)
        assert nxg.number_of_edges() == 14

        nxg = self.g.return_nx_multigraph(["color"], ["dx", "dt"], True)
        assert nxg.number_of_edges() == 8

    def test_return_gt_graph(self):
        # only testing that return_gt_graph produces no errors
        #     work in progress!

        pytest.importorskip("graph_tool")

        self.g.return_gt_graph(False, False)
        self.g.return_gt_graph(False, True)
        self.g.return_gt_graph(True, False)
        self.g.return_gt_graph("color", "dx")
        self.g.return_gt_graph(["color"], ["dx", "dt"])
        self.g_si.return_gt_graph(["color"], ["dx", "dt"], node_indices=True, edge_indices=True)

        # test dropna
        gtg = self.g.return_gt_graph(["color"], ["dx", "dt"], dropna="none")
        assert gtg.num_edges() == 7

        gtg = self.g.return_gt_graph(["color"], ["dx", "dt"], dropna="all")
        assert gtg.num_edges() == 6

        gtg = self.g.return_gt_graph(["color"], ["dx", "dt"], dropna="any")
        assert gtg.num_edges() == 2


class TestAppendCP:
    e = g.e.iloc[[0, 3]]
    g = DeepGraph(v, e)

    def test_defaults(self):
        pytest.importorskip("scipy")

        self.g.append_cp()
        cp = g.v.cp.values

        assert np.all(
            np.equal(
                np.unique(cp, return_counts=True),
                (np.array([0, 1, 2]), np.array([3, 1, 1])),
            )
        )
        assert np.all(cp[:3] == cp[0])

    def test_consolidate_singles(self):
        pytest.importorskip("scipy")

        cp_true = np.array([1, 1, 1, 0, 0])

        self.g.append_cp(consolidate_singles=True)
        cp_test = g.v.cp.values

        assert_allclose(cp_true, cp_test)


class TestTriuIndices:
    def test_random(self):
        from deepgraph._triu_indices import _triu_indices

        N = np.random.randint(900, 1100)
        n = N * (N - 1) / 2
        start = np.random.randint(0, n)
        end = np.random.randint(start, n)

        indices_true = np.triu_indices(N, k=1)
        sources_true = indices_true[0][start:end]
        targets_true = indices_true[1][start:end]

        indices_test = _triu_indices(N, start, end)
        sources_test = indices_test[0]
        targets_test = indices_test[1]

        assert (sources_true == sources_test).all() and (targets_true == targets_test).all()

    def test_border_cases(self):
        from deepgraph._triu_indices import _triu_indices

        Ns = [2, 5, 1004, 1523]
        starts = [0, 0, 0, 9, 9, 9, 9]
        ends = [0, 1, 2, 9, 10, 11, 12]

        for N in Ns:
            for start, end in zip(starts, ends):
                indices_true = np.triu_indices(N, k=1)
                sources_true = indices_true[0][start:end]
                targets_true = indices_true[1][start:end]

                indices_test = _triu_indices(N, start, end)
                sources_test = indices_test[0]
                targets_test = indices_test[1]

                assert (sources_true == sources_test).all() and (targets_true == targets_test).all()
