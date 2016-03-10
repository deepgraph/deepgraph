"""Test create_edges and create_edges_ft."""

# Copyright (C) 2016 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD license.

import pytest

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from deepgraph import DeepGraph


# test node table
v = pd.DataFrame({'si': [0,1,2,3,4,5,6,7],
                  'i': [5,4,9,6,7,8,2,4],
                  'sf': [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5],
                  'f': [.38,.84,.98,.67,.97,.61,.37,.82],
                  's': ['g','g','b','g','r','r','g','b'],
                  'c': [1,3,2,3,1,3,2,3],
                  'mcs': [0,3,6,7,7,8,12,12],
                  'o': [np.array([1,2,3]), np.array([2,2,2]),
                        np.array([3,1,2]), np.array([1,2,3]),
                        np.array([3,3,2]), np.array([1,2,3]),
                        np.array([3,1,1]), np.array([1,1,1])]
                  }
                 )

dts = pd.date_range('1998-01-01', periods=v.mcs.max()+1, freq='1H')
tdtdic = {i: j for i, j in enumerate(dts.values)}
v['sdt'] = v.si.apply(lambda x: tdtdic[x])
v['dtmcs'] = v.mcs.apply(lambda x: tdtdic[x])

# test edge table
s = np.triu_indices(len(v), k=1)[0]
t = np.triu_indices(len(v), k=1)[1]
dsi = [1,2,3,4,5,6,7,1,2,3,4,5,6,1,2,3,4,5,1,2,3,4,1,2,3,1,2,1]
dsf = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,0.5,1.0,1.5,2.0,2.5,3.0,0.5,1.0,1.5,2.0,2.5,
       0.5,1.0,1.5,2.0,0.5,1.0,1.5,0.5,1.0,0.5]
velo = np.repeat(2, len(s)).astype(float)
e_full_true = pd.DataFrame({'s': s, 't': t, 'dsi': dsi, 'dsf': dsf,
                            'velo': velo})
e_full_true.set_index(['s','t'], inplace=True)


# connectors
def dsi(si_s, si_t):
    dsi = si_t - si_s
    return dsi


def dsf(sf_s, sf_t):
    dsf = sf_t - sf_s
    return dsf


def velo(dsi, dsf):
    velo = dsi/dsf
    return velo


def ft_r_dep(ft_r, sf_s, sf_t):
    dsf = sf_t - sf_s
    velo = ft_r/dsf
    return velo, dsf


# selectors
def dsi_t(dsi, sources, targets):
    sources = sources[dsi <= 3]
    targets = targets[dsi <= 3]
    return sources, targets


def dsf_t(dsf, sources, targets):
    sources = sources[dsf <= 1]
    targets = targets[dsf <= 1]
    return sources, targets


def dsf_velo_t(dsi, sf_s, sf_t, sources, targets):
    dsf = sf_t - sf_s
    velo = dsi/dsf
    sources = sources[dsf <= 1]
    targets = targets[dsf <= 1]
    return dsf, velo, sources, targets


def dsi_dsf_t(si_s, si_t, sf_s, sf_t, sources, targets):
    dsi = si_t - si_s
    dsf = sf_t - sf_s
    velo = dsi/dsf
    sources = sources[(dsi <= 3) & (dsf <= 1)]
    targets = targets[(dsi <= 3) & (dsf <= 1)]
    return dsi, dsf, velo, sources, targets


def fail_selector_order(dsi, sources, targets):
    sources = sources[dsi <= 3]
    targets = targets[dsi <= 3]
    return sources, targets


def fail_connector_selector_order(velo, sources, targets):
    sources = sources[dsi <= 3]
    targets = targets[dsi <= 3]
    return sources, targets


def fail_r_shape(dsi, sf_s, sf_t, sources, targets):
    dsf = sf_t - sf_s
    velo = dsi/dsf
    sources = sources[dsf <= 1]
    targets = targets[dsf <= 1]
    # the length of dsf must not be modified
    dsf = dsf[:-1]
    return dsf, velo, sources, targets


def fail_ind_shape(dsi, sf_s, sf_t, sources, targets):
    dsf = sf_t - sf_s
    velo = dsi/dsf
    sources = sources[dsf <= 1]
    # targets must be reduced as well
    return dsf, velo, sources, targets

# parameter
p_step_size = [1,2,3,4,5,10,28,100]
p_from_pos = [0,7,14,21,27]
p_to_pos = [1,7,14,21,28]
p_min_chunk_size = np.arange(1, len(v))
p_ft_from_pos = np.arange(len(v))
p_ft_to_pos = np.arange(len(v)) + 1
p_max_pairs = np.arange(28) + 1


@pytest.fixture(params=p_step_size)
def step_size(request):
    return request.param


@pytest.fixture(params=p_from_pos)
def from_pos(request):
    return request.param


@pytest.fixture(params=p_to_pos)
def to_pos(request):
    return request.param


@pytest.fixture(params=p_min_chunk_size)
def min_chunk_size(request):
    return request.param


@pytest.fixture(params=p_ft_from_pos)
def ft_from_pos(request):
    return request.param


@pytest.fixture(params=p_ft_to_pos)
def ft_to_pos(request):
    return request.param


@pytest.fixture(params=p_max_pairs)
def max_pairs(request):
    return request.param


class TestCreateEdges(object):

    def test_no_arguments(self):

        e_true = e_full_true.iloc[:, 0:0]

        g = DeepGraph(v)
        g.create_edges()
        e_test = g.e

        pdt.assert_frame_equal(e_test, e_true)

        g.create_edges_ft(('si', v.si.max()))
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test, e_true)

    def test_connector(self):

        e_true = e_full_true[['dsi']]

        g = DeepGraph(v)
        g.create_edges(connectors=dsi)
        e_test = g.e

        pdt.assert_frame_equal(e_test, e_true)

        g.create_edges_ft(('si', v.si.max()),
                          connectors=dsi)
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test, e_true)

    def test_connectors(self):

        e_true = e_full_true[['dsf', 'dsi']]

        g = DeepGraph(v)
        g.create_edges(connectors=[dsi, dsf])
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi, dsf])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_connector_selector(self):

        e_true = e_full_true[e_full_true.dsi <= 3]

        g = DeepGraph(v)
        g.create_edges(connectors=[dsi, dsf, velo],
                       selectors=dsi_t)
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi, dsf, velo],
                          selectors=dsi_t)
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_connector_selectors(self):

        e_true = e_full_true[(e_full_true.dsi <= 3) &
                             (e_full_true.dsf <= 1)]

        g = DeepGraph(v)
        g.create_edges(connectors=[dsi, dsf, velo],
                       selectors=[dsi_t, dsf_t])
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi, dsf, velo],
                          selectors=[dsi_t, dsf_t])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_r_dep_selector(self):

        e_true = e_full_true[(e_full_true.dsi <= 3) &
                             (e_full_true.dsf <= 1)]

        g = DeepGraph(v)
        g.create_edges(connectors=[dsi],
                       selectors=[dsi_t, dsf_velo_t])
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi],
                          selectors=[dsi_t, dsf_velo_t])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_selector(self):

        e_true = e_full_true[(e_full_true.dsi <= 3) &
                             (e_full_true.dsf <= 1)]

        g = DeepGraph(v)
        g.create_edges(selectors=[dsi_dsf_t])
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          selectors=[dsi_dsf_t])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_transfer_feature(self):

        e_true = e_full_true.iloc[:, 0:0]
        s = e_true.index.get_level_values(0)
        t = e_true.index.get_level_values(1)
        e_true['f_s'] = v.loc[s, 'f'].values
        e_true['f_t'] = v.loc[t, 'f'].values

        g = DeepGraph(v)
        g.create_edges(transfer_features='f')
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          transfer_features='f')
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_transfer_features(self):

        e_true = e_full_true.iloc[:, 0:0]
        s = e_true.index.get_level_values(0)
        t = e_true.index.get_level_values(1)
        e_true['f_s'] = v.loc[s, 'f'].values
        e_true['f_t'] = v.loc[t, 'f'].values
        e_true['s_s'] = v.loc[s, 's'].values
        e_true['s_t'] = v.loc[t, 's'].values
        e_true['o_s'] = v.loc[s, 'o'].values
        e_true['o_t'] = v.loc[t, 'o'].values

        g = DeepGraph(v)
        g.create_edges(transfer_features=['f', 's', 'o'])
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          transfer_features=['f', 's', 'o'])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_r_dtype_dic(self):

        e_true = e_full_true[['dsf', 'dsi']]
        e_true['dsf'] = e_true.dsf.astype(np.uint8)
        e_true['dsi'] = e_true.dsi.astype(np.float32)

        r_dtype_dic = {'dsf': np.uint8, 'dsi': np.float32}
        g = DeepGraph(v)
        g.create_edges(connectors=[dsi, dsf], r_dtype_dic=r_dtype_dic)
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi, dsf], r_dtype_dic=r_dtype_dic)
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_no_transfer_r(self):

        e_true = e_full_true.drop('dsf', axis=1)

        g = DeepGraph(v)
        g.create_edges(connectors=[dsi, dsf, velo], no_transfer_rs='dsf')
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi, dsf, velo],
                          no_transfer_rs='dsf')
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_no_transfer_rs(self):

        e_true = e_full_true.drop(['dsf', 'velo'], axis=1)

        g = DeepGraph(v)
        g.create_edges(connectors=[dsi, dsf, velo],
                       no_transfer_rs=['dsf', 'velo'])
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          connectors=[dsi, dsf, velo],
                          no_transfer_rs=['dsf', 'velo'])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_step_size(self, step_size):

        e_true = e_full_true.iloc[:, 0:0]

        g = DeepGraph(v)
        g.create_edges(step_size=step_size)
        e_test = g.e

        pdt.assert_frame_equal(e_test, e_true)

    def test_from_pos(self, from_pos):

        e_true = e_full_true.iloc[from_pos:, 0:0]

        g = DeepGraph(v)
        g.create_edges(from_pos=from_pos)
        e_test = g.e

        pdt.assert_frame_equal(e_test, e_true)

    def test_to_pos(self, to_pos):

        e_true = e_full_true.iloc[:to_pos, 0:0]

        g = DeepGraph(v)
        g.create_edges(to_pos=to_pos)
        e_test = g.e

        pdt.assert_frame_equal(e_test, e_true)

    def test_hdf_key(self, tmpdir):

        pytest.importorskip('tables')

        # tmp hdf store
        folder = str(tmpdir)
        vo = v.drop('o', axis=1)
        vs = pd.HDFStore(folder + 'vs.h5', mode='w')
        vs.put('v', vo, format='t', data_columns=True, index=False)

        e_true = e_full_true[(e_full_true.dsi <= 3) &
                             (e_full_true.dsf <= 1)]

        g = DeepGraph(vs)
        g.create_edges(selectors=[dsi_dsf_t], hdf_key='v')
        e_test = g.e

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(('si', v.si.max()),
                          selectors=[dsi_dsf_t], hdf_key='v')
        e_test = g.e.drop('ft_r', axis=1)
        vs.close()

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_hdf_step_size(self, tmpdir, step_size):

        pytest.importorskip('tables')

        # tmp hdf store
        folder = str(tmpdir)
        vo = v.drop('o', axis=1)
        vs = pd.HDFStore(folder + 'vs.h5', mode='w')
        vs.put('v', vo, format='t', index=False)

        e_true = e_full_true[(e_full_true.dsi <= 3) &
                             (e_full_true.dsf <= 1)]

        g = DeepGraph(vs)
        g.create_edges(selectors=[dsi_dsf_t], step_size=step_size,
                       hdf_key='v')
        e_test = g.e
        vs.close()

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_logging(self, tmpdir):

        e_true = e_full_true.iloc[:, 0:0]

        g = DeepGraph(v)
        folder = str(tmpdir)
        g.create_edges(verbose=True, logfile=folder+'lf.txt')
        e_test = g.e

        pdt.assert_frame_equal(e_test, e_true)

        g.create_edges_ft(('si', v.si.max()),
                          verbose=True, logfile=folder+'lf.txt')
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_fail_selector_r_shape(self):

        g = DeepGraph(v)
        with pytest.raises(AssertionError):
            g.create_edges(connectors=dsi, selectors=fail_r_shape)
            g.create_edges_ft(('si', v.si.max()), connectors=dsi,
                              selectors=fail_r_shape)

    def test_fail_selector_ind_shape(self):

        g = DeepGraph(v)
        with pytest.raises(AssertionError):
            g.create_edges(connectors=dsi, selectors=fail_ind_shape)
            g.create_edges_ft(('si', v.si.max()), connectors=dsi,
                              selectors=fail_ind_shape)

    def test_fail_common_output_rs(self):

        g = DeepGraph(v)
        with pytest.raises(AssertionError):
            g.create_edges(connectors=[dsi, dsf], selectors=[dsf_velo_t])
            g.create_edges_ft(('si', v.si.max()), connectors=[dsi, dsf],
                              selectors=[dsf_velo_t])

    def test_fail_connectors_order(self):

        g = DeepGraph(v)
        with pytest.raises(KeyError):
            g.create_edges(connectors=[dsi, velo, dsf])
            g.create_edges_ft(('si', v.si.max()),
                              connectors=[dsi, velo, dsf])

    def test_fail_selectors_order(self):

        g = DeepGraph(v)
        with pytest.raises(KeyError):
            g.create_edges(selectors=[fail_selector_order, dsi_dsf_t])
            g.create_edges_ft(('si', v.si.max()),
                              selectors=[fail_selector_order, dsi_dsf_t])

    def test_fail_connector_selector_order(self):

        g = DeepGraph(v)
        with pytest.raises(KeyError):
            g.create_edges(connectors=[dsi, dsf, velo],
                           selectors=[fail_connector_selector_order])
            g.create_edges_ft(('si', v.si.max()),
                              connectors=[dsi, dsf, velo],
                              selectors=[fail_connector_selector_order])

    def test_ft_si(self):

        e_true = e_full_true[e_full_true.dsi <= 3][['dsi']]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('si', 3))
        e_test = g.e.rename(columns={'ft_r': 'dsi'})

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_sf(self):

        e_true = e_full_true[e_full_true.dsf <= 2.][['dsf']]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('sf', 2.))
        e_test = g.e.rename(columns={'ft_r': 'dsf'})

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_sdt(self):

        e_true = e_full_true[e_full_true.dsi <= 3][['dsi']]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('sdt', 3, 'h'))
        e_test = g.e.rename(columns={'ft_r': 'dsi'})
        e_test['dsi'] = e_test.dsi.astype(int)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_sdt_hdf(self, tmpdir):

        pytest.importorskip('tables')

        # tmp hdf store
        folder = str(tmpdir)
        vo = v.drop('o', axis=1)
        vs = pd.HDFStore(folder + 'vs.h5', mode='w')
        vs.put('v', vo, format='t', data_columns=True, index=False)

        e_true = e_full_true[(e_full_true.dsi <= 3) &
                             (e_full_true.dsf <= 1)]

        g = DeepGraph(vs)
        g.create_edges_ft(ft_feature=('sdt', 3, 'h'),
                          selectors=dsi_dsf_t, hdf_key='v')
        e_test = g.e.drop('ft_r', axis=1)
        vs.close()

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_r_dep_connector(self):

        e_true = e_full_true[e_full_true.dsi <= 5]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('sdt', 5, 'h'),
                          connectors=ft_r_dep)
        e_test = g.e.rename(columns={'ft_r': 'dsi'})
        e_test['dsi'] = e_test.dsi.astype(int)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_ftt_order(self):

        e_true = e_full_true[e_full_true.dsf <= 1]

        g = DeepGraph(v)
        g.create_edges_ft(('si', v.si.max()), connectors=[dsi],
                          selectors=[dsf_velo_t, 'ft_selector'])
        e_test = g.e.drop('ft_r', axis=1)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_min_chunk_size(self, min_chunk_size):

        e_true = e_full_true.copy()
        e_true['dmcs'] = [3,6,7,7,8,12,12,3,4,4,5,9,9,1,1,2,6,6,0,1,5,5,1,
                          5,5,4,4,0]
        e_true = e_true[e_true.dmcs <= 2][['dmcs']]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('mcs', 2),
                          min_chunk_size=min_chunk_size)
        e_test = g.e.rename(columns={'ft_r': 'dmcs'})
        e_test['dmcs'] = e_test.dmcs.astype(int)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

        g.create_edges_ft(ft_feature=('dtmcs', 2, 'h'),
                          min_chunk_size=min_chunk_size)
        e_test = g.e.rename(columns={'ft_r': 'dmcs'})
        e_test['dmcs'] = e_test.dmcs.astype(int)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_dt_min_chunk_size_hdf(self, tmpdir, min_chunk_size):

        pytest.importorskip('tables')

        # tmp hdf store
        folder = str(tmpdir)
        vo = v.drop('o', axis=1)
        vs = pd.HDFStore(folder + 'vs.h5', mode='w')
        vs.put('v', vo, format='t', data_columns=True, index=False)

        e_true = e_full_true.copy()
        e_true['dmcs'] = [3,6,7,7,8,12,12,3,4,4,5,9,9,1,1,2,6,6,0,1,5,5,1,
                          5,5,4,4,0]
        e_true = e_true[e_true.dmcs <= 2][['dmcs']]

        g = DeepGraph(vs)
        g.create_edges_ft(ft_feature=('dtmcs', 2, 'h'),
                          min_chunk_size=min_chunk_size)
        e_test = g.e.rename(columns={'ft_r': 'dmcs'})
        e_test['dmcs'] = e_test.dmcs.astype(int)

        pdt.assert_frame_equal(e_test.sort(axis=1), e_true.sort(axis=1))

    def test_ft_from_pos(self, ft_from_pos):

        e_true = e_full_true[['dsi']]
        s = e_true.index.get_level_values(0)
        e_true = e_true[s >= ft_from_pos]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('si', v.si.max()),
                          from_pos=ft_from_pos)
        e_test = g.e.rename(columns={'ft_r': 'dsi'})

        pdt.assert_frame_equal(e_test, e_true)

    def test_ft_to_pos(self, ft_to_pos):

        e_true = e_full_true[['dsi']]
        s = e_true.index.get_level_values(0)
        e_true = e_true[s < ft_to_pos]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('si', v.si.max()),
                          to_pos=ft_to_pos)
        e_test = g.e.rename(columns={'ft_r': 'dsi'})

        pdt.assert_frame_equal(e_test, e_true)

    def test_ft_max_pairs(self, max_pairs):

        e_true = e_full_true[['dsi']]

        g = DeepGraph(v)
        g.create_edges_ft(ft_feature=('si', v.si.max()),
                          max_pairs=max_pairs)
        e_test = g.e.rename(columns={'ft_r': 'dsi'})

        pdt.assert_frame_equal(e_test, e_true)

    def test_ft_fail_sorted(self):

        g = DeepGraph(v)
        with pytest.raises(AssertionError):
            g.create_edges_ft(ft_feature=('f', 1))
