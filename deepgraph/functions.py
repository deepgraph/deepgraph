"""Auxiliary **connector** and **selector** functions to create edges.

This module provides auxiliary **connector** and **selector** functions
for the ``dg.DeepGraph.create_edges`` and
``dg.DeepGraph.create_ft_edges`` methods.

They are described in their corresponding docstrings.

"""

from __future__ import print_function, division, absolute_import

# Copyright (C) 2016 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD license.

# py2/3 compatibility
try:
    range = xrange
except NameError:
    pass

import numpy as np

__all__ = ['great_circle_dist',
           'cp_node_intersection',
           'cp_intersection_strength',
           'hypergeometric_p_value',
           ]


# ============================================================================
# CONNECTORS
# ============================================================================

def great_circle_dist(lat_s, lat_t, lon_s, lon_t):
    """Return the great circle distance between nodes.

    The latitude and longitude values in the node table have to be in signed
    decimal degrees without compass direction (the sign indicates west/south).
    The great circle distance is calculated using the spherical law of cosines.

    """

    # dtypes
    lat_s = np.array(lat_s, dtype=float)
    lat_t = np.array(lat_t, dtype=float)
    lon_s = np.array(lon_s, dtype=float)
    lon_t = np.array(lon_t, dtype=float)

    # select by event_indices
    phi_i = np.radians(lat_s)
    phi_j = np.radians(lat_t)

    delta_alpha = np.radians(lon_t) - np.radians(lon_s)

    # earth's radius
    R = 6371

    # spatial distance of nodes
    gcd = np.arccos(np.sin(phi_i) * np.sin(phi_j) +
                    np.cos(phi_i) * np.cos(phi_j) *
                    np.cos(delta_alpha)) * R

    # for 0 gcd, there might be nans, convert to 0.
    gcd = np.nan_to_num(gcd)

    return gcd


def cp_node_intersection(supernode_ids, sources, targets):
    """Work in progress!

    """
    nodess = supernode_ids[sources]
    nodest = supernode_ids[targets]

    identical_nodes = (nodess == nodest)

    intsec = np.zeros(len(sources), dtype=object)
    intsec_card = np.zeros(len(sources), dtype=np.int)

    for i in range(len(sources)):
        intsec[i] = nodess[i].intersection(nodest[i])
        intsec_card[i] = len(intsec[i])

    return intsec, intsec_card, identical_nodes


def cp_intersection_strength(n_unique_nodes, intsec_card, sources, targets):
    """Work in progress!

    """
    us = n_unique_nodes[sources]
    ut = n_unique_nodes[targets]

    # min cardinality
    min_card = np.array(np.vstack((us, ut)).min(axis=0), dtype=np.float64)

    # intersection strength
    intsec_strength = intsec_card / min_card

    return intsec_strength


def hypergeometric_p_value(n_unique_nodes, intsec_card, sources, targets):
    """Work in progress!

    """
    from scipy.stats import hypergeom

    us = n_unique_nodes[sources]
    ut = n_unique_nodes[targets]

    # population size
    M = 220*220
    # number of success states in population
    n = np.vstack((us, ut)).max(axis=0)
    # total draws
    N = np.vstack((us, ut)).min(axis=0)
    # successes
    x = intsec_card

    hg_p = np.zeros(len(sources))
    for i in range(len(sources)):
        hg_p[i] = hypergeom.sf(x[i], M, n[i], N[i])

    return hg_p


# ============================================================================
# Selectors
# ============================================================================
