"""Auxiliary **connector** and **selector** functions to create edges.

This module provides auxiliary **connector** and **selector** functions
for the ``dg.DeepGraph.create_edges`` and
``dg.DeepGraph.create_ft_edges`` methods.

They are described in their corresponding docstrings.

"""

# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

import numpy as np


def _ft_connector(ft_feature_s, ft_feature_t):
    ft_r = ft_feature_t - ft_feature_s
    return ft_r


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
    gcd = np.arccos(np.sin(phi_i) * np.sin(phi_j) + np.cos(phi_i) * np.cos(phi_j) * np.cos(delta_alpha)) * R

    # for 0 gcd, there might be nans, convert to 0.
    gcd = np.nan_to_num(gcd)

    return gcd


def _ft_selector(ft_r, ftt, sources, targets):
    sources = sources[ft_r <= ftt]
    targets = targets[ft_r <= ftt]
    return sources, targets
