# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

from collections.abc import Iterable

import numpy as np


def _create_bin_edges(x, bins, log_bins, floor):
    xmax = x.max()
    xmin = x.min()
    if log_bins is False:
        if floor is False:
            bin_edges = np.linspace(xmin, xmax, bins)
        else:
            bin_edges = np.unique(np.floor(np.linspace(xmin, xmax, bins)))
            bin_edges[-1] = xmax
    else:
        log_xmin = np.log10(xmin)
        log_xmax = np.log10(xmax)
        bins = int(np.ceil((log_xmax - log_xmin) * bins))
        if floor is False:
            bin_edges = np.logspace(log_xmin, log_xmax, bins)
        else:
            bin_edges = np.unique(np.floor(np.logspace(log_xmin, log_xmax, bins)))
            bin_edges[-1] = xmax

    return bin_edges


def _dic_translator(x, dic):
    x = x.copy()
    for i in range(len(x)):
        x[i] = dic[x[i]]
    return x


def _is_array_like(x):
    return isinstance(x, Iterable) and not isinstance(x, str)


def _flatten(x):
    result = []
    for el in x:
        if _is_array_like(el):
            result.extend(_flatten(el))
        else:
            result.append(el)
    return result


def _merge_dicts(*dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result
