# encoding: utf-8
# cython: profile=False
# filename: _find_selected_indices.pyx

# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.uint64
ctypedef np.uint64_t DTYPE_t

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _find_selected_indices(np.ndarray[DTYPE_t, ndim=1] s,
                           np.ndarray[DTYPE_t, ndim=1] t,
                           np.ndarray[DTYPE_t, ndim=1] ns,
                           np.ndarray[DTYPE_t, ndim=1] nt):

    cdef unsigned long long i, j, k, ledger, ns_i, nt_i, s_j, t_k, lns, ls
    cdef np.ndarray[DTYPE_t, ndim=1] ind = np.zeros(len(ns), dtype=DTYPE)
    lns = len(ns)
    ls = len(s)

    ledger = 0
    for i in range(lns):
        ns_i = ns[i]
        nt_i = nt[i]
        for j in range(ledger, ls):
            s_j = s[j]
            if s_j == ns_i:
                ledger = j
                for k in range(ledger, ls):
                    t_k = t[k]
                    if t_k == nt_i:
                        ledger = k
                        ind[i] = ledger
                        ledger += 1
                        break
                break
    return ind
