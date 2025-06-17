# encoding: utf-8
# cython: profile=False
# filename: _triu_indices.pyx

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
@cython.cdivision(True)
cpdef tuple _triu_indices(unsigned long long N,
                          unsigned long long start,
                          unsigned long long end):
    """Upper-triangle indices from start to end.

    Return the indices for the upper-triangle of an (N, N) array with a
    diagonal offset of k=1, from start to end position (excluded).

    Equivalent to (np.triu_indices(N, k=1)[0][start:end], np.triu_indices(N,
    k=1)[1][start:end]), without the overhead memory consumption of creating
    the entire range of indices first.

    See ``np.triu_indices`` for details.

    """

    cdef unsigned long long i, j_s, j_e, npairs, row
    cdef unsigned long long i_s = 0
    cdef unsigned long long i_e = 0

    if end > N*(N-1)//2:
        end = N*(N-1)//2

    if start >= end:
        return (np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE))

    cdef np.ndarray[DTYPE_t, ndim=1] sources = np.zeros(end-start, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] targets = np.zeros(end-start, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] cumsums = np.zeros(N-1, dtype=DTYPE)

    cumsums[0] = N - 1
    for i in range(1, N-1):
        cumsums[i] += cumsums[i-1] + N - i - 1

    # starting line
    for i in range(N-1):
        if start < cumsums[i]:
            i_s = i
            break

    # starting column
    if i_s == 0:
        j_s = start + 1
    else:
        j_s = start - cumsums[i_s - 1] + i_s + 1

    # ending line
    for i in range(N-1):
        if end <= cumsums[i]:
            i_e = i
            break

    # ending column
    if i_e == 0:
        j_e = end + 1
    else:
        j_e = end - cumsums[i_e - 1] + i_e + 1

    # create triu indices
    npairs = 0
    for row in range(i_s, i_e + 1):

        if i_s == i_e:
            for i in range(j_e - j_s):
                sources[npairs] = row
                targets[npairs] = j_s + i
                npairs += 1

        elif row == i_s:
            for i in range(N - j_s):
                sources[npairs] = row
                targets[npairs] = j_s + i
                npairs += 1

        elif row == i_e:
            for i in range(j_e - row - 1):
                sources[npairs] = row
                targets[npairs] = row + i + 1
                npairs += 1

        else:
            for i in range(N - row - 1):
                sources[npairs] = row
                targets[npairs] = row + i + 1
                npairs += 1

    return (sources, targets)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _reduce_triu_indices(np.ndarray[DTYPE_t, ndim=1] sources,
                         np.ndarray[DTYPE_t, ndim=1] targets,
                         np.ndarray[DTYPE_t, ndim=1] indices):

    if len(sources) == 0:
        return (np.zeros(0, dtype=DTYPE), np.zeros(0, dtype=DTYPE))

    cdef unsigned long long i, n, n_pairs, f_target
    cdef unsigned long long start = 0
    cdef np.ndarray[DTYPE_t, ndim=1] rsources
    cdef np.ndarray[DTYPE_t, ndim=1] rtargets

    n = len(indices)
    n_pairs = len(sources)
    f_target = targets[0]
    for i in range(n):
        if indices[i] == f_target:
            start = i - 1
            break

    rsources, rtargets = _triu_indices(n, start, start + n_pairs)

    return (rsources, rtargets)


@cython.boundscheck(False)
@cython.nonecheck(False)
def _union_of_indices(unsigned long long N,
                      np.ndarray[DTYPE_t, ndim=1] sources,
                      np.ndarray[DTYPE_t, ndim=1] targets):

    if len(sources) == 0:
        return np.zeros(0, dtype=DTYPE)

    cdef unsigned long long i_s = sources[0]
    cdef unsigned long long i_e = sources[-1]
    cdef unsigned long long j_s = targets[0]
    cdef unsigned long long j_e = targets[-1]
    cdef unsigned long long i = 0
    cdef unsigned long long c = 0
    cdef np.ndarray[DTYPE_t, ndim=1] indices = np.zeros(N, dtype=DTYPE)

    if i_s == i_e:
        indices[c] = i_s
        c += 1
        for i in range(j_s, j_e + 1):
            indices[c] = i
            c += 1
    else:
        for i in range(i_s, i_e + 1):
            indices[c] = i
            c += 1
        if i_e - i_s == 1:
            for i in range(i_e + 1, j_e + 1):
                indices[c] = i
                c += 1
            if j_s <= j_e:
                for i in range(j_e + 1, N):
                    indices[c] = i
                    c += 1
            else:
                for i in range(j_s, N):
                    indices[c] = i
                    c += 1
        else:
            for i in range(i_e + 1, N):
                indices[c] = i
                c += 1

    return indices[:c]
