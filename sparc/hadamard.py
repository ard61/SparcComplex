"""
Functions to calculate the product of a vector and a Hadamard matrix.
Copied from Adam Greig, https://github.com/sigproc/sparc-amp/blob/master/sparc_amp.ipynb
"""

import numpy as np

try:
    from pyfht import fht_inplace
except ImportError:
    import warnings
    warnings.warn("Using very slow Python version of fht, please install pyfht")
    def fht_inplace(x):
        N = len(x)
        i = N>>1
        while i:
            for k in range(0, N, 2*i):
                for j in range(k, i+k):
                    ij = i|j
                    temp = x[j]
                    x[j] += x[ij]
                    x[ij] = temp - x[ij]
            i = i >> 1

def fht(n, m, ordering):
    """
    Returns functions to compute the sub-sampled Walsh-Hadamard transform,
    i.e., operating with a wide rectangular matrix of random +/-1 entries.

    n: number of rows
    m: number of columns

    It is most efficient (but not required) for max(m+1,n+1) to be a power of 2.

    ordering: n-long array of row indices in [1, max(m,n)] to
              implement subsampling

    Returns (Ax, Ay, ordering):
        Ax(x): computes A.x (of length n), with x having length m
        Ay(y): computes A'.y (of length m), with y having length n
        ordering: the ordering in use, which may have been generated from seed
    """
    assert n > 0, "n must be positive"
    assert m > 0, "m must be positive"
    w = 2**int(np.ceil(np.log2(max(m+1, n+1))))

    assert ordering.shape == (n,)

    def Ax(x):
        assert x.size == m, "x must be m long"
        y = np.zeros(w)
        y[w-m:] = x.reshape(m)
        fht_inplace(y)
        return y[ordering]

    def Ay(y):
        assert y.size == n, "input must be n long"
        x = np.zeros(w)
        x[ordering] = y
        fht_inplace(x)
        return x[w-m:]

    return Ax, Ay

def fht_block(n, m, l, ordering):
    """
    As `fht`, but computes in `l` blocks of size `n` by `m`, potentially
    offering substantial speed improvements.

    n: number of rows
    m: number of columns per block
    l: number of blocks

    It is most efficient (though not required) when max(m+1,n+1) is a power of 2.

    ordering: (l, n) shaped array of row indices in [1, max(m, n)] to
              implement subsampling

    Returns (Ax, Ay, ordering):
        Ax(x): computes A.x (of length n), with x having length l*m
        Ay(y): computes A'.y (of length l*m), with y having length n
        ordering: the ordering in use, which may have been generated from seed
    """
    assert n > 0, "n must be positive"
    assert m > 0, "m must be positive"
    assert l > 0, "l must be positive"
    assert ordering.shape == (l, n)

    def Ax(x):
        assert x.size == l*m
        out = np.zeros(n)
        for ll in range(l):
            ax, ay = fht(n, m, ordering=ordering[ll])
            out += ax(x[ll*m:(ll+1)*m])
        return out

    def Ay(y):
        assert y.size == n
        out = np.empty(l*m)
        for ll in range(l):
            ax, ay = fht(n, m, ordering=ordering[ll])
            out[ll*m:(ll+1)*m] = ay(y)
        return out

    return Ax, Ay

def sparc_transforms(n, M, L, ordering):
    Ax, Ay = fht_block(n, M, L, ordering=ordering)
    def Ab(b):
        return Ax(b) / np.sqrt(n)
    def Az(z):
        return Ay(z) / np.sqrt(n)
    return Ab, Az

def gen_ordering(n, m, l, seed=0):
    w = 2**int(np.ceil(np.log2(max(m+1, n+1))))
    rng = np.random.RandomState(seed)
    ordering = np.empty((l, n), dtype=np.uint32)
    idxs = np.arange(1, w, dtype=np.uint32)
    for ll in range(l):
        rng.shuffle(idxs)
        ordering[ll] = idxs[:n]
    return ordering