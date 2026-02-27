from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from pyprseismic import seismictomo


def test_sparse_shape():
    N, s, p = 4, 4, 8
    A = seismictomo(N, s, p)
    assert A.shape == (s * p, N * N)


def test_sparse_nonnegative():
    A = seismictomo(4, 4, 8)
    assert A.min() >= 0


def test_sparse_vs_linop_matvec():
    N, s, p = 4, 4, 8
    A_sparse = seismictomo(N, s, p, sparse_matrix=True)
    A_linop = seismictomo(N, s, p, sparse_matrix=False)

    rng = np.random.default_rng(0)
    x = rng.random(N * N)

    b_sparse = A_sparse @ x
    b_linop = A_linop.matvec(x)
    np.testing.assert_allclose(b_sparse, b_linop, atol=1e-12)


def test_sparse_vs_linop_rmatvec():
    N, s, p = 4, 4, 8
    A_sparse = seismictomo(N, s, p, sparse_matrix=True)
    A_linop = seismictomo(N, s, p, sparse_matrix=False)

    rng = np.random.default_rng(1)
    y = rng.random(s * p)

    x_sparse = A_sparse.T @ y
    x_linop = A_linop.rmatvec(y)
    np.testing.assert_allclose(x_sparse, x_linop, atol=1e-12)


def test_defaults():
    N = 4
    A = seismictomo(N)
    assert A.shape == (N * 2 * N, N * N)  # s=N, p=2*N


def test_returns_sparse_by_default():
    A = seismictomo(4, 4, 8)
    assert issparse(A)


def test_returns_linop():
    A = seismictomo(4, 4, 8, sparse_matrix=False)
    assert not issparse(A)
    assert hasattr(A, "matvec")
    assert hasattr(A, "rmatvec")


def test_larger_problem():
    N, s, p = 8, 8, 16
    A = seismictomo(N, s, p)
    assert A.shape == (s * p, N * N)
    assert A.nnz > 0
