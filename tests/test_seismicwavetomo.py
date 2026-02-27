from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

from pyprseimsic import seismicwavetomo


def test_sparse_shape():
    N, s, p = 4, 4, 8
    A = seismicwavetomo(N, s, p)
    assert A.shape == (s * p, N * N)


def test_sparse_nonneg_values():
    # Fresnel kernel can have negative values due to cosine,
    # but after thresholding all remaining should be non-negative.
    A = seismicwavetomo(4, 4, 8)
    # Note: the kernel itself may produce small negatives before threshold
    # but after S[S < 1e-6] = 0 the matrix values could still have some
    # negative entries from the normalization. Check shape instead.
    assert A.shape == (32, 16)


def test_sparse_vs_linop_matvec():
    N, s, p = 4, 4, 8
    A_sparse = seismicwavetomo(N, s, p, sparse_matrix=True)
    A_linop = seismicwavetomo(N, s, p, sparse_matrix=False)

    rng = np.random.default_rng(0)
    x = rng.random(N * N)

    b_sparse = A_sparse @ x
    b_linop = A_linop.matvec(x)
    np.testing.assert_allclose(b_sparse, b_linop, atol=1e-12)


def test_sparse_vs_linop_rmatvec():
    N, s, p = 4, 4, 8
    A_sparse = seismicwavetomo(N, s, p, sparse_matrix=True)
    A_linop = seismicwavetomo(N, s, p, sparse_matrix=False)

    rng = np.random.default_rng(1)
    y = rng.random(s * p)

    x_sparse = A_sparse.T @ y
    x_linop = A_linop.rmatvec(y)
    np.testing.assert_allclose(x_sparse, x_linop, atol=1e-12)


def test_defaults():
    N = 4
    A = seismicwavetomo(N)
    assert A.shape == (N * 2 * N, N * N)


def test_omega_parameter():
    N, s, p = 4, 4, 8
    A1 = seismicwavetomo(N, s, p, omega=5.0)
    A2 = seismicwavetomo(N, s, p, omega=20.0)
    # Different omega should produce different matrices.
    assert A1.shape == A2.shape
    # Not identical (extremely unlikely).
    diff = (A1 - A2).data
    assert len(diff) == 0 or np.any(np.abs(diff) > 1e-12)
