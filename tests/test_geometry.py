from __future__ import annotations

import numpy as np

from pyprseismic._geometry import compute_geometry


def test_source_positions():
    geom = compute_geometry(8, 4, 8)
    # Sources on the right boundary: x = N/2 = 4
    assert np.all(geom.x0 == 4.0)
    assert len(geom.x0) == 4


def test_receiver_count():
    geom = compute_geometry(8, 4, 10)
    assert len(geom.xp) == 10
    assert len(geom.yp) == 10


def test_receiver_split():
    N, s, p = 8, 4, 10
    geom = compute_geometry(N, s, p)
    p2 = p // 2  # floor(10/2) = 5 on left
    p1 = (p + 1) // 2  # ceil(10/2) = 5 on top

    # First p2 receivers on left boundary.
    np.testing.assert_allclose(geom.xp[:p2], -N / 2)
    # Last p1 receivers on top boundary.
    np.testing.assert_allclose(geom.yp[p2:], N / 2)
    assert len(geom.yp[p2:]) == p1


def test_geometry_default_N():
    geom = compute_geometry(4, 4, 8)
    assert geom.N == 4
    assert geom.s == 4
    assert geom.p == 8
