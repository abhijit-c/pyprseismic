from __future__ import annotations

import numpy as np
import pytest

from pyprseismic import phantom_gallery


PHANTOM_NAMES = [
    "tectonic",
    "shepplogan",
    "smooth",
    "grains",
    "ppower",
    "threephases",
    "threephasessmooth",
]


@pytest.mark.parametrize("name", PHANTOM_NAMES)
def test_phantom_shape(name):
    N = 16
    im = phantom_gallery(name, N, seed=42)
    assert im.shape == (N, N)


@pytest.mark.parametrize("name", PHANTOM_NAMES)
def test_phantom_range(name):
    N = 16
    im = phantom_gallery(name, N, seed=42)
    assert im.min() >= 0.0 - 1e-12
    assert im.max() <= 1.0 + 1e-12


def test_tectonic_deterministic():
    im1 = phantom_gallery("tectonic", 32)
    im2 = phantom_gallery("tectonic", 32)
    np.testing.assert_array_equal(im1, im2)


def test_shepplogan_deterministic():
    im1 = phantom_gallery("shepplogan", 32)
    im2 = phantom_gallery("shepplogan", 32)
    np.testing.assert_array_equal(im1, im2)


def test_seeded_phantoms_reproducible():
    for name in ["grains", "ppower", "threephases", "threephasessmooth"]:
        im1 = phantom_gallery(name, 16, seed=123)
        im2 = phantom_gallery(name, 16, seed=123)
        np.testing.assert_array_equal(im1, im2)


def test_unknown_phantom_raises():
    with pytest.raises(ValueError, match="Unknown phantom"):
        phantom_gallery("nonexistent", 16)


def test_tectonic_has_expected_values():
    im = phantom_gallery("tectonic", 64)
    unique = set(np.unique(im))
    # Should contain at least 0, 0.75, 1.0
    assert 0.0 in unique
    assert 0.75 in unique
    assert 1.0 in unique


def test_smooth_num_gaussians():
    im1 = phantom_gallery("smooth", 16, num_gaussians=1)
    im4 = phantom_gallery("smooth", 16, num_gaussians=4)
    # Both valid but may differ
    assert im1.shape == (16, 16)
    assert im4.shape == (16, 16)
