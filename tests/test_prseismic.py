from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

from pyprseismic import PhantomType, SeismicProblem, WaveModel, prseismic


def test_default_call():
    prob = prseismic(8)
    assert isinstance(prob, SeismicProblem)
    assert prob.A.shape == (8 * 2 * 8, 8 * 8)
    assert prob.b.shape == (8 * 2 * 8,)
    assert prob.x.shape == (8 * 8,)
    assert prob.info.x_size == (8, 8)


def test_ray_model():
    prob = prseismic(4, wave_model="ray", s=4, p=8)
    assert prob.A.shape == (32, 16)
    assert issparse(prob.A)


def test_fresnel_model():
    prob = prseismic(4, wave_model="fresnel", s=4, p=8)
    assert prob.A.shape == (32, 16)


def test_enum_inputs():
    prob = prseismic(
        4,
        phantom=PhantomType.TECTONIC,
        wave_model=WaveModel.RAY,
        s=4,
        p=8,
    )
    assert prob.A.shape == (32, 16)


def test_custom_phantom():
    custom = np.ones((4, 4))
    prob = prseismic(phantom=custom, s=4, p=8)
    assert prob.A.shape == (32, 16)
    np.testing.assert_allclose(prob.x, 1.0)


def test_custom_phantom_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        prseismic(phantom=np.ones((3, 4)))


def test_custom_phantom_3d_raises():
    with pytest.raises(ValueError, match="2-D"):
        prseismic(phantom=np.ones((3, 3, 3)))


def test_linear_operator_mode():
    prob = prseismic(4, s=4, p=8, sparse_matrix=False)
    assert not issparse(prob.A)
    assert prob.b.shape == (32,)


def test_b_equals_Ax():
    prob = prseismic(4, s=4, p=8)
    b_check = prob.A @ prob.x
    np.testing.assert_allclose(prob.b, b_check, atol=1e-12)


def test_b_equals_Ax_linop():
    prob = prseismic(4, s=4, p=8, sparse_matrix=False)
    b_check = prob.A.matvec(prob.x)
    np.testing.assert_allclose(prob.b, b_check, atol=1e-12)


def test_problem_info():
    prob = prseismic(8, s=4, p=16)
    assert prob.info.problem_type == "tomography"
    assert prob.info.x_size == (8, 8)
    assert prob.info.b_size == (16, 4)


def test_all_phantoms():
    for name in [
        "tectonic",
        "shepplogan",
        "smooth",
        "grains",
        "ppower",
        "threephases",
        "threephasessmooth",
    ]:
        prob = prseismic(8, phantom=name, s=4, p=8, seed=42)
        assert prob.A.shape == (32, 64)
        assert prob.b.shape == (32,)


def test_unknown_wave_model_raises():
    with pytest.raises(ValueError, match="Unknown wave model"):
        prseismic(4, wave_model="invalid")
