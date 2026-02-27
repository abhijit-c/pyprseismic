from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from pyprseismic import PhantomType, ProblemInfo, SeismicProblem, WaveModel


def test_wave_model_values():
    assert WaveModel.RAY.value == "ray"
    assert WaveModel.FRESNEL.value == "fresnel"


def test_phantom_type_values():
    assert PhantomType.TECTONIC.value == "tectonic"
    assert PhantomType.SHEPPLOGAN.value == "shepplogan"


def test_problem_info_frozen():
    info = ProblemInfo("tomography", (8, 8), (16, 8))
    assert info.x_size == (8, 8)
    assert info.b_size == (16, 8)


def test_seismic_problem():
    A = csr_matrix((4, 9))
    b = np.zeros(4)
    x = np.zeros(9)
    info = ProblemInfo("tomography", (3, 3), (4, 1))
    prob = SeismicProblem(A=A, b=b, x=x, info=info)
    assert prob.A.shape == (4, 9)
    assert prob.b.shape == (4,)
    assert prob.x.shape == (9,)
