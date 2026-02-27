from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator


class WaveModel(Enum):
    RAY = "ray"
    FRESNEL = "fresnel"


class PhantomType(Enum):
    TECTONIC = "tectonic"
    SHEPPLOGAN = "shepplogan"
    SMOOTH = "smooth"
    GRAINS = "grains"
    PPOWER = "ppower"
    THREEPHASES = "threephases"
    THREEPHASESSMOOTH = "threephasessmooth"


@dataclass(frozen=True)
class ProblemInfo:
    problem_type: str
    x_size: tuple[int, int]
    b_size: tuple[int, int]


@dataclass(frozen=True)
class SeismicProblem:
    A: csr_matrix | LinearOperator
    b: NDArray[np.float64]
    x: NDArray[np.float64]
    info: ProblemInfo
