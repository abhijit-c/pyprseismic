from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from ._phantoms import phantom_gallery
from ._seismictomo import seismictomo
from ._seismicwavetomo import seismicwavetomo
from ._types import PhantomType, ProblemInfo, SeismicProblem, WaveModel


def prseismic(
    N: int = 256,
    *,
    phantom: str | PhantomType | NDArray[np.float64] = "tectonic",
    wave_model: str | WaveModel = "ray",
    s: int | None = None,
    p: int | None = None,
    omega: float = 10.0,
    sparse_matrix: bool = True,
    seed: int | None = None,
    **phantom_kwargs: float | int,
) -> SeismicProblem:
    """Generate a 2D seismic travel-time tomography test problem.

    Parameters
    ----------
    N : int
        Image size (N x N pixels). Default 256.
    phantom : str, PhantomType, or ndarray
        Phantom image specification. Either a name (string or enum) or a
        custom 2D square array.
    wave_model : str or WaveModel
        ``'ray'`` for straight-ray model, ``'fresnel'`` for Fresnel-zone
        model.
    s : int or None
        Number of sources on the right boundary. Default: N.
    p : int or None
        Number of receivers on the left and top boundaries. Default: 2*N.
    omega : float
        Dominant frequency (Fresnel model only). Default: 10.
    sparse_matrix : bool
        If True return a sparse CSR matrix for A, otherwise a
        ``LinearOperator``.
    seed : int or None
        Random seed passed to phantom generators.
    **phantom_kwargs
        Extra keyword arguments forwarded to ``phantom_gallery``.

    Returns
    -------
    SeismicProblem
        Named tuple with fields ``A``, ``b``, ``x``, ``info``.
    """
    # Resolve phantom image.
    if isinstance(phantom, np.ndarray):
        if phantom.ndim != 2:
            raise ValueError("Custom phantom must be a 2-D array")
        if phantom.shape[0] != phantom.shape[1]:
            raise ValueError("Custom phantom must be square")
        N = phantom.shape[0]
        image = phantom.astype(np.float64)
    else:
        name = phantom.value if isinstance(phantom, PhantomType) else phantom
        image = phantom_gallery(name, N, seed=seed, **phantom_kwargs)

    # Defaults for s and p.
    if s is None:
        s = N
    if p is None:
        p = 2 * N

    # Resolve wave model.
    wm = wave_model.value if isinstance(wave_model, WaveModel) else wave_model

    # Build system matrix.
    A: csr_matrix | LinearOperator
    if wm == "ray":
        A = seismictomo(N, s, p, sparse_matrix=sparse_matrix)
    elif wm == "fresnel":
        A = seismicwavetomo(N, s, p, omega, sparse_matrix=sparse_matrix)
    else:
        raise ValueError(f"Unknown wave model '{wm}'. Use 'ray' or 'fresnel'.")

    # Flatten image in column-major order (matching MATLAB's x(:)).
    x = image.ravel(order="F")

    # Compute measurements.
    if sparse_matrix:
        b = A @ x
    else:
        b = A.matvec(x)

    info = ProblemInfo(
        problem_type="tomography",
        x_size=(N, N),
        b_size=(p, s),
    )

    return SeismicProblem(A=A, b=b, x=x, info=info)
