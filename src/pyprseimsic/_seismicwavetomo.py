from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator

from ._geometry import compute_geometry


def _fresnel_kernel(
    source: tuple[float, float],
    receiver: tuple[float, float],
    omega_scaled: float,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the Fresnel sensitivity kernel for a source-receiver pair.

    Ref: seismicwavetomo.m lines 273-289 (simple_fat_kernel).
    """
    alpha = 10.0

    tSX = np.sqrt((xx - source[0]) ** 2 + (yy - source[1]) ** 2)
    tRX = np.sqrt((xx - receiver[0]) ** 2 + (yy - receiver[1]) ** 2)
    distSR = np.sqrt((source[0] - receiver[0]) ** 2 + (source[1] - receiver[1]) ** 2)
    delta_t = tSX + tRX - distSR

    S = np.cos(2 * np.pi * delta_t * omega_scaled) * np.exp(
        -(alpha * delta_t * omega_scaled) ** 2
    )

    s_sum = S.sum()
    if s_sum != 0:
        S = distSR * S / s_sum

    return S


def _compute_kernel_entries(
    source: tuple[float, float],
    receiver: tuple[float, float],
    omega_scaled: float,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    N: int,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """Compute nonzero kernel values and their pixel indices for one ray."""
    S = _fresnel_kernel(source, receiver, omega_scaled, xx, yy)
    S[S < tol] = 0.0

    # np.nonzero returns (row_indices, col_indices) in array coords.
    # In the MATLAB code: [ym, xm, aval] = find(sens)
    # MATLAB find returns (row, col) in 1-indexed form.
    # np.nonzero on our (N, N) array returns 0-indexed (ym, xm).
    ym, xm = np.nonzero(S)
    if len(ym) == 0:
        return None

    aval = S[ym, xm]

    # Pixel index: MATLAB (xm-1)*N + ym → Python xm*N + ym (already 0-indexed).
    col = xm.astype(np.int64) * N + ym.astype(np.int64)

    return aval, col


def seismicwavetomo(
    N: int,
    s: int | None = None,
    p: int | None = None,
    omega: float = 10.0,
    *,
    sparse_matrix: bool = True,
) -> csr_matrix | LinearOperator:
    """Build the system matrix for 2D seismic Fresnel-zone tomography.

    Parameters
    ----------
    N : int
        Number of grid cells per dimension.
    s : int or None
        Number of sources (default: N).
    p : int or None
        Number of receivers (default: 2*N).
    omega : float
        Dominant frequency of the propagating wave (default: 10).
    sparse_matrix : bool
        If True return a sparse CSR matrix, otherwise a
        ``LinearOperator``.

    Returns
    -------
    csr_matrix or LinearOperator
        System matrix of shape ``(s*p, N*N)``.
    """
    if s is None:
        s = N
    if p is None:
        p = 2 * N

    geom = compute_geometry(N, s, p)
    omega_scaled = omega / N

    N2 = N / 2
    xrange = np.linspace(-N2 + 0.5, N2 - 0.5, N)
    yrange = np.linspace(N2 - 0.5, -N2 + 0.5, N)
    xx, yy = np.meshgrid(xrange, yrange)

    if sparse_matrix:
        return _build_sparse(N, s, p, omega_scaled, xx, yy, geom)
    return _build_linear_operator(N, s, p, omega_scaled, xx, yy, geom)


def _build_sparse(
    N: int,
    s: int,
    p: int,
    omega_scaled: float,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    geom: object,
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i in range(s):
        source = (geom.x0[i], geom.y0[i])
        for j in range(p):
            receiver = (geom.xp[j], geom.yp[j])
            result = _compute_kernel_entries(
                source, receiver, omega_scaled, xx, yy, N
            )
            if result is not None:
                aval, col = result
                row_idx = i * p + j
                rows.extend([row_idx] * len(aval))
                cols.extend(col.tolist())
                vals.extend(aval.tolist())

    if not rows:
        return csr_matrix((s * p, N * N))

    return coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(s * p, N * N),
    ).tocsr()


def _build_linear_operator(
    N: int,
    s: int,
    p: int,
    omega_scaled: float,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    geom: object,
) -> LinearOperator:
    # Precompute all kernel data.
    kernel_data: dict[int, tuple[NDArray[np.float64], NDArray[np.int64]]] = {}
    for i in range(s):
        source = (geom.x0[i], geom.y0[i])
        for j in range(p):
            receiver = (geom.xp[j], geom.yp[j])
            result = _compute_kernel_entries(
                source, receiver, omega_scaled, xx, yy, N
            )
            if result is not None:
                kernel_data[i * p + j] = result

    m = s * p
    n = N * N

    def matvec(x: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.zeros(m)
        for row_idx, (aval, col) in kernel_data.items():
            y[row_idx] = aval @ x[col]
        return y

    def rmatvec(y: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.zeros(n)
        for row_idx, (aval, col) in kernel_data.items():
            x[col] += y[row_idx] * aval
        return x

    return LinearOperator(
        shape=(m, n),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
