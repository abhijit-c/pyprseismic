from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator

from ._geometry import compute_geometry


def _compute_ray_segments(
    x0_i: float,
    y0_i: float,
    xp_j: float,
    yp_j: float,
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """Trace a single ray and return (segment_lengths, pixel_indices).

    Returns None if the ray does not intersect any pixel.
    """
    N2 = N / 2

    # Grid lines.
    x_lines = np.arange(-N2, N2 + 1, dtype=np.float64)
    y_lines = np.arange(-N2, N2 + 1, dtype=np.float64)

    # Direction vector.
    a = xp_j - x0_i
    b = yp_j - y0_i

    # Parametric intersections with vertical grid lines (x = const).
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (x_lines - x0_i) / a
        yx = b * tx + y0_i

        # Parametric intersections with horizontal grid lines (y = const).
        ty = (y_lines - y0_i) / b
        xy = a * ty + x0_i

    # Collect all intersections.
    t = np.concatenate([tx, ty])
    xxy = np.concatenate([x_lines, xy])
    yxy = np.concatenate([yx, y_lines])

    # Sort by parameter t.
    order = np.argsort(t)
    xxy = xxy[order]
    yxy = yxy[order]

    # Clip to domain.
    inside = (xxy >= -N2) & (xxy <= N2) & (yxy >= -N2) & (yxy <= N2)
    xxy = xxy[inside]
    yxy = yxy[inside]

    if len(xxy) < 2:
        return None

    # Remove duplicate intersection points.
    dups = (np.abs(np.diff(xxy)) <= 1e-10) & (np.abs(np.diff(yxy)) <= 1e-10)
    keep = np.concatenate([[True], ~dups])
    xxy = xxy[keep]
    yxy = yxy[keep]

    if len(xxy) < 2:
        return None

    # Segment lengths.
    aval = np.sqrt(np.diff(xxy) ** 2 + np.diff(yxy) ** 2)

    # Midpoints shifted to [0, N] coordinates.
    xm = 0.5 * (xxy[:-1] + xxy[1:]) + N2
    ym = 0.5 * (yxy[:-1] + yxy[1:]) + N2

    # Pixel index (column-major, matching MATLAB's x(:) ordering).
    # MATLAB: col = floor(xm)*N + (N - floor(ym))  [1-indexed]
    # Python: col = floor(xm)*N + (N - 1 - floor(ym))  [0-indexed]
    col = np.floor(xm).astype(np.int64) * N + (N - 1 - np.floor(ym).astype(np.int64))

    # Filter out any segments with zero length.
    nonzero = aval > 0
    if not nonzero.any():
        return None

    return aval[nonzero], col[nonzero]


def seismictomo(
    N: int,
    s: int | None = None,
    p: int | None = None,
    *,
    sparse_matrix: bool = True,
) -> csr_matrix | LinearOperator:
    """Build the system matrix for 2D seismic ray tomography.

    Parameters
    ----------
    N : int
        Number of grid cells per dimension.
    s : int or None
        Number of sources (default: N).
    p : int or None
        Number of receivers (default: 2*N).
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

    if sparse_matrix:
        return _build_sparse(N, s, p, geom)
    return _build_linear_operator(N, s, p, geom)


def _build_sparse(
    N: int,
    s: int,
    p: int,
    geom: object,
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i in range(s):
        for j in range(p):
            result = _compute_ray_segments(
                geom.x0[i], geom.y0[i], geom.xp[j], geom.yp[j], N
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
    geom: object,
) -> LinearOperator:
    # Precompute all ray data.
    ray_data: dict[int, tuple[NDArray[np.float64], NDArray[np.int64]]] = {}
    for i in range(s):
        for j in range(p):
            result = _compute_ray_segments(
                geom.x0[i], geom.y0[i], geom.xp[j], geom.yp[j], N
            )
            if result is not None:
                ray_data[i * p + j] = result

    m = s * p
    n = N * N

    def matvec(x: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.zeros(m)
        for row_idx, (aval, col) in ray_data.items():
            y[row_idx] = aval @ x[col]
        return y

    def rmatvec(y: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.zeros(n)
        for row_idx, (aval, col) in ray_data.items():
            x[col] += y[row_idx] * aval
        return x

    return LinearOperator(
        shape=(m, n),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
