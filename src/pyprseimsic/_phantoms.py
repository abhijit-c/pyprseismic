from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def phantom_gallery(
    name: str,
    N: int,
    *,
    seed: int | None = None,
    **kwargs: float | int,
) -> NDArray[np.float64]:
    """Generate a phantom image of the given type.

    Parameters
    ----------
    name : str
        Phantom type. One of ``'tectonic'``, ``'shepplogan'``, ``'smooth'``,
        ``'grains'``, ``'ppower'``, ``'threephases'``, ``'threephasessmooth'``.
    N : int
        Image size (N x N).
    seed : int or None
        Random seed for reproducibility (random phantoms only).
    **kwargs
        Additional keyword arguments forwarded to the specific phantom
        generator.

    Returns
    -------
    NDArray
        Phantom image of shape ``(N, N)`` with values in ``[0, 1]``.
    """
    generators = {
        "tectonic": _tectonic,
        "shepplogan": _shepplogan,
        "smooth": _smooth,
        "grains": _grains,
        "ppower": _ppower,
        "threephases": _threephases,
        "threephasessmooth": _threephasessmooth,
    }
    if name not in generators:
        raise ValueError(
            f"Unknown phantom '{name}'. Choose from: {list(generators)}"
        )
    return generators[name](N, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Tectonic phantom
# ---------------------------------------------------------------------------

def _tectonic(N: int, *, seed: int | None = None) -> NDArray[np.float64]:
    """Two tectonic plates with a subduction zone.

    Ref: phantomgallery.m lines 422-454.
    """
    x = np.zeros((N, N), dtype=np.float64)

    N5 = round(N / 5)
    N13 = round(N / 13)
    N7 = round(N / 7)
    N20 = round(N / 20)

    # The right plate (MATLAB: x(N5:N5+N7, 5*N13:end) = 0.75).
    # MATLAB 1-indexed inclusive → Python 0-indexed: rows [N5-1, N5+N7-1],
    # cols [5*N13-1, N-1].
    x[N5 - 1 : N5 + N7, 5 * N13 - 1 :] = 0.75

    # The angle of the right plate.
    # MATLAB: i starts at N5, decrements for odd j.
    i = N5  # MATLAB 1-indexed value
    for j_one in range(1, N20 + 1):  # j_one is 1-based like MATLAB j
        if j_one % 2 != 0:
            i -= 1
            # MATLAB: x(i, 5*N13+j:end) = 0.75
            x[i - 1, 5 * N13 + j_one - 1 :] = 0.75

    # The left plate before the break.
    # MATLAB: x(N5:N5+N5, 1:5*N13) = 1
    x[N5 - 1 : N5 + N5, 0 : 5 * N13] = 1

    # The break from the left plate.
    # MATLAB: vector = N5:N5+N5 (1-indexed)
    vector = np.arange(N5, N5 + N5 + 1)  # 1-indexed values
    for j_one in range(5 * N13, min(12 * N13, N) + 1):  # 1-indexed
        if j_one % 2 != 0:
            vector = vector + 1
        # MATLAB: x(vector, j) = 1
        x[vector - 1, j_one - 1] = 1

    return x


# ---------------------------------------------------------------------------
# Shepp-Logan phantom
# ---------------------------------------------------------------------------

def _shepplogan(N: int, *, seed: int | None = None) -> NDArray[np.float64]:
    """Modified Shepp-Logan phantom.

    Ref: phantomgallery.m lines 155-214.
    """
    #         A      a      b     x0     y0    phi(deg)
    ellipses = np.array([
        [ 1.0,  0.69,   0.92,   0.0,    0.0,    0.0],
        [-0.8,  0.6624, 0.8740, 0.0,   -0.0184, 0.0],
        [-0.2,  0.1100, 0.3100, 0.22,   0.0,  -18.0],
        [-0.2,  0.1600, 0.4100,-0.22,   0.0,   18.0],
        [ 0.1,  0.2100, 0.2500, 0.0,    0.35,   0.0],
        [ 0.1,  0.0460, 0.0460, 0.0,    0.1,    0.0],
        [ 0.1,  0.0460, 0.0460, 0.0,   -0.1,    0.0],
        [ 0.1,  0.0460, 0.0230,-0.08,  -0.605,  0.0],
        [ 0.1,  0.0230, 0.0230, 0.0,   -0.606,  0.0],
        [ 0.1,  0.0230, 0.0460, 0.06,  -0.605,  0.0],
    ])

    # Coordinate grid: xn = ((0:N-1) - (N-1)/2) / ((N-1)/2)
    xn = (np.arange(N) - (N - 1) / 2) / ((N - 1) / 2)
    Xn = np.tile(xn, (N, 1))        # rows constant
    Yn = np.rot90(Xn)               # columns constant

    X = np.zeros((N, N), dtype=np.float64)

    for row in ellipses:
        A_val = row[0]
        a2 = row[1] ** 2
        b2 = row[2] ** 2
        x0 = row[3]
        y0 = row[4]
        phi = row[5] * np.pi / 180

        xc = Xn - x0
        yc = Yn - y0

        mask = (
            (xc * np.cos(phi) + yc * np.sin(phi)) ** 2 / a2
            + (yc * np.cos(phi) - xc * np.sin(phi)) ** 2 / b2
        ) <= 1.0

        X[mask] += A_val

    X[X < 0] = 0.0
    return X


# ---------------------------------------------------------------------------
# Smooth phantom
# ---------------------------------------------------------------------------

def _smooth(
    N: int,
    *,
    seed: int | None = None,
    num_gaussians: int = 4,
) -> NDArray[np.float64]:
    """Smooth image from sum of anisotropic Gaussians.

    Ref: phantomgallery.m lines 216-232.
    """
    p = min(num_gaussians, 4)
    I_grid, J_grid = np.meshgrid(
        np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij"
    )
    sigma = 0.25 * N
    c = np.array([
        [0.6 * N, 0.6 * N],
        [0.5 * N, 0.3 * N],
        [0.2 * N, 0.7 * N],
        [0.8 * N, 0.2 * N],
    ])
    a = np.array([1.0, 0.5, 0.7, 0.9])

    im = np.zeros((N, N), dtype=np.float64)
    for k in range(p):
        im += a[k] * np.exp(
            -(I_grid - c[k, 0]) ** 2 / (1.2 * sigma) ** 2
            - (J_grid - c[k, 1]) ** 2 / sigma**2
        )
    im /= im.max()
    return im


# ---------------------------------------------------------------------------
# Grains (Voronoi) phantom
# ---------------------------------------------------------------------------

def _grains(
    N: int,
    *,
    seed: int | None = None,
    num_cells: int | None = None,
) -> NDArray[np.float64]:
    """Voronoi cell image.

    Ref: phantomgallery.m lines 352-389.
    """
    rng = np.random.default_rng(seed)
    if num_cells is None:
        num_cells = round(3 * np.sqrt(N))

    dN = round(N / 10)
    Nbig = N + 2 * dN

    # Random grain centers.
    xG = rng.integers(1, Nbig + 1, size=num_cells)
    yG = rng.integers(1, Nbig + 1, size=num_cells)

    # Pixel coordinate grid.
    coords = np.arange(1, Nbig + 1)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Compute distance to each grain center and find nearest.
    dist = np.empty((X_flat.size, num_cells), dtype=np.float64)
    for k in range(num_cells):
        dist[:, k] = (X_flat - xG[k]) ** 2 + (Y_flat - yG[k]) ** 2

    min_idx = np.argmin(dist, axis=1)
    im_big = min_idx.reshape(Nbig, Nbig).astype(np.float64)

    # Extract center, scale to [0, 1].
    im = im_big[dN : dN + N, dN : dN + N]
    mx = im.max()
    if mx > 0:
        im = im / mx
    return im


# ---------------------------------------------------------------------------
# Ppower phantom
# ---------------------------------------------------------------------------

def _ppower(
    N: int,
    *,
    seed: int | None = None,
    rel_nonzero: float = 0.3,
    smoothness: float = 2.0,
) -> NDArray[np.float64]:
    """Power-law filtered random image.

    Ref: phantomgallery.m lines 393-418.
    """
    rng = np.random.default_rng(seed)

    Nodd = N % 2 != 0
    Nwork = N + 1 if Nodd else N

    P = rng.standard_normal((Nwork, Nwork))
    I_grid, J_grid = np.meshgrid(
        np.arange(1, Nwork + 1), np.arange(1, Nwork + 1), indexing="ij"
    )
    U = (
        ((2 * I_grid - 1) / Nwork - 1) ** 2
        + ((2 * J_grid - 1) / Nwork - 1) ** 2
    ) ** (-smoothness / 2)
    F = U * np.exp(2 * np.pi * 1j * P)
    F = np.abs(np.fft.ifft2(F))

    f = np.sort(F.ravel())[::-1]
    k = round(rel_nonzero * Nwork**2)
    F[F < f[k - 1]] = 0  # k-1 because 0-indexed
    F = F / f[0]

    if Nodd:
        F = F[:N, :N]

    return F


# ---------------------------------------------------------------------------
# Three phases phantom
# ---------------------------------------------------------------------------

def _threephases(
    N: int,
    *,
    seed: int | None = None,
    num_domains: int = 100,
) -> NDArray[np.float64]:
    """Three-phase image with values {0, 0.5, 1}.

    Ref: phantomgallery.m lines 257-292.
    """
    rng = np.random.default_rng(seed)
    p = num_domains

    I_grid, J_grid = np.meshgrid(
        np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij"
    )

    # First image (exponential/cubic decay).
    sigma1 = 0.025 * N
    c1 = rng.random((p, 2)) * N
    im1 = np.zeros((N, N), dtype=np.float64)
    for k in range(p):
        im1 += np.exp(
            -np.abs(I_grid - c1[k, 0]) ** 3 / (2.5 * sigma1) ** 3
            - np.abs(J_grid - c1[k, 1]) ** 3 / sigma1**3
        )
    t1 = 0.35
    im1[im1 < t1] = 0.0
    im1[im1 >= t1] = 2.0

    # Second image (Gaussian decay).
    sigma2 = 0.025 * N
    c2 = rng.random((p, 2)) * N
    im2 = np.zeros((N, N), dtype=np.float64)
    for k in range(p):
        im2 += np.exp(
            -(I_grid - c2[k, 0]) ** 2 / (2 * sigma2) ** 2
            - (J_grid - c2[k, 1]) ** 2 / sigma2**2
        )
    t2 = 0.55
    im2[im2 < t2] = 0.0
    im2[im2 >= t2] = 1.0

    # Combine.
    im = im1 + im2
    im[im == 3] = 1.0
    mx = im.max()
    if mx > 0:
        im = im / mx
    return im


# ---------------------------------------------------------------------------
# Three phases smooth phantom
# ---------------------------------------------------------------------------

def _threephasessmooth(
    N: int,
    *,
    seed: int | None = None,
    num_domains: int = 100,
    intensity_variation: float = 1.8,
) -> NDArray[np.float64]:
    """Smoothly varying three-phase image on a ppower background.

    Ref: phantomgallery.m lines 296-335.
    """
    rng = np.random.default_rng(seed)
    p = num_domains
    v = intensity_variation

    I_grid, J_grid = np.meshgrid(
        np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij"
    )

    # First image (exponential/cubic decay).
    sigma1 = 0.025 * N
    c1 = rng.random((p, 2)) * N
    im1 = np.zeros((N, N), dtype=np.float64)
    for k in range(p):
        im1 += np.exp(
            -np.abs(I_grid - c1[k, 0]) ** 3 / (2.5 * sigma1) ** 3
            - np.abs(J_grid - c1[k, 1]) ** 3 / sigma1**3
        )
    t1 = 0.35
    im1[im1 < t1] = 0.0
    mask1 = im1 >= t1
    if mask1.any():
        im1[mask1] = (im1[mask1] - im1[mask1].min()) / im1.max() * v + 0.8

    # Second image (Gaussian decay).
    sigma2 = 0.025 * N
    c2 = rng.random((p, 2)) * N
    im2 = np.zeros((N, N), dtype=np.float64)
    for k in range(p):
        im2 += np.exp(
            -(I_grid - c2[k, 0]) ** 2 / (2 * sigma2) ** 2
            - (J_grid - c2[k, 1]) ** 2 / sigma2**2
        )
    t2 = 0.55
    im2[im2 < t2] = 0.0
    mask2 = im2 >= t2
    if mask2.any():
        im2[mask2] = (im2[mask2] - im2[mask2].min()) / im2.max() * v + 0.3

    # Combine onto ppower background.
    # Use a child seed derived from the rng so that the ppower call is
    # deterministic when a seed is provided.
    ppower_seed = int(rng.integers(0, 2**31)) if seed is not None else None
    im = (v / 3) * _ppower(N, seed=ppower_seed, rel_nonzero=1.0, smoothness=2.5)
    im[im1 > 0] = im1[im1 > 0]
    im[im2 > 0] = im2[im2 > 0]
    mx = im.max()
    if mx > 0:
        im = im / mx
    return im
