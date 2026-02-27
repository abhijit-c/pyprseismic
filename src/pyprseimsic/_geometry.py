from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SeismicGeometry:
    x0: NDArray[np.float64]
    y0: NDArray[np.float64]
    xp: NDArray[np.float64]
    yp: NDArray[np.float64]
    N: int
    s: int
    p: int


def compute_geometry(N: int, s: int, p: int) -> SeismicGeometry:
    """Compute source and receiver positions for seismic tomography.

    Sources are placed on the right boundary (x = N/2).
    Receivers are split between the left boundary (x = -N/2) and the top
    boundary (y = N/2).

    Parameters
    ----------
    N : int
        Number of grid cells per dimension.
    s : int
        Number of sources.
    p : int
        Number of receivers.

    Returns
    -------
    SeismicGeometry
        Dataclass containing source/receiver positions.
    """
    N2 = N / 2

    # Source positions on right boundary.
    Ns = (N / s) / 2
    x0 = N2 * np.ones(s)
    y0 = np.linspace(-N2 + Ns, N2 - Ns, s)

    # Receiver positions: floor(p/2) on left, ceil(p/2) on top.
    p1 = int(np.ceil(p / 2))
    p2 = int(np.floor(p / 2))
    Np1 = (N / p1) / 2
    Np2 = (N / p2) / 2

    xp = np.concatenate([
        -N2 * np.ones(p2),
        np.linspace(-N2 + Np1, N2 - Np1, p1),
    ])
    yp = np.concatenate([
        np.linspace(-N2 + Np2, N2 - Np2, p2),
        N2 * np.ones(p1),
    ])

    return SeismicGeometry(x0=x0, y0=y0, xp=xp, yp=yp, N=N, s=s, p=p)
