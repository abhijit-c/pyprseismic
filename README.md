# PyPRSeismic

A Python port of the MATLAB [IRTools](https://github.com/jnagy1/IRtools) `PRseismic` function for generating 2D seismic travel-time tomography test problems.

Given an N-by-N pixel domain, `pyprseimsic` builds a system matrix **A** relating pixel slowness values to measured travel times, along with a phantom image **x** and synthetic measurements **b = Ax**. Two forward models are supported: straight-ray tracing and Fresnel-zone sensitivity kernels.

## Installation

Requires Python >= 3.13, numpy, and scipy.

```bash
uv sync
```

## Quick start

```python
from pyprseimsic import prseismic

# Default: 256x256 tectonic phantom, ray model
prob = prseismic()

# Smaller problem with Fresnel model
prob = prseismic(64, wave_model="fresnel", omega=10)

prob.A    # sparse CSR matrix (s*p, N*N)
prob.b    # measurement vector (s*p,)
prob.x    # phantom image flattened in column-major order (N*N,)
prob.info # ProblemInfo with sizes and type metadata
```

## API reference

### `prseismic`

```python
prseismic(
    N=256, *,
    phantom="tectonic",    # str | PhantomType | ndarray
    wave_model="ray",      # "ray" | "fresnel"
    s=None,                # number of sources (default: N)
    p=None,                # number of receivers (default: 2*N)
    omega=10.0,            # dominant frequency (Fresnel only)
    sparse_matrix=True,    # False → LinearOperator
    seed=None,             # random seed for phantom generation
    **phantom_kwargs,
) -> SeismicProblem
```

Main entry point. Returns a `SeismicProblem` dataclass with fields `A`, `b`, `x`, `info`.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `N` | Image size (N x N pixels). Default 256. |
| `phantom` | Phantom name (string or `PhantomType` enum), or a custom 2D square ndarray. |
| `wave_model` | `"ray"` for straight-ray model, `"fresnel"` for Fresnel-zone model. |
| `s` | Number of sources on the right boundary. Default: N. |
| `p` | Number of receivers on left + top boundaries. Default: 2N. |
| `omega` | Dominant frequency of the propagating wave (Fresnel model only). |
| `sparse_matrix` | If `True`, `A` is a `scipy.sparse.csr_matrix`. If `False`, a `LinearOperator`. |
| `seed` | Random seed forwarded to phantom generators for reproducibility. |

### `seismictomo`

```python
seismictomo(N, s=None, p=None, *, sparse_matrix=True) -> csr_matrix | LinearOperator
```

Build the system matrix for 2D seismic **ray** tomography. Each row of the matrix corresponds to one source-receiver pair; each entry is the length of the ray segment passing through that pixel.

- Sources: `s` points equally spaced on the right boundary (`x = N/2`).
- Receivers: `floor(p/2)` on the left boundary, `ceil(p/2)` on the top boundary.
- Matrix shape: `(s*p, N*N)`.

### `seismicwavetomo`

```python
seismicwavetomo(N, s=None, p=None, omega=10.0, *, sparse_matrix=True) -> csr_matrix | LinearOperator
```

Build the system matrix for 2D seismic **Fresnel-zone** tomography. Instead of infinitely thin rays, each source-receiver pair contributes a sensitivity kernel based on the first Fresnel zone:

```
S = cos(2*pi*delta_t*omega/N) * exp(-(10*delta_t*omega/N)^2)
```

where `delta_t` is the travel-time detour through each pixel. Entries below `1e-6` are zeroed for sparsity.

### `phantom_gallery`

```python
phantom_gallery(name, N, *, seed=None, **kwargs) -> NDArray
```

Generate an N-by-N phantom image with pixel values in [0, 1].

**Available phantoms:**

| Name | Description | Extra kwargs |
|------|-------------|--------------|
| `tectonic` | Two tectonic plates with a subduction zone | -- |
| `shepplogan` | Modified Shepp-Logan head phantom (10 ellipses) | -- |
| `smooth` | Sum of anisotropic Gaussians | `num_gaussians` (1--4, default 4) |
| `grains` | Voronoi cell texture | `num_cells` (default `3*sqrt(N)`) |
| `ppower` | Power-law FFT filtered random pattern | `rel_nonzero` (default 0.3), `smoothness` (default 2.0) |
| `threephases` | Three-phase image with values {0, 0.5, 1} | `num_domains` (default 100) |
| `threephasessmooth` | Smooth three-phase image on ppower background | `num_domains` (default 100), `intensity_variation` (default 1.8) |

### Types

```python
class SeismicProblem:    # frozen dataclass
    A: csr_matrix | LinearOperator
    b: NDArray[np.float64]
    x: NDArray[np.float64]
    info: ProblemInfo

class ProblemInfo:        # frozen dataclass
    problem_type: str     # "tomography"
    x_type: str           # "image2D"
    b_type: str           # "image2D"
    x_size: tuple[int, int]
    b_size: tuple[int, int]

class WaveModel(Enum):   # RAY, FRESNEL
class PhantomType(Enum): # TECTONIC, SHEPPLOGAN, SMOOTH, GRAINS, PPOWER, ...
```

## Examples

```python
from pyprseimsic import prseismic, phantom_gallery, seismictomo

# Use a specific phantom with the ray model
prob = prseismic(128, phantom="shepplogan", s=64, p=128)

# Use enums instead of strings
from pyprseimsic import WaveModel, PhantomType
prob = prseismic(64, phantom=PhantomType.GRAINS, wave_model=WaveModel.FRESNEL, seed=42)

# Supply a custom phantom image
import numpy as np
custom = np.random.default_rng(0).random((32, 32))
prob = prseismic(phantom=custom)

# Use the matrix-free LinearOperator for large problems
prob = prseismic(512, sparse_matrix=False)
y = prob.A.matvec(prob.x)   # forward projection
z = prob.A.rmatvec(prob.b)  # back projection

# Generate a phantom independently
im = phantom_gallery("ppower", 128, seed=7, rel_nonzero=0.5, smoothness=3.0)

# Build just the system matrix
A = seismictomo(64, s=32, p=64)
```

## Tests

```bash
uv run pytest
```

## Provenance

Ported from the MATLAB packages:

- [IRTools](https://github.com/jnagy1/IRtools) -- `PRseismic.m`
- [AIR Tools II](https://github.com/jakobsj/AIRToolsII) -- `seismictomo.m`, `seismicwavetomo.m`, `phantomgallery.m`

Original authors: Silvia Gazzola, Per Christian Hansen, James G. Nagy, Jakob Sauer Jorgensen, Maria Saxild-Hansen.

## AI disclosure

This codebase was written with the assistance of Claude (Anthropic). Claude was
used to port the MATLAB reference implementations to Python, write unit tests,
and draft this README. All generated code was reviewed and validated by the
repository owner.
