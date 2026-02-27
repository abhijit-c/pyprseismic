# Notebooks

This folder contains runnable, rendered demos for `pyprseimsic`.

## Contents

- `00_overview_prseismic.ipynb`: End-to-end `prseismic` usage and data interpretation.
- `01_phantom_gallery.ipynb`: All built-in phantoms and key parameter sweeps.
- `02_forward_models_ray_vs_fresnel.ipynb`: Geometry and matrix diagnostics for ray vs Fresnel models.

## Run locally

```bash
.venv/bin/jupyter lab notebooks
```

Or execute in batch:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/00_overview_prseismic.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_phantom_gallery.ipynb --inplace
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/02_forward_models_ray_vs_fresnel.ipynb --inplace
```
