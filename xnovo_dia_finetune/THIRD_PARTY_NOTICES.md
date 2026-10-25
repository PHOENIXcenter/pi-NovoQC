# Third-party notices

This repository vendors a small number of third-party components for convenience and reproducibility.

## Vendored code

- `pyteomics_master/` — a copy/fork of **Pyteomics**.
  - License: see `pyteomics_master/LICENSE`.
  - Upstream project: https://pyteomics.readthedocs.io/
  - Local compatibility change: read the public `pandas.__version__` attribute.
  - Local compatibility change: use `pkgutil.extend_path` for the package namespace.

- `depthcharge_new/` — a local copy/fork based on **Depthcharge** (Transformer models for proteomics).
  - License: see `depthcharge_new/transformer_generalization/LICENSE`.

## Python dependencies

All other dependencies are installed via `pip` (see `requirements.txt`) and keep their original licenses.
