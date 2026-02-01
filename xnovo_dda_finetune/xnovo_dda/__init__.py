"""π-xNovo (DDA) fine-tuning code.

This package contains:
  - dataset.py: MGF dataset loader and batching
  - model.py: Transformer-based spectrum-to-sequence model wrapper
  - utils.py: training / validation loops
  - evaluate.py: peptide/AA-level evaluation utilities

The repository is prepared for public release on GitHub.
"""

__all__ = [
    "dataset",
    "model",
    "utils",
    "evaluate",
]

__version__ = "0.1.0"
