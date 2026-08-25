"""Quantisation accuracy curves for one small model, on CPU.

The question the repository exists to answer: *how much does the choice of
calibration corpus change the measured degradation?* Quantisation levels get
benchmarked constantly; the calibration corpus that produced the importance
matrix is usually a footnote, and this measures what that footnote is worth.

Four modules:

- :mod:`.quantise`    real llama.cpp k-quant kernels via ctypes
- :mod:`.calibrate`   the calibration corpora and imatrix comparison
- :mod:`.evaluate`    activation collection, weight replacement, perplexity
- :mod:`.granularity` per-tensor / per-channel / per-group affine quantisation

Nothing here needs a GPU.
"""

from __future__ import annotations

__all__ = ["calibrate", "evaluate", "granularity", "quantise"]
