"""Per-tensor, per-channel and per-group affine quantisation.

The ggml k-quants in :mod:`.quantise` are fixed at a 256-wide group, so they
cannot answer the granularity question on their own -- there is no per-tensor
Q4_K. This module implements the three granularities explicitly at a chosen
bit width, which makes the scale/zero-point arithmetic exactly testable
against hand-computed fixtures and lets the group size sweep continuously.

The arithmetic is standard asymmetric affine quantisation:

    scale = (max - min) / (2**bits - 1)
    zero  = round(-min / scale)
    q     = clip(round(x / scale) + zero, 0, 2**bits - 1)
    x_hat = (q - zero) * scale

What changes between granularities is only the set over which ``max`` and
``min`` are taken. That is the entire mechanism, and it is why the ranking is
forced rather than empirical: a finer partition minimises the same objective
over a strictly finer set, so it cannot do worse. The measurement is about
*how much* is gained per unit of scale-storage overhead, not about which way
the inequality points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Granularity:
    """How the weight matrix is partitioned before scales are chosen.

    ``group_size`` is ``None`` for per-tensor (one partition for everything)
    and for per-channel (one partition per output row). An integer means each
    row is further split into contiguous blocks of that many input features.
    """

    name: str
    per_row: bool
    group_size: int | None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


PER_TENSOR = Granularity("per-tensor", per_row=False, group_size=None)
PER_CHANNEL = Granularity("per-channel", per_row=True, group_size=None)


def per_group(size: int) -> Granularity:
    """Per-channel, further split into contiguous groups of ``size`` inputs."""
    if size <= 0:
        raise ValueError(f"group size must be positive, got {size}")
    return Granularity(f"per-group-{size}", per_row=True, group_size=size)


def affine_quantise_dequantise(
    weight: np.ndarray, bits: int, granularity: Granularity
) -> np.ndarray:
    """Round-trip ``weight`` through ``bits``-bit affine quantisation.

    Returns a float32 array of the same shape holding the reconstructed
    values. Degenerate partitions -- where every weight in a group is
    identical, so ``max == min`` -- are reconstructed exactly rather than
    dividing by a zero scale.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {weight.shape}")
    if not 2 <= bits <= 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")

    x = np.ascontiguousarray(weight, dtype=np.float32).astype(np.float64)
    nrow, ncol = x.shape
    qmax = float(2**bits - 1)

    if not granularity.per_row:
        blocks = x.reshape(1, -1)
    elif granularity.group_size is None:
        blocks = x.reshape(nrow, ncol)
    else:
        g = granularity.group_size
        if ncol % g != 0:
            raise ValueError(
                f"{granularity.name} needs the input dimension to be a multiple of {g}, "
                f"got {ncol}"
            )
        blocks = x.reshape(nrow * (ncol // g), g)

    lo = blocks.min(axis=1, keepdims=True)
    hi = blocks.max(axis=1, keepdims=True)

    scale = (hi - lo) / qmax
    degenerate = scale <= 0
    # Substitute 1.0 so the divide is well-defined; those rows are overwritten
    # with their exact original values below.
    safe_scale = np.where(degenerate, 1.0, scale)

    zero = np.rint(-lo / safe_scale)
    q = np.clip(np.rint(blocks / safe_scale) + zero, 0.0, qmax)
    recon = (q - zero) * safe_scale
    recon = np.where(degenerate, blocks, recon)

    return np.ascontiguousarray(recon.reshape(nrow, ncol), dtype=np.float32)


def scale_overhead_bits(
    weight_shape: tuple[int, int], bits: int, granularity: Granularity
) -> float:
    """Extra bits per weight spent storing scales and zero points.

    Counted as one float16 scale plus one float16 zero point per partition,
    which is the layout most integer-quantised formats use. This is the
    number that makes the granularity comparison honest: per-group-32 always
    reconstructs better than per-group-128, but it costs three extra bits per
    weight to do it, and at that price you could simply have used a wider
    integer.
    """
    nrow, ncol = weight_shape
    total = nrow * ncol
    if not granularity.per_row:
        partitions = 1
    elif granularity.group_size is None:
        partitions = nrow
    else:
        if ncol % granularity.group_size != 0:
            raise ValueError(
                f"{granularity.name} does not divide an input dimension of {ncol}"
            )
        partitions = nrow * (ncol // granularity.group_size)
    return 32.0 * partitions / total


def effective_bits(weight_shape: tuple[int, int], bits: int, granularity: Granularity) -> float:
    """Payload bits plus scale overhead, per weight."""
    return bits + scale_overhead_bits(weight_shape, bits, granularity)
