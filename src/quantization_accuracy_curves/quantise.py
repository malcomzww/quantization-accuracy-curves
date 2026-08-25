"""Real llama.cpp k-quant kernels, driven from Python.

This module does not reimplement quantisation. It calls the same
``ggml_quantize_chunk`` entry point that ``llama-quantize`` calls, exported
from the ``ggml-base`` shared library that ships inside the
``llama-cpp-python`` wheel. The bit-packing, the block layout, the k-quant
super-block scale search and the importance-weighted error minimisation are
all upstream C code. What lives here is the plumbing: shape checks, buffer
allocation, and the importance vector.

Why go through ctypes rather than the ``gguf`` Python package: ``gguf.quants``
can *dequantise* every k-quant type but its ``quantize_blocks`` raises
``NotImplementedError`` for all of Q2_K..Q6_K. Only the C kernels can produce
them, and only the C kernels accept an imatrix. Since the imatrix is the
entire subject of this repository, the C path is the only honest one.

The round trip implemented here is quantise-then-dequantise into float32.
That is deliberate: it isolates *the error the quantiser introduces* from the
error a particular integer GEMM kernel introduces, and it lets the quantised
weights go straight back into a PyTorch model for a real forward pass. It is
not a speed claim, and this repository makes none.

Key fact the rest of the repo depends on: ``ggml_quantize_chunk`` takes an
importance vector of length ``n_per_row``. Passing a different vector for the
same weights produces different bits. That is calibration, and it is why the
calibration corpus can move a measured accuracy number.
"""

from __future__ import annotations

import ctypes
import functools
import os
from dataclasses import dataclass

import numpy as np

# --- quantisation levels ----------------------------------------------

#: GGML type enum values, taken from ``ggml.h``. Hard-coded rather than
#: imported so that the numeric contract this module relies on is visible in
#: the source and testable, instead of silently tracking an upstream rename.
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_IQ4_NL = 20


@dataclass(frozen=True)
class QuantLevel:
    """One rung of the quantisation ladder.

    ``bits_per_weight`` is computed from the block layout rather than quoted
    from documentation, because k-quant super-blocks carry sub-scales whose
    cost is easy to forget: Q4_K is not 4.0 bits per weight, it is 4.5.
    """

    name: str
    ggml_type: int
    #: True for types whose kernel consults the importance vector. Q8_0 has
    #: enough headroom that upstream does not weight it; F16 is not quantised
    #: at all. Calibration cannot matter for those, and claiming otherwise
    #: would be the easiest way to fake a result in this repo.
    uses_imatrix: bool

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


#: Ordered coarse-to-fine. Index 0 is the reference; degradation is always
#: measured against it.
LADDER: tuple[QuantLevel, ...] = (
    QuantLevel("F16", GGML_TYPE_F16, uses_imatrix=False),
    QuantLevel("Q8_0", GGML_TYPE_Q8_0, uses_imatrix=False),
    QuantLevel("Q6_K", GGML_TYPE_Q6_K, uses_imatrix=True),
    QuantLevel("Q5_K", GGML_TYPE_Q5_K, uses_imatrix=True),
    QuantLevel("Q4_K", GGML_TYPE_Q4_K, uses_imatrix=True),
    QuantLevel("IQ4_NL", GGML_TYPE_IQ4_NL, uses_imatrix=True),
    QuantLevel("Q3_K", GGML_TYPE_Q3_K, uses_imatrix=True),
    QuantLevel("Q2_K", GGML_TYPE_Q2_K, uses_imatrix=True),
)

BY_NAME: dict[str, QuantLevel] = {q.name: q for q in LADDER}


# --- the shared library -----------------------------------------------


class GgmlUnavailable(RuntimeError):
    """Raised when the ggml shared library cannot be located or loaded.

    Carried as a distinct type so tests can skip cleanly rather than fail on
    a machine without the wheel, while the results script can treat it as
    fatal -- a results file generated without the real kernels would be a
    fabrication.
    """


@functools.lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    """Load ``ggml-base`` from the installed ``llama-cpp-python`` wheel.

    Cached because ``CDLL`` on the same path twice is wasteful, and because
    the bound argtypes below should be set exactly once.
    """
    try:
        import llama_cpp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GgmlUnavailable("llama-cpp-python is not installed") from exc

    libdir = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    candidates = ("ggml-base.dll", "libggml-base.so", "libggml-base.dylib")
    for name in candidates:
        path = os.path.join(libdir, name)
        if os.path.exists(path):
            break
    else:  # pragma: no cover - environment dependent
        raise GgmlUnavailable(f"no ggml-base library in {libdir}; looked for {candidates}")

    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:  # pragma: no cover - environment dependent
        raise GgmlUnavailable(f"could not load {path}: {exc}") from exc

    f32p = ctypes.POINTER(ctypes.c_float)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,  # type
        f32p,  # src
        ctypes.c_void_p,  # dst
        ctypes.c_int64,  # start
        ctypes.c_int64,  # nrows
        ctypes.c_int64,  # n_per_row
        f32p,  # imatrix
    ]
    lib.ggml_type_size.restype = ctypes.c_size_t
    lib.ggml_type_size.argtypes = [ctypes.c_int]
    lib.ggml_blck_size.restype = ctypes.c_int64
    lib.ggml_blck_size.argtypes = [ctypes.c_int]
    return lib


def block_size(ggml_type: int) -> int:
    """Weights per block for a ggml type. 32 for Q8_0, 256 for the k-quants."""
    return int(_lib().ggml_blck_size(ggml_type))


def type_size(ggml_type: int) -> int:
    """Bytes per block, including that block's scales."""
    return int(_lib().ggml_type_size(ggml_type))


def bits_per_weight(level: QuantLevel) -> float:
    """Effective bits per weight, derived from the block layout.

    Includes the super-block scale overhead, which is the difference between
    the marketing number and the file size. Q4_K stores 4-bit weights but
    costs 4.5 bits/weight once its 6-bit sub-scales are counted.
    """
    if level.ggml_type == GGML_TYPE_F16:
        return 16.0
    return 8.0 * type_size(level.ggml_type) / block_size(level.ggml_type)


# --- the round trip ---------------------------------------------------


def quantise_dequantise(
    weight: np.ndarray,
    level: QuantLevel,
    importance: np.ndarray | None = None,
) -> np.ndarray:
    """Quantise ``weight`` to ``level`` and immediately dequantise to float32.

    ``weight`` is 2-D ``(out_features, in_features)`` in PyTorch ``nn.Linear``
    convention. ggml quantises along the last axis, so the importance vector
    is indexed by *input* feature -- which is exactly what an activation
    statistic gives you, since activations are what multiply the input axis.

    ``importance`` is the imatrix: one non-negative float per input feature,
    or ``None`` to quantise unweighted. Passing ``None`` is not the same as
    passing all-ones; upstream takes a different code path when the pointer
    is null, and the outputs differ.

    Returns a float32 array of the same shape. The values are exactly the
    ones a llama.cpp kernel would dequantise at inference time.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {weight.shape}")

    src = np.ascontiguousarray(weight, dtype=np.float32)
    nrow, ncol = src.shape

    if level.ggml_type == GGML_TYPE_F16:
        # Not a ggml block type; the round trip is a plain float16 cast.
        # Kept on the ladder because it is the reference every degradation
        # number is measured against, and because it is not free -- F16 is
        # itself lossy relative to the float32 the model was loaded in.
        return src.astype(np.float16).astype(np.float32)

    bs = block_size(level.ggml_type)
    if ncol % bs != 0:
        # Real transformer hidden sizes are frequently not multiples of 256:
        # SmolLM2-135M is 576, Qwen2.5-0.5B is 896. A k-quant kernel cannot
        # express a partial super-block, so llama.cpp falls back to a
        # finer-blocked type for the leftover columns rather than refusing
        # the tensor. Mirroring that here is what keeps the whole weight
        # matrix quantised: the alternative -- skipping these layers -- left
        # 76% of SmolLM2's linear weights at full precision and reported the
        # result as a quantised model, which understated every degradation
        # number in the sweep.
        return _quantise_split(src, level, importance, bs)

    return _quantise_aligned(src, level, importance, bs)


def _quantise_aligned(
    src: np.ndarray, level: QuantLevel, importance: np.ndarray | None, bs: int
) -> np.ndarray:
    """Round trip a matrix whose input dimension is a whole number of blocks."""
    nrow, ncol = src.shape

    imp_ptr = None
    if importance is not None:
        if importance.shape != (ncol,):
            raise ValueError(
                f"importance must have shape ({ncol},) to match the input dimension, "
                f"got {importance.shape}"
            )
        if not np.all(np.isfinite(importance)):
            raise ValueError("importance vector contains non-finite values")
        if np.any(importance < 0):
            raise ValueError("importance must be non-negative")
        imp = np.ascontiguousarray(importance, dtype=np.float32)
        imp_ptr = imp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    nbytes = (ncol // bs) * type_size(level.ggml_type) * nrow
    dst = (ctypes.c_char * nbytes)()

    written = _lib().ggml_quantize_chunk(
        level.ggml_type,
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(dst, ctypes.c_void_p),
        0,
        nrow,
        ncol,
        imp_ptr,
    )
    if written != nbytes:
        raise RuntimeError(f"{level.name}: kernel wrote {written} bytes, layout implies {nbytes}")

    import gguf.quants
    from gguf.constants import GGMLQuantizationType

    packed = np.frombuffer(bytes(dst), dtype=np.uint8).reshape(nrow, -1)
    out = gguf.quants.dequantize(packed, GGMLQuantizationType(level.ggml_type))
    return np.ascontiguousarray(out.reshape(nrow, ncol), dtype=np.float32)


#: Type the split path falls back to for leftover columns. Q8_0 has a 32-wide
#: block, which divides every hidden size this repo touches, and it is the
#: highest-fidelity block type available -- so the fallback cannot be what
#: causes a degradation, and any measured loss is attributable to the k-quant
#: on the aligned majority of the row.
_FALLBACK = QuantLevel("Q8_0", GGML_TYPE_Q8_0, uses_imatrix=False)


def _quantise_split(
    src: np.ndarray, level: QuantLevel, importance: np.ndarray | None, bs: int
) -> np.ndarray:
    """Quantise the block-aligned prefix at ``level``, the remainder at Q8_0.

    ``bs`` is the target type's block size. The prefix is the largest multiple
    of it that fits; on SmolLM2's 576-wide layers that is 512 columns at the
    k-quant and 64 at Q8_0, so 89% of each row carries the level under test.
    The fraction is reported in the results file rather than hidden, because
    a mixed-precision tensor is not the same object as a uniformly quantised
    one and the difference belongs in the record.
    """
    nrow, ncol = src.shape
    head = (ncol // bs) * bs
    if head == 0:
        raise ValueError(
            f"{level.name} has a {bs}-wide block but the input dimension is only {ncol}"
        )

    fb = block_size(_FALLBACK.ggml_type)
    tail = ncol - head
    if tail % fb != 0:
        raise ValueError(
            f"cannot split a {ncol}-wide row for {level.name}: the {tail}-column "
            f"remainder is not a multiple of the Q8_0 block size {fb}"
        )

    imp_head = importance[:head] if importance is not None else None
    out = np.empty_like(src)
    out[:, :head] = _quantise_aligned(
        np.ascontiguousarray(src[:, :head]), level, imp_head, bs
    )
    out[:, head:] = _quantise_aligned(
        np.ascontiguousarray(src[:, head:]), _FALLBACK, None, fb
    )
    return out


def aligned_fraction(n_features: int, level: QuantLevel) -> float:
    """Fraction of each row that ``level`` itself quantises, rest being Q8_0.

    1.0 when the input dimension is a whole number of blocks. Used by the
    results script to state the mixed-precision caveat as a number.
    """
    if level.ggml_type == GGML_TYPE_F16:
        return 1.0
    bs = block_size(level.ggml_type)
    return ((n_features // bs) * bs) / n_features


def weighted_mse(
    original: np.ndarray, reconstructed: np.ndarray, importance: np.ndarray | None = None
) -> float:
    """Mean squared reconstruction error, optionally weighted per input feature.

    The unweighted version is what a naive quantisation report shows. The
    weighted version is what actually predicts downstream loss, because an
    error on a channel the model never excites costs nothing. The gap between
    the two is the reason calibration exists at all.
    """
    diff = (original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    if importance is None:
        return float(diff.mean())
    w = importance.astype(np.float64)
    total = w.sum() * original.shape[0]
    if total <= 0:
        raise ValueError("importance weights sum to zero")
    return float((diff * w).sum() / total)
