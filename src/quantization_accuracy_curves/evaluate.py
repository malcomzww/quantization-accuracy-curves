"""Collect imatrices from a real model and measure what quantising it costs.

Three operations, in the order the experiment uses them:

1. ``collect_importance`` runs the calibration text through the unmodified
   model and records the mean squared activation entering every linear layer.
2. ``quantise_model`` replaces each linear layer's weights with their
   quantise-dequantise round trip, steered by an imatrix.
3. ``perplexity`` scores held-out text through whatever weights are loaded.

Perplexity rather than a task metric, on purpose. A 135M model scores near
chance on multiple-choice benchmarks, so a task eval would report the noise
floor moving and call it degradation. Token-level negative log-likelihood is
dense -- every token contributes -- which is what makes a small model's
degradation measurable at all. Per-window losses are kept rather than
averaged away so that the harness can put a bootstrap interval around the
result instead of reporting a bare point estimate.

Everything runs on CPU in float32. No GPU is required or used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"

#: Layers excluded from quantisation. ``lm_head`` and the token embedding are
#: what llama.cpp itself keeps at higher precision by default: they are read
#: once per token rather than per layer, so quantising them buys little space
#: while directly distorting the output distribution the perplexity measures.
#: Excluding them keeps this experiment comparable to a real GGUF build.
SKIP_LAYERS = ("lm_head",)


@dataclass(frozen=True)
class PerplexityResult:
    """Perplexity plus the per-window losses that produced it.

    The windows are retained because a single perplexity number has no error
    bar, and a degradation of 0.3% is meaningless without knowing whether the
    measurement noise is 0.01% or 3%.
    """

    perplexity: float
    window_losses: tuple[float, ...]
    n_windows: int
    n_tokens: int


def load_model(model_id: str = DEFAULT_MODEL):
    """Load the model and tokenizer on CPU in float32.

    float32 rather than the model's native dtype so that the F16 rung of the
    ladder is a genuine measurement rather than a no-op: if the baseline were
    already float16, casting to float16 would cost exactly zero and the
    reference row would be silently degenerate.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def quantisable_layers(model) -> list[str]:
    """Names of the linear layers this experiment quantises.

    Every linear layer in the transformer stack, minus :data:`SKIP_LAYERS`.
    Layers whose input dimension is not a multiple of the k-quant super-block
    are *not* excluded -- :func:`..quantise.quantise_dequantise` splits them,
    quantising the aligned prefix at the target level and the remainder at
    Q8_0. An earlier version of this function filtered them out, which on
    SmolLM2-135M silently left 30 of 211 layers as the only quantised ones
    and reported the result as a quantised model.

    The one hard requirement is that the input dimension be a multiple of 32,
    the Q8_0 block width, so that the fallback can cover the remainder. A
    layer failing that raises rather than being skipped.
    """
    import torch

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name.endswith(skip) for skip in SKIP_LAYERS):
            continue
        n_in = module.weight.shape[1]
        if n_in % 32 != 0:
            raise ValueError(
                f"layer {name} has input dimension {n_in}, not a multiple of the Q8_0 "
                f"block size 32; no ggml block type can cover it"
            )
        names.append(name)
    return names


def _get_module(model, name: str):
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def tokenize_windows(tokenizer, text: str, window: int = 128) -> list:
    """Split text into non-overlapping token windows of a fixed length.

    Non-overlapping so that windows are independent samples, which is what
    the bootstrap in the harness assumes. A trailing partial window is
    dropped rather than padded: padding would contribute tokens the model was
    never asked to predict and would dilute the loss by an amount that
    depends on the text length.
    """

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = (len(ids) // window) * window
    if n == 0:
        raise ValueError(
            f"text tokenises to {len(ids)} tokens, fewer than one {window}-token window"
        )
    return [ids[i : i + window].unsqueeze(0) for i in range(0, n, window)]


def collect_importance(model, tokenizer, text: str, window: int = 128) -> dict[str, np.ndarray]:
    """Run ``text`` through ``model`` and return one imatrix per linear layer.

    Implemented with forward pre-hooks so the statistic is taken from the
    tensor that actually enters each layer, including whatever normalisation
    and residual mixing precede it. Reconstructing that analytically would be
    guesswork; hooking it is exact.

    Accumulates sums rather than retaining activations, so memory stays flat
    in the amount of calibration text.
    """
    import torch

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    names = quantisable_layers(model)

    def make_hook(name: str):
        def hook(_module, args):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            sq = (flat * flat).sum(dim=0).numpy().astype(np.float64)
            if name in sums:
                sums[name] += sq
                counts[name] += flat.shape[0]
            else:
                sums[name] = sq
                counts[name] = flat.shape[0]

        return hook

    handles = [
        _get_module(model, name).register_forward_pre_hook(make_hook(name)) for name in names
    ]
    try:
        with torch.no_grad():
            for chunk in tokenize_windows(tokenizer, text, window=window):
                model(chunk)
    finally:
        for handle in handles:
            handle.remove()

    if not sums:
        raise RuntimeError("no activations collected; model has no quantisable linear layers")

    return {
        name: (sums[name] / counts[name]).astype(np.float32)
        for name in sums
    }


def quantise_model(model, level, importance: dict[str, np.ndarray] | None = None) -> dict:
    """Round-trip every quantisable layer's weights in place.

    Returns the original weights, keyed by layer name, so the caller can
    restore them. Restoring is how the sweep avoids reloading a model per
    rung: loading SmolLM2 takes about five seconds and the sweep does it
    dozens of times, but the correctness requirement is that every rung starts
    from *identical* float32 weights, which restore-from-saved guarantees more
    strictly than a reload would.
    """
    import torch

    from .quantise import quantise_dequantise

    original: dict[str, torch.Tensor] = {}
    for name in quantisable_layers(model):
        module = _get_module(model, name)
        w = module.weight.detach().cpu().numpy()
        original[name] = module.weight.detach().clone()
        imp = None
        if importance is not None and level.uses_imatrix:
            imp = importance.get(name)
            if imp is None:
                raise KeyError(f"no importance vector collected for layer {name}")
        recon = quantise_dequantise(w, level, imp)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(recon))
    return original


def restore_model(model, original: dict) -> None:
    """Put the saved float32 weights back."""
    import torch

    with torch.no_grad():
        for name, weight in original.items():
            _get_module(model, name).weight.copy_(weight)


def perplexity(model, tokenizer, text: str, window: int = 128) -> PerplexityResult:
    """Token-level perplexity of ``text``, with per-window losses retained.

    Each window is scored independently with no context carried across the
    boundary. That makes the windows exchangeable, which the bootstrap needs;
    a sliding-window perplexity would give a lower number but correlated
    samples, and a confidence interval computed over correlated samples is
    too narrow.
    """
    import torch

    windows = tokenize_windows(tokenizer, text, window=window)
    losses: list[float] = []
    with torch.no_grad():
        for chunk in windows:
            out = model(chunk, labels=chunk)
            losses.append(float(out.loss))

    mean_loss = sum(losses) / len(losses)
    return PerplexityResult(
        perplexity=math.exp(mean_loss),
        window_losses=tuple(losses),
        n_windows=len(windows),
        n_tokens=len(windows) * window,
    )
