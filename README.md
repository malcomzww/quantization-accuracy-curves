# quantization-accuracy-curves

A k-quant ladder swept against three calibration corpora, built around one
question:

> **How much does the calibration set choice change measured degradation?**

Everyone benchmarks quantisation levels. Almost nobody reports which corpus
they calibrated on — and on this model that omission is the larger term.

## The answer

At the coarse end of the ladder, the choice of calibration corpus moves
measured degradation by **38.56 percentage points** — **61.9× more than a
whole step of quantisation level**.

A paper reporting "Q2_K costs X%" without naming its calibration set has
published a number whose uncontrolled term is sixty times larger than the
effect it is measuring.

## The ladder is not monotone

| level | degradation |
|---|---|
| Q6_K | 2.15% |
| **Q5_K** | **1.53%** |
| Q4_K | 6.27% |
| Q3_K | 57.25% |
| Q2_K | 181.36% |

**Q5_K beats Q6_K.** Fewer bits, less degradation.

That is not a kernel bug. k-quants differ in block structure and super-block
scaling, not only in bit width, so the ladder is not a single ordered axis. An
earlier version of this script *asserted* monotonicity as a correctness check —
which would have crashed rather than reported a real measurement. The
assertion is gone and the inversion is in the results file.

## Why the corpora disagree

The disagreement is concentrated in **`down_proj`** (median cosine similarity
0.6672 between the encyclopedic and chat importance matrices, against 0.1062
spread). Attention projections agree closely; the MLP down-projection is where
corpora hold different opinions about which intermediate features matter.

That gives the effect a mechanism rather than leaving it as a correlation:
calibration selects which MLP features survive quantisation, and different text
selects differently.

Matched calibration — evaluating on the corpus you calibrated on — wins at
every level, which the script asserts, so this file fails rather than
publishing if that ever stops being true.

## Granularity: payload beats scales

Sweeping per-tensor, per-channel and per-group scaling at 3 and 4 bits, the
crossover is **per-group-32 at 3 bits**: finer grouping costs bits to store the
scales themselves, and past that point the scales cost more than the extra
precision buys. **Spending bits on payload beats spending them on scales.**

## Quickstart

```bash
uv sync --extra dev            # arithmetic tests only, fast
uv run pytest -q               # 36 tests

uv sync --extra measure        # llama-cpp-python, gguf, torch
uv run python scripts/generate_results.py    # ~240s
```

Full output: [`results/calibration-sensitivity.md`](results/calibration-sensitivity.md).

## Limitations

- **One model.** Everything here is measured on a single small model; the
  60× ratio is not claimed to generalise, only the fact that the calibration
  term can dominate.
- **Absolute perplexities are not committed.** The same model on the same text
  returns 17.761073017 on one thread and 17.761070597 on twenty-four — float
  reduction order depends on how BLAS splits the GEMM, so an absolute
  perplexity is a property of the runner's core count. Ratios, orderings and
  imatrix cosines are portable and are what this file carries.
- **Three corpora is a small sample** of the space of calibration text. The
  spread is a lower bound on what a deliberately adversarial choice could do.
- **No GPU.** All measurement is CPU llama.cpp; kernel behaviour on a GPU
  backend is not established here.
- The eval corpus is held out from calibration but drawn from the same pool,
  which biases matched-calibration results optimistically.

## License

MIT
