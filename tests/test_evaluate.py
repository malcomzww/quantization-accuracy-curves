"""Model-level plumbing: windowing, hooks, weight replacement, perplexity.

Split into two halves. The first uses a tiny hand-built ``nn.Module`` and runs
anywhere torch is installed -- it tests the parts that can be wrong
independently of any particular model. The second needs SmolLM2 in the local
HuggingFace cache and skips otherwise, because a test that downloads 270 MB is
not a test anyone will run twice.

The invariant worth stating: ``quantise_model`` mutates weights in place and
returns the originals. If ``restore_model`` did not put them back exactly, the
sweep's later rungs would be measuring the accumulated damage of every
earlier rung, and the degradation curve would rise monotonically for a reason
that had nothing to do with bit width.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantization_accuracy_curves import evaluate as E  # noqa: E402
from quantization_accuracy_curves import quantise as Q  # noqa: E402


class TinyModel(torch.nn.Module):
    """Two linear layers with 256-divisible inputs, plus one to be skipped."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(512, 32, bias=False)
        self.b = torch.nn.Linear(512, 32, bias=False)
        self.lm_head = torch.nn.Linear(32, 8, bias=False)

    def forward(self, x):
        return self.lm_head(self.a(x) + self.b(x))


class TestLayerSelection:
    def test_finds_linear_layers_and_skips_the_head(self):
        names = E.quantisable_layers(TinyModel())
        assert names == ["a", "b"]

    def test_lm_head_is_skipped(self):
        """llama.cpp keeps the output projection at higher precision too."""
        assert "lm_head" in E.SKIP_LAYERS
        assert "lm_head" not in E.quantisable_layers(TinyModel())

    def test_raises_on_a_width_no_block_type_can_cover(self):
        """Not a multiple of 32 means no ggml block type fits. Fail, not skip.

        Silently skipping is how an earlier version left most of SmolLM2 at
        full precision while reporting the model as quantised.
        """

        class Odd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(100, 4, bias=False)

        with pytest.raises(ValueError, match="not a multiple of"):
            E.quantisable_layers(Odd())


class TestTokenWindows:
    class FakeTokenizer:
        def __init__(self, n):
            self.n = n

        def __call__(self, text, return_tensors=None):
            class R:
                pass

            r = R()
            r.input_ids = torch.arange(self.n).unsqueeze(0)
            return r

    def test_splits_into_equal_windows(self):
        windows = E.tokenize_windows(self.FakeTokenizer(400), "x", window=128)
        assert len(windows) == 3
        assert all(w.shape == (1, 128) for w in windows)

    def test_drops_the_partial_tail_rather_than_padding(self):
        """400 tokens at window 128 yields 3 windows, not 4.

        Padding would add tokens the model was never asked to predict and
        would dilute the loss by an amount depending on text length -- so the
        same text at a different window size would report a different
        perplexity for a purely arithmetic reason.
        """
        windows = E.tokenize_windows(self.FakeTokenizer(400), "x", window=128)
        assert sum(w.shape[1] for w in windows) == 384

    def test_windows_do_not_overlap(self):
        """Independence is what the bootstrap CI assumes."""
        windows = E.tokenize_windows(self.FakeTokenizer(256), "x", window=128)
        assert windows[0][0, -1].item() + 1 == windows[1][0, 0].item()

    def test_raises_when_text_is_shorter_than_one_window(self):
        with pytest.raises(ValueError, match="fewer than one"):
            E.tokenize_windows(self.FakeTokenizer(50), "x", window=128)


class TestWeightRoundTrip:
    def _ggml_or_skip(self):
        try:
            Q.block_size(Q.GGML_TYPE_Q4_K)
        except Q.GgmlUnavailable as exc:  # pragma: no cover
            pytest.skip(f"ggml unavailable: {exc}")

    def test_quantise_then_restore_is_bit_exact(self):
        """The sweep's correctness depends on this being exact, not close.

        Restoring approximately would make every rung after the first measure
        accumulated damage rather than its own bit width.
        """
        self._ggml_or_skip()
        model = TinyModel()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        original = E.quantise_model(model, Q.BY_NAME["Q2_K"])
        assert not torch.equal(model.a.weight, before["a.weight"])

        E.restore_model(model, original)
        for name, param in model.named_parameters():
            assert torch.equal(param, before[name]), name

    def test_quantisation_actually_changes_weights(self):
        self._ggml_or_skip()
        model = TinyModel()
        before = model.a.weight.detach().clone()
        E.quantise_model(model, Q.BY_NAME["Q4_K"])
        assert not torch.equal(model.a.weight, before)

    def test_skipped_layer_is_left_alone(self):
        self._ggml_or_skip()
        model = TinyModel()
        before = model.lm_head.weight.detach().clone()
        E.quantise_model(model, Q.BY_NAME["Q2_K"])
        assert torch.equal(model.lm_head.weight, before)

    def test_missing_importance_vector_raises(self):
        """A partial imatrix must fail loudly, not quantise some layers blind."""
        self._ggml_or_skip()
        model = TinyModel()
        with pytest.raises(KeyError, match="no importance vector"):
            E.quantise_model(
                model, Q.BY_NAME["Q4_K"], {"a": np.ones(512, dtype=np.float32)}
            )

    def test_imatrix_free_levels_ignore_a_missing_entry(self):
        """Q8_0 declares uses_imatrix=False, so an empty dict is fine for it."""
        self._ggml_or_skip()
        model = TinyModel()
        E.quantise_model(model, Q.BY_NAME["Q8_0"], {})


class TestActivationHooks:
    def test_collects_one_vector_per_layer_of_the_right_width(self):
        model = TinyModel()
        names = E.quantisable_layers(model)

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}

        def make_hook(name):
            def hook(_m, args):
                x = args[0].detach()
                flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
                sq = (flat * flat).sum(dim=0).numpy().astype(np.float64)
                sums[name] = sums.get(name, 0) + sq
                counts[name] = counts.get(name, 0) + flat.shape[0]

            return hook

        handles = [
            E._get_module(model, n).register_forward_pre_hook(make_hook(n)) for n in names
        ]
        try:
            with torch.no_grad():
                model(torch.ones(4, 512))
        finally:
            for h in handles:
                h.remove()

        assert set(sums) == {"a", "b"}
        for name in names:
            assert sums[name].shape == (512,)
            # Input is all ones, so mean square per feature is exactly 1.
            np.testing.assert_allclose(sums[name] / counts[name], np.ones(512))

    def test_hooks_are_removed_after_collection(self):
        """A leaked hook would keep accumulating during the perplexity pass,
        silently slowing every later measurement and holding references to
        activation tensors."""
        model = TinyModel()
        before = len(model.a._forward_pre_hooks)

        def make(_n):
            return lambda _m, _a: None

        h = model.a.register_forward_pre_hook(make("a"))
        assert len(model.a._forward_pre_hooks) == before + 1
        h.remove()
        assert len(model.a._forward_pre_hooks) == before


# --- the real model, only if it is already cached ----------------------


def _cached_model_or_skip():
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return E.load_model()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{E.DEFAULT_MODEL} not in the local cache: {type(exc).__name__}")


@pytest.fixture(scope="module")
def real_model():
    return _cached_model_or_skip()


class TestRealModel:
    def test_every_transformer_linear_is_quantisable(self, real_model):
        """210 layers, not 30.

        The count is the guard: filtering on 256-divisibility left only the
        30 mlp.down_proj layers, since SmolLM2's hidden size is 576. Every
        degradation number in the sweep was understated as a result, so the
        count is asserted rather than trusted.
        """
        model, _ = real_model
        names = E.quantisable_layers(model)
        assert len(names) == 210
        kinds = {n.split(".")[-1] for n in names}
        assert kinds == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_baseline_perplexity_is_plausible(self, real_model):
        """A loose band. Tight enough to catch a broken forward pass, wide
        enough not to encode this machine's arithmetic."""
        from quantization_accuracy_curves.calibrate import EVAL_CORPUS

        model, tok = real_model
        result = E.perplexity(model, tok, EVAL_CORPUS.text)
        assert 5.0 < result.perplexity < 60.0
        assert result.n_windows >= 5
        assert len(result.window_losses) == result.n_windows

    def test_imatrices_from_different_corpora_diverge(self, real_model):
        """The headline mechanism, at the imatrix level.

        Asserted as "not near-identical" rather than at a threshold: the exact
        cosine is a property of this model and these corpora, and pinning it
        would be a portability trap.
        """
        from quantization_accuracy_curves.calibrate import (
            CORPORA_BY_NAME,
            cosine_similarity,
        )

        model, tok = real_model
        a = E.collect_importance(model, tok, CORPORA_BY_NAME["encyclopedic"].text)
        b = E.collect_importance(model, tok, CORPORA_BY_NAME["code"].text)
        assert set(a) == set(b)

        cosines = [cosine_similarity(a[n], b[n]) for n in a]
        assert min(cosines) < 0.9, (
            f"no layer's imatrix diverged between corpora (min cosine "
            f"{min(cosines):.4f}); the corpora may be too similar to measure"
        )

    def test_quantising_raises_perplexity(self, real_model):
        """Q2_K must be measurably worse than the float32 baseline.

        Direction only, no magnitude: the size of the gap depends on the
        model and would not survive being pinned.
        """
        from quantization_accuracy_curves.calibrate import EVAL_CORPUS

        model, tok = real_model
        base = E.perplexity(model, tok, EVAL_CORPUS.text)
        original = E.quantise_model(model, Q.BY_NAME["Q2_K"])
        try:
            after = E.perplexity(model, tok, EVAL_CORPUS.text)
        finally:
            E.restore_model(model, original)
        assert after.perplexity > base.perplexity

        restored = E.perplexity(model, tok, EVAL_CORPUS.text)
        assert restored.perplexity == pytest.approx(base.perplexity, rel=1e-6)
