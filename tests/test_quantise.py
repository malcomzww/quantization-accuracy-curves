"""The ggml k-quant path: block layout, the imatrix contract, the split fallback.

These tests need the real shared library, so they skip rather than fail when
it is absent. What they are actually defending is the claim the whole
repository rests on: that passing a different importance vector to the same
weights produces different output. If that were false -- if the imatrix
argument were being silently ignored -- every calibration-sensitivity number
here would be measuring nothing, and it would look exactly like a real result.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantization_accuracy_curves import quantise as Q

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _skip_without_ggml():
    try:
        Q.block_size(Q.GGML_TYPE_Q4_K)
    except Q.GgmlUnavailable as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ggml library unavailable: {exc}")


@pytest.fixture(scope="module")
def ggml():
    _skip_without_ggml()
    return True


@pytest.fixture(scope="module")
def weights():
    """A fixed 32x512 matrix. 512 is a whole number of 256-wide super-blocks."""
    rng = np.random.default_rng(20250825)
    return rng.standard_normal((32, 512)).astype(np.float32)


class TestBlockLayout:
    def test_kquants_use_256_wide_superblocks(self, ggml):
        for name in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
            assert Q.block_size(Q.BY_NAME[name].ggml_type) == 256, name

    def test_q8_0_uses_32_wide_blocks(self, ggml):
        """The fallback type's block width. The split path depends on 32
        dividing every hidden size in play, so it is asserted rather than
        assumed."""
        assert Q.block_size(Q.GGML_TYPE_Q8_0) == 32

    def test_bits_per_weight_includes_subscale_overhead(self, ggml):
        """Q4_K costs 4.5 bits/weight, not 4.0.

        The half-bit is the 6-bit sub-scales inside the super-block. Quoting
        4.0 would understate every k-quant file size, and the ladder in the
        results file is ordered by this number.
        """
        assert Q.bits_per_weight(Q.BY_NAME["Q4_K"]) == pytest.approx(4.5)
        assert Q.bits_per_weight(Q.BY_NAME["Q5_K"]) == pytest.approx(5.5)
        assert Q.bits_per_weight(Q.BY_NAME["Q8_0"]) == pytest.approx(8.5)
        assert Q.bits_per_weight(Q.BY_NAME["F16"]) == pytest.approx(16.0)

    def test_ladder_is_ordered_coarse_to_fine_by_bits(self, ggml):
        bpw = [Q.bits_per_weight(lv) for lv in Q.LADDER]
        for hi, lo in zip(bpw, bpw[1:], strict=False):
            assert lo <= hi, f"ladder not monotone in bits/weight: {bpw}"

    def test_f16_is_the_reference_rung(self):
        assert Q.LADDER[0].name == "F16"
        assert not Q.LADDER[0].uses_imatrix


class TestRoundTrip:
    def test_shape_and_dtype_preserved(self, ggml, weights):
        for lv in Q.LADDER:
            out = Q.quantise_dequantise(weights, lv)
            assert out.shape == weights.shape, lv.name
            assert out.dtype == np.float32, lv.name
            assert np.all(np.isfinite(out)), lv.name

    def test_f16_round_trip_is_a_float16_cast(self):
        """The reference rung must be exactly the float16 cast, no more.

        Not a no-op: float32 -> float16 -> float32 loses mantissa bits, and
        the results file measures that loss rather than assuming it is zero.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        out = Q.quantise_dequantise(x, Q.BY_NAME["F16"])
        np.testing.assert_array_equal(out, x.astype(np.float16).astype(np.float32))

    def test_error_grows_as_bits_shrink(self, ggml, weights):
        """Coarser k-quants must reconstruct worse. Monotone across the ladder.

        A machine-independent claim: it is a property of the quantiser, not of
        the hardware, so it is safe to assert exactly.
        """
        errs = {}
        for lv in Q.LADDER:
            out = Q.quantise_dequantise(weights, lv)
            errs[lv.name] = float(((weights - out) ** 2).mean())

        ordered = [errs[lv.name] for lv in Q.LADDER]
        # IQ4_NL and Q4_K share 4.5 bits/weight, so their order is not forced.
        # Compare only the strictly-decreasing-bits subsequence.
        strict = ["F16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"]
        seq = [errs[n] for n in strict]
        for finer, coarser in zip(seq, seq[1:], strict=False):
            assert coarser > finer, f"{strict}: {ordered}"

    def test_q2_k_is_much_worse_than_q6_k(self, ggml, weights):
        """A sanity floor: 2-bit must be at least an order of magnitude worse.

        Catches a silent fallthrough where every level ends up using the same
        kernel -- which would still produce a monotone-looking table if the
        differences were tiny.
        """
        q6 = Q.quantise_dequantise(weights, Q.BY_NAME["Q6_K"])
        q2 = Q.quantise_dequantise(weights, Q.BY_NAME["Q2_K"])
        e6 = float(((weights - q6) ** 2).mean())
        e2 = float(((weights - q2) ** 2).mean())
        assert e2 > 10 * e6


class TestImatrixContract:
    """The claim the repository stands on."""

    def test_different_importance_gives_different_bits(self, ggml, weights):
        """Two importance vectors must produce two different reconstructions.

        If this fails, the imatrix pointer is being ignored and every
        calibration result in the repo is noise dressed as signal.
        """
        rng = np.random.default_rng(7)
        flat = np.ones(weights.shape[1], dtype=np.float32)
        peaked = (rng.random(weights.shape[1]).astype(np.float32) * 10.0 + 0.01)

        for name in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL"):
            lv = Q.BY_NAME[name]
            a = Q.quantise_dequantise(weights, lv, flat)
            b = Q.quantise_dequantise(weights, lv, peaked)
            assert not np.array_equal(a, b), f"{name} ignored the importance vector"

    def test_matched_importance_lowers_weighted_error(self, ggml, weights):
        """Quantising under the imatrix you will be judged by must win.

        This is the mechanism, stated as a test: under importance vector ``w``,
        the reconstruction produced *using* ``w`` has lower w-weighted error
        than one produced using a different vector. That is why calibration
        choice can move a perplexity, and it is asserted at the level where it
        is deterministic rather than only observed downstream.
        """
        rng = np.random.default_rng(11)
        n = weights.shape[1]
        w_eval = (rng.random(n).astype(np.float32) * 10.0 + 0.01)
        w_other = (rng.random(n).astype(np.float32) * 10.0 + 0.01)

        for name in ("Q2_K", "Q3_K", "Q4_K"):
            lv = Q.BY_NAME[name]
            matched = Q.quantise_dequantise(weights, lv, w_eval)
            mismatched = Q.quantise_dequantise(weights, lv, w_other)
            e_matched = Q.weighted_mse(weights, matched, w_eval)
            e_mismatched = Q.weighted_mse(weights, mismatched, w_eval)
            assert e_matched < e_mismatched, (
                f"{name}: matched calibration did not win "
                f"({e_matched:.6g} vs {e_mismatched:.6g})"
            )

    def test_none_differs_from_all_ones(self, ggml, weights):
        """A null pointer is not the same as a uniform vector.

        Upstream takes a different code path when no imatrix is supplied. The
        results file distinguishes "uncalibrated" from "calibrated on a flat
        prior", so the distinction has to be real.
        """
        lv = Q.BY_NAME["Q4_K"]
        none = Q.quantise_dequantise(weights, lv)
        ones = Q.quantise_dequantise(weights, lv, np.ones(weights.shape[1], dtype=np.float32))
        assert not np.array_equal(none, ones)

    def test_types_declared_imatrix_free_ignore_it(self, ggml, weights):
        """Q8_0 declares uses_imatrix=False, so it must actually ignore it.

        Guards against overclaiming: if the ladder said Q8_0 were
        calibration-sensitive, the results file would report a spread for it
        that could only be noise.
        """
        rng = np.random.default_rng(3)
        peaked = rng.random(weights.shape[1]).astype(np.float32) + 0.01
        lv = Q.BY_NAME["Q8_0"]
        assert not lv.uses_imatrix
        a = Q.quantise_dequantise(weights, lv)
        b = Q.quantise_dequantise(weights, lv, peaked)
        np.testing.assert_array_equal(a, b)

    def test_rejects_wrong_length_importance(self, ggml, weights):
        with pytest.raises(ValueError, match="importance must have shape"):
            Q.quantise_dequantise(weights, Q.BY_NAME["Q4_K"], np.ones(7, dtype=np.float32))

    def test_rejects_negative_importance(self, ggml, weights):
        bad = np.ones(weights.shape[1], dtype=np.float32)
        bad[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            Q.quantise_dequantise(weights, Q.BY_NAME["Q4_K"], bad)

    def test_rejects_nonfinite_importance(self, ggml, weights):
        bad = np.ones(weights.shape[1], dtype=np.float32)
        bad[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            Q.quantise_dequantise(weights, Q.BY_NAME["Q4_K"], bad)


class TestSplitFallback:
    """Non-256-divisible hidden sizes, which is most real models."""

    def test_576_wide_row_is_quantised_not_rejected(self, ggml):
        """SmolLM2's hidden size. 576 = 2*256 + 64, so it needs the split."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal((16, 576)).astype(np.float32)
        out = Q.quantise_dequantise(x, Q.BY_NAME["Q4_K"])
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_aligned_fraction_reports_the_split(self, ggml):
        """512 of 576 columns carry the k-quant: 8/9 of each row."""
        assert Q.aligned_fraction(576, Q.BY_NAME["Q4_K"]) == pytest.approx(512 / 576)
        assert Q.aligned_fraction(1536, Q.BY_NAME["Q4_K"]) == pytest.approx(1.0)
        assert Q.aligned_fraction(896, Q.BY_NAME["Q4_K"]) == pytest.approx(768 / 896)

    def test_f16_is_always_fully_aligned(self, ggml):
        assert Q.aligned_fraction(577, Q.BY_NAME["F16"]) == 1.0

    def test_tail_is_higher_fidelity_than_head(self, ggml):
        """The Q8_0 remainder must reconstruct better than the Q2_K prefix.

        Confirms the fallback cannot be the cause of a measured degradation:
        error is concentrated in the columns carrying the level under test.
        """
        rng = np.random.default_rng(9)
        x = rng.standard_normal((16, 576)).astype(np.float32)
        out = Q.quantise_dequantise(x, Q.BY_NAME["Q2_K"])
        head_err = float(((x[:, :512] - out[:, :512]) ** 2).mean())
        tail_err = float(((x[:, 512:] - out[:, 512:]) ** 2).mean())
        assert tail_err < head_err

    def test_split_still_honours_the_imatrix(self, ggml):
        """The head must be steered even though the tail is not."""
        rng = np.random.default_rng(13)
        x = rng.standard_normal((16, 576)).astype(np.float32)
        a = Q.quantise_dequantise(x, Q.BY_NAME["Q3_K"], np.ones(576, dtype=np.float32))
        b = Q.quantise_dequantise(
            x, Q.BY_NAME["Q3_K"], rng.random(576).astype(np.float32) * 5 + 0.01
        )
        assert not np.array_equal(a[:, :512], b[:, :512])
        np.testing.assert_array_equal(a[:, 512:], b[:, 512:])

    def test_rejects_row_narrower_than_one_block(self, ggml):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((4, 128)).astype(np.float32)
        with pytest.raises(ValueError, match="256-wide block"):
            Q.quantise_dequantise(x, Q.BY_NAME["Q4_K"])

    def test_rejects_remainder_not_divisible_by_32(self, ggml):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((4, 300)).astype(np.float32)
        with pytest.raises(ValueError, match="remainder"):
            Q.quantise_dequantise(x, Q.BY_NAME["Q4_K"])


class TestWeightedMse:
    def test_unweighted_matches_plain_mean(self):
        a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        b = np.array([[0.0, 2.0], [2.0, 5.0]], dtype=np.float32)
        # squared diffs: 0, 1, 0, 4 -> mean 1.25
        assert Q.weighted_mse(a, b) == pytest.approx(1.25)

    def test_weight_on_one_column_only(self):
        """Importance [0, 1] must ignore column 0 entirely.

        Hand-computed: only column 1 contributes, squared diffs 1 and 4,
        weight sum 1 times 2 rows = 2, so (1 + 4) / 2 = 2.5.
        """
        a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        b = np.array([[9.0, 2.0], [9.0, 5.0]], dtype=np.float32)
        w = np.array([0.0, 1.0], dtype=np.float32)
        assert Q.weighted_mse(a, b, w) == pytest.approx(2.5)

    def test_uniform_weights_equal_unweighted(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 8))
        b = a + rng.standard_normal((4, 8)) * 0.1
        w = np.ones(8)
        assert Q.weighted_mse(a, b, w) == pytest.approx(Q.weighted_mse(a, b))

    def test_zero_weights_rejected(self):
        a = np.zeros((2, 2))
        with pytest.raises(ValueError, match="sum to zero"):
            Q.weighted_mse(a, a, np.zeros(2))


class TestInputValidation:
    def test_rejects_non_2d_weight(self, ggml):
        with pytest.raises(ValueError, match="2-D"):
            Q.quantise_dequantise(np.zeros(256, dtype=np.float32), Q.BY_NAME["Q4_K"])
