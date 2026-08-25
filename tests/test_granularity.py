"""Affine quantisation arithmetic, against fixtures computed by hand.

The point of hand-computed fixtures rather than round-trip property checks:
an off-by-one in the zero point, or a scale using ``2**bits`` instead of
``2**bits - 1``, still round-trips plausibly and still produces a smooth
degradation curve. It just produces the wrong one. Only comparing against a
value derived on paper catches that.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantization_accuracy_curves.granularity import (
    PER_CHANNEL,
    PER_TENSOR,
    affine_quantise_dequantise,
    effective_bits,
    per_group,
    scale_overhead_bits,
)


class TestHandComputedFixtures:
    def test_two_bit_per_tensor_exact(self):
        """A row whose values land exactly on the 2-bit grid must round-trip exactly.

        Worked by hand: min = 0.0, max = 3.0, qmax = 3, so scale = 1.0 and
        zero = round(-0.0 / 1.0) = 0. Every input is already an integer
        multiple of the scale, so reconstruction is lossless. If the
        implementation divided by 2**bits = 4 instead of 3, scale would be
        0.75 and none of these would come back exact.
        """
        x = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, PER_TENSOR)
        np.testing.assert_allclose(out, x, atol=0)

    def test_two_bit_asymmetric_range(self):
        """min = -1, max = 2 over 2 bits: scale = 1.0, zero = 1.

        Grid is {-1, 0, 1, 2}. Inputs sit on it, so the round trip is exact
        and the zero point must be exactly 1 for that to happen. A zero point
        of 0 would shift the whole grid and clip -1 to 0.
        """
        x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, PER_TENSOR)
        np.testing.assert_allclose(out, x, atol=0)

    def test_two_bit_midpoint_rounds_to_grid(self):
        """0.5 with min = 0, max = 3 must snap to the nearest grid point.

        scale = 1.0, so the grid is {0, 1, 2, 3}. numpy's rint uses
        round-half-to-even, so 0.5 -> 0 and 1.5 -> 2. Asserted explicitly
        because a switch to round-half-away-from-zero would change measured
        error on every tensor in the repo, and it should not happen silently.
        """
        x = np.array([[0.0, 0.5, 1.5, 3.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, PER_TENSOR)
        np.testing.assert_allclose(out, [[0.0, 0.0, 2.0, 3.0]], atol=1e-6)

    def test_four_bit_scale_is_range_over_fifteen(self):
        """min = 0, max = 15 over 4 bits gives scale exactly 1.0."""
        x = np.arange(16, dtype=np.float32).reshape(1, 16)
        out = affine_quantise_dequantise(x, 4, PER_TENSOR)
        np.testing.assert_allclose(out, x, atol=0)

    def test_three_bit_known_scale(self):
        """min = 0, max = 7 over 3 bits: qmax = 7, scale = 1.0, exact."""
        x = np.arange(8, dtype=np.float32).reshape(1, 8)
        out = affine_quantise_dequantise(x, 3, PER_TENSOR)
        np.testing.assert_allclose(out, x, atol=0)

    def test_per_channel_uses_each_row_own_range(self):
        """Two rows on different scales must each round-trip exactly.

        Row 0 spans [0, 3] and row 1 spans [0, 300]. Per-channel gives each
        its own scale (1.0 and 100.0), so both are exact. Per-tensor would
        share one scale of 100.0 and destroy row 0 entirely -- which the next
        test asserts, making this pair the actual evidence that the
        granularity flag is wired to anything.
        """
        x = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 200.0, 300.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, PER_CHANNEL)
        np.testing.assert_allclose(out, x, atol=1e-4)

    def test_per_tensor_shares_one_scale_and_destroys_the_small_row(self):
        """The same input under per-tensor: scale = 100, row 0 collapses to 0.

        Hand-computed: min = 0, max = 300, qmax = 3, scale = 100, zero = 0.
        Row 0's values 1.0 and 2.0 are far below half a step, so they quantise
        to 0 and reconstruct as 0.0. This is the failure mode per-channel
        exists to fix.
        """
        x = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 200.0, 300.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, PER_TENSOR)
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(out[1], [0.0, 100.0, 200.0, 300.0], atol=1e-4)

    def test_per_group_isolates_a_spike_to_its_own_group(self):
        """A large value in the second half must not corrupt the first half.

        Eight columns, group size 4. Group 0 spans [0, 3] -> scale 1.0, exact.
        Group 1 contains the 300 spike and takes its own scale. Under
        per-channel the whole row would share scale 100 and the first four
        values would collapse.
        """
        x = np.array([[0.0, 1.0, 2.0, 3.0, 0.0, 100.0, 200.0, 300.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, per_group(4))
        np.testing.assert_allclose(out[0, :4], [0.0, 1.0, 2.0, 3.0], atol=1e-4)

        coarse = affine_quantise_dequantise(x, 2, PER_CHANNEL)
        assert coarse[0, 1] == pytest.approx(0.0, abs=1e-4)


class TestDegenerateAndInvalid:
    def test_constant_group_reconstructs_exactly(self):
        """max == min means scale 0. Must return the input, not NaN."""
        x = np.full((2, 8), 2.5, dtype=np.float32)
        out = affine_quantise_dequantise(x, 4, PER_CHANNEL)
        np.testing.assert_allclose(out, x, atol=0)
        assert np.all(np.isfinite(out))

    def test_all_zeros_is_finite(self):
        x = np.zeros((2, 8), dtype=np.float32)
        out = affine_quantise_dequantise(x, 4, PER_TENSOR)
        np.testing.assert_allclose(out, x, atol=0)

    def test_mixed_constant_and_varying_rows(self):
        """A degenerate row alongside a normal one: both must be right.

        Guards the vectorised ``where`` that patches degenerate partitions --
        an implementation that fell back to the input for *all* rows once any
        row was degenerate would pass the all-constant test above and fail
        here.
        """
        x = np.array([[1.0] * 4, [0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        out = affine_quantise_dequantise(x, 2, PER_CHANNEL)
        np.testing.assert_allclose(out[0], [1.0] * 4, atol=0)
        np.testing.assert_allclose(out[1], [0.0, 1.0, 2.0, 3.0], atol=1e-4)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            affine_quantise_dequantise(np.zeros(8, dtype=np.float32), 4, PER_TENSOR)

    def test_rejects_out_of_range_bits(self):
        x = np.zeros((1, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="bits"):
            affine_quantise_dequantise(x, 1, PER_TENSOR)
        with pytest.raises(ValueError, match="bits"):
            affine_quantise_dequantise(x, 17, PER_TENSOR)

    def test_rejects_indivisible_group_size(self):
        x = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="multiple of 4"):
            affine_quantise_dequantise(x, 4, per_group(4))

    def test_rejects_nonpositive_group_size(self):
        with pytest.raises(ValueError, match="positive"):
            per_group(0)


class TestGranularityOrdering:
    def test_finer_granularity_never_reconstructs_worse(self):
        """per-tensor >= per-channel >= per-group-64 >= per-group-16, in MSE.

        This ordering is forced rather than empirical: each finer partition
        minimises the same squared error over a strictly finer set, so it
        cannot do worse. Asserted as an invariant precisely because it is not
        a measurement -- if it ever fails, the implementation is wrong, not
        the machine.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16, 128)).astype(np.float32)
        levels = [PER_TENSOR, PER_CHANNEL, per_group(64), per_group(16)]
        errors = [
            float(((x - affine_quantise_dequantise(x, 4, g)) ** 2).mean()) for g in levels
        ]
        for coarse, fine in zip(errors, errors[1:], strict=False):
            assert fine <= coarse * 1.000001, f"{errors}"

    def test_more_bits_never_reconstructs_worse(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((8, 64)).astype(np.float32)
        errors = [
            float(((x - affine_quantise_dequantise(x, b, PER_CHANNEL)) ** 2).mean())
            for b in (2, 3, 4, 6, 8)
        ]
        for coarse, fine in zip(errors, errors[1:], strict=False):
            assert fine <= coarse * 1.000001, f"{errors}"


class TestScaleOverhead:
    def test_per_tensor_overhead_is_negligible(self):
        """One 32-bit scale+zero pair over 1024 weights."""
        assert scale_overhead_bits((32, 32), 4, PER_TENSOR) == pytest.approx(32.0 / 1024)

    def test_per_channel_overhead_is_one_pair_per_row(self):
        """32 rows of 32: 32 pairs over 1024 weights = 1 bit/weight."""
        assert scale_overhead_bits((32, 32), 4, PER_CHANNEL) == pytest.approx(1.0)

    def test_per_group_32_costs_exactly_one_bit(self):
        """A 32-bit scale pair per 32 weights is 1.0 bits/weight, by definition.

        The number that makes the granularity trade-off concrete: 4-bit
        per-group-32 costs 5 effective bits, so it must be compared against
        5-bit per-channel rather than against 4-bit anything.
        """
        assert scale_overhead_bits((8, 256), 4, per_group(32)) == pytest.approx(1.0)
        assert effective_bits((8, 256), 4, per_group(32)) == pytest.approx(5.0)

    def test_group_64_costs_half_a_bit(self):
        assert scale_overhead_bits((8, 256), 4, per_group(64)) == pytest.approx(0.5)

    def test_overhead_shrinks_as_groups_widen(self):
        shape = (16, 256)
        wide = scale_overhead_bits(shape, 4, per_group(128))
        narrow = scale_overhead_bits(shape, 4, per_group(32))
        assert wide < narrow

    def test_overhead_rejects_indivisible_group(self):
        with pytest.raises(ValueError, match="does not divide"):
            scale_overhead_bits((4, 10), 4, per_group(4))
