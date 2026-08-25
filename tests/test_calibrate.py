"""Calibration corpora and the imatrix comparison.

Mostly guarding properties the experiment silently assumes: that the corpora
are actually distinct, that none of them overlaps the held-out evaluation
text, and that the cosine comparison is scale-invariant. A calibration
sensitivity result computed against an evaluation set that shared sentences
with one of the corpora would be measuring leakage, and would look exactly
like a strong finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantization_accuracy_curves.calibrate import (
    CORPORA,
    CORPORA_BY_NAME,
    EVAL_CORPUS,
    cosine_similarity,
    importance_from_activations,
)


def _sentences(text: str) -> set[str]:
    return {s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 25}


class TestCorpora:
    def test_three_corpora_present(self):
        assert len(CORPORA) == 3
        assert {c.name for c in CORPORA} == {"encyclopedic", "code", "chat"}

    def test_lookup_matches_the_tuple(self):
        assert set(CORPORA_BY_NAME) == {c.name for c in CORPORA}
        for c in CORPORA:
            assert CORPORA_BY_NAME[c.name] is c

    def test_every_corpus_is_substantial(self):
        """Enough text that an imatrix is not dominated by a handful of tokens."""
        for c in CORPORA:
            assert len(c.text) > 800, f"{c.name} is only {len(c.text)} chars"

    def test_eval_corpus_is_longer_than_a_single_window(self):
        """Perplexity needs several 128-token windows for a bootstrap.

        Roughly four characters per token, so 128 tokens is about 500
        characters and the bootstrap wants several windows. An earlier version
        of the eval text tokenised to 247 tokens and could not fill even one
        256-token window.
        """
        assert len(EVAL_CORPUS.text) > 3000

    def test_corpora_are_pairwise_distinct(self):
        texts = [c.text for c in CORPORA]
        assert len(set(texts)) == len(texts)

    def test_no_calibration_sentence_appears_in_the_eval_text(self):
        """No leakage. The headline result is void if the eval text is seen.

        Sentence-level rather than substring: short fragments like "It was"
        legitimately recur, but a whole shared clause would mean one corpus
        had an unfair advantage that would masquerade as calibration
        sensitivity.
        """
        eval_sentences = _sentences(EVAL_CORPUS.text)
        for c in CORPORA:
            overlap = _sentences(c.text) & eval_sentences
            assert not overlap, f"{c.name} shares sentences with the eval text: {overlap}"

    def test_every_corpus_has_a_description(self):
        """Carried into the results file so a spread number is interpretable."""
        for c in (*CORPORA, EVAL_CORPUS):
            assert c.description and len(c.description) > 10

    def test_code_corpus_is_punctuation_dense(self):
        """A cheap check that the corpora really are different distributions.

        If someone replaced the code corpus with more prose, the headline
        spread would shrink and the repo would report a weaker finding for the
        wrong reason.
        """
        def density(t: str) -> float:
            return sum(ch in "(){}[]=<>;:_." for ch in t) / len(t)

        code = density(CORPORA_BY_NAME["code"].text)
        prose = density(CORPORA_BY_NAME["encyclopedic"].text)
        assert code > 2 * prose, f"code {code:.3f} vs prose {prose:.3f}"

    def test_chat_corpus_uses_dialogue_markers(self):
        chat = CORPORA_BY_NAME["chat"].text
        assert chat.count("User:") >= 4
        assert chat.count("Assistant:") >= 4
        assert "User:" not in CORPORA_BY_NAME["encyclopedic"].text


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_scaling_does_not_change_the_cosine(self):
        """The reason cosine is used rather than a distance.

        The quantiser only consults the imatrix's shape -- scaling every
        importance by a constant cannot change which feature is relatively
        more important -- so the comparison must be blind to magnitude. A raw
        L2 distance would report two identically-shaped imatrices from corpora
        of different lengths as wildly different.
        """
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v * 1000.0) == pytest.approx(1.0)
        assert cosine_similarity(v, v * 0.001) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_hand_computed_value(self):
        """[1,0] vs [1,1]: cos = 1/sqrt(2)."""
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / np.sqrt(2))

    def test_flattens_before_comparing(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="differ in shape"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_rejects_zero_vector(self):
        with pytest.raises(ValueError, match="zero importance"):
            cosine_similarity([0.0, 0.0], [1.0, 2.0])


class TestImportanceReduction:
    def test_divides_sum_by_count(self):
        """Mean squared activation: sum of squares over token count."""
        got = importance_from_activations([("layer.0", np.array([4.0, 8.0]), 4)])
        np.testing.assert_allclose(got["layer.0"], [1.0, 2.0])

    def test_handles_several_layers(self):
        got = importance_from_activations(
            [("a", np.array([2.0]), 2), ("b", np.array([9.0]), 3)]
        )
        assert set(got) == {"a", "b"}
        np.testing.assert_allclose(got["a"], [1.0])
        np.testing.assert_allclose(got["b"], [3.0])

    def test_rejects_zero_token_count(self):
        with pytest.raises(ValueError, match="no tokens observed"):
            importance_from_activations([("layer.0", np.array([1.0]), 0)])
