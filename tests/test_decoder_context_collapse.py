"""Independent contract tests for the requested two-pass context decoder."""
from __future__ import annotations

import math
import unittest

from decoder import ContextProbabilityModel, TwoPassContextDecoder, WordPath


def wp(text: str, probability: float = 0.4, confidence: float | None = 0.4) -> WordPath:
    return WordPath(text, math.log(probability), confidence)


class TwoPassContextContractTests(unittest.TestCase):
    def test_one_context_slot_cannot_hide_multiple_whitespace_separated_words(self):
        for text in ("ම ගත", "ගත\n", "\tගත"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                TwoPassContextDecoder([]).decode_word_or_line([wp(text, confidence=0.99)])

    def test_inclusive_high_confidence_shield_preserves_oov(self):
        paths = [wp("කත", confidence=0.85), wp("ගත", 0.9)]
        result = TwoPassContextDecoder(["ගත"]).decode_word_or_line(paths)
        word = result.words[0]
        self.assertEqual(word.text, "කත")
        self.assertTrue(word.anchor)
        self.assertFalse(word.needs_review)
        self.assertEqual(word.confidence, 0.85)

    def test_disabled_gating_preserves_raw_as_policy_anchor(self):
        for mode, enabled in [(mode, True) for mode in ("B0", "B1", "B2", "B3")] + [("B4", False)]:
            with self.subTest(mode=mode, enabled=enabled):
                raw = WordPath("කත", None, None)
                result = TwoPassContextDecoder(["ගත"], mode=mode, enabled=enabled).decode_word_or_line([raw, wp("ගත")])
                self.assertEqual(result.text, "කත")
                self.assertTrue(result.words[0].anchor)
                self.assertFalse(result.words[0].needs_review)

    def test_missing_confidence_is_reviewed_and_never_fabricated(self):
        result = TwoPassContextDecoder(["ගත"]).decode_word_or_line([wp("කත", confidence=None), wp("ගත")])
        word = result.words[0]
        self.assertEqual(word.text, "කත")
        self.assertIsNone(word.confidence)
        self.assertFalse(word.anchor)
        self.assertTrue(word.needs_review)

    def test_only_first_three_scored_model_paths_are_eligible(self):
        result = TwoPassContextDecoder(["ගත"]).decode_word_or_line([
            wp("කත"), wp("පත"), wp("බත"), wp("ගත", 0.99)])
        self.assertEqual(result.text, "කත")
        self.assertTrue(result.words[0].needs_review)

    def test_dictionary_neighbour_is_not_invented_without_model_path(self):
        result = TwoPassContextDecoder(["ගත"]).decode_word_or_line([wp("කත")])
        self.assertEqual(result.text, "කත")
        self.assertTrue(result.words[0].needs_review)

    def test_edit_radius_and_length_prefilter(self):
        result = TwoPassContextDecoder(["ගන", "කතනම"]).decode_word_or_line([
            wp("කත"), wp("ගන", 0.8), wp("කතනම", 0.9)])
        self.assertEqual(result.text, "කත")
        self.assertTrue(result.words[0].needs_review)

    def test_raw_in_lexicon_is_valid_zero_edit_candidate(self):
        result = TwoPassContextDecoder(["කත"]).decode_word_or_line([wp("කත")])
        word = result.words[0]
        self.assertEqual(word.text, "කත")
        self.assertFalse(word.needs_review)
        self.assertEqual(word.edit_distance, 0)
        self.assertFalse(word.anchor)

    def test_exact_isolated_score_zeros_neighbour_terms(self):
        context = ContextProbabilityModel(bigrams={("ම", "ගත"): 0.9}, domains={"letter": {"ගත": 0.8}})
        word = TwoPassContextDecoder(["ගත"], context, eta_lex=0.25,
                                     beta_left=3.0, beta_right=4.0, beta_doc=2.0).decode_word_or_line(
                                         [wp("කත"), wp("ගත", 0.2)], doc_type="letter").words[0]
        expected = 1 - 0.25 * math.log(0.2 + 1e-12) - 2.0 * math.log(0.8 + 1e-6)
        self.assertAlmostEqual(word.score, expected, places=12)
        self.assertIsNone(word.left_anchor)
        self.assertIsNone(word.right_anchor)

    def test_absent_document_type_disables_domain_term(self):
        context = ContextProbabilityModel(domains={"letter": {"ගත": 0.9}})
        word = TwoPassContextDecoder(["ගත"], context, beta_doc=100.0).decode_word_or_line([
            wp("කත"), wp("ගත", 0.2)]).words[0]
        self.assertAlmostEqual(word.score, 1 - 0.1 * math.log(0.2 + 1e-12), places=12)

    def test_no_probability_model_deactivates_context_terms(self):
        line = [[wp("ම", confidence=0.99)], [wp("කත"), wp("ගත", 0.2)], [wp("ය", confidence=0.99)]]
        word = TwoPassContextDecoder(["ගත"], beta_left=100.0, beta_right=100.0,
                                     beta_doc=100.0).decode_word_or_line(line, doc_type="letter").words[1]
        self.assertAlmostEqual(word.score, 1 - 0.1 * math.log(0.2 + 1e-12), places=12)

    def test_left_bigram_uses_candidate_given_left(self):
        context = ContextProbabilityModel(bigrams={("ම", "ගත"): 0.001, ("ම", "කන"): 0.9})
        line = [[wp("ම", confidence=0.99)], [wp("කත"), wp("ගත", 0.6), wp("කන", 0.4)]]
        word = TwoPassContextDecoder(["ගත", "කන"], context, beta_left=1.0).decode_word_or_line(line).words[1]
        self.assertEqual(word.text, "කන")
        self.assertEqual(word.left_anchor, "ම")
        self.assertIsNone(word.right_anchor)

    def test_right_bigram_uses_right_given_candidate(self):
        context = ContextProbabilityModel(bigrams={("ගත", "ය"): 0.001, ("කන", "ය"): 0.9})
        line = [[wp("කත"), wp("ගත", 0.6), wp("කන", 0.4)], [wp("ය", confidence=0.99)]]
        word = TwoPassContextDecoder(["ගත", "කන"], context, beta_right=1.0).decode_word_or_line(line).words[0]
        self.assertEqual(word.text, "කන")
        self.assertEqual(word.right_anchor, "ය")
        self.assertIsNone(word.left_anchor)

    def test_both_neighbours_and_domain_exact_formula(self):
        context = ContextProbabilityModel(bigrams={("ම", "ගත"): 0.2, ("ගත", "ය"): 0.3},
                                          domains={"letter": {"ගත": 0.4}})
        line = [[wp("ම", confidence=0.99)], [wp("කත"), wp("ගත", 0.5)], [wp("ය", confidence=0.99)]]
        word = TwoPassContextDecoder(["ගත"], context, eta_lex=0.12, beta_left=0.2,
                                     beta_right=0.3, beta_doc=0.4).decode_word_or_line(line, doc_type="letter").words[1]
        expected = 1 - 0.12 * math.log(0.5 + 1e-12) - 0.2 * math.log(0.2 + 1e-6) - 0.3 * math.log(0.3 + 1e-6) - 0.4 * math.log(0.4 + 1e-6)
        self.assertAlmostEqual(word.score, expected, places=12)
        self.assertEqual(word.confidence, 0.4)
        self.assertFalse(word.anchor)

    def test_pass_two_correction_never_becomes_anchor_for_next_word(self):
        context = ContextProbabilityModel(bigrams={("ගත", "කන"): 1.0, ("ගත", "පත"): 0.0},
                                          domains={"letter": {"ගත": 0.3, "පත": 0.3, "කන": 0.3}})
        line = [[wp("කත"), wp("ගත", 0.6)], [wp("බත"), wp("පත", 0.6), wp("කන", 0.4)]]
        result = TwoPassContextDecoder(["ගත", "පත", "කන"], context,
                                       beta_left=100.0, beta_doc=1.0).decode_word_or_line(line, doc_type="letter")
        self.assertEqual([w.text for w in result.words], ["ගත", "පත"])
        self.assertIsNone(result.words[1].left_anchor)
        self.assertFalse(result.words[0].anchor)

    def test_only_immediate_anchors_used(self):
        context = ContextProbabilityModel(bigrams={("ම", "කන"): 1.0, ("ම", "පත"): 0.0})
        line = [[wp("ම", confidence=0.99)], [wp("කත")], [wp("බත"), wp("පත", 0.6), wp("කන", 0.4)]]
        word = TwoPassContextDecoder(["පත", "කන"], context, beta_left=100.0).decode_word_or_line(line).words[2]
        self.assertEqual(word.text, "පත")
        self.assertIsNone(word.left_anchor)

    def test_extremely_small_model_probability_uses_exact_additive_floor(self):
        context = ContextProbabilityModel(bigrams={("ම", "ගත"): 0.0, ("ගත", "ය"): 0.0},
                                          domains={"letter": {"ගත": 0.0}})
        line = [[wp("ම", confidence=0.99)], [wp("කත"), WordPath("ගත", -1e300, 0.2)], [wp("ය", confidence=0.99)]]
        word = TwoPassContextDecoder(["ගත"], context, eta_lex=0.1, beta_left=0.2,
                                     beta_right=0.3, beta_doc=0.4).decode_word_or_line(line, doc_type="letter").words[1]
        expected = 1 - 0.1 * math.log(1e-12) - 0.9 * math.log(1e-6)
        self.assertTrue(math.isfinite(word.score))
        self.assertAlmostEqual(word.score, expected, places=12)

    def test_empty_line_is_valid(self):
        result = TwoPassContextDecoder([]).decode_word_or_line([])
        self.assertEqual(result.text, "")
        self.assertEqual(result.words, ())

    def test_empty_word_is_reviewed_and_never_anchor(self):
        result = TwoPassContextDecoder([]).decode_word_or_line([[], [wp("කත", confidence=0.99)]])
        self.assertTrue(result.words[0].needs_review)
        self.assertFalse(result.words[0].anchor)
        self.assertIsNone(result.words[1].left_anchor)

    def test_no_matching_lexicon_preserves_raw_with_review(self):
        for lexicon in (None, []):
            with self.subTest(lexicon=lexicon):
                word = TwoPassContextDecoder(lexicon).decode_word_or_line([wp("කත")]).words[0]
                self.assertEqual(word.text, "කත")
                self.assertTrue(word.needs_review)

    def test_negative_or_nonfinite_weights_rejected(self):
        for parameter in ("eta_lex", "beta_left", "beta_right", "beta_doc"):
            for value in (-0.1, float("nan"), float("inf")):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        TwoPassContextDecoder([], **{parameter: value})

    def test_scores_saturate_for_display_without_changing_order(self):
        paths = [wp("කත"), WordPath("ගත", -10.0, 0.1), WordPath("කන", -5.0, 0.1)]
        word = TwoPassContextDecoder(["ගත", "කන"], eta_lex=1e308).decode_word_or_line(paths).words[0]
        self.assertEqual(word.text, "කන")
        self.assertTrue(math.isfinite(word.score))
        self.assertTrue(all(candidate.metadata["score_saturated"] for candidate in word.candidates))
        self.assertEqual(word.candidates[0].score, word.candidates[1].score)

    def test_zero_weights_disable_all_context_lookups(self):
        class RefuseLookups(ContextProbabilityModel):
            def bigram_probability(self, word, previous):
                raise AssertionError("zero bigram weight must disable lookup")

            def domain_probability(self, word, doc_type):
                raise AssertionError("zero domain weight must disable lookup")

        model = RefuseLookups(bigrams={}, domains={})
        line = [[wp("ම", confidence=0.99)], [wp("කත"), wp("ගත", 0.2)], [wp("ය", confidence=0.99)]]
        word = TwoPassContextDecoder(["ගත"], model, beta_left=0.0, beta_right=0.0,
                                     beta_doc=0.0).decode_word_or_line(line, doc_type="letter").words[1]
        self.assertAlmostEqual(word.score, 1 - 0.1 * math.log(0.2 + 1e-12), places=12)

    def test_explicit_empty_context_source_assigns_zero_events(self):
        model = ContextProbabilityModel(bigrams={}, domains={})
        line = [[wp("ම", confidence=0.99)], [wp("කත"), wp("ගත", 0.2)], [wp("ය", confidence=0.99)]]
        word = TwoPassContextDecoder(["ගත"], model, beta_left=0.2, beta_right=0.3,
                                     beta_doc=0.4).decode_word_or_line(line, doc_type="letter").words[1]
        expected = 1 - 0.1 * math.log(0.2 + 1e-12) - 0.9 * math.log(1e-6)
        self.assertAlmostEqual(word.score, expected, places=12)

    def test_invalid_document_type_rejected(self):
        for document_type in ("", " ", "a" * 257, 23, True):
            with self.subTest(document_type=document_type):
                with self.assertRaises((TypeError, ValueError)):
                    TwoPassContextDecoder([]).decode_word_or_line([wp("කත")], doc_type=document_type)

    def test_line_and_beam_budgets_rejected(self):
        decoder = TwoPassContextDecoder([])
        with self.assertRaises(ValueError):
            decoder.decode_word_or_line([[wp("කත")]] * 4097)
        with self.assertRaises(ValueError):
            decoder.decode_word_or_line([wp("කත")] * 129)

    def test_bad_path_groups_rejected(self):
        for groups in ("කත", [wp("කත"), []], [["කත"]], [23]):
            with self.subTest(groups=groups):
                with self.assertRaises((TypeError, ValueError)):
                    TwoPassContextDecoder([]).decode_word_or_line(groups)


if __name__ == "__main__":
    unittest.main()
