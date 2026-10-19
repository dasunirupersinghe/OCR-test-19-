"""Numeric prototype-to-context integration; these do not measure OCR accuracy."""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from config import get_config
from decoder import (CalibrationSample, ContextProbabilityModel, DecoderPipeline,
                     PrototypeCalibrationManager, decode_word_or_line)
from features import CandidateIdentity, FeatureBatchResult, FeaturePipeline
from segmentation import SegmentationHypothesis


M, K, G, T, Y = "ම", "ක", "ග", "ත", "ය"


def calibrated_line(*, mode: str = "B4", decoder_overrides=None, gate_enabled=True):
    """Five real calibrated means with ambiguous K/G/T and strong M/Y anchors."""
    settings = {"distance_metric": "standardized_euclidean", "gamma_sim": 5.,
                "eta_lex": .1, "beta_left": 1., "beta_right": 1., "beta_doc": .2}
    settings.update(decoder_overrides or {})
    config = get_config(mode, overrides={
        "features": {"micro_windows": 1},
        "flags": {"topology_features": False, "lexicon_gate": gate_enabled},
        "decoder": settings,
    })
    reference = FeaturePipeline(config).extract(np.eye(8, dtype=np.uint8))

    def feature(value, *, word_id=None):
        vector = np.array([value], dtype=np.float64)
        vector.setflags(write=False)
        identity = None if word_id is None else CandidateIdentity(
            f"word-{word_id}", word_id, (0, 0, 8, 8), (0, 0, 8, 8),
            ((0., 0.), (8., 0.), (8., 8.), (0., 8.)), ())
        return replace(reference, vector=vector, identity=identity)

    samples = tuple(CalibrationSample(label, feature(value))
                    for label, value in ((M, -20.), (K, 0.), (G, .5), (T, 1.), (Y, 20.)))
    book = PrototypeCalibrationManager(config).fit(samples)
    observations = tuple(feature(value, word_id=word_id)
                         for word_id, value in ((10, -20.), (20, 0.), (30, 20.)))
    batch = FeatureBatchResult(
        observations, {f.identity.candidate_id: f for f in observations},
        {f.identity.word_id: (SegmentationHypothesis(
            f"hyp-{f.identity.word_id}", (f.identity.candidate_id,), "intact", False),)
         for f in observations},
        {10: (), 20: (), 30: ()}, {"candidate_count": 3})
    context = ContextProbabilityModel(
        bigrams={(M, G): .001, (M, T): .9, (G, Y): .001, (T, Y): .9},
        domains={"report": {K: .05, G: .05, T: .9}})
    return config, book, batch, context


class DecoderContextPipelineIntegration(unittest.TestCase):
    def test_numeric_prototypes_generate_three_paths_and_two_immutable_anchors(self):
        config, book, batch, context = calibrated_line()
        result = DecoderPipeline(book, (G, T), config,
                                 context_model=context).decode_word_or_line(batch, doc_type="report")
        self.assertEqual(result.text, f"{M} {T} {Y}")
        self.assertEqual([word.word_id for word in result.words], [10, 20, 30])
        self.assertEqual([word.is_context_anchor for word in result.words], [True, False, True])
        self.assertEqual([word.needs_review for word in result.words], [False, False, False])
        center = result.words[1]
        self.assertEqual(center.raw_text, K)
        self.assertEqual([path.text for path in center.paths], [K, G, T])
        self.assertGreater(center.confidence, .4)
        self.assertLess(center.confidence, .6)
        self.assertEqual(center.confidence, center.paths[0].confidence)
        for word in (result.words[0], result.words[2]):
            self.assertGreaterEqual(word.confidence, .85)
            self.assertNotIn(word.text, (G, T))
            self.assertEqual(word.metadata["context_decision"]["reason"],
                             "high_confidence_oov_shield")

    def test_all_context_score_terms_match_original_model_word_probabilities(self):
        config, book, batch, context = calibrated_line()
        result = DecoderPipeline(book, (G, T), config,
                                 context_model=context).run(batch, doc_type="report")
        word = result.words[1]
        decision = word.metadata["context_decision"]
        self.assertEqual(decision["left_anchor"], M)
        self.assertEqual(decision["right_anchor"], Y)
        self.assertEqual({entry["text"] for entry in decision["candidate_scores"]}, {G, T})
        paths = {path.text: path for path in word.paths}
        for entry in decision["candidate_scores"]:
            label = entry["text"]
            probability = math.exp(paths[label].log_probability)
            expected = {
                "edit": 1.,
                "model": -.1 * math.log(probability + 1e-12),
                "left": -math.log(context.bigram_probability(label, M) + 1e-6),
                "right": -math.log(context.bigram_probability(Y, label) + 1e-6),
                "domain": -.2 * math.log(context.domain_probability(label, "report") + 1e-6),
            }
            for name, value in expected.items():
                self.assertAlmostEqual(entry["terms"][name], value, places=12)
            self.assertAlmostEqual(entry["score"], sum(expected.values()), places=12)
        selected = next(entry for entry in decision["candidate_scores"] if entry["text"] == T)
        self.assertEqual(decision["score"], selected["score"])
        self.assertEqual(word.lexicon_decision.selected_path, paths[T])
        self.assertEqual(word.confidence, paths[K].confidence)

    def test_raw_word_confidence_is_independent_mean_max_class_probability(self):
        config, book, batch, context = calibrated_line()
        result = DecoderPipeline(book, (G, T), config, context_model=context).run(batch)
        for observation, word in zip(batch.features, result.words):
            standardized = (observation.vector - book.center) / book.scale
            distances = np.linalg.norm(book.means - standardized, axis=1)
            weights = np.exp(-2 * 5. * (distances - distances.min()))
            expected_confidence = float(weights.max() / weights.sum())
            self.assertAlmostEqual(word.confidence, expected_confidence, places=14)
            self.assertEqual(word.confidence, word.paths[0].confidence)

    def test_multiple_glyph_confidence_averages_maxima_and_keeps_probability_product(self):
        config, book, batch, context = calibrated_line()
        left = batch.features[1]
        right = replace(batch.features[2], identity=replace(
            batch.features[2].identity, word_id=20, candidate_id="word-20-second",
            normalized_bbox=(8, 0, 16, 8)))
        together = replace(batch, features=(left, right),
                           by_candidate_id={f.identity.candidate_id: f for f in (left, right)},
                           hypotheses={20: (SegmentationHypothesis(
                               "two-glyph-word", (left.identity.candidate_id, right.identity.candidate_id),
                               "components", False),)}, diacritic_links={20: ()})
        result = DecoderPipeline(book, None, config, context_model=context).run(together)
        word = result.words[0]
        independent_maxima = []
        for observation in (left, right):
            query = (observation.vector-book.center)/book.scale
            distances = np.linalg.norm(book.means-query, axis=1)
            weights = np.exp(-10. * (distances-distances.min()))
            independent_maxima.append(float(weights.max()/weights.sum()))
        self.assertEqual(word.raw_text, K + Y)
        self.assertAlmostEqual(word.confidence, sum(independent_maxima)/2, places=14)
        self.assertAlmostEqual(math.exp(word.paths[0].log_probability),
                               math.prod(independent_maxima), places=14)
        self.assertFalse(word.is_context_anchor)
        self.assertTrue(word.needs_review)

    def test_convenience_function_matches_reusable_pipeline(self):
        config, book, batch, context = calibrated_line()
        expected = DecoderPipeline(book, (G, T), config,
                                   context_model=context).run(batch, doc_type="report")
        result = decode_word_or_line(batch, book, (G, T), config,
                                     context_model=context, doc_type="report")
        self.assertEqual(result.text, expected.text)
        self.assertEqual(result.metadata, expected.metadata)
        for actual, wanted in zip(result.words, expected.words):
            self.assertEqual(actual.metadata, wanted.metadata)
            self.assertEqual(actual.confidence, wanted.confidence)

    def test_unavailable_lexicon_retains_raw_and_marks_only_ambiguous_word_for_review(self):
        config, book, batch, context = calibrated_line()
        result = DecoderPipeline(book, None, config, context_model=context).run(batch)
        self.assertEqual(result.text, f"{M} {K} {Y}")
        self.assertEqual([word.needs_review for word in result.words], [False, True, False])
        self.assertEqual(tuple(result.metadata["review_word_ids"]), (20,))
        self.assertEqual(result.words[1].metadata["context_decision"]["reason"], "lexicon_unavailable")

    def test_lexicon_neighbour_without_generated_model_path_is_not_invented(self):
        config, book, batch, context = calibrated_line()
        result = DecoderPipeline(book, ("ප",), config, context_model=context).run(batch)
        center = result.words[1]
        self.assertEqual(center.text, K)
        self.assertTrue(center.needs_review)
        self.assertEqual(center.metadata["context_decision"]["reason"], "no_eligible_model_path")
        self.assertEqual(tuple(center.metadata["context_decision"]["candidate_scores"]), ())

    def test_single_word_has_no_left_or_right_context_terms(self):
        config, book, batch, context = calibrated_line()
        middle = batch.features[1]
        isolated = replace(batch, features=(middle,),
                           by_candidate_id={middle.identity.candidate_id: middle},
                           hypotheses={20: batch.hypotheses[20]}, diacritic_links={20: ()})
        result = DecoderPipeline(book, (G, T), config,
                                 context_model=context).run(isolated, doc_type="report")
        self.assertEqual(len(result.words), 1)
        self.assertEqual(result.words[0].word_id, 20)
        decision = result.words[0].metadata["context_decision"]
        self.assertIsNone(decision["left_anchor"])
        self.assertIsNone(decision["right_anchor"])
        self.assertEqual(decision["effective_weights"]["beta_left"], 0.)
        self.assertEqual(decision["effective_weights"]["beta_right"], 0.)
        for entry in decision["candidate_scores"]:
            self.assertEqual(entry["terms"]["left"], 0.)
            self.assertEqual(entry["terms"]["right"], 0.)
        self.assertEqual(result.text, T)

    def test_b3_mode_or_disabled_b4_gate_preserves_all_raw_words_as_anchors(self):
        for mode, enabled in (("B3", True), ("B4", False)):
            with self.subTest(mode=mode, enabled=enabled):
                config, book, batch, context = calibrated_line(mode=mode, gate_enabled=enabled)
                result = DecoderPipeline(book, (G, T), config,
                                         context_model=context).run(batch, doc_type="report")
                self.assertEqual(result.text, f"{M} {K} {Y}")
                self.assertTrue(all(word.is_context_anchor for word in result.words))
                self.assertFalse(any(word.needs_review for word in result.words))
                self.assertEqual(result.words[1].metadata["context_decision"]["reason"], "disabled")

    def test_eta_lex_override_and_legacy_eta_fallback_reach_context_decoder(self):
        for eta_lex, legacy, expected in ((.23, .79, .23), (None, .79, .79)):
            with self.subTest(eta_lex=eta_lex):
                config, book, batch, context = calibrated_line(
                    decoder_overrides={"eta_lex": eta_lex, "eta": legacy})
                result = DecoderPipeline(book, (G, T), config,
                                         context_model=context).run(batch, doc_type="report")
                decision = result.words[1].metadata["context_decision"]
                self.assertEqual(decision["effective_weights"]["eta_lex"], expected)
                for entry in decision["candidate_scores"]:
                    path = next(path for path in result.words[1].paths if path.text == entry["text"])
                    wanted = -expected * math.log(math.exp(path.log_probability) + 1e-12)
                    self.assertAlmostEqual(entry["terms"]["model"], wanted, places=12)

    def test_negative_context_weights_fail_at_configuration_boundary(self):
        for name in ("eta_lex", "beta_left", "beta_right", "beta_doc"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                calibrated_line(decoder_overrides={name: -.1})


if __name__ == "__main__":
    unittest.main()
