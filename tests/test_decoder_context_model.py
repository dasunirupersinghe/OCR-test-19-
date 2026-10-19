"""Offline context evidence contracts, independent of OCR model predictions."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import decoder


from decoder import ContextProbabilityModel


class ContextProbabilityContracts(unittest.TestCase):
    def test_directional_bigram_and_domain_lookup(self):
        model = ContextProbabilityModel(
            {("ශ්‍රී", "ලංකා"): .8, ("ලංකා", "ශ්‍රී"): .1},
            {"address": {"ලංකා": .7, "ගම": .2}},
        )
        self.assertEqual(model.bigram_probability("ලංකා", "ශ්‍රී"), .8)
        self.assertEqual(model.bigram_probability("ශ්‍රී", "ලංකා"), .1)
        self.assertEqual(model.domain_probability("ලංකා", "address"), .7)
        self.assertEqual(model.domain_probability("ලංකා", "unknown"), 0.)
        self.assertEqual(model.bigram_probability("ගම", "ශ්‍රී"), 0.)
        self.assertEqual(model.bigram_probability("ගම", None), 0.)
        self.assertEqual(model.domain_probability("ගම", None), 0.)

    def test_missing_sources_and_observed_empty_tables_differ(self):
        missing, empty = ContextProbabilityModel(), ContextProbabilityModel({}, {})
        self.assertFalse(missing.has_bigrams)
        self.assertFalse(missing.has_domains)
        self.assertTrue(empty.has_bigrams)
        self.assertTrue(empty.has_domains)
        self.assertEqual(empty.bigram_probability("ක", "ග"), 0.)
        self.assertEqual(empty.domain_probability("ක", "form"), 0.)

    def test_tables_are_defensive_deep_immutable_copies(self):
        bigrams = {("ක", "ග"): .4}
        domains = {"form": {"ක": .6}}
        model = ContextProbabilityModel(bigrams, domains)
        bigrams[("ක", "ග")] = .7
        domains["form"]["ක"] = .1
        self.assertEqual(model.bigram_probability("ග", "ක"), .4)
        self.assertEqual(model.domain_probability("ක", "form"), .6)
        with self.assertRaises(TypeError):
            model.bigrams[("ක", "ග")] = .8
        with self.assertRaises(TypeError):
            model.domains["form"]["ක"] = .8
        with self.assertRaises(FrozenInstanceError):
            model.bigrams = {}

    def test_nfc_normalizes_queries_and_keys(self):
        model = ContextProbabilityModel({("ක", "ක\u0dd9\u0dcf"): .8},
                                        {"form": {"ක\u0dd9\u0dcf": .7}})
        self.assertEqual(model.bigram_probability("කො", "ක"), .8)
        self.assertEqual(model.domain_probability("කො", "form"), .7)
        self.assertIn(("ක", "කො"), model.bigrams)

    def test_duplicate_keys_after_normalization_are_rejected(self):
        pairs = {("ක", "කො"): .4, ("ක", "ක\u0dd9\u0dcf"): .4}
        domains = {"form": {"කො": .4, "ක\u0dd9\u0dcf": .4}}
        for args in ((pairs, None), (None, domains),
                     (None, {"é": {"ක": .2}, "e\u0301": {"ග": .2}})):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ContextProbabilityModel(*args)

    def test_invalid_words_and_context_identifiers_are_rejected(self):
        for word in ("", " ", "ක ග", "ක\t", "\nක", 1):
            with self.subTest(word=word), self.assertRaises((ValueError, TypeError)):
                ContextProbabilityModel({("ක", word): .2})
        for doc_type in ("", "   ", 1, None):
            with self.subTest(doc_type=doc_type), self.assertRaises((ValueError, TypeError)):
                ContextProbabilityModel(domains={doc_type: {"ක": .2}})
        for pair in ("කග", ("ක",), ("ක", "ග", "ත")):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                ContextProbabilityModel({pair: .2})

    def test_invalid_probabilities_and_excess_row_mass_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), -1., 1.01, True, "0.2"):
            for kwargs in ({"bigrams": {("ක", "ග"): value}},
                           {"domains": {"form": {"ක": value}}}):
                with self.subTest(value=value, kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                    ContextProbabilityModel(**kwargs)
        with self.assertRaises(ValueError):
            ContextProbabilityModel({("ක", "ග"): .6, ("ක", "ත"): .5})
        with self.assertRaises(ValueError):
            ContextProbabilityModel(domains={"form": {"ක": .6, "ග": .5}})

    def test_rows_are_conditioned_independently_and_partial_mass_preserved(self):
        model = ContextProbabilityModel({("ක", "ග"): .8, ("ග", "ත"): .8},
                                        {"one": {"ක": .2}, "two": {"ග": .8}})
        self.assertEqual(model.bigram_probability("ග", "ක"), .8)
        self.assertEqual(model.domain_probability("ක", "one"), .2)
        self.assertEqual(sum(model.domains["one"].values()), .2)

    def test_count_rows_are_normalized_and_large_counts_do_not_overflow(self):
        model = ContextProbabilityModel.from_counts(
            {("ක", "ග"): 3, ("ක", "ත"): 1, ("ග", "ක"): 0},
            {"form": {"ක": 1e308, "ග": 1e308}, "empty": {"ක": 0}},
        )
        self.assertEqual(model.bigram_probability("ග", "ක"), .75)
        self.assertEqual(model.bigram_probability("ත", "ක"), .25)
        self.assertEqual(model.domain_probability("ක", "form"), .5)
        self.assertEqual(model.domain_probability("ග", "form"), .5)
        self.assertEqual(dict(model.domains["empty"]), {})
        self.assertNotIn(("ග", "ක"), model.bigrams)

    def test_count_rows_merge_normalized_previous_contexts_before_normalizing(self):
        model = ContextProbabilityModel.from_counts({("é", "ක"): 3, ("e\u0301", "ග"): 1})
        self.assertEqual(model.bigram_probability("ක", "é"), .75)
        self.assertEqual(model.bigram_probability("ග", "e\u0301"), .25)

    def test_invalid_counts_and_data_splits_rejected(self):
        for value in (-1, float("nan"), float("inf"), True, "2"):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                ContextProbabilityModel.from_counts({("ක", "ග"): value})
        for split in ("test", "evaluation", "train", None):
            with self.subTest(split=split), self.assertRaises(ValueError):
                ContextProbabilityModel.from_counts({("ක", "ග"): 1}, split=split)
        self.assertFalse(ContextProbabilityModel.from_counts().has_bigrams)
        self.assertTrue(ContextProbabilityModel.from_counts(bigrams={}).has_bigrams)

    def test_wrong_container_types_rejected(self):
        for kwargs in ({"bigrams": []}, {"domains": []}, {"domains": {"form": []}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                ContextProbabilityModel(**kwargs)


if __name__ == "__main__":
    unittest.main()
