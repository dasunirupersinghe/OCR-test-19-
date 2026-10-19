"""Independent decoder contracts; synthetic checks do not measure OCR accuracy."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unicodedata
import unittest
from unittest import mock
import zipfile

import numpy as np

from config import (COMPOSITE_VOWEL_COMPONENTS, CONJUNCT_PREFIXES,
                    CONJUNCT_SUFFIXES, CONSONANTS, INDEPENDENT_VOWELS,
                    VOWEL_SIGNS, compose_consonant_vowel, get_config)
from decoder import (LexiconGate, SinhalaAssembler, UnicodeAssemblyError,
                     WordPath, levenshtein_distance, CalibrationSample,
                     PrototypeCalibrationManager, PrototypeBook, PrototypeScorer,
                     DecoderPipeline, DecoderOptions)
from features import FeaturePipeline
from segmentation import (BoundingBox, DiacriticLink, SegmentationHypothesis,
                          SegmentationPipeline)


def controlled_config(k: int = 1, **decoder: object):
    return get_config("B4", overrides={
        "features": {"micro_windows": k},
        "flags": {"topology_features": False},
        "decoder": decoder,
    })


def controlled_feature(values: list[float] | np.ndarray, *, k: int = 1,
                       view: str = "whole_glyph"):
    pipeline = FeaturePipeline(controlled_config(k))
    reference = pipeline.extract(np.eye(8, dtype=np.uint8), view=view)
    vector = np.asarray(values, dtype=np.float64)
    vector.setflags(write=False)
    return replace(reference, vector=vector)


def controlled_book(*, k: int = 1, per_class: int = 3, view: str = "whole_glyph", **decoder: object):
    dimension = k * k
    samples = []
    for label, offset in (("ක", 0.), ("ග", 10.)):
        for index in range(per_class):
            values = np.arange(dimension, dtype=float) * .3 + offset + index
            samples.append(CalibrationSample(label, controlled_feature(values, k=k, view=view)))
    config = controlled_config(k, **decoder)
    return PrototypeCalibrationManager(config).fit(samples), config


class DecoderCalibrationContracts(unittest.TestCase):
    def test_global_standardization_and_class_means_are_training_only(self) -> None:
        book, config = controlled_book()
        all_values = np.array((0., 1., 2., 10., 11., 12.))
        expected_center = all_values.mean()
        expected_scale = all_values.std(ddof=0)
        np.testing.assert_allclose(book.center, (expected_center,))
        np.testing.assert_allclose(book.scale, (expected_scale,))
        means = {label: book.means[i, 0] for i, label in enumerate(book.labels)}
        self.assertAlmostEqual(means["ක"], (1. - expected_center) / expected_scale)
        self.assertAlmostEqual(means["ග"], (11. - expected_center) / expected_scale)
        original_center = book.center.copy()
        PrototypeScorer(book, config).score(controlled_feature([100.]))
        np.testing.assert_array_equal(book.center, original_center)

    def test_calibration_splits_and_feature_schema_views_are_enforced(self) -> None:
        manager = PrototypeCalibrationManager(controlled_config())
        for split in ("test", "evaluation", "train"):
            with self.subTest(split=split), self.assertRaises((ValueError, TypeError)):
                manager.fit((CalibrationSample("ක", controlled_feature([0.]), split),))
        for second in (controlled_feature([1., 2., 3., 4.], k=2),
                       controlled_feature([1.], view="zone:core")):
            with self.subTest(view=second.metadata["view"]), self.assertRaises((ValueError, TypeError)):
                manager.fit((CalibrationSample("ක", controlled_feature([0.])),
                             CalibrationSample("ග", second)))
        with self.assertRaises((ValueError, TypeError)):
            manager.fit(())

    def test_nonfinite_or_wrong_dimension_samples_are_rejected(self) -> None:
        manager = PrototypeCalibrationManager(controlled_config())
        for values in ([float("nan")], [float("inf")], [1., 2.]):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                manager.fit((CalibrationSample("ක", controlled_feature(values)),))

    def test_covariance_requires_dimension_aware_sample_count(self) -> None:
        insufficient, _ = controlled_book(k=2, per_class=3)
        self.assertFalse(insufficient.covariance_available.any())
        sufficient, _ = controlled_book(k=2, per_class=5)
        self.assertTrue(sufficient.covariance_available.all())

    def test_covariance_matches_independent_global_space_reference(self) -> None:
        book, config = controlled_book(k=2, per_class=5)
        data = np.asarray([np.arange(4)*.3 + index for index in range(5)])
        standardized = (data-book.center)/book.scale
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        settings = config.decoder
        expected = ((1-settings.covariance_shrinkage)*covariance
                    + settings.covariance_shrinkage*np.diag(np.diag(covariance))
                    + settings.covariance_regularization*np.eye(4))
        np.testing.assert_allclose(book.covariances[book.labels.index("ක")], expected)


class DecoderScorerContracts(unittest.TestCase):
    def test_standardized_euclidean_distances_and_amplitudes_match_formula(self) -> None:
        book, config = controlled_book(distance_metric="standardized_euclidean", gamma_sim=.7)
        query = controlled_feature([4.])
        result = PrototypeScorer(book, config).score(query)
        standardized = (query.vector-book.center)/book.scale
        distances = np.linalg.norm(book.means-standardized, axis=1)
        np.testing.assert_allclose(result.distances, distances)
        weights = np.exp(-config.decoder.gamma_sim*(distances-distances.min()))
        amplitudes = weights / np.linalg.norm(weights)
        np.testing.assert_allclose(result.amplitudes, amplitudes)
        np.testing.assert_allclose(result.probabilities, amplitudes**2)
        self.assertAlmostEqual(float(np.sum(result.probabilities)), 1.)
        self.assertEqual(result.top_candidates[0].label,
                         book.labels[int(np.argmin(distances))])

    def test_nearest_baselines_do_not_invent_probabilities(self) -> None:
        book, _ = controlled_book(distance_metric="standardized_euclidean")
        for mode in ("B0", "B1", "B2"):
            config = get_config(mode, overrides={"features": {"micro_windows": 1}})
            result = PrototypeScorer(book, config).score(controlled_feature([1.]))
            self.assertIsNone(result.amplitudes)
            self.assertIsNone(result.probabilities)
            self.assertIsNone(result.confidence)
            for candidate in result.top_candidates:
                self.assertIsNone(candidate.probability)
                self.assertIsNone(candidate.log_probability)

    def test_mahalanobis_distance_matches_independent_solve(self) -> None:
        book, config = controlled_book(k=2, per_class=5)
        query = controlled_feature([4., 4.2, 5.1, 4.7], k=2)
        result = PrototypeScorer(book, config).score(query)
        standardized = (query.vector-book.center)/book.scale
        expected = []
        for mean, covariance in zip(book.means, book.covariances):
            delta = standardized-mean
            expected.append(np.sqrt(delta@np.linalg.solve(covariance, delta)))
        np.testing.assert_allclose(result.distances, expected)

    def test_query_view_schema_nonfinite_and_dimension_mismatch_fail_closed(self) -> None:
        book, config = controlled_book()
        scorer = PrototypeScorer(book, config)
        for query in (controlled_feature([1.], view="zone:core"),
                      controlled_feature([1., 2., 3., 4.], k=2),
                      controlled_feature([float("nan")])):
            with self.subTest(shape=query.vector.shape), self.assertRaises((ValueError, TypeError)):
                scorer.score(query)

    def test_near_and_far_scores_remain_finite_without_exponent_underflow_failure(self) -> None:
        book, config = controlled_book(distance_metric="standardized_euclidean", gamma_sim=1000.)
        result = PrototypeScorer(book, config).score(controlled_feature([1e100]))
        self.assertTrue(np.isfinite(result.distances).all())
        self.assertTrue(np.isfinite(result.amplitudes).all())
        self.assertTrue(np.isfinite(result.probabilities).all())
        self.assertAlmostEqual(float(result.probabilities.sum()), 1.)


class DecoderPersistenceContracts(unittest.TestCase):
    def test_npz_manifest_roundtrip_keeps_model_and_scores(self) -> None:
        book, config = controlled_book(k=2, per_class=5)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/"model.npz"
            manifest = Path(directory)/"model.json"
            book.save(archive, manifest)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIsInstance(document, dict)
            restored = PrototypeBook.load(archive, manifest)
            for field in ("means", "center", "scale", "covariances", "covariance_available", "counts"):
                np.testing.assert_array_equal(getattr(book, field), getattr(restored, field))
            for field in ("labels", "schema_id", "view", "model_view_key", "feature_names"):
                self.assertEqual(getattr(book, field), getattr(restored, field))
            query = controlled_feature([2., 2.1, 2.5, 3.], k=2)
            original = PrototypeScorer(book, config).score(query)
            reloaded = PrototypeScorer(restored, config).score(query)
            np.testing.assert_array_equal(original.distances, reloaded.distances)
            np.testing.assert_array_equal(original.probabilities, reloaded.probabilities)

    def test_tampered_archive_is_rejected(self) -> None:
        book, _ = controlled_book()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/"model.npz"
            manifest = Path(directory)/"model.json"
            book.save(archive, manifest)
            contents = bytearray(archive.read_bytes())
            contents[-1] ^= 1
            archive.write_bytes(contents)
            with self.assertRaises((ValueError, OSError)):
                PrototypeBook.load(archive, manifest)

    def test_overwrite_is_explicit_and_load_honors_resource_budget(self) -> None:
        book, _ = controlled_book()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/"model.npz"
            manifest = Path(directory)/"model.json"
            book.save(archive, manifest)
            with self.assertRaises((ValueError, OSError)):
                book.save(archive, manifest)
            with self.assertRaises((ValueError, OSError)):
                PrototypeBook.load(archive, manifest, max_model_bytes=1)

    def test_wrong_array_shape_or_dtype_is_rejected_before_numpy_allocation(self) -> None:
        book, _ = controlled_book()
        for replacement in (np.zeros((10, 10), np.float64), np.zeros(book.means.shape, np.float16)):
            with self.subTest(shape=replacement.shape, dtype=replacement.dtype), tempfile.TemporaryDirectory() as directory:
                archive = Path(directory)/"model.npz"
                manifest = Path(directory)/"model.json"
                book.save(archive, manifest)
                with zipfile.ZipFile(archive) as original:
                    members = {info.filename: original.read(info) for info in original.infolist()}
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, replacement, allow_pickle=False)
                members["means.npy"] = buffer.getvalue()
                payload = io.BytesIO()
                with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as altered:
                    for name, data in members.items():
                        altered.writestr(name, data)
                archive.write_bytes(payload.getvalue())
                document = json.loads(manifest.read_text(encoding="utf-8"))
                document["archive_sha256"] = hashlib.sha256(payload.getvalue()).hexdigest()
                manifest.write_text(json.dumps(document), encoding="utf-8")
                with mock.patch("decoder.np.load", side_effect=AssertionError("array allocation attempted")):
                    with self.assertRaises((ValueError, OSError)):
                        PrototypeBook.load(archive, manifest)


class DecoderPipelineContracts(unittest.TestCase):
    @staticmethod
    def mask() -> np.ndarray:
        mask = np.zeros((32, 24), np.uint8)
        mask[8:24, 8:11] = 1
        mask[21:24, 8:18] = 1
        return mask

    @staticmethod
    def zoned_batch():
        config = controlled_config()
        mask = np.zeros((64, 40), np.uint8)
        mask[22:43, 10:13] = 1
        mask[40:43, 10:30] = 1
        mask[4:8, 18:22] = 1
        mask[55:59, 18:22] = 1
        segmentation = SegmentationPipeline(config).run(mask)
        batch = FeaturePipeline(config).run(segmentation)
        assert len(batch.features) == 1
        assert all(zone.metadata["foreground_pixels"] > 0 for zone in batch.features[0].zones.values())
        return config, batch

    @staticmethod
    def zone_models(config, feature, *, upper: str = "ි", lower: str = ""):
        return {f"zone:{name}": PrototypeCalibrationManager(config).fit((CalibrationSample(label, feature.zones[name]),))
                for name, label in (("core", "ක"), ("upper", upper), ("lower", lower))}

    def test_one_glyph_whole_model_decodes_with_identity(self) -> None:
        config = controlled_config()
        segmentation = SegmentationPipeline(config).run(self.mask())
        batch = FeaturePipeline(config).run(segmentation)
        feature = batch.features[0]
        samples = (CalibrationSample("ක", feature),
                   CalibrationSample("ග", replace(feature, vector=feature.vector+10.)))
        book = PrototypeCalibrationManager(config).fit(samples)
        decoded = DecoderPipeline(book, config=config).run(batch)
        self.assertEqual(decoded.text, "ක")
        self.assertEqual(decoded.words[0].word_id, 0)
        self.assertIn(feature.identity.candidate_id, decoded.words[0].paths[0].candidate_ids)

    def test_empty_segmentation_has_empty_decoding(self) -> None:
        book, config = controlled_book()
        segmentation = SegmentationPipeline(config).run(np.zeros((32, 24), np.uint8))
        batch = FeaturePipeline(config).run(segmentation)
        decoded = DecoderPipeline(book, config=config).run(batch)
        self.assertEqual(decoded.text, "")
        self.assertEqual(tuple(decoded.words), ())

    def test_baseline_overlapping_windows_are_alternatives_not_all_concatenated(self) -> None:
        config = get_config("B0", overrides={"features": {"micro_windows": 1}})
        mask = np.zeros((32, 100), np.uint8)
        mask[12:20, 3:97] = 1
        segmentation = SegmentationPipeline(config).run(mask)
        original = FeaturePipeline(config).run(segmentation)
        features = tuple(replace(feature, vector=np.zeros(1)) for feature in original.features)
        batch = replace(original, features=features,
                        by_candidate_id={f.identity.candidate_id: f for f in features})
        sample = features[0]
        book = PrototypeCalibrationManager(config).fit((
            CalibrationSample("ක", sample),
            CalibrationSample("ග", replace(sample, vector=np.full(1, 10.))),
        ))
        decoded = DecoderPipeline(book, config=config).run(batch)
        self.assertTrue(decoded.words)
        path = decoded.words[0].paths[0]
        self.assertLess(len(path.candidate_ids), len(batch.features))
        self.assertIsNone(decoded.words[0].confidence)
        selected = [batch.by_candidate_id[cid] for cid in path.candidate_ids]
        boxes = [feature.identity.normalized_bbox for feature in selected]
        self.assertEqual(boxes[0][0], 0)
        self.assertEqual(boxes[-1][2], segmentation.words[0].normalized_mask.shape[1])
        for left, right in zip(boxes, boxes[1:]):
            self.assertGreater(right[2], left[2])
            self.assertLessEqual(right[0], left[2])

    def test_intact_and_split_hypothesis_candidates_are_both_retained(self) -> None:
        config = controlled_config()
        segmentation = SegmentationPipeline(config).run(self.mask())
        original = FeaturePipeline(config).run(segmentation)
        prototype = original.features[0]
        width = prototype.identity.normalized_bbox[2]
        height = prototype.identity.normalized_bbox[3]
        candidates = []
        for cid, box in (("whole", (0, 0, width, height)),
                         ("left", (0, 0, width//2, height)),
                         ("right", (width//2, 0, width, height))):
            identity = replace(prototype.identity, candidate_id=cid, normalized_bbox=box)
            candidates.append(replace(prototype, identity=identity, vector=np.zeros(1)))
        hypotheses = (SegmentationHypothesis("intact", ("whole",), "intact", True),
                      SegmentationHypothesis("split", ("left", "right"), "split", True))
        batch = replace(original, features=tuple(candidates),
                        by_candidate_id={f.identity.candidate_id: f for f in candidates},
                        hypotheses={0: hypotheses},
                        metadata={**original.metadata, "candidate_count": 3})
        book = PrototypeCalibrationManager(config).fit((
            CalibrationSample("ක", candidates[0]),
            CalibrationSample("ග", replace(candidates[0], vector=np.full(1, 10.))),
        ))
        decoded = DecoderPipeline(book, config=config).run(batch)
        retained = {path.hypothesis_id for path in decoded.words[0].paths}
        self.assertIn("intact", retained)
        self.assertIn("split", retained)

    def test_multizone_uses_actual_per_view_books_and_combines_linked_marks(self) -> None:
        config, batch = self.zoned_batch()
        models = self.zone_models(config, batch.features[0])
        self.assertEqual(set(book.view for book in models.values()),
                         {"zone:core", "zone:upper", "zone:lower"})
        decoder = DecoderPipeline(models, config=config, options=DecoderOptions(recognition_mode="multi_zone"))
        decoded = decoder.run(batch)
        self.assertEqual(decoded.text, "කි")
        self.assertIn(batch.features[0].identity.candidate_id, decoded.words[0].paths[0].candidate_ids)

    def test_nonempty_upper_and_lower_ink_can_be_explicitly_calibrated_null_marks(self) -> None:
        config, batch = self.zoned_batch()
        models = self.zone_models(config, batch.features[0], upper="", lower="")
        decoded = DecoderPipeline(models, config=config,
                                  options=DecoderOptions(recognition_mode="multi_zone")).run(batch)
        self.assertEqual(decoded.text, "ක")
        self.assertEqual(models["zone:upper"].labels, ("",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"null-mark.npz"
            models["zone:upper"].save(path)
            restored = PrototypeBook.load(path)
            self.assertEqual(restored.labels, ("",))
            self.assertEqual(restored.view, "zone:upper")

    def test_null_labels_are_forbidden_for_whole_and_core_models(self) -> None:
        for view in ("whole_glyph", "zone:core"):
            with self.subTest(view=view), self.assertRaises((ValueError, TypeError)):
                PrototypeCalibrationManager(controlled_config()).fit((
                    CalibrationSample("", controlled_feature([0.], view=view)),
                ))

    def test_missing_nonempty_mark_zone_model_does_not_silently_emit_bare_base(self) -> None:
        config, batch = self.zoned_batch()
        models = self.zone_models(config, batch.features[0])
        del models["zone:upper"]
        try:
            decoded = DecoderPipeline(models, config=config,
                                      options=DecoderOptions(recognition_mode="multi_zone")).run(batch)
        except ValueError as error:
            self.assertIn("zone", str(error).lower())
        else:
            self.assertEqual(decoded.text, "")
            self.assertTrue(decoded.words)
            self.assertNotEqual(decoded.words[0].text, "ක")
            self.assertIn("zone", str(decoded.words[0].metadata).lower())

    def test_shared_mark_rejects_contaminated_split_but_keeps_intact_hypothesis(self) -> None:
        config, original = self.zoned_batch()
        prototype = original.features[0]
        models = self.zone_models(config, prototype)
        candidates = []
        for cid, box in (("whole", (0, 0, 40, 64)),
                         ("left", (0, 0, 20, 64)), ("right", (20, 0, 40, 64))):
            identity = replace(prototype.identity, candidate_id=cid, normalized_bbox=box,
                               associated_mark_ids=("shared-upper",))
            zones = {name: replace(zone, identity=replace(zone.identity, candidate_id=cid,
                                                         associated_mark_ids=("shared-upper",)))
                     for name, zone in prototype.zones.items()}
            candidates.append(replace(prototype, identity=identity, zones=zones))
        hypotheses = (SegmentationHypothesis("intact", ("whole",), "intact", True),
                      SegmentationHypothesis("split", ("left", "right"), "split", True))
        mark = DiacriticLink("shared-upper", "upper", BoundingBox(18, 4, 22, 8),
                             BoundingBox(18, 4, 22, 8), ("whole", "left", "right"), True)
        batch = replace(original, features=tuple(candidates),
                        by_candidate_id={feature.identity.candidate_id: feature for feature in candidates},
                        hypotheses={0: hypotheses}, diacritic_links={0: (mark,)},
                        metadata={**original.metadata, "candidate_count": 3})
        decoded = DecoderPipeline(models, config=config,
                                  options=DecoderOptions(recognition_mode="multi_zone")).run(batch)
        self.assertEqual(decoded.text, "කි")
        retained = {path.hypothesis_id for path in decoded.words[0].paths}
        self.assertIn("intact", retained)
        self.assertNotIn("split", retained)

    def test_whole_glyph_paths_separate_mean_max_confidence_and_selected_label_evidence(self) -> None:
        config = controlled_config()
        batch = FeaturePipeline(config).run(SegmentationPipeline(config).run(self.mask()))
        feature = batch.features[0]
        book = PrototypeCalibrationManager(config).fit((
            CalibrationSample("ක", feature),
            CalibrationSample("ග", replace(feature, vector=feature.vector+2.)),
        ))
        decoded = DecoderPipeline(book, config=config).run(batch)
        result = decoded.words[0].scores[f"{feature.identity.candidate_id}:whole_glyph"]
        probabilities = dict(zip(result.labels, result.probabilities))
        self.assertEqual({path.text for path in decoded.words[0].paths}, {"ක", "ග"})
        for path in decoded.words[0].paths:
            self.assertAlmostEqual(path.confidence, max(probabilities.values()))
            self.assertAlmostEqual(path.metadata["selected_label_confidence"], probabilities[path.text])

    def test_multizone_paths_separate_mean_max_confidence_and_selected_joint_evidence(self) -> None:
        config, batch = self.zoned_batch()
        feature = batch.features[0]
        models = self.zone_models(config, feature)
        for zone, labels in (("core", ("ක", "ග")), ("upper", ("ි", "ී"))):
            reference = feature.zones[zone]
            models[f"zone:{zone}"] = PrototypeCalibrationManager(config).fit((
                CalibrationSample(labels[0], reference),
                CalibrationSample(labels[1], replace(reference, vector=reference.vector+2.)),
            ))
        decoded = DecoderPipeline(models, config=config,
                                  options=DecoderOptions(recognition_mode="multi_zone")).run(batch)
        scores = decoded.words[0].scores
        probabilities = {zone: dict(zip(scores[f"{feature.identity.candidate_id}:zone:{zone}"].labels,
                                          scores[f"{feature.identity.candidate_id}:zone:{zone}"].probabilities))
                         for zone in ("core", "upper", "lower")}
        self.assertEqual(len(decoded.words[0].paths), 3)
        for path in decoded.words[0].paths:
            expected = probabilities["core"][path.text[0]] * probabilities["upper"][path.text[1]]
            expected *= probabilities["lower"][""]
            maximum = max(probabilities["core"].values()) * max(probabilities["upper"].values())
            maximum *= max(probabilities["lower"].values())
            self.assertAlmostEqual(path.confidence, maximum)
            self.assertAlmostEqual(path.metadata["selected_label_confidence"], expected)

    def test_explicit_null_zone_marks_allow_visual_kombuwa_before_its_base(self) -> None:
        config, original = self.zoned_batch()
        prototype = original.features[0]
        candidates = []
        for cid, box, value in (("prebase", (0, 0, 20, 64), 0.),
                                ("base", (20, 0, 40, 64), 2.)):
            identity = replace(prototype.identity, candidate_id=cid, normalized_bbox=box,
                               associated_mark_ids=())
            zones = {name: replace(zone, identity=replace(zone.identity, candidate_id=cid,
                                                         associated_mark_ids=()),
                                   vector=np.array([value]) if name == "core" else zone.vector)
                     for name, zone in prototype.zones.items()}
            candidates.append(replace(prototype, identity=identity, zones=zones))
        hypotheses = (SegmentationHypothesis("components", ("prebase", "base"), "components", False),)
        batch = replace(original, features=tuple(candidates),
                        by_candidate_id={feature.identity.candidate_id: feature for feature in candidates},
                        hypotheses={0: hypotheses}, diacritic_links={0: ()},
                        metadata={**original.metadata, "candidate_count": 2})
        models = self.zone_models(config, prototype, upper="", lower="")
        models["zone:core"] = PrototypeCalibrationManager(config).fit((
            CalibrationSample("ෙ", candidates[0].zones["core"]),
            CalibrationSample("ක", candidates[1].zones["core"]),
        ))
        decoded = DecoderPipeline(models, config=config,
                                  options=DecoderOptions(recognition_mode="multi_zone")).run(batch)
        self.assertEqual(decoded.text, "කෙ")
        self.assertEqual(decoded.words[0].paths[0].candidate_ids, ("prebase", "base"))

    def test_pruned_best_unicode_label_keeps_specified_max_confidence_but_exposes_selected_evidence(self) -> None:
        config = controlled_config()
        batch = FeaturePipeline(config).run(SegmentationPipeline(config).run(self.mask()))
        reference = batch.features[0]
        book = PrototypeCalibrationManager(config).fit((
            CalibrationSample("ි", reference),
            CalibrationSample("ක", replace(reference, vector=reference.vector+2.)),
            CalibrationSample("ග", replace(reference, vector=reference.vector+2.2)),
        ))
        decoded = DecoderPipeline(book, lexicon=("ග",), config=config).run(batch)
        word = decoded.words[0]
        self.assertEqual(word.raw_text, "ක")
        self.assertGreaterEqual(word.paths[0].confidence, config.decoder.tau_oov)
        self.assertLess(word.paths[0].metadata["selected_label_confidence"], config.decoder.tau_oov)
        self.assertEqual(word.text, "ක")
        self.assertFalse(word.lexicon_decision.corrected)
        self.assertTrue(word.lexicon_decision.bypassed)


class DecoderUnicodeContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.assembler = SinhalaAssembler()

    def test_all_declared_consonant_vowel_pairs_match_configuration(self) -> None:
        for consonant in CONSONANTS.values():
            for vowel, sign in VOWEL_SIGNS.items():
                with self.subTest(consonant=consonant, vowel=vowel):
                    actual = self.assembler.compose(consonant, core_marks=(sign,) if sign else ())
                    self.assertEqual(actual, compose_consonant_vowel(consonant, vowel))

    def test_composite_sign_components_recombine_across_zones(self) -> None:
        for sign, components in COMPOSITE_VOWEL_COMPONENTS.items():
            with self.subTest(sign=sign):
                actual = self.assembler.compose("ක", core_marks=components[:1],
                                                upper=components[1:])
                self.assertEqual(actual, unicodedata.normalize("NFC", "ක" + sign))

    def test_joiners_and_configured_conjuncts_remain_explicit(self) -> None:
        for joiner in ("\u200c", "\u200d"):
            self.assertEqual(self.assembler.compose("ක්" + joiner), "ක්" + joiner)
            self.assertEqual(self.assembler.compose("ක්" + joiner + "ය"), "ක්" + joiner + "ය")
        for prefix in CONJUNCT_PREFIXES.values():
            self.assertEqual(self.assembler.compose("ක", prefix=prefix), prefix + "ක")
        for suffix in CONJUNCT_SUFFIXES.values():
            self.assertEqual(self.assembler.compose("ක", suffix=suffix), "ක" + suffix)

    def test_independent_vowels_reject_dependent_signs(self) -> None:
        for vowel in INDEPENDENT_VOWELS.values():
            self.assertEqual(self.assembler.compose(vowel), vowel)
            with self.assertRaises(UnicodeAssemblyError):
                self.assembler.compose(vowel, upper=("ි",))

    def test_visual_prebase_order_becomes_logical_unicode_order(self) -> None:
        self.assertEqual(self.assembler.assemble_visual(("ෙ", "ක", "ා", "ෙ", "ග")), "කොගෙ")
        self.assertEqual(self.assembler.assemble_visual(("ෙ", "ෙ", "ක")), "කෛ")
        self.assertEqual(self.assembler.assemble_visual(("කෙ", "ග")), "කෙග")
        self.assertEqual(self.assembler.assemble_visual(()), "")

    def test_incompatible_or_orphan_marks_are_never_silently_discarded(self) -> None:
        for kwargs in ({"upper": ("ි", "ී")}, {"lower": ("ු", "ූ")},
                       {"core_marks": ("ෙ", "ෙ")}, {"upper": ("\u200d",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(UnicodeAssemblyError):
                self.assembler.compose("ක", **kwargs)
        for tokens in (("ෙ",), ("ි", "ක"), ("ka",), ("1",)):
            with self.subTest(tokens=tokens), self.assertRaises(UnicodeAssemblyError):
                self.assembler.assemble_visual(tokens)


class DecoderLexiconContracts(unittest.TestCase):
    @staticmethod
    def path(text: str, probability: float, confidence: float | None = None) -> WordPath:
        return WordPath(text, math.log(probability), confidence=confidence)

    def test_high_confidence_oov_is_preserved_inclusively(self) -> None:
        paths = (self.path("කම", .6, .85), self.path("ගම", .4))
        decision = LexiconGate(("ගම",)).apply(paths)
        self.assertEqual(decision.text, "කම")
        self.assertFalse(decision.corrected)
        self.assertTrue(decision.bypassed)

    def test_only_scored_model_paths_can_be_corrections(self) -> None:
        raw = self.path("කම", .6, .2)
        missing = LexiconGate(("ගම",)).apply((raw,))
        self.assertEqual(missing.text, "කම")
        decision = LexiconGate(("ගම",)).apply((raw, self.path("ගම", .4)))
        self.assertEqual(decision.text, "ගම")
        self.assertEqual(decision.edit_distance, 1)

    def test_edit_radius_and_top_three_bounds(self) -> None:
        paths = (self.path("කම", .4, .2), self.path("ගය", .3),
                 self.path("තර", .2), self.path("ගම", .1))
        decision = LexiconGate(("ගය", "තර", "ගම")).apply(paths)
        self.assertEqual(decision.text, "කම")
        for radius in (-1, 2):
            with self.assertRaises((ValueError, TypeError)):
                LexiconGate((), d_edit_max=radius)

    def test_empty_missing_and_disabled_lexicons_preserve_raw(self) -> None:
        paths = (self.path("කම", .6, .2), self.path("ගම", .4))
        for lexicon, enabled in ((None, True), ((), True), (("ගම",), False)):
            with self.subTest(lexicon=lexicon, enabled=enabled):
                self.assertEqual(LexiconGate(lexicon, enabled=enabled).apply(paths).text, "කම")

    def test_codepoint_distance_uses_nfc_and_preserves_joiners(self) -> None:
        self.assertEqual(levenshtein_distance("කො", "කො"), 0)
        self.assertEqual(levenshtein_distance("ක්\u200dය", "ක්ය"), 1)
        self.assertEqual(levenshtein_distance("කම", "ගය"), 2)
        self.assertEqual(levenshtein_distance("", "ක"), 1)


if __name__ == "__main__":
    unittest.main()
