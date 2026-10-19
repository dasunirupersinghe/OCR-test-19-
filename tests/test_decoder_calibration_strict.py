"""Independent tests of the strict Master-Spec decoder calibration profile.

These checks use small synthetic development samples, not OCR accuracy claims.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from config import get_config
import decoder
from decoder import (CalibrationError, CalibrationSample, DecoderError,
                     DecoderPipeline, DevelopmentFolderSpec,
                     DevelopmentSetCalibrationManager, ModelCompatibilityError,
                     PrototypeBook, PrototypeBundle, PrototypeScorer,
                     make_master_feature_pipeline)
from features import FeaturePipeline
from preprocessing import PreprocessingPipeline
from segmentation import SegmentationPipeline


ARCHIVE_LIMIT = 2_000_000


def glyph_image() -> np.ndarray:
    image = np.full((96, 96), 255, np.uint8)
    cv2.ellipse(image, (48, 48), (15, 15), 0, 0, 360, 0, 3)
    cv2.circle(image, (46, 12), 4, 0, -1)
    cv2.circle(image, (47, 80), 4, 0, -1)
    return image


def strict_feature(vector: np.ndarray | None = None, *, view: str = "whole_glyph"):
    pipeline = make_master_feature_pipeline()
    feature = pipeline.extract((glyph_image() == 0).astype(np.uint8), view=view)
    if vector is None:
        return feature
    values = np.asarray(vector, dtype=np.float64)
    values.setflags(write=False)
    return replace(feature, vector=values)


def strict_book(*, epsilon: float = .025):
    generator = np.random.default_rng(190)
    rows = generator.normal(size=(4, 46))
    rows[2:] += np.linspace(0, 3, 46)
    samples = tuple(CalibrationSample(label, strict_feature(vector))
                    for label, vector in zip(("ක", "ක", "ග", "ග"), rows))
    manager = DevelopmentSetCalibrationManager(cov_reg_eps=epsilon)
    return manager.fit(samples), manager, rows


class StrictFeatureProfileTests(unittest.TestCase):
    def test_master_profile_is_real_46_dimensional_v1_features(self) -> None:
        pipeline = make_master_feature_pipeline()
        self.assertIsInstance(pipeline, FeaturePipeline)
        self.assertEqual(pipeline.schema.dimension, 46)
        self.assertEqual(pipeline.schema.version, "v1")
        self.assertEqual(pipeline.settings.micro_windows, 4)
        self.assertEqual(pipeline.settings.max_hole_centroids, 9)
        self.assertFalse(pipeline.settings.include_occupancy)
        self.assertTrue(pipeline.flags.slope_features)
        self.assertTrue(pipeline.flags.topology_features)
        feature = strict_feature()
        self.assertEqual(feature.vector.shape, (46,))
        self.assertTrue(np.isfinite(feature.vector).all())
        self.assertEqual(feature.schema.slope_dimension, 16)
        self.assertEqual(feature.schema.topology_dimension, 30)

    def test_strict_manager_exposes_profile_and_rejects_36_dimensional_input(self) -> None:
        manager = DevelopmentSetCalibrationManager()
        self.assertEqual(manager.feature_pipeline.schema.dimension, 46)
        old_feature = FeaturePipeline().extract(np.eye(16, dtype=np.uint8))
        self.assertEqual(old_feature.schema.dimension, 36)
        with self.assertRaises((CalibrationError, ModelCompatibilityError)):
            manager.fit((CalibrationSample("ක", old_feature),))

    def test_explicit_wrong_semantics_are_rejected_even_if_dimension_is_46(self) -> None:
        config = get_config("B4", overrides={"features": {
            "micro_windows": 2, "include_occupancy": True, "max_hole_centroids": 13}})
        self.assertEqual(FeaturePipeline(config).schema.dimension, 46)
        with self.assertRaises(ModelCompatibilityError):
            make_master_feature_pipeline(config)
        feature = FeaturePipeline(config).extract(np.eye(16, dtype=np.uint8))
        with self.assertRaises(ModelCompatibilityError):
            DevelopmentSetCalibrationManager().fit((CalibrationSample("ක", feature),))

    def test_no_config_master_profile_keeps_46_features_across_distance_ablations(self) -> None:
        for mode in ("B0", "B1", "B2", "B3", "B4"):
            with self.subTest(mode=mode):
                pipeline = make_master_feature_pipeline(mode=mode)
                self.assertEqual(pipeline.mode, mode)
                self.assertEqual(pipeline.schema.dimension, 46)


class StrictCovarianceTests(unittest.TestCase):
    def test_two_examples_fit_full_regularized_sample_covariance_without_shrinkage(self) -> None:
        epsilon = .025
        book, _, raw = strict_book(epsilon=epsilon)
        np.testing.assert_array_equal(book.counts, (2, 2))
        self.assertTrue(book.covariance_available.all())
        for index, (start, stop) in enumerate(((0, 2), (2, 4))):
            standardized = (raw[start:stop] - book.center) / book.scale
            expected = np.cov(standardized, rowvar=False, ddof=1) + epsilon*np.eye(46)
            np.testing.assert_allclose(book.covariances[index], expected, atol=1e-12, rtol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(book.covariances[index]).min(), 0)
        np.testing.assert_allclose(book.raw_means, np.stack((raw[:2].mean(0), raw[2:].mean(0))))
        self.assertFalse(book.raw_means.flags.writeable)

    def test_single_example_is_explicit_euclidean_fallback(self) -> None:
        manager = DevelopmentSetCalibrationManager(cov_reg_eps=.1)
        book = manager.fit((CalibrationSample("ක", strict_feature(np.arange(46.))),))
        self.assertFalse(book.covariance_available[0])
        result = PrototypeScorer(book, manager.config).score(strict_feature(np.arange(46.)+1))
        np.testing.assert_allclose(result.distances, (np.sqrt(46),))
        self.assertEqual(result.metadata["class_distance_metrics"], ("standardized_euclidean",))

    def test_b0_forces_standardized_euclidean_even_with_mahalanobis_override(self) -> None:
        book, _, _ = strict_book()
        config = get_config("B0", overrides={"decoder": {"distance_metric": "regularized_mahalanobis"}})
        query = strict_feature(np.linspace(-1, 4, 46))
        result = PrototypeScorer(book, config).score(query)
        expected = np.linalg.norm((query.vector-book.center)/book.scale-book.means, axis=1)
        np.testing.assert_allclose(result.distances, expected, atol=1e-11)
        self.assertTrue(all(metric == "standardized_euclidean"
                            for metric in result.metadata["class_distance_metrics"]))
        self.assertIsNone(result.probabilities)

    def test_strict_calibration_still_rejects_test_split_and_mismatched_view(self) -> None:
        manager = DevelopmentSetCalibrationManager()
        with self.assertRaises(CalibrationError):
            manager.fit((CalibrationSample("ක", strict_feature(), "test"),))
        with self.assertRaises(ModelCompatibilityError):
            manager.fit((CalibrationSample("ක", strict_feature()),
                         CalibrationSample("ග", strict_feature(view="zone:core"))))


class StrictFolderCalibrationTests(unittest.TestCase):
    def test_real_png_folders_calibrate_whole_glyph_and_each_zone_without_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "class-1"
            folder.mkdir()
            path = folder / "example.png"
            self.assertTrue(cv2.imwrite(str(path), glyph_image()))
            manager = DevelopmentSetCalibrationManager()
            whole = manager.calibrate_from_folders(root, {"class-1": "ක"})
            self.assertEqual(whole.dimension, 46)
            self.assertEqual(whole.view, "whole_glyph")
            np.testing.assert_array_equal(whole.counts, (1,))
            mask, _ = PreprocessingPipeline(manager.config, mode=manager.mode).run(path)
            segmentation = SegmentationPipeline(manager.config, mode=manager.mode).run(mask)
            self.assertEqual(len(segmentation.words), 1)
            for zone, label in (("upper", "ි"), ("core", "ක"), ("lower", "ු")):
                with self.subTest(zone=zone):
                    book = manager.calibrate_from_folders(root, {"class-1": label}, view=f"zone:{zone}")
                    region = segmentation.words[0].zones[zone]
                    reference = manager.feature_pipeline.extract(region.image, view=f"zone:{zone}")
                    self.assertEqual(book.view, f"zone:{zone}")
                    self.assertEqual(book.dimension, 46)
                    self.assertEqual(book.model_view_key, reference.metadata["model_view_key"])
                    np.testing.assert_allclose(book.raw_means[0], reference.vector, atol=1e-12)

    def test_explicit_feature_pipeline_matches_query_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "c"
            folder.mkdir()
            cv2.imwrite(str(folder / "one.png"), glyph_image())
            manager = DevelopmentSetCalibrationManager()
            pipeline = make_master_feature_pipeline()
            book = manager.calibrate_from_folders(root, {"c": "ක"}, pipeline)
            self.assertEqual(book.schema_id, pipeline.schema.schema_id)

    def test_blank_unreadable_and_invalid_label_images_fail_explicitly(self) -> None:
        for case in ("blank", "unreadable", "invalid-label"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                folder = root / "c"
                folder.mkdir()
                path = folder / "one.png"
                if case == "unreadable":
                    path.write_bytes(b"this is not an image")
                else:
                    image = np.full((64, 64), 255, np.uint8) if case == "blank" else glyph_image()
                    cv2.imwrite(str(path), image)
                manager = DevelopmentSetCalibrationManager()
                with self.assertRaises((ValueError, OSError)):
                    manager.calibrate_from_folders(root, {"c": "001" if case == "invalid-label" else "ක"})

    def test_folder_map_cannot_silently_ignore_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("a", "b"):
                (root / folder).mkdir()
                cv2.imwrite(str(root / folder / "one.png"), glyph_image())
            with self.assertRaises(CalibrationError):
                DevelopmentSetCalibrationManager().calibrate_from_folders(root, {"a": "ක"})


class StrictMultiViewIntegrationTests(unittest.TestCase):
    def test_folder_bundle_roundtrip_decodes_native_zone_geometry(self) -> None:
        epsilon = .02
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "character"
            folder.mkdir()
            for index in range(2):
                self.assertTrue(cv2.imwrite(str(folder/f"{index}.png"), glyph_image()))
            manager = DevelopmentSetCalibrationManager(cov_reg_eps=epsilon)
            folder_specs = {view: DevelopmentFolderSpec(root, {"character": label})
                            for view, label in (("zone:core", "ක"), ("zone:upper", "ි"),
                                                ("zone:lower", ""))}
            bundle = manager.calibrate_bundle_from_folders(folder_specs)
            self.assertEqual(set(bundle), set(folder_specs))
            for view, book in bundle.items():
                with self.subTest(view=view):
                    self.assertEqual(book.dimension, 46)
                    np.testing.assert_array_equal(book.counts, (2,))
                    np.testing.assert_array_equal(book.covariance_available, (True,))
                    np.testing.assert_allclose(book.covariances[0], epsilon*np.eye(46), atol=1e-12)
            archive = root/"prototypes.npz"
            bundle.save(archive)
            self.assertLess(archive.stat().st_size, ARCHIVE_LIMIT)
            restored = PrototypeBundle.load(archive)
            self.assertEqual(set(restored.modifier_signatures), {"zone:upper", "zone:lower"})
            for view, label in (("zone:upper", "ි"), ("zone:lower", "")):
                signature = restored.modifier_signatures[view][label]
                self.assertEqual(signature.shape, (46,))
                self.assertFalse(signature.flags.writeable)
                np.testing.assert_array_equal(signature, bundle[view].raw_means[0])
            mask, _ = PreprocessingPipeline(manager.config, mode=manager.mode).run(folder/"0.png")
            segmentation = SegmentationPipeline(manager.config, mode=manager.mode).run(mask)
            features = manager.feature_pipeline.run(segmentation)
            result = DecoderPipeline(restored, config=manager.config).run(features)
            self.assertEqual(result.text, "කි")
            self.assertEqual(result.words[0].status, "decoded")
            self.assertEqual(result.words[0].metadata["recognition_mode"], "multi_zone")
            bindings = result.words[0].metadata["modifier_bindings_by_candidate_path"][0]["bindings"]
            self.assertEqual({binding["zone"] for binding in bindings}, {"upper", "lower"})
            self.assertEqual(len({binding["parent_candidate_id"] for binding in bindings}), 1)
            self.assertTrue(all(binding["overlap_pixels"] > 0 for binding in bindings))

    def test_fit_views_calibrates_explicit_null_and_modifier_signatures(self) -> None:
        manager = DevelopmentSetCalibrationManager(cov_reg_eps=.03)
        sample_map = {}
        vectors = {}
        for index, (view, label) in enumerate((("zone:core", "ක"), ("zone:upper", "ි"),
                                             ("zone:lower", ""))):
            vector = np.arange(46.)+index*10
            vectors[view] = vector
            feature = strict_feature(vector, view=view)
            sample_map[view] = (CalibrationSample(label, feature), CalibrationSample(label, feature))
        bundle = manager.fit_views(sample_map)
        self.assertIsInstance(bundle, PrototypeBundle)
        for view, book in bundle.items():
            self.assertEqual(book.view, view)
            self.assertEqual(book.dimension, 46)
            np.testing.assert_array_equal(book.raw_means[0], vectors[view])
        self.assertEqual(bundle["zone:lower"].labels, ("",))
        self.assertEqual(bundle["zone:upper"].labels, ("ි",))
        self.assertEqual(set(bundle.modifier_signatures), {"zone:upper", "zone:lower"})

    def test_fit_views_rejects_missing_samples_and_disagreeing_view_keys(self) -> None:
        manager = DevelopmentSetCalibrationManager()
        core = CalibrationSample("ක", strict_feature(view="zone:core"))
        upper = CalibrationSample("ි", strict_feature(view="zone:upper"))
        cases = ({}, {"zone:core": ()}, {"zone:upper": (core,)},
                 {"zone:core": (core, upper)})
        for sample_map in cases:
            with self.subTest(keys=tuple(sample_map)), self.assertRaises((CalibrationError, ModelCompatibilityError)):
                manager.fit_views(sample_map)


class StrictArchiveQuotaTests(unittest.TestCase):
    def test_real_small_archive_roundtrips_under_decimal_two_megabytes(self) -> None:
        book, _, _ = strict_book()
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "model.npz"
            book.save(archive)
            self.assertLess(archive.stat().st_size, ARCHIVE_LIMIT)
            restored = PrototypeBook.load(archive)
            for attribute in ("means", "covariances", "raw_means"):
                np.testing.assert_array_equal(getattr(restored, attribute), getattr(book, attribute))
            self.assertEqual(restored.schema_id, book.schema_id)

    def test_save_rejects_exact_limit_before_writing_either_model_file(self) -> None:
        book, _, _ = strict_book()
        def oversized_compression(buffer, **arrays):
            buffer.write(b"x" * ARCHIVE_LIMIT)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "model.npz"
            with mock.patch.object(decoder.np, "savez_compressed", side_effect=oversized_compression):
                with self.assertRaises(DecoderError):
                    book.save(archive)
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".json").exists())

    def test_load_rejects_archive_limit_even_when_expanded_budget_is_larger(self) -> None:
        book, _, _ = strict_book()
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "model.npz"
            _, manifest_path = book.save(archive)
            payload = archive.read_bytes()
            payload += bytes(ARCHIVE_LIMIT-len(payload))
            archive.write_bytes(payload)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["archive_sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(decoder.np, "load", side_effect=AssertionError("limit checked too late")):
                with self.assertRaises(DecoderError):
                    PrototypeBook.load(archive, max_model_bytes=32_000_000)

    def test_expanded_budget_remains_separate_from_compressed_archive_quota(self) -> None:
        book, _, _ = strict_book()
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "model.npz"
            book.save(archive)
            declared_arrays = sum(getattr(book, name).nbytes for name in
                                  ("means", "center", "scale", "covariances", "covariance_available", "counts"))
            self.assertLess(archive.stat().st_size, ARCHIVE_LIMIT)
            with self.assertRaises(DecoderError):
                PrototypeBook.load(archive, max_model_bytes=declared_arrays-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
