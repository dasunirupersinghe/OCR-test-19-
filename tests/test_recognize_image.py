"""Real synthetic image inference smoke tests; these do not measure OCR accuracy."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from decoder import (CalibrationSample, DecoderError, DevelopmentSetCalibrationManager,
                     ModelCompatibilityError)
from preprocessing import PreprocessingPipeline
from segmentation import SegmentationPipeline
import recognize_image as cli


class RecognitionCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "glyph.png"
        array = np.full((96, 96), 255, np.uint8)
        cv2.ellipse(array, (48, 48), (18, 24), 0, 0, 360, 0, 3)
        self.assertTrue(cv2.imwrite(str(self.image), array))
        self.manager = DevelopmentSetCalibrationManager(mode="B4")
        mask, _ = PreprocessingPipeline(self.manager.config, mode="B4").run(self.image)
        segmentation = SegmentationPipeline(self.manager.config, mode="B4").run(mask)
        feature = self.manager.feature_pipeline.run(segmentation).features[0]
        other = replace(feature, vector=feature.vector+5.)
        self.book = self.manager.fit((CalibrationSample("ක", feature), CalibrationSample("ක", feature),
                                     CalibrationSample("ග", other), CalibrationSample("ග", other)))
        self.model = self.root / "prototypes.npz"
        self.book.save(self.model)
        self.output = self.root / "recognition.json"

    def args(self, *extra: str):
        return cli.build_parser().parse_args(["--image", str(self.image), "--model", str(self.model),
                                              "--output", str(self.output), *extra])

    def test_real_image_pipeline_records_actual_text_paths_and_provenance(self) -> None:
        report = cli.recognize(self.args())
        self.assertEqual(report["text"], "ක")
        self.assertEqual(report["raw_text"], "ක")
        self.assertEqual(report["word_count"], 1)
        word = report["words"][0]
        self.assertEqual(word["status"], "decoded")
        self.assertTrue(word["paths"][0]["candidate_ids"])
        self.assertGreater(word["class_relative_confidence"], .85)
        self.assertIn("not calibrated correctness", report["confidence_interpretation"])
        self.assertIsNone(report["lexicon"])
        self.assertFalse(report["context_statistics_supplied"])
        self.assertEqual(report["model"]["dimension"], 46)
        self.assertEqual(report["model"]["class_count"], 2)
        self.assertEqual(report["image"]["sha256"], hashlib.sha256(self.image.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        self.output.write_text("preserve me")
        with self.assertRaisesRegex(DecoderError, "output exists"):
            cli.recognize(self.args())
        self.assertEqual(self.output.read_text(), "preserve me")
        report = cli.recognize(self.args("--overwrite"))
        self.assertEqual(report["text"], "ක")

    def test_debug_images_are_real_pngs_and_collision_is_not_overwritten(self) -> None:
        directory = self.root / "debug"
        report = cli.recognize(self.args("--debug-dir", str(directory)))
        self.assertIn("binary_mask", report["debug_images"])
        self.assertIn("word_0000_candidate_boxes", report["debug_images"])
        for path in report["debug_images"].values():
            self.assertIsNotNone(cv2.imread(path))
        self.output.unlink()
        with self.assertRaisesRegex(DecoderError, "debug images already exist"):
            cli.recognize(self.args("--debug-dir", str(directory)))

    def test_empty_image_is_reported_for_review_without_fabricated_text(self) -> None:
        cv2.imwrite(str(self.image), np.full((96, 96), 255, np.uint8))
        report = cli.recognize(self.args())
        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["text"], "")
        self.assertEqual(report["words"], [])
        self.assertTrue(report["needs_review"])

    def test_corrupt_input_produces_clear_failure_and_no_json(self) -> None:
        self.image.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            cli.recognize(self.args())
        self.assertFalse(self.output.exists())

    def test_checksum_mismatch_fails_before_recognition(self) -> None:
        self.model.write_bytes(self.model.read_bytes()+b"changed")
        with self.assertRaisesRegex(DecoderError, "SHA256"):
            cli.recognize(self.args())
        self.assertFalse(self.output.exists())

    def test_zone_only_book_is_rejected_as_incomplete_model_mapping(self) -> None:
        zone = replace(self.book, view="zone:core")
        zone.save(self.model, overwrite=True)
        with self.assertRaisesRegex(ModelCompatibilityError, "whole_glyph"):
            cli.recognize(self.args())

    def test_optional_lexicon_is_explicit_bounded_and_recorded(self) -> None:
        path = self.root / "lexicon.txt"
        path.write_text("ක\nග\nක\n", encoding="utf-8")
        report = cli.recognize(self.args("--lexicon-text", str(path)))
        self.assertEqual(report["lexicon"]["word_count"], 2)
        self.assertEqual(report["lexicon"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        path.write_text("two words\n")
        with self.assertRaisesRegex(DecoderError, "one word"):
            cli.load_lexicon(path)

    def test_main_missing_model_returns_error_without_creating_placeholder(self) -> None:
        self.model.unlink()
        result = cli.main(["--image", str(self.image), "--model", str(self.model), "--output", str(self.output)])
        self.assertEqual(result, 1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
