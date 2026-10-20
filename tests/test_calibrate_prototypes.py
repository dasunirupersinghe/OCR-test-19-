"""CLI safety/split tests and a real image-to-reloaded-prototype smoke test."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import calibrate_prototypes as cli
from decoder import CalibrationError, PrototypeBook


class CalibrationCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "development"
        self.mapping = {"001": "ක", "002": "ග"}
        for class_index, folder in enumerate(self.mapping):
            directory = self.data / folder
            directory.mkdir(parents=True)
            for sample in range(5):
                image = np.full((64, 64), 255, np.uint8)
                if class_index == 0:
                    cv2.ellipse(image, (31, 32), (14+sample, 17), 0, 0, 360, 0, 2)
                else:
                    cv2.line(image, (16+sample, 12), (45, 48), 0, 3)
                    cv2.line(image, (45, 12), (17+sample, 48), 0, 3)
                self.assertTrue(cv2.imwrite(str(directory / f"sample_{sample}.png"), image))
        self.map_path = self.root / "labels.json"
        self.map_path.write_text(json.dumps(self.mapping, ensure_ascii=False), encoding="utf-8")
        self.output = self.root / "artifacts" / "prototypes.npz"

    def args(self, *extra: str):
        return cli.build_parser().parse_args(["--data-dir", str(self.data), "--label-map", str(self.map_path),
                                              "--output", str(self.output), "--mode", "B0", *extra])

    def test_split_is_reproducible_disjoint_and_cap_is_calibration_only(self) -> None:
        first = cli.select_records(self.data, self.mapping, seed=9, max_samples_per_class=2)
        second = cli.select_records(self.data, self.mapping, seed=9, max_samples_per_class=2)
        self.assertEqual(first, second)
        fit, holdout, duplicates = first
        self.assertEqual(len(fit), 4)
        self.assertEqual(len(holdout), 2)
        self.assertFalse(duplicates)
        self.assertFalse({record.sha256 for record in fit} & {record.sha256 for record in holdout})

    def test_duplicate_files_never_leak_into_holdout(self) -> None:
        source = self.data / "001" / "sample_0.png"
        (source.parent / "duplicate.png").write_bytes(source.read_bytes())
        fit, holdout, duplicates = cli.select_records(self.data, self.mapping)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(fit), 8)
        self.assertFalse({record.sha256 for record in fit} & {record.sha256 for record in holdout})

    def test_duplicate_conflicting_labels_are_rejected(self) -> None:
        (self.data / "002" / "copy.png").write_bytes((self.data / "001" / "sample_0.png").read_bytes())
        with self.assertRaisesRegex(CalibrationError, "conflicting labels"):
            cli.select_records(self.data, self.mapping)

    def test_exact_folder_map_and_supported_files_are_required(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "match class folders"):
            cli.select_records(self.data, {"001": "ක"})
        (self.data / "001" / "unrecognized.json").write_text("{}")
        with self.assertRaisesRegex(CalibrationError, "unsupported file"):
            cli.select_records(self.data, self.mapping)

    def test_json_duplicate_keys_and_path_escape_are_rejected(self) -> None:
        self.map_path.write_text('{"001":"ක", "001":"ග"}', encoding="utf-8")
        with self.assertRaisesRegex(CalibrationError, "duplicate JSON key"):
            cli.load_label_map(self.map_path)
        self.map_path.write_text('{"../escape":"ක"}', encoding="utf-8")
        with self.assertRaisesRegex(CalibrationError, "single folder"):
            cli.load_label_map(self.map_path)

    def test_actual_images_fit_save_reload_and_report_heldout_glyphs(self) -> None:
        report = cli.calibrate(self.args("--max-samples-per-class", "2"))
        self.assertEqual(report["feature_dimension"], 46)
        self.assertEqual(report["calibration_sample_count"], 4)
        self.assertEqual(report["heldout_sample_count"], 2)
        self.assertEqual(len(report["heldout_predictions"]), 2)
        self.assertTrue(report["load_roundtrip_verified"])
        self.assertLess(report["archive_bytes"], 2_000_000)
        loaded = PrototypeBook.load(self.output)
        self.assertEqual(loaded.labels, ("ක", "ග"))
        np.testing.assert_array_equal(loaded.counts, [2, 2])
        self.assertTrue(loaded.covariance_available.all())
        saved = json.loads((self.output.parent / "calibration_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        fit_hashes = {item["sha256"] for item in report["calibration_samples"]}
        self.assertFalse(fit_hashes & {item["sha256"] for item in report["heldout_predictions"]})
        with self.assertRaisesRegex(CalibrationError, "refusing to replace"):
            cli.calibrate(self.args())

    def test_bad_image_fails_without_model_artifact(self) -> None:
        for path in (self.data / "001").iterdir():
            path.write_bytes(f"not an image: {path.name}".encode())
        with self.assertRaisesRegex(CalibrationError, "feature extraction failed"):
            cli.calibrate(self.args())
        self.assertFalse(self.output.exists())

    def test_invalid_label_fails_without_model_artifact(self) -> None:
        self.mapping["001"] = "100"
        self.map_path.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(CalibrationError, "actual Sinhala"):
            cli.calibrate(self.args())
        self.assertFalse(self.output.exists())

    def test_main_reports_failure_and_does_not_fake_dataset(self) -> None:
        status = cli.main(["--data-dir", str(self.root / "missing"), "--label-map", str(self.map_path)])
        self.assertEqual(status, 1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
