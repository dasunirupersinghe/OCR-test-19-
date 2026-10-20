"""Dataset preparation verifies source labels and never overwrites stale data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import prepare_reference_data as reference


def mapping_scripts() -> tuple[str, str]:
    prefixes = list(reference.EXPECTED_MAPPING) + list(reference.EXCLUDED_CLASSES)
    feature = "% letters = ['bogus'];\nletters = [" + ";".join(f"'{p:4}'" for p in prefixes) + "];"
    blocks: list[str] = []
    for index, prefix in enumerate(prefixes, 1):
        condition = "if" if index == 1 else "elseif"
        if prefix in reference.EXPECTED_MAPPING:
            value = ord(reference.EXPECTED_MAPPING[prefix])
            expression = f"fwrite(fid, unicode2native(char(hex2dec('{value:04X}')),'UTF-8'));"
        else:
            expression = f"fprintf(fid,'{reference.EXCLUDED_CLASSES[prefix]}');"
        blocks.append(f"{condition} (I=={index})\n{expression}\nfclose(fid);")
    return feature, "\n".join(blocks) + "\nelse\ndisp('not found');\nend"


class ReferencePreparationTests(unittest.TestCase):
    def test_source_mapping_ignores_comments_and_uses_explicit_indices(self) -> None:
        self.assertEqual(reference.derive_label_mapping(*mapping_scripts()), reference.EXPECTED_MAPPING)
        self.assertEqual(reference.EXPECTED_MAPPING["a"], "අ")
        self.assertEqual(reference.EXPECTED_MAPPING["af"], "ඝ")
        self.assertEqual(reference.EXCLUDED_CLASSES, {"o": "O", "num3": "3", "num4": "4", "num5": "5"})

    def test_changed_or_duplicate_source_labels_are_rejected(self) -> None:
        feature, output = mapping_scripts()
        with self.assertRaisesRegex(reference.ReferenceDataError, "verified Unicode"):
            reference.derive_label_mapping(feature, output.replace("0D85", "0D86"))
        with self.assertRaisesRegex(reference.ReferenceDataError, "exactly once"):
            reference.derive_label_mapping(feature, output.replace("I==2)", "I==1)"))

    def test_missing_checkout_and_stale_revision_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(reference.ReferenceDataError, "does not exist"):
                reference.obtain_checkout(Path(temporary) / "missing", Path(temporary) / "cache")
            with patch.object(reference, "_git", return_value="0" * 40):
                with self.assertRaisesRegex(reference.ReferenceDataError, "pinned"):
                    reference.verify_checkout(Path(temporary))

    def test_local_mapping_changes_fail_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "myFeatureScript.m").write_text("altered")
            with patch.object(reference, "_git", side_effect=[reference.SOURCE_COMMIT, ""]):
                with self.assertRaisesRegex(reference.ReferenceDataError, "checksum"):
                    reference.verify_checkout(root)

    def make_images(self, root: Path) -> None:
        training = root / "training_set"
        training.mkdir()
        for class_index, prefix in enumerate(sorted(reference.EXPECTED_MAPPING)):
            for index in range(1, 26):
                number = class_index * 25 + index
                image = np.full((4, 4), 255, dtype=np.uint8)
                image[0, 0], image[0, 1] = number // 256, number % 256
                success, encoded = cv2.imencode(".bmp", image)
                self.assertTrue(success)
                (training / f"{prefix}{index}.bmp").write_bytes(encoded.tobytes())

    def test_complete_preparation_is_idempotent_and_prefix_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            destination = root / "prepared"
            with patch.object(reference, "verify_checkout", return_value=reference.EXPECTED_MAPPING):
                first = reference.prepare_dataset(root, destination)
                before = (destination / "af" / "af1.bmp").stat().st_mtime_ns
                second = reference.prepare_dataset(root, destination)
            self.assertEqual(first, second)
            self.assertEqual(first["sample_count"], 750)
            self.assertEqual((destination / "af" / "af1.bmp").stat().st_mtime_ns, before)
            self.assertFalse((destination / "a" / "af1.bmp").exists())
            self.assertEqual(json.loads((destination / "labels.json").read_text()), reference.EXPECTED_MAPPING)
            for record in first["samples"]:
                self.assertEqual(hashlib.sha256((destination / record["path"]).read_bytes()).hexdigest(),
                                 record["sha256"])

    def test_missing_image_fails_before_any_destination_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            (root / "training_set" / "af1.bmp").unlink()
            destination = root / "prepared"
            with patch.object(reference, "verify_checkout", return_value=reference.EXPECTED_MAPPING):
                with self.assertRaisesRegex(reference.ReferenceDataError, "missing"):
                    reference.prepare_dataset(root, destination)
            self.assertFalse(destination.exists())

    def test_pixel_duplicates_with_different_file_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            training = root / "training_set"
            (training / "a2.bmp").write_bytes((training / "a1.bmp").read_bytes() + b"different metadata")
            with patch.object(reference, "verify_checkout", return_value=reference.EXPECTED_MAPPING):
                with self.assertRaisesRegex(reference.ReferenceDataError, "pixel_sha256"):
                    reference.prepare_dataset(root, root / "prepared")

    def test_existing_mismatches_are_preserved_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_images(root)
            destination = root / "prepared"
            (destination / "a").mkdir(parents=True)
            stale = destination / "a" / "a1.bmp"
            stale.write_bytes(b"user data")
            with patch.object(reference, "verify_checkout", return_value=reference.EXPECTED_MAPPING):
                with self.assertRaisesRegex(reference.ReferenceDataError, "differs from source"):
                    reference.prepare_dataset(root, destination)
            self.assertEqual(stale.read_bytes(), b"user data")
            self.assertFalse((destination / "labels.json").exists())


if __name__ == "__main__":
    unittest.main()
