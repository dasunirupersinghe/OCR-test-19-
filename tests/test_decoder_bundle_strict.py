"""Independent strict-bundle persistence contracts; no OCR accuracy assertion."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import decoder as d



def book(view="whole_glyph", *, dimension=46, schema_id="a" * 64):
    rng = np.random.default_rng(47)
    labels = ("ි", "ී") if view == "zone:upper" else ("ු", "ූ") if view == "zone:lower" else ("ක", "ග")
    factors = rng.normal(size=(len(labels), dimension, dimension))
    cov = factors @ factors.transpose(0, 2, 1) + np.eye(dimension)[None] * .1
    return d.PrototypeBook(labels, rng.normal(size=(2, dimension)), np.zeros(dimension), np.ones(dimension),
                           cov, np.ones(2, bool), np.full(2, 60, np.int64), schema_id, view,
                           tuple(f"feature.{index:02d}" for index in range(dimension)), {"calibration_split": "development"})


def rewrite_archive(archive, sidecar, transform):
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    transform(arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    archive.write_bytes(buffer.getvalue())
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["archive_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")


class StrictBundleContracts(unittest.TestCase):
    def test_real_three_zone_models_roundtrip_losslessly_in_single_archive(self):
        models = {view: book(view) for view in ("zone:core", "zone:upper", "zone:lower")}
        bundle = d.PrototypeBundle(models)
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar = bundle.save(Path(directory) / "bundle.npz")
            self.assertLess(archive.stat().st_size, 2_000_000)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(len(zipped.namelist()), 19)
                self.assertFalse(any(name.endswith(".npz") for name in zipped.namelist()))
            restored = d.PrototypeBundle.load(archive, sidecar)
            self.assertEqual(set(restored), set(models))
            for view, original in models.items():
                reloaded = restored[view]
                for field in ("means", "center", "scale", "covariances", "covariance_available", "counts", "precisions"):
                    np.testing.assert_array_equal(getattr(original, field), getattr(reloaded, field))
                self.assertEqual(original.labels, reloaded.labels)
                self.assertEqual(original.model_view_key, reloaded.model_view_key)
                self.assertEqual(dict(original.metadata), dict(reloaded.metadata))
            with self.assertRaises(TypeError):
                restored["zone:core"] = models["zone:core"]
            signatures = restored.modifier_signatures
            self.assertEqual(set(signatures), {"zone:upper", "zone:lower"})
            np.testing.assert_array_equal(signatures["zone:upper"]["ි"], models["zone:upper"].raw_means[0])
            self.assertFalse(signatures["zone:upper"]["ි"].flags.writeable)
            with self.assertRaises(TypeError):
                signatures["zone:upper"]["ි"] = np.zeros(46)
            pipeline = d.DecoderPipeline(restored)
            self.assertEqual(pipeline.recognition_mode, "multi_zone")

    def test_whole_glyph_can_coexist_with_modifier_books(self):
        bundle = d.PrototypeBundle({"whole_glyph": book(), "zone:upper": book("zone:upper")})
        self.assertEqual(len(bundle), 2)
        self.assertEqual(d.DecoderPipeline(bundle).recognition_mode, "whole_glyph")

    def test_generic_non46_books_are_rejected_without_padding(self):
        with self.assertRaises(d.ModelCompatibilityError):
            d.PrototypeBundle({"whole_glyph": book(dimension=36)})

    def test_view_and_schema_identity_are_enforced(self):
        with self.assertRaises(d.CalibrationError):
            d.PrototypeBundle({"zone:core": book()})
        with self.assertRaises(d.ModelCompatibilityError):
            d.PrototypeBundle({"whole_glyph": book(), "zone:core": book("zone:core", schema_id="b" * 64)})
        with self.assertRaises(d.CalibrationError):
            d.PrototypeBundle({})

    def test_exact_2mb_boundary_is_rejected_before_any_write(self):
        bundle = d.PrototypeBundle({"whole_glyph": book()})
        def oversized(buffer, **arrays):
            buffer.write(b"x" * 2_000_000)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(d.np, "savez_compressed", side_effect=oversized), mock.patch.object(d, "_atomic_write") as write:
            target = Path(directory) / "new-dir" / "bundle.npz"
            with self.assertRaisesRegex(d.DecoderError, "<2000000"):
                bundle.save(target)
            write.assert_not_called()
            self.assertFalse(target.parent.exists())

    def test_no_overwrite_and_atomic_new_save_cleanup(self):
        bundle = d.PrototypeBundle({"whole_glyph": book()})
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar = bundle.save(Path(directory) / "bundle.npz")
            prior = archive.read_bytes(), sidecar.read_bytes()
            with self.assertRaises(FileExistsError):
                bundle.save(archive)
            self.assertEqual(prior, (archive.read_bytes(), sidecar.read_bytes()))
            new_archive = Path(directory) / "new.npz"
            original_write = d._atomic_write
            def fail_manifest(path, data, **kwargs):
                if path.suffix == ".json":
                    raise OSError("simulated sidecar commit failure")
                return original_write(path, data, **kwargs)
            with mock.patch.object(d, "_atomic_write", side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    bundle.save(new_archive)
            self.assertFalse(new_archive.exists())

    def test_wrong_checksum_is_rejected_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar = d.PrototypeBundle({"whole_glyph": book()}).save(Path(directory) / "bundle.npz")
            archive.write_bytes(archive.read_bytes() + b"modified")
            with mock.patch.object(d.np, "load") as load:
                with self.assertRaisesRegex(d.DecoderError, "SHA256"):
                    d.PrototypeBundle.load(archive)
                load.assert_not_called()

    def test_dtype_and_shape_are_rejected_before_numpy_load(self):
        for mutate in (lambda arrays: arrays.__setitem__("b0__means", arrays["b0__means"].astype(np.float32)),
                       lambda arrays: arrays.__setitem__("b0__means", arrays["b0__means"].T)):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                archive, sidecar = d.PrototypeBundle({"whole_glyph": book()}).save(Path(directory) / "bundle.npz")
                rewrite_archive(archive, sidecar, mutate)
                with mock.patch.object(d.np, "load") as load:
                    with self.assertRaisesRegex(d.DecoderError, "shape/dtype"):
                        d.PrototypeBundle.load(archive)
                    load.assert_not_called()

    def test_missing_unexpected_and_object_arrays_are_rejected(self):
        mutations = (lambda arrays: arrays.pop("b0__means"),
                     lambda arrays: arrays.__setitem__("unexpected", np.array([1.])),
                     lambda arrays: arrays.__setitem__("b0__means", np.full((2, 46), "object", dtype=object)))
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                archive, sidecar = d.PrototypeBundle({"whole_glyph": book()}).save(Path(directory) / "bundle.npz")
                rewrite_archive(archive, sidecar, mutate)
                with mock.patch.object(d.np, "load") as load:
                    with self.assertRaises(d.DecoderError):
                        d.PrototypeBundle.load(archive)
                    load.assert_not_called()

    def test_compressed_and_expanded_budgets_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar = d.PrototypeBundle({"whole_glyph": book()}).save(Path(directory) / "bundle.npz")
            with mock.patch.object(d.np, "load") as load:
                with self.assertRaisesRegex(d.DecoderError, "allocation budget"):
                    d.PrototypeBundle.load(archive, max_model_bytes=100)
                load.assert_not_called()
            archive.write_bytes(b"x" * 2_000_000)
            with mock.patch.object(d.np, "load") as load:
                with self.assertRaisesRegex(d.DecoderError, "strictly smaller"):
                    d.PrototypeBundle.load(archive)
                load.assert_not_called()

    def test_manifest_generation_mismatch_and_duplicate_views_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar = d.PrototypeBundle({"whole_glyph": book()}).save(Path(directory) / "bundle.npz")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            manifest["books"][0]["metadata"]["generation"] = "wrong"
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(d.DecoderError, "different model generations"):
                d.PrototypeBundle.load(archive)
            manifest["books"].append(manifest["books"][0])
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(d.np, "load") as load:
                with self.assertRaises(d.DecoderError):
                    d.PrototypeBundle.load(archive)
                load.assert_not_called()

    def test_nonpositive_covariance_is_rejected_after_safe_load(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, sidecar = d.PrototypeBundle({"whole_glyph": book()}).save(Path(directory) / "bundle.npz")
            def mutate(arrays):
                arrays["b0__covariances"][0] = -np.eye(46)
            rewrite_archive(archive, sidecar, mutate)
            with self.assertRaisesRegex(d.DecoderError, "positive definite"):
                d.PrototypeBundle.load(archive)


if __name__ == "__main__":
    unittest.main()
