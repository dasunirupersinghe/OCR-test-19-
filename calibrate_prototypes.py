"""Calibrate the project's strict 46-value Sinhala prototype model on CPU.

Example::

    python calibrate_prototypes.py --data-dir data/development \
        --label-map data/folder_to_unicode.json --output artifacts/prototypes.npz

The label map is a JSON object containing actual Sinhala Unicode values, e.g.
``{"001": "ක", "002": "ග"}``. Folder numbers are never guessed. Each class
folder must contain only supported image files. At least two distinct files
per class are fitted; a separate, deterministic held-out sample is reserved by
default. Exact duplicate files in a class are deduplicated before splitting.
The report is glyph-level validation on this split, not word OCR accuracy.

Preparation matches decoder.py's whole-glyph development-folder policy:
preprocessing followed by proportional height normalization of the entire
crop. Use ``prepare_feature`` for isolated query crops to reproduce it. This
does not calibrate separate upper/core/lower zone models from unannotated data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from decoder import (CalibrationError, CalibrationSample, DecoderError,
                     DevelopmentSetCalibrationManager, PrototypeBook,
                     PrototypeScorer)
from features import FeatureResult
from preprocessing import PreprocessingPipeline
from segmentation import SegmentationPipeline


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
REPORT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    relative_path: str
    folder: str
    label: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.relative_path, "folder": self.folder,
                "label": self.label, "sha256": self.sha256}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"duplicate JSON key in label map: {key!r}")
        result[key] = value
    return result


def load_label_map(path: Path) -> tuple[dict[str, str], str]:
    """Read a bounded explicit folder→Unicode map without lossy label inference."""
    with path.open("rb") as handle:
        data = handle.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise CalibrationError("label map exceeds the 1 MB limit")
    try:
        mapping = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_json,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 CalibrationError(f"invalid JSON constant: {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"invalid UTF-8 JSON label map: {path}") from exc
    if not isinstance(mapping, dict) or not mapping or len(mapping) > 4096:
        raise CalibrationError("label map must be a nonempty JSON object with at most 4096 classes")
    normalized: dict[str, str] = {}
    for folder, label in mapping.items():
        if (not isinstance(folder, str) or folder in ("", ".", "..")
                or folder.startswith(".") or "/" in folder or "\\" in folder):
            raise CalibrationError("label-map keys must be visible single folder names")
        if not isinstance(label, str) or not label or len(label) > 128:
            raise CalibrationError(f"folder {folder!r} needs an actual Sinhala Unicode label")
        normalized[folder] = unicodedata.normalize("NFC", label)
    # All linguistic label validation is still performed by the manager's fit.
    return normalized, hashlib.sha256(data).hexdigest()


def select_records(data_dir: Path, label_map: dict[str, str], *, seed: int = 42,
                   max_samples_per_class: int | None = None,
                   holdout_per_class: int = 1) -> tuple[list[ImageRecord], list[ImageRecord], list[dict[str, str]]]:
    """Split unique files before feature extraction; held-out files never fit."""
    if not data_dir.is_dir():
        raise CalibrationError(f"development directory does not exist: {data_dir}")
    if holdout_per_class < 0 or (max_samples_per_class is not None and max_samples_per_class < 2):
        raise CalibrationError("holdout count must be nonnegative and calibration cap must be >=2")
    folders = {entry.name for entry in data_dir.iterdir()
               if entry.is_dir() and not entry.name.startswith(".")}
    if folders != set(label_map):
        raise CalibrationError(f"label map must match class folders exactly; unmapped={sorted(folders-set(label_map))}, "
                               f"missing={sorted(set(label_map)-folders)}")
    fit: list[ImageRecord] = []
    holdout: list[ImageRecord] = []
    duplicates: list[dict[str, str]] = []
    global_hashes: dict[str, ImageRecord] = {}
    for folder, label in sorted(label_map.items()):
        directory = data_dir / folder
        if directory.is_symlink():
            raise CalibrationError(f"symbolic-link class folders are not supported: {directory}")
        unique: list[ImageRecord] = []
        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise CalibrationError(f"class folder contains an unsupported file or nested directory: {path}")
            record = ImageRecord(path, path.relative_to(data_dir).as_posix(), folder, label, _sha256_file(path))
            previous = global_hashes.get(record.sha256)
            if previous is not None:
                if previous.label != label:
                    raise CalibrationError(f"identical file bytes have conflicting labels: {previous.path}, {path}")
                duplicates.append({"path": record.relative_path, "duplicate_of": previous.relative_path,
                                   "sha256": record.sha256})
                continue
            global_hashes[record.sha256] = record
            unique.append(record)
        if len(unique) < holdout_per_class + 2:
            raise CalibrationError(f"class {folder!r} has {len(unique)} unique images; need at least "
                                   f"{holdout_per_class+2} for two calibration files and the requested holdout")
        # A folder-specific seed keeps other classes stable when a class is added.
        key = f"{seed}\0{folder}".encode("utf-8")
        generator = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        indices = generator.permutation(len(unique))
        holdout.extend(unique[int(index)] for index in indices[:holdout_per_class])
        fitting = indices[holdout_per_class:]
        if max_samples_per_class is not None:
            fitting = fitting[:max_samples_per_class]
        fit.extend(unique[int(index)] for index in fitting)
    return fit, holdout, duplicates


class WholeGlyphFeatureBuilder:
    """Reusable CPU preparation shared by calibration and isolated queries."""

    def __init__(self, manager: DevelopmentSetCalibrationManager) -> None:
        self.manager = manager
        self.preprocessor = PreprocessingPipeline(manager.config, mode=manager.mode)
        self.normalizer = SegmentationPipeline(manager.config, mode=manager.mode).zone_segmenter

    def __call__(self, path: Path) -> FeatureResult:
        try:
            mask, metadata = self.preprocessor.run(path)
            if metadata["status"] != "ok":
                raise CalibrationError("image has no usable foreground ink")
            normalized, _ = self.normalizer.normalize(mask)
            feature = self.manager.feature_pipeline.extract(normalized, view="whole_glyph")
            if feature.metadata.get("empty_ink") or feature.metadata.get("empty_canvas"):
                raise CalibrationError("image produced empty glyph features")
            return feature
        except (OSError, ValueError, RuntimeError) as exc:
            raise CalibrationError(f"feature extraction failed for {path}: {exc}") from exc


def prepare_feature(path: str | Path, manager: DevelopmentSetCalibrationManager) -> FeatureResult:
    """Prepare one isolated query crop with exactly the calibration policy."""
    return WholeGlyphFeatureBuilder(manager)(Path(path))


def _write_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.expanduser().resolve()
    label_map_path = args.label_map.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise CalibrationError("--output must end in .npz")
    manifest_path = output.with_suffix(".json")
    report_path = output.parent / "calibration_report.json"
    for target in (output, manifest_path, report_path):
        if target.exists() and not args.overwrite:
            raise CalibrationError(f"refusing to replace existing artifact; use --overwrite explicitly: {target}")
    label_map, map_hash = load_label_map(label_map_path)
    fit_records, heldout_records, duplicates = select_records(
        data_dir, label_map, seed=args.seed, max_samples_per_class=args.max_samples_per_class,
        holdout_per_class=args.holdout_per_class)
    manager = DevelopmentSetCalibrationManager(mode=args.mode, cov_reg_eps=args.cov_reg_eps)
    builder = WholeGlyphFeatureBuilder(manager)
    LOGGER.info("Extracting %d development glyphs; holding out %d distinct files",
                len(fit_records), len(heldout_records))
    book = manager.fit(CalibrationSample(record.label, builder(record.path), "development")
                       for record in fit_records)
    metadata = dict(book.metadata)
    metadata["dataset_provenance"] = {
        "data_directory": str(data_dir), "label_map_sha256": map_hash,
        "source_url": args.source_url, "source_license": args.source_license,
        "selection_seed": args.seed, "heldout_count": len(heldout_records),
        "preparation": "preprocess_then_entire_crop_aspect_ratio_height_normalize",
        "view": "whole_glyph", "script": "calibrate_prototypes.py"}
    book = replace(book, metadata=metadata)
    # Validate held-out files before committing any model artifacts, then score
    # these exact features with the reloaded model rather than fitting them.
    heldout_features = [(record, builder(record.path)) for record in heldout_records]
    # Save enforces the strict compressed <2,000,000-byte archive limit.
    output.parent.mkdir(parents=True, exist_ok=True)
    book.save(output, overwrite=args.overwrite)
    loaded = PrototypeBook.load(output)
    for name in ("means", "center", "scale", "covariances", "covariance_available", "counts"):
        if not np.array_equal(getattr(book, name), getattr(loaded, name)):
            raise DecoderError(f"model round-trip changed the {name} array")
    if loaded.labels != book.labels or loaded.schema_id != book.schema_id:
        raise DecoderError("model round-trip changed the label/schema contract")
    scorer = PrototypeScorer(loaded, manager.config, mode=manager.mode)
    predictions: list[dict[str, Any]] = []
    for record, feature in heldout_features:
        scores = scorer.score(feature)
        candidate = scores.top_candidates[0]
        predictions.append({**record.to_dict(), "predicted_label": candidate.label,
                            "correct": candidate.label == record.label,
                            "distance": candidate.distance, "class_relative_score": scores.confidence})
    correct = sum(item["correct"] for item in predictions)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION, "mode": manager.mode, "view": loaded.view,
        "feature_dimension": loaded.dimension, "schema_id": loaded.schema_id,
        "labels": list(loaded.labels), "folder_to_unicode": label_map,
        "label_map_path": str(label_map_path), "label_map_sha256": map_hash,
        "data_directory": str(data_dir), "source_url": args.source_url,
        "source_license": args.source_license, "seed": args.seed,
        "max_calibration_samples_per_class": args.max_samples_per_class,
        "requested_holdout_per_class": args.holdout_per_class,
        "calibration_sample_count": len(fit_records), "heldout_sample_count": len(heldout_records),
        "per_class_calibration_count": dict(zip(loaded.labels, map(int, loaded.counts))),
        "covariance_available": dict(zip(loaded.labels, map(bool, loaded.covariance_available))),
        "cov_reg_eps": manager.settings.covariance_regularization,
        "calibration_samples": [record.to_dict() for record in fit_records],
        "heldout_predictions": predictions, "deduplicated_files": duplicates,
        "heldout_correct": correct,
        "heldout_glyph_accuracy": correct/len(predictions) if predictions else None,
        "validation_scope": "held-out isolated glyphs from this dataset; not word OCR accuracy",
        "scores_are_calibrated_correctness": False,
        "archive_path": str(output), "manifest_path": str(manifest_path),
        "archive_bytes": output.stat().st_size, "archive_sha256": _sha256_file(output),
        "strict_archive_limit_bytes": 2_000_000, "load_roundtrip_verified": True,
    }
    _write_report(report_path, report, overwrite=args.overwrite)
    LOGGER.info("Saved %s (%d bytes); %d/%d held-out glyph predictions matched their labels",
                output, report["archive_bytes"], correct, len(predictions))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root containing one visible folder per class")
    parser.add_argument("--label-map", type=Path, required=True, help="UTF-8 JSON folder→actual Sinhala Unicode object")
    parser.add_argument("--output", type=Path, default=Path("artifacts/prototypes.npz"))
    parser.add_argument("--mode", choices=("B0", "B1", "B2", "B3", "B4"), default="B4")
    parser.add_argument("--cov-reg-eps", type=float, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Optional cap on calibration samples per class, >=2")
    parser.add_argument("--holdout-per-class", type=int, default=1, help="Reserve this many files per class; 0 disables validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-url", default=None, help="Dataset provenance URL, recorded without making a network request")
    parser.add_argument("--source-license", default=None, help="Verified dataset license identifier or description")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace the archive, manifest and report")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        for name in ("preprocessing", "segmentation"):
            logging.getLogger(name).setLevel(logging.WARNING)
    if args.cov_reg_eps is not None and (not math.isfinite(args.cov_reg_eps) or args.cov_reg_eps <= 0):
        parser.error("--cov-reg-eps must be finite and positive")
    if args.holdout_per_class < 0:
        parser.error("--holdout-per-class must be nonnegative")
    if args.max_samples_per_class is not None and args.max_samples_per_class < 2:
        parser.error("--max-samples-per-class must be >=2")
    try:
        report = calibrate(args)
    except (OSError, DecoderError, ValueError) as exc:
        LOGGER.error("Calibration failed: %s", exc)
        return 1
    print(json.dumps({"archive": report["archive_path"], "bytes": report["archive_bytes"],
                      "dimension": report["feature_dimension"], "classes": len(report["labels"]),
                      "heldout_glyph_accuracy": report["heldout_glyph_accuracy"],
                      "report": str(args.output.expanduser().resolve().parent / "calibration_report.json")},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
