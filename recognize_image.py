"""Run the offline Sinhala OCR pipeline with an explicitly calibrated model.

    python recognize_image.py --image sample.png --model artifacts/prototypes.npz

This command reads an existing whole-glyph PrototypeBook (.npz plus .json),
preprocesses and segments the image, extracts the strict 46-dimensional profile,
and executes DecoderPipeline's two-pass decoder. An optional UTF-8 word list
enables dictionary matching; no dictionary or context statistics are invented.
Confidence values are class-relative scores, not measured OCR accuracy.

An isolated-glyph calibration corpus and unseen word segmentation can have
different margins and character boundaries. The output records both preparation
policies; successful execution alone does not establish word recognition quality.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence
import unicodedata

import cv2
import numpy as np

from decoder import (DecoderError, DecoderPipeline, DevelopmentSetCalibrationManager,
                     ModelCompatibilityError, PrototypeBook)
from preprocessing import PreprocessingPipeline
from segmentation import SegmentationPipeline, SegmentationResult


LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert only supported diagnostic types; never emit NaN/Infinity."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.floating):
        return _json_value(float(value))
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise DecoderError("diagnostics contain a nonfinite number")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise DecoderError(f"unsupported diagnostic value type: {type(value).__name__}")


def _write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_lexicon(path: Path | None) -> tuple[tuple[str, ...] | None, dict[str, Any] | None]:
    """Read an explicit offline word list, with one whitespace-free word/line."""
    if path is None:
        return None, None
    with path.open("rb") as handle:
        payload = handle.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise DecoderError("lexicon text exceeds the 32 MB limit")
    try:
        lines = payload.decode("utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise DecoderError(f"lexicon is not valid UTF-8: {path}") from exc
    words: dict[str, None] = {}
    for number, line in enumerate(lines, 1):
        word = unicodedata.normalize("NFC", line.strip())
        if not word:
            continue
        if len(word) > 128 or any(char.isspace() for char in word):
            raise DecoderError(f"lexicon line {number} must contain one word of at most 128 code points")
        words[word] = None
        if len(words) > 1_000_000:
            raise DecoderError("lexicon exceeds the one-million-word limit")
    if not words:
        raise DecoderError("the supplied lexicon has no words")
    return tuple(words), {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                          "word_count": len(words)}


def _debug_images(mask: np.ndarray, segmentation: SegmentationResult, directory: Path,
                  *, overwrite: bool, protected_paths: Sequence[Path] = ()) -> dict[str, str]:
    images: dict[str, np.ndarray] = {"binary_mask": ((1-mask)*255).astype(np.uint8)}
    for word in segmentation.words:
        prefix = f"word_{word.word_id:04d}"
        images[f"{prefix}_normalized"] = ((1-word.normalized_mask)*255).astype(np.uint8)
        images[f"{prefix}_skeleton_overlay"] = word.diagnostics.skeleton_overlay
        boxes = cv2.cvtColor(images[f"{prefix}_normalized"], cv2.COLOR_GRAY2BGR)
        for index, candidate in enumerate(word.candidates):
            box = candidate.normalized_bbox
            cv2.rectangle(boxes, (box.x_min, box.y_min), (box.x_max-1, box.y_max-1), (0, 0, 255), 1)
            cv2.putText(boxes, str(index), (box.x_min, max(10, box.y_min+10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0), 1, cv2.LINE_AA)
        images[f"{prefix}_candidate_boxes"] = boxes
    paths: dict[str, str] = {}
    targets = {name: directory / f"{name}.png" for name in images}
    if any(path in protected_paths for path in targets.values()):
        raise DecoderError("debug images must not replace any input or model file")
    if not overwrite and any(path.exists() for path in targets.values()):
        raise DecoderError("debug images already exist; use --overwrite explicitly")
    for name, image in images.items():
        path = targets[name]
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise DecoderError(f"could not encode debug image: {name}")
        _write_bytes(path, encoded.tobytes(), overwrite=overwrite)
        paths[name] = str(path)
    return paths


def recognize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    image_path = args.image.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise DecoderError("--output must end in .json")
    if output_path in (image_path, model_path, model_path.with_suffix(".json")):
        raise DecoderError("output must not replace the input image or model files")
    if output_path.exists() and not args.overwrite:
        raise DecoderError(f"output exists; use --overwrite explicitly: {output_path}")
    if not image_path.is_file():
        raise DecoderError(f"input image does not exist: {image_path}")
    book = PrototypeBook.load(model_path)
    if book.view != "whole_glyph":
        raise ModelCompatibilityError("this CLI requires a whole_glyph PrototypeBook; zone models need a complete model mapping")
    manager = DevelopmentSetCalibrationManager(mode=args.mode)
    if (book.schema_id != manager.feature_pipeline.schema.schema_id
            or book.feature_names != manager.feature_pipeline.schema.feature_names):
        raise ModelCompatibilityError("model does not match this mode's strict 46-dimensional feature profile")
    lexicon_path = args.lexicon_text.expanduser().resolve() if args.lexicon_text is not None else None
    lexicon, lexicon_metadata = load_lexicon(lexicon_path)
    try:
        mask, preprocessing_metadata = PreprocessingPipeline(manager.config, mode=manager.mode).run(image_path)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise DecoderError(f"could not preprocess input image {image_path}: {exc}") from exc
    segmentation = SegmentationPipeline(manager.config, mode=manager.mode).run(mask)
    batch = manager.feature_pipeline.run(segmentation)
    result = DecoderPipeline(book, lexicon=lexicon, config=manager.config, mode=manager.mode).run(batch)
    words: list[dict[str, Any]] = []
    for word in result.words:
        words.append({"word_id": word.word_id, "raw_text": word.raw_text, "text": word.text,
                      "status": word.status, "class_relative_confidence": word.confidence,
                      "needs_review": word.needs_review, "is_context_anchor": word.is_context_anchor,
                      "paths": [{"text": path.text, "log_probability": path.log_probability,
                                 "class_relative_confidence": path.confidence,
                                 "candidate_ids": path.candidate_ids,
                                 "hypothesis_id": path.hypothesis_id,
                                 "metadata": path.metadata} for path in word.paths],
                      "metadata": word.metadata})
    debug_images = (_debug_images(mask, segmentation, args.debug_dir.expanduser().resolve(), overwrite=args.overwrite,
                                  protected_paths=(image_path, model_path, model_path.with_suffix(".json"), output_path))
                    if args.debug_dir is not None else {})
    report = _json_value({
        "report_version": 1, "mode": manager.mode, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("empty" if not words else "partial" if any(word["status"] == "rejected" for word in words) else "completed"),
        "raw_text": " ".join(word.raw_text for word in result.words if word.raw_text), "text": result.text,
        "needs_review": not words or any(word.needs_review or word.status != "decoded" for word in result.words),
        "word_count": len(words), "words": words,
        "confidence_interpretation": "class-relative prototype scores; not calibrated correctness or OCR accuracy",
        "image": {"path": str(image_path), "sha256": _sha256_file(image_path),
                  "preprocessed_shape": list(mask.shape)},
        "model": {"path": str(model_path), "sha256": _sha256_file(model_path),
                  "manifest_path": str(model_path.with_suffix(".json")), "class_count": len(book.labels),
                  "labels": book.labels, "dimension": book.dimension, "schema_id": book.schema_id,
                  "view": book.view, "dataset_provenance": book.metadata.get("dataset_provenance"),
                  "calibration_preparation": book.metadata.get("dataset_provenance", {}).get("preparation")},
        "query_preparation": "preprocess_then_word_localization_and_candidate_segmentation",
        "lexicon": lexicon_metadata, "context_statistics_supplied": False,
        "preprocessing": preprocessing_metadata, "segmentation": segmentation.metadata,
        "decoder": result.metadata, "debug_images": debug_images,
        "elapsed_seconds": time.perf_counter()-started,
    })
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    _write_bytes(output_path, payload, overwrite=args.overwrite)
    LOGGER.info("Recognition completed: %d words; review=%s; saved %s",
                len(words), report["needs_review"], output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/prototypes.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/recognition_result.json"))
    parser.add_argument("--mode", choices=("B0", "B1", "B2", "B3", "B4"), default="B4")
    parser.add_argument("--lexicon-text", type=Path, default=None, help="Optional actual offline word list, one word per line")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        for name in ("preprocessing", "segmentation"):
            logging.getLogger(name).setLevel(logging.WARNING)
    try:
        report = recognize(args)
    except (OSError, DecoderError, ValueError, RuntimeError, cv2.error) as exc:
        LOGGER.error("Recognition failed: %s", exc)
        return 1
    print(json.dumps({"text": report["text"], "word_count": report["word_count"],
                      "needs_review": report["needs_review"], "status": report["status"],
                      "output": str(args.output.expanduser().resolve())}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
