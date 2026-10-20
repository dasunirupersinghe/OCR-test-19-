"""Prepare the pinned, partial-alphabet Sinhala handwriting reference dataset.

Downloads source files into a temporary-directory cache, verifies the source
revision and explicit MATLAB Unicode mapping, and copies 750 original BMPs
into 30 class folders. No upstream program is executed. The source contains
no explicit dataset license: images are for local calibration, not bundled
with this project. These are upstream training images; the calibration CLI
must reserve a holdout before fitting. Writer identities are unavailable.

    python prepare_reference_data.py --destination data/reference
    python calibrate_prototypes.py --data-dir data/reference \
        --label-map data/reference/labels.json --holdout-per-class 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np


SOURCE_URL = "https://github.com/asithishantha/Sinhala-Handwriting-Character-Recognition-based-on-Feature-Extraction"
SOURCE_COMMIT = "d2c87bb1eed2d6177ffda4ad07f79e38cf1d993d"
SOURCE_HASHES = {
    "myFeatureScript.m": "95b3e8eeb289f5e8f6911009095328b56a9a2798fd4f491b267a055da83117c9",
    "test.m": "7e85ac8a154cff7793054364dbbd9d60364af05cf043c152e50ead69fc2182a6",
}
EXPECTED_MAPPING = dict(zip(
    "a b c d e f g h i j k l m n p q r s t u v w x y z af ag ae ac aa".split(),
    "අ බ ධ ද එ ෆ ග හ ඉ ජ ළ ල ම ෂ ප ට ර ස ත උ ව ෙ ක ය ණ ඝ භ න ශ ඩ".split(),
))
EXCLUDED_CLASSES = {"o": "O", "num3": "3", "num4": "4", "num5": "5"}


class ReferenceDataError(ValueError):
    """The source or destination violates the pinned dataset contract."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive_label_mapping(feature_script: str, output_script: str) -> dict[str, str]:
    """Join the active MATLAB filename array to explicit indexed Unicode outputs."""
    active = "\n".join(line.split("%", 1)[0] for line in feature_script.splitlines())
    arrays = re.findall(r"\bletters\s*=\s*\[(.*?)\]\s*;", active, re.S)
    if len(arrays) != 1:
        raise ReferenceDataError("expected exactly one active MATLAB class-order array")
    prefixes = [value.strip() for value in re.findall(r"'([^']+)'", arrays[0])]
    if len(prefixes) != 34 or len(set(prefixes)) != 34:
        raise ReferenceDataError("expected 34 unique source class prefixes")
    output_active = "\n".join(line.split("%", 1)[0] for line in output_script.splitlines())
    blocks = re.findall(r"(?:elseif|if)\s*\(I\s*==\s*(\d+)\)(.*?)(?=elseif|\belse\b|$)",
                        output_active, re.S)
    if sorted(int(index) for index, _ in blocks) != list(range(1, 35)):
        raise ReferenceDataError("source output mapping must cover each class index exactly once")
    mapping: dict[str, str] = {}
    exclusions: dict[str, str] = {}
    for index, block in blocks:
        prefix = prefixes[int(index) - 1]
        unicode_codes = re.findall(r"hex2dec\('([0-9A-Fa-f]+)'\)", block)
        ascii_labels = re.findall(r"fprintf\(fid\s*,\s*'([^']+)'\)", block)
        if len(unicode_codes) == 1 and not ascii_labels:
            mapping[prefix] = chr(int(unicode_codes[0], 16))
        elif len(ascii_labels) == 1 and not unicode_codes:
            exclusions[prefix] = ascii_labels[0]
        else:
            raise ReferenceDataError(f"ambiguous output label for source prefix {prefix!r}")
    if mapping != EXPECTED_MAPPING or exclusions != EXCLUDED_CLASSES:
        raise ReferenceDataError("source labels differ from the independently verified Unicode mapping")
    return mapping


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(checkout), *arguments], check=True,
                            capture_output=True, text=True, timeout=120)
    return result.stdout.strip()


def obtain_checkout(checkout: Path | None, cache_dir: Path) -> Path:
    """Reuse an exact checkout or download the immutable public revision."""
    if checkout is not None:
        if not checkout.is_dir():
            raise ReferenceDataError(f"source checkout does not exist: {checkout}")
        return checkout.resolve()
    target = cache_dir.expanduser().resolve() / SOURCE_COMMIT
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Clone into a new temporary directory, so failed downloads never poison
        # an existing cache. A source revision is checked, never executed.
        temporary = Path(tempfile.mkdtemp(prefix=".reference-", dir=target.parent))
        try:
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            SOURCE_URL + ".git", str(temporary)], check=True, timeout=120)
            _git(temporary, "checkout", "--detach", SOURCE_COMMIT)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def verify_checkout(checkout: Path) -> dict[str, str]:
    if _git(checkout, "rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ReferenceDataError("source checkout HEAD does not match the pinned dataset revision")
    dirty = _git(checkout, "status", "--porcelain", "--untracked-files=no", "--",
                 "training_set", *SOURCE_HASHES)
    if dirty:
        raise ReferenceDataError("source dataset or Unicode mapping has local modifications")
    for name, expected in SOURCE_HASHES.items():
        if _digest((checkout / name).read_bytes()) != expected:
            raise ReferenceDataError(f"source mapping checksum mismatch: {name}")
    return derive_label_mapping((checkout / "myFeatureScript.m").read_text(),
                                (checkout / "test.m").read_text())


def _image_record(path: Path, relative_path: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceDataError(f"missing or symbolic-link source image: {path}")
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ReferenceDataError(f"unreadable source image: {path}")
    pixel_digest = _digest(str(image.shape).encode() + image.dtype.str.encode() + image.tobytes())
    return {"path": relative_path, "label": label, "sha256": _digest(payload),
            "pixel_sha256": pixel_digest, "bytes": len(payload)}


def prepare_dataset(checkout: Path, destination: Path) -> dict[str, Any]:
    """Verify all inputs and existing outputs before copying; mismatches fail closed."""
    checkout, destination = checkout.resolve(), destination.expanduser().absolute()
    mapping = verify_checkout(checkout)
    records = [_image_record(checkout / "training_set" / f"{prefix}{index}.bmp",
                             f"{prefix}/{prefix}{index}.bmp", label)
               for prefix, label in sorted(mapping.items()) for index in range(1, 26)]
    for key in ("sha256", "pixel_sha256"):
        if len({record[key] for record in records}) != len(records):
            raise ReferenceDataError(f"duplicate source images detected by {key}")
    provenance: dict[str, Any] = {
        "format_version": 1, "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
        "source_mapping_sha256": SOURCE_HASHES, "folder_to_unicode": mapping,
        "mapping_evidence": ["myFeatureScript.m: active ordered letters array",
                             "test.m: indexed explicit UTF-8 Unicode hex values"],
        "excluded_ascii_classes": EXCLUDED_CLASSES, "sample_count": len(records),
        "class_count": len(mapping), "samples_per_class": 25,
        "source_license": "Not specified in the source repository; raw images are not redistributed",
        "source_split": "upstream training_set; reserve held-out files before calibration",
        "writer_ids_available": False, "coverage": "partial alphabet; not a complete Sinhala recognizer",
        "byte_and_decoded_pixel_duplicates": 0, "samples": records,
    }
    json_files = {"labels.json": mapping, "dataset_provenance.json": provenance}
    if destination.is_symlink():
        raise ReferenceDataError("destination must not be a symbolic link")
    if destination.exists():
        if not destination.is_dir():
            raise ReferenceDataError("destination must be a directory")
        expected = set(mapping) | set(json_files)
        unexpected = {p.name for p in destination.iterdir()} - expected
        if unexpected:
            raise ReferenceDataError(f"unexpected destination entries: {sorted(unexpected)}")
    for prefix in mapping:
        folder = destination / prefix
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ReferenceDataError(f"invalid destination class folder: {folder}")
        expected_names = {f"{prefix}{index}.bmp" for index in range(1, 26)}
        if folder.exists() and {p.name for p in folder.iterdir()} - expected_names:
            raise ReferenceDataError(f"unexpected files in destination class: {prefix}")
    for record in records:
        target = destination / record["path"]
        if target.is_symlink() or (target.exists() and (not target.is_file() or
                _digest(target.read_bytes()) != record["sha256"])):
            raise ReferenceDataError(f"existing destination image differs from source: {target}")
    for name, value in json_files.items():
        target = destination / name
        if target.is_symlink() or (target.exists() and json.loads(target.read_text()) != value):
            raise ReferenceDataError(f"existing destination metadata differs: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        target = destination / record["path"]
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            with target.open("xb") as handle:
                handle.write((checkout / "training_set" / target.name).read_bytes())
        if _digest(target.read_bytes()) != record["sha256"]:
            raise ReferenceDataError(f"copied destination image checksum mismatch: {target}")
    for name, value in json_files.items():
        target = destination / name
        if not target.exists():
            with target.open("x", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, help="Existing source checkout, pinned revision required")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "sinhala-reference-cache")
    parser.add_argument("--destination", type=Path, default=Path("data/reference"))
    args = parser.parse_args(argv)
    try:
        checkout = obtain_checkout(args.checkout, args.cache_dir)
        report = prepare_dataset(checkout, args.destination)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Reference preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"destination": str(args.destination.absolute()),
                      "classes": report["class_count"], "samples": report["sample_count"],
                      "source_commit": SOURCE_COMMIT}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
