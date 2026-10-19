"""Offline prototype decoding and structurally validated Sinhala word assembly.

Python >=3.10; NumPy and the preceding pipeline modules are required. No model,
lexicon, image directory, or network service is opened merely by importing this
module. Calibration is explicit and must use development/calibration data with
actual Unicode labels. Numeric dataset folder IDs are never inferred as labels.

    from decoder import CalibrationSample, PrototypeCalibrationManager, DecoderPipeline
    book = PrototypeCalibrationManager().fit(samples)  # labelled FeatureResults
    book.save("artifacts/prototypes.npz")
    result = DecoderPipeline(book, lexicon=words).run(feature_batch)

Models are bound to both FeatureSchema.schema_id AND feature view. Whole-glyph
models already include vowel marks. Zoned models are separate books keyed by
zone:core/upper/lower. A missing model for nonempty ink rejects that path. Marks
spanning multiple selected candidates reject a contaminated split hypothesis;
the preserved intact hypothesis remains available. No mark is silently dropped
or counted twice. Folder calibration defaults to whole-glyph preparation;
custom/zone preparation requires an explicit feature_builder callback.

The quantum-inspired state is a classical REAL vector, not quantum hardware:
alpha=exp(-gamma*D)/sqrt(sum(exp(-2*gamma*D))), P=alpha**2. Evaluate after
subtracting the minimum distance. These class-relative scores are NOT calibrated
correctness or open-set probabilities; a one-class book gives P=1 even far away.
B0--B2 use nearest prototypes and report confidence/probability as unavailable.
Word confidence follows C=(1/T)*sum_t max_k P_t(c_k), as specified. Selected
label probabilities are reported separately because Unicode filtering can
retain a lower-ranked label. The prescribed maximum-score shield can therefore
be high even when the selected label score is low; it is not an accuracy claim.

Raw prototype statistics are globally standardized using calibration mean and
population std. Constant/tiny-variance dimensions use scale1. Class covariance
requires at least max(configured_minimum,D+1) samples; use standardized Euclidean
otherwise. Eligible covariance is shrunk and regularized in standardized space.
No feature weighting is repeated: features.py already applied block weights.

Cross-segmentation paths rank by mean token log probability (or mean negative
distance for baselines), with a documented overlap penalty for window proposals.
Summed log probabilities remain separately stored for the requested lexicon
objective. Beam search is bounded, not claimed globally optimal or statistically
calibrated across segment counts. Overlapping baseline windows are selected as
covering interval paths rather than blindly concatenated.

Persistence uses pickle-free NPZ and strict JSON, SHA256, shape/dtype/resource
checks, and fsynced temporary files. The archive commits before its manifest;
interruption/concurrent replacement can cause a checksum mismatch, which fails
closed. A pair of files cannot be replaced atomically as one transaction.
Levenshtein gating uses NFC Unicode CODE POINTS, d<=1 and at most three scored
model paths. Full UAX29 grapheme mode is explicitly unsupported here.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import logging
import math
import os
import tempfile
import unicodedata
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from features import FeatureBatchResult, FeaturePipeline, FeatureResult

try:
    import config as _project_config
except ModuleNotFoundError as exc:
    if exc.name != "config":
        raise
    _project_config = None

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
PIPELINE_VERSION = "1.0.0"
MODEL_FORMAT_VERSION = 1
DEFAULT_MAX_MODEL_BYTES = 256 * 1024 * 1024


class DecoderError(ValueError):
    """Invalid model, feature evidence, or decoding contract."""


class CalibrationError(DecoderError):
    """Calibration data or its split/view/schema is invalid."""


class ModelCompatibilityError(DecoderError):
    """A prototype book does not match the supplied feature schema and view."""


def _number(name: str, value: Any, low: float = 0, high: float = math.inf,
            *, integer: bool = False, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or (integer and type(value) is not int):
        raise DecoderError(f"{name} has an invalid numeric type")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise DecoderError(f"invalid {name}: {value!r}")


def _freeze(array: Any, dtype: Any = None) -> NDArray[Any]:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(item) for item in value)
    return value


def _strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise DecoderError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result
    def constant(value: str) -> Any:
        raise DecoderError(f"nonfinite JSON constant: {value}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise DecoderError("invalid model JSON") from exc


@dataclass(frozen=True, slots=True)
class DecoderOptions:
    recognition_mode: str = "auto"
    overlap_penalty: float = 1.0
    max_tokens: int = 256
    max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES

    def __post_init__(self) -> None:
        if self.recognition_mode not in ("auto", "whole_glyph", "multi_zone"):
            raise DecoderError("recognition_mode must be auto, whole_glyph or multi_zone")
        _number("overlap_penalty", self.overlap_penalty)
        _number("max_tokens", self.max_tokens, 1, 4096, integer=True)
        _number("max_model_bytes", self.max_model_bytes, 1, integer=True)


@dataclass(frozen=True, slots=True)
class _Settings:
    distance_metric: str = "regularized_mahalanobis"
    covariance_shrinkage: float = .10
    covariance_regularization: float = 1e-4
    covariance_condition_limit: float = 1e10
    min_covariance_samples: int = 3
    gamma_sim: float = 1.0
    glyph_top_k: int = 3
    beam_width: int = 3
    lexicon_top_paths: int = 3
    tau_oov: float = .85
    d_edit_max: int = 1
    eta: float = .10
    log_probability_floor: float = 1e-300
    lexicon_edit_unit: str = "codepoint"
    standardization_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.distance_metric not in ("regularized_mahalanobis", "standardized_euclidean"):
            raise DecoderError("unsupported distance_metric")
        _number("covariance_shrinkage", self.covariance_shrinkage, 0, 1)
        for name in ("covariance_regularization", "gamma_sim", "standardization_floor"):
            _number(name, getattr(self, name), 0, positive=True)
        _number("covariance_condition_limit", self.covariance_condition_limit, 1)
        _number("min_covariance_samples", self.min_covariance_samples, 2, integer=True)
        _number("glyph_top_k", self.glyph_top_k, 1, 4096, integer=True)
        _number("beam_width", self.beam_width, 1, 128, integer=True)
        _number("lexicon_top_paths", self.lexicon_top_paths, 1, 3, integer=True)
        if self.lexicon_top_paths > self.beam_width:
            raise DecoderError("lexicon_top_paths exceeds beam_width")
        _number("tau_oov", self.tau_oov, 0, 1)
        _number("d_edit_max", self.d_edit_max, 0, 1, integer=True)
        _number("eta", self.eta)
        _number("log_probability_floor", self.log_probability_floor, 0, 1, positive=True)
        if self.lexicon_edit_unit != "codepoint":
            raise DecoderError("only codepoint lexicon edits are supported; full UAX29 segmentation is required for graphemes")


@dataclass(frozen=True, slots=True)
class _Flags:
    probability_scoring: bool = True
    unicode_assembly: bool = True
    lexicon_gate: bool = True


def _lookup(source: Any, section: str, name: str, default: Any, missing: list[str]) -> Any:
    sentinel = object()
    group = source.get(section, sentinel) if isinstance(source, Mapping) else getattr(source, section, sentinel)
    value = group.get(name, sentinel) if isinstance(group, Mapping) else getattr(group, name, sentinel)
    if value is sentinel:
        missing.append(f"{section}.{name}")
        return default
    return value


def _resolve(config: Any, mode: str | None) -> tuple[Any, str, _Settings, _Flags, list[str]]:
    supplied = config.get("mode") if isinstance(config, Mapping) else getattr(config, "mode", None)
    selected = mode if mode is not None else supplied if supplied is not None else "B4"
    if not isinstance(selected, str) or selected.upper() not in ("B0", "B1", "B2", "B3", "B4"):
        raise DecoderError("mode must be B0, B1, B2, B3 or B4")
    selected = selected.upper()
    if supplied is not None and (not isinstance(supplied, str) or supplied.upper() != selected):
        raise DecoderError("mode conflicts with config.mode")
    if config is None and _project_config is not None:
        factory = getattr(_project_config, "get_config", None)
        config = factory(selected) if callable(factory) else getattr(_project_config, "DEFAULT_CONFIG", _project_config)
    defaults = _Settings()
    missing: list[str] = []
    values = {}
    for item in fields(defaults):
        section = "features" if item.name == "standardization_floor" else "decoder"
        fallback = getattr(defaults, item.name)
        if item.name == "distance_metric" and selected in ("B0", "B1", "B2"):
            fallback = "standardized_euclidean"
        values[item.name] = _lookup(config, section, item.name, fallback, missing)
    flags = {}
    for name, fallback in (("probability_scoring", selected in ("B3", "B4")),
                           ("unicode_assembly", True), ("lexicon_gate", selected == "B4")):
        value = _lookup(config, "flags", name, fallback, missing)
        if type(value) is not bool:
            raise DecoderError(f"flags.{name} must be bool")
        flags[name] = value
    resolved = _Flags(**flags)
    if resolved.lexicon_gate and not (resolved.probability_scoring and resolved.unicode_assembly):
        raise DecoderError("lexicon gate requires probability scoring and Unicode assembly")
    return config, selected, _Settings(**values), resolved, missing


def _validate_label(label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(label, str) or (not label and not allow_empty) or len(label) > 128:
        raise CalibrationError("class labels must be nonempty Unicode strings of at most 128 code points")
    label = unicodedata.normalize("NFC", label)
    if any(char not in _SI_ALLOWED for char in label):
        raise CalibrationError("supply actual Sinhala Unicode labels; numeric folder IDs and Roman guesses are invalid")
    return label


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    label: str
    feature: FeatureResult
    split: str = "development"


@dataclass(frozen=True, slots=True)
class PrototypeBook:
    labels: tuple[str, ...]
    means: NDArray[np.float64]
    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    covariances: NDArray[np.float64]
    covariance_available: NDArray[np.bool_]
    counts: NDArray[np.int64]
    schema_id: str
    view: str
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]
    precisions: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        labels = tuple(_validate_label(label, allow_empty=self.view in ("zone:upper", "zone:lower")) for label in self.labels)
        if not labels or len(labels) > 4096 or len(set(labels)) != len(labels):
            raise CalibrationError("prototype classes must be nonempty, unique and <=4096")
        means = np.asarray(self.means, dtype=np.float64)
        m = len(labels)
        if means.ndim != 2 or means.shape[0] != m or not 1 <= means.shape[1] <= 4096:
            raise CalibrationError("invalid prototype mean shape")
        dimension = means.shape[1]
        arrays = {
            "means": (means, (m, dimension)), "center": (self.center, (dimension,)),
            "scale": (self.scale, (dimension,)), "covariances": (self.covariances, (m, dimension, dimension)),
            "covariance_available": (self.covariance_available, (m,)), "counts": (self.counts, (m,)),
        }
        if np.asarray(self.covariance_available).dtype != np.bool_:
            raise CalibrationError("covariance_available must be boolean")
        if np.asarray(self.counts).dtype.kind not in "iu":
            raise CalibrationError("sample counts must be integers")
        for name, (array, shape) in arrays.items():
            data = np.asarray(array)
            if data.shape != shape or not np.isfinite(data).all():
                raise CalibrationError(f"invalid {name} shape or nonfinite values")
            dtype = np.bool_ if name == "covariance_available" else np.int64 if name == "counts" else np.float64
            object.__setattr__(self, name, _freeze(data, dtype))
        if np.any(self.scale <= 0) or np.any(self.counts <= 0):
            raise CalibrationError("scales and sample counts must be positive")
        if not isinstance(self.schema_id, str) or len(self.schema_id) != 64:
            raise CalibrationError("schema_id must be the feature schema SHA256")
        try:
            int(self.schema_id, 16)
        except ValueError as exc:
            raise CalibrationError("schema_id is not hexadecimal") from exc
        if self.view not in ("whole_glyph", "zone:upper", "zone:core", "zone:lower"):
            raise CalibrationError("invalid feature view")
        names = tuple(self.feature_names)
        if len(names) != dimension or len(set(names)) != dimension or any(not isinstance(name, str) for name in names):
            raise CalibrationError("feature names must be unique and match dimension")
        precision = np.broadcast_to(np.eye(dimension), (m, dimension, dimension)).copy()
        for index, covariance in enumerate(self.covariances):
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                raise CalibrationError("covariance matrices must be symmetric")
            try:
                factor = np.linalg.cholesky(covariance)
                if self.covariance_available[index]:
                    inverse_factor = np.linalg.solve(factor, np.eye(dimension))
                    precision[index] = inverse_factor.T @ inverse_factor
            except np.linalg.LinAlgError as exc:
                raise CalibrationError("covariance matrices must be positive definite") from exc
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "feature_names", names)
        if not np.isfinite(precision).all():
            raise CalibrationError("covariance inverse exceeds floating-point range")
        object.__setattr__(self, "precisions", _freeze(precision))
        object.__setattr__(self, "metadata", _immutable(self.metadata))

    @property
    def dimension(self) -> int:
        return self.means.shape[1]

    @property
    def model_view_key(self) -> str:
        return f"{self.schema_id}:{self.view}"

    def save(self, archive_path: str | Path, metadata_path: str | Path | None = None,
             *, overwrite: bool = False) -> tuple[Path, Path]:
        return _save_book(self, archive_path, metadata_path, overwrite=overwrite)

    @classmethod
    def load(cls, archive_path: str | Path, metadata_path: str | Path | None = None,
             *, max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES) -> PrototypeBook:
        return _load_book(archive_path, metadata_path, max_model_bytes=max_model_bytes)


class PrototypeCalibrationManager:
    def __init__(self, config: Any = None, *, mode: str | None = None) -> None:
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve(config, mode)

    def fit(self, samples: Iterable[CalibrationSample]) -> PrototypeBook:
        records = list(itertools.islice(iter(samples), 1_000_001))
        if not records or len(records) > 1_000_000:
            raise CalibrationError("calibration requires 1..1,000,000 labelled samples")
        first = records[0]
        if not isinstance(first, CalibrationSample) or not isinstance(first.feature, FeatureResult):
            raise CalibrationError("samples must contain CalibrationSample and FeatureResult")
        schema = first.feature.schema
        view = first.feature.metadata.get("view")
        key = f"{schema.schema_id}:{view}"
        vectors: list[NDArray[np.float64]] = []
        labels: list[str] = []
        splits: set[str] = set()
        for record in records:
            if not isinstance(record, CalibrationSample) or not isinstance(record.feature, FeatureResult):
                raise CalibrationError("every sample must be CalibrationSample with extracted features")
            if record.split not in ("development", "dev", "calibration"):
                raise CalibrationError("only explicit development/dev/calibration splits may fit prototypes")
            feature = record.feature
            if (feature.schema.schema_id != schema.schema_id or feature.schema.feature_names != schema.feature_names
                    or feature.metadata.get("model_view_key") != key):
                raise ModelCompatibilityError("calibration mixes incompatible feature schemas or views")
            if feature.metadata.get("empty_ink", False) or feature.metadata.get("empty_canvas", False):
                raise CalibrationError("blank calibration samples must be removed explicitly")
            vector = np.asarray(feature.vector, dtype=np.float64)
            if vector.shape != (schema.dimension,) or not np.isfinite(vector).all():
                raise CalibrationError("calibration vectors must be finite and match their schema")
            labels.append(_validate_label(record.label, allow_empty=view in ("zone:upper", "zone:lower")))
            vectors.append(vector)
            splits.add(record.split)
        raw = np.stack(vectors)
        center = raw.mean(axis=0)
        standard_deviation = raw.std(axis=0, ddof=0)
        scale = np.where(standard_deviation > self.settings.standardization_floor, standard_deviation, 1.0)
        standardized = (raw-center)/scale
        classes = tuple(sorted(set(labels)))
        if len(classes) > 4096:
            raise CalibrationError("too many prototype classes")
        dimension = schema.dimension
        if len(classes)*dimension*dimension*8 > DEFAULT_MAX_MODEL_BYTES:
            raise CalibrationError("class covariance tensor exceeds model resource budget")
        means = np.zeros((len(classes), dimension), np.float64)
        covariances = np.broadcast_to(np.eye(dimension), (len(classes), dimension, dimension)).copy()
        available = np.zeros(len(classes), bool)
        counts = np.zeros(len(classes), np.int64)
        reasons: list[str] = []
        label_array = np.asarray(labels)
        minimum = max(self.settings.min_covariance_samples, dimension+1)
        for index, label in enumerate(classes):
            examples = standardized[label_array == label]
            means[index] = examples.mean(axis=0)
            counts[index] = examples.shape[0]
            reason = "insufficient_class_samples"
            if counts[index] >= minimum:
                centered = examples-means[index]
                covariance = centered.T @ centered/(counts[index]-1)
                shrink = self.settings.covariance_shrinkage
                covariance = ((1-shrink)*covariance + shrink*np.diag(np.diag(covariance))
                              + self.settings.covariance_regularization*np.eye(dimension))
                if np.linalg.cond(covariance) <= self.settings.covariance_condition_limit:
                    try:
                        np.linalg.cholesky(covariance)
                        available[index] = True
                        covariances[index] = covariance
                        reason = "regularized_covariance_available"
                    except np.linalg.LinAlgError:
                        reason = "non_positive_covariance_fallback"
                else:
                    reason = "ill_conditioned_covariance_fallback"
            reasons.append(reason)
        metadata = {
            "format_version": MODEL_FORMAT_VERSION, "pipeline_version": PIPELINE_VERSION,
            "schema": schema.to_dict(), "view": view, "model_view_key": key,
            "calibration_splits": sorted(splits), "sample_count": len(records),
            "covariance_minimum_samples": minimum, "covariance_status": reasons,
            "calibration_settings": asdict(self.settings),
            "standardization": "global_population_std_constant_dimensions_scale1",
            "config_fingerprint": getattr(self.config, "fingerprint", None),
            "probabilities_are_calibrated_correctness": False,
        }
        return PrototypeBook(classes, means, center, scale, covariances, available, counts,
                             schema.schema_id, view, schema.feature_names, metadata)

    def calibrate_from_folders(self, root: str | Path, label_map: Mapping[str, str],
                               feature_pipeline: FeaturePipeline, *, split: str = "development",
                               view: str = "whole_glyph",
                               feature_builder: Callable[[Path, str], FeatureResult] | None = None) -> PrototypeBook:
        """Read isolated class folders using an EXPLICIT folder→Unicode map.

        Whole-glyph default: preprocess then height-normalize each entire crop,
        preserving margins. Other/canonical preparation uses feature_builder;
        the callback must reproduce query-view preparation, not fit a new scaler.
        All nonhidden class directories must be mapped; unreadable/blank images
        fail with their path rather than being silently skipped.
        """
        if not isinstance(label_map, Mapping) or not label_map:
            raise CalibrationError("an explicit nonempty folder-to-Unicode label_map is required")
        directory = Path(root)
        if not directory.is_dir():
            raise CalibrationError("calibration root is not a directory")
        if view != "whole_glyph" and feature_builder is None:
            raise CalibrationError("zone folder calibration needs an explicit feature_builder matching query preparation")
        subfolders = {entry.name for entry in directory.iterdir() if entry.is_dir() and not entry.name.startswith('.')}
        if subfolders != set(label_map):
            raise CalibrationError("label_map must match all nonhidden class folders exactly")
        if feature_builder is None:
            from preprocessing import PreprocessingPipeline
            from segmentation import SegmentationPipeline
            preprocessor = PreprocessingPipeline(self.config, mode=self.mode)
            normalizer = SegmentationPipeline(self.config, mode=self.mode).zone_segmenter
            def build(path: Path, declared_view: str) -> FeatureResult:
                mask, metadata = preprocessor.run(path)
                if metadata["status"] != "ok":
                    raise CalibrationError(f"calibration image has unusable ink: {path}")
                normalized, _ = normalizer.normalize(mask)
                return feature_pipeline.extract(normalized, view=declared_view)
            feature_builder = build
        samples: list[CalibrationSample] = []
        for folder, label in sorted(label_map.items()):
            if not isinstance(folder, str) or folder in ("", ".", "..") or Path(folder).name != folder:
                raise CalibrationError("class folder keys must be single path components")
            _validate_label(label, allow_empty=view in ("zone:upper", "zone:lower"))
            paths = sorted(path for path in (directory/folder).iterdir()
                           if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            if not paths:
                raise CalibrationError(f"class folder has no supported images: {folder}")
            for path in paths:
                samples.append(CalibrationSample(label, feature_builder(path, view), split))
        return self.fit(samples)

    @staticmethod
    def save(book: PrototypeBook, archive_path: str | Path, metadata_path: str | Path | None = None,
             *, overwrite: bool = False) -> tuple[Path, Path]:
        return book.save(archive_path, metadata_path, overwrite=overwrite)

    @staticmethod
    def load(archive_path: str | Path, metadata_path: str | Path | None = None,
             *, max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES) -> PrototypeBook:
        return PrototypeBook.load(archive_path, metadata_path, max_model_bytes=max_model_bytes)


def _book_manifest(book: PrototypeBook) -> dict[str, Any]:
    return {"format_version": MODEL_FORMAT_VERSION, "labels": list(book.labels),
            "schema_id": book.schema_id, "view": book.view, "model_view_key": book.model_view_key,
            "dimension": book.dimension, "feature_names": list(book.feature_names),
            "metadata": _plain(book.metadata)}


def _atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # Hard linking is an atomic no-overwrite commit on this filesystem.
            os.link(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_book(book: PrototypeBook, archive_path: str | Path, metadata_path: str | Path | None,
               *, overwrite: bool) -> tuple[Path, Path]:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    archive = Path(archive_path)
    manifest_path = Path(metadata_path) if metadata_path is not None else archive.with_suffix(".json")
    if archive.resolve() == manifest_path.resolve():
        raise DecoderError("archive and JSON manifest must be separate files")
    if not overwrite and (archive.exists() or manifest_path.exists()):
        raise FileExistsError("prototype archive or manifest already exists")
    manifest = _book_manifest(book)
    embedded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, means=book.means, center=book.center, scale=book.scale,
                        covariances=book.covariances, covariance_available=book.covariance_available,
                        counts=book.counts, metadata=np.frombuffer(embedded, dtype=np.uint8))
    payload = buffer.getvalue()
    manifest["archive_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False).encode("utf-8")
    _atomic_write(archive, payload, overwrite=overwrite)
    try:
        _atomic_write(manifest_path, manifest_bytes, overwrite=overwrite)
    except Exception:
        # A new incomplete save can be removed without touching earlier models.
        # Overwrites fail closed through the checksum until a complete retry.
        if not overwrite:
            archive.unlink(missing_ok=True)
        raise
    return archive, manifest_path


def _load_book(archive_path: str | Path, metadata_path: str | Path | None,
               *, max_model_bytes: int) -> PrototypeBook:
    _number("max_model_bytes", max_model_bytes, 1, integer=True)
    archive = Path(archive_path)
    manifest_path = Path(metadata_path) if metadata_path is not None else archive.with_suffix(".json")
    required = {"format_version", "labels", "schema_id", "view", "model_view_key", "dimension", "feature_names", "metadata"}
    try:
        if manifest_path.stat().st_size > min(max_model_bytes, 4*1024*1024):
            raise DecoderError("prototype manifest exceeds resource budget")
        manifest = _strict_json(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != required | {"archive_sha256"}:
            raise DecoderError("invalid prototype manifest fields")
        if type(manifest["format_version"]) is not int or manifest["format_version"] != MODEL_FORMAT_VERSION:
            raise DecoderError("unsupported prototype manifest format")
        labels = manifest["labels"]
        dimension = manifest["dimension"]
        names = manifest["feature_names"]
        if (not isinstance(labels, list) or not 1 <= len(labels) <= 4096
                or any(not isinstance(label, str) or len(label) > 128 for label in labels)
                or len(set(labels)) != len(labels)):
            raise DecoderError("invalid prototype class labels")
        if type(dimension) is not int or not 1 <= dimension <= 4096:
            raise DecoderError("invalid prototype dimension")
        if (not isinstance(names, list) or len(names) != dimension
                or any(not isinstance(name, str) for name in names) or len(set(names)) != dimension):
            raise DecoderError("invalid prototype feature names")
        schema_id = manifest["schema_id"]
        checksum = manifest["archive_sha256"]
        for value in (schema_id, checksum):
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise DecoderError("invalid prototype SHA256 field")
        view = manifest["view"]
        if (view not in ("whole_glyph", "zone:upper", "zone:core", "zone:lower")
                or manifest["model_view_key"] != f"{schema_id}:{view}"):
            raise DecoderError("invalid prototype view or model key")
        if any(not label for label in labels) and view not in ("zone:upper", "zone:lower"):
            raise DecoderError("empty class labels require an explicit upper/lower null-mark model")
        if not isinstance(manifest["metadata"], dict):
            raise DecoderError("invalid prototype metadata object")
        classes = len(labels)
        contracts = {
            "means.npy": ((classes, dimension), np.dtype(np.float64)),
            "center.npy": ((dimension,), np.dtype(np.float64)),
            "scale.npy": ((dimension,), np.dtype(np.float64)),
            "covariances.npy": ((classes, dimension, dimension), np.dtype(np.float64)),
            "covariance_available.npy": ((classes,), np.dtype(np.bool_)),
            "counts.npy": ((classes,), np.dtype(np.int64)),
        }
        expected_numeric_bytes = sum(math.prod(shape)*dtype.itemsize for shape, dtype in contracts.values())
        if expected_numeric_bytes > max_model_bytes:
            raise DecoderError("declared prototype arrays exceed resource budget")
        if archive.stat().st_size > max_model_bytes:
            raise DecoderError("prototype archive exceeds resource budget")
        with archive.open("rb") as handle:
            payload = handle.read(max_model_bytes+1)
    except OSError as exc:
        raise DecoderError("prototype files could not be read") from exc
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, DecoderError):
            raise
        raise DecoderError("invalid strict JSON prototype manifest") from exc
    if len(payload) > max_model_bytes or hashlib.sha256(payload).hexdigest() != checksum:
        raise DecoderError("prototype archive SHA256 mismatch or resource limit exceeded")
    expected = {"means", "center", "scale", "covariances", "covariance_available", "counts", "metadata"}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            infos = zipped.infolist()
            if len(infos) != len(expected) or {info.filename for info in infos} != {name+".npy" for name in expected}:
                raise DecoderError("prototype NPZ has missing, duplicate or unexpected arrays")
            if sum(info.file_size for info in infos) > max_model_bytes:
                raise DecoderError("expanded prototype archive exceeds resource budget")
            for info in infos:
                with zipped.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=10000)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member, max_header_size=10000)
                    else:
                        raise DecoderError("unsupported NPY header version")
                    if any(type(side) is not int or side < 0 for side in shape):
                        raise DecoderError("invalid NPY shape")
                    if (dtype.hasobject or dtype.kind not in "fbiu" or len(shape) > 3
                            or math.prod(shape)*dtype.itemsize > max_model_bytes):
                        raise DecoderError("unsafe prototype dtype, dimensions or advertised allocation")
                    if info.filename == "metadata.npy":
                        if dtype != np.dtype(np.uint8) or len(shape) != 1 or not 1 <= shape[0] <= 4*1024*1024:
                            raise DecoderError("invalid embedded prototype metadata header")
                    else:
                        expected_shape, expected_dtype = contracts[info.filename]
                        # Permit byte-order portability without allowing a cast
                        # from smaller/wider numerical types after allocation.
                        if (shape != expected_shape or dtype.kind != expected_dtype.kind
                                or dtype.itemsize != expected_dtype.itemsize):
                            raise DecoderError(f"prototype array header violates {info.filename} shape/dtype contract")
                    if info.file_size-member.tell() != math.prod(shape)*dtype.itemsize:
                        raise DecoderError("NPY payload length disagrees with its declared shape")
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name], copy=True) for name in expected}
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
        if isinstance(exc, DecoderError):
            raise
        raise DecoderError("invalid pickle-free prototype archive") from exc
    metadata_array = arrays.pop("metadata")
    if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1 or metadata_array.size > 4*1024*1024:
        raise DecoderError("invalid embedded prototype metadata")
    try:
        embedded = _strict_json(metadata_array.tobytes().decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, DecoderError):
            raise
        raise DecoderError("invalid embedded strict JSON prototype metadata") from exc
    outer = dict(manifest)
    outer.pop("archive_sha256", None)
    if embedded != outer:
        raise DecoderError("prototype archive and manifest describe different model generations")
    if not isinstance(embedded, dict) or set(embedded) != required:
        raise DecoderError("invalid prototype metadata fields")
    book = PrototypeBook(tuple(embedded["labels"]), **arrays, schema_id=embedded["schema_id"],
                         view=embedded["view"], feature_names=tuple(embedded["feature_names"]), metadata=embedded["metadata"])
    if book.dimension != embedded["dimension"] or book.model_view_key != embedded["model_view_key"]:
        raise DecoderError("prototype dimensions/view disagree with manifest")
    return book


@dataclass(frozen=True, slots=True)
class ClassPrediction:
    label: str
    distance: float
    probability: float | None
    log_probability: float | None


@dataclass(frozen=True, slots=True)
class ScoringResult:
    labels: tuple[str, ...]
    distances: NDArray[np.float64]
    amplitudes: NDArray[np.float64] | None
    probabilities: NDArray[np.float64] | None
    top_candidates: tuple[ClassPrediction, ...]
    confidence: float | None
    metadata: Mapping[str, Any]


class PrototypeScorer:
    """Numerically stable standardized distances and real L2-normalized states."""

    def __init__(self, book: PrototypeBook, config: Any = None, *, mode: str | None = None) -> None:
        if not isinstance(book, PrototypeBook):
            raise TypeError("book must be PrototypeBook")
        self.book = book
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve(config, mode)

    def score(self, feature: FeatureResult) -> ScoringResult:
        if not isinstance(feature, FeatureResult):
            raise TypeError("feature must be FeatureResult")
        book = self.book
        if (feature.schema.schema_id != book.schema_id or tuple(feature.schema.feature_names) != book.feature_names
                or feature.metadata.get("model_view_key") != book.model_view_key):
            raise ModelCompatibilityError("prototype and query schema/view do not match")
        vector = np.asarray(feature.vector, dtype=np.float64)
        if vector.shape != (book.dimension,) or not np.isfinite(vector).all():
            raise DecoderError("query feature must be finite and match prototype dimensions")
        if feature.metadata.get("empty_ink", False) or feature.metadata.get("empty_canvas", False):
            raise DecoderError("blank candidate ink cannot be scored as a character")
        # Extended accumulation and row rescaling avoid squaring huge doubles.
        query = (vector.astype(np.longdouble)-book.center)/book.scale
        differences = query[None, :]-book.means
        magnitude = np.max(np.abs(differences), axis=1)
        normalized = np.divide(differences, magnitude[:, None], out=np.zeros_like(differences),
                               where=magnitude[:, None] > 0)
        euclidean_squared = np.sum(normalized*normalized, axis=1)
        use_covariance = book.covariance_available & (self.settings.distance_metric == "regularized_mahalanobis")
        squared = euclidean_squared.copy()
        if np.any(use_covariance):
            selected = normalized[use_covariance]
            squared[use_covariance] = np.einsum("md,mdk,mk->m", selected,
                                               book.precisions[use_covariance].astype(np.longdouble), selected)
        exact_distances = magnitude*np.sqrt(np.maximum(squared, 0))
        if not np.isfinite(exact_distances).all():
            raise DecoderError("distance calculation exceeded the platform's extended floating-point range")
        maximum = np.finfo(np.float64).max
        distances = np.asarray(np.minimum(exact_distances, maximum), np.float64)
        order = np.lexsort((np.asarray(book.labels), exact_distances))
        probabilities: NDArray[np.float64] | None = None
        amplitudes: NDArray[np.float64] | None = None
        logs: NDArray[np.float64] | None = None
        if self.flags.probability_scoring:
            shifted = -np.longdouble(self.settings.gamma_sim)*(exact_distances-exact_distances.min())
            with np.errstate(under="ignore"):
                weights = np.exp(shifted)
                normalization = np.sum(weights*weights)
                amplitudes = np.asarray(weights/np.sqrt(normalization), np.float64)
            probabilities = amplitudes*amplitudes
            log_values = 2*shifted-np.log(normalization)
            logs = np.asarray(np.maximum(log_values, -maximum), np.float64)
        predictions = tuple(ClassPrediction(book.labels[index], float(distances[index]),
                                             None if probabilities is None else float(probabilities[index]),
                                             None if logs is None else float(logs[index]))
                            for index in order[:self.settings.glyph_top_k])
        confidence = None if probabilities is None else float(probabilities.max())
        metadata = {
            "mode": self.mode, "model_view_key": book.model_view_key,
            "distance_metric": self.settings.distance_metric,
            "class_distance_metrics": ["regularized_mahalanobis" if value else "standardized_euclidean" for value in use_covariance],
            "distance_display_saturated": bool(np.any(exact_distances > maximum)),
            "probability_scoring": self.flags.probability_scoring,
            "probabilities_are_calibrated_correctness": False,
            "normalization": "real_L2_amplitudes_squared" if probabilities is not None else "none_nearest_prototype",
            "gamma_sim": self.settings.gamma_sim, "log_probability_saturation": "negative_float64_max",
        }
        return ScoringResult(book.labels, _freeze(distances), None if amplitudes is None else _freeze(amplitudes),
                             None if probabilities is None else _freeze(probabilities), predictions, confidence, _immutable(metadata))


_unicode_config = _project_config

_SI_CONSONANTS = frozenset(
    getattr(_unicode_config, "CONSONANTS", {}).values()
) or frozenset(chr(value) for value in (
    *range(0x0D9A, 0x0DB2), 0x0DB3, *range(0x0DB4, 0x0DBC),
    0x0DBD, *range(0x0DC0, 0x0DC7),
))
_SI_VOWELS = frozenset(
    getattr(_unicode_config, "INDEPENDENT_VOWELS", {}).values()
) or frozenset(map(chr, range(0x0D85, 0x0D97)))
_SI_FINALS = frozenset(
    getattr(_unicode_config, "FINAL_SIGNS", {}).values()
) or frozenset(("\u0d81", "\u0d82", "\u0d83"))
_SI_SIGNS = frozenset(filter(None, getattr(
    _unicode_config, "VOWEL_SIGNS", {}
).values())) or frozenset(map(chr, (
    0x0DCF, 0x0DD0, 0x0DD1, 0x0DD2, 0x0DD3, 0x0DD4, 0x0DD6,
    0x0DD8, 0x0DD9, 0x0DDA, 0x0DDB, 0x0DDC, 0x0DDD, 0x0DDE,
    0x0DDF, 0x0DF2, 0x0DF3,
)))
_SI_VIRAMA = "\u0dca"
_SI_JOINERS = frozenset(("\u200c", "\u200d"))
_SI_E = "\u0dd9"
_SI_AI = "\u0ddb"
_SI_AA = "\u0dcf"
_SI_GAYA = "\u0ddf"
_SI_PREFIXES = frozenset(getattr(
    _unicode_config, "CONJUNCT_PREFIXES", {}
).values()) or frozenset(("\u0dbb\u0dca\u200d",))
_SI_SUFFIXES = frozenset(getattr(
    _unicode_config, "CONJUNCT_SUFFIXES", {}
).values()) or frozenset(("\u0dca\u200d\u0dba", "\u0dca\u200d\u0dbb"))
_SI_PREBASE_SIGNS = frozenset((_SI_E, "\u0dda", _SI_AI))
_SI_ALLOWED = _SI_CONSONANTS | _SI_VOWELS | _SI_FINALS | _SI_SIGNS | _SI_JOINERS | {_SI_VIRAMA}


class UnicodeAssemblyError(ValueError):
    """A classifier-label combination has no supported Sinhala assembly."""


class SinhalaAssembler:
    """Compose actual Unicode labels; never infer numeric or Roman aliases.

    ``compose`` consumes one consonant cluster or independent-vowel grapheme
    and its linked zone labels. ``assemble_visual`` consumes left-to-right
    classifier tokens: standalone kombuwa-family tokens precede their base;
    other separate vowel marks follow their base. Use complete Unicode
    grapheme labels for already logically ordered training labels.

    Explicit ZWJ/ZWNJ are preserved. An unjoined virama+consonant sequence is
    also structurally accepted. No joiners are inserted implicitly.
    """

    def __init__(self, *, max_tokens: int = 4096, max_token_length: int = 128) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_token_length, bool) or not isinstance(max_token_length, int) or max_token_length < 1:
            raise ValueError("max_token_length must be a positive integer")
        self.max_tokens = max_tokens
        self.max_token_length = max_token_length

    def _text(self, value: str, *, label: str, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{label} must be a Unicode string")
        if not value and not allow_empty:
            raise UnicodeAssemblyError(f"{label} cannot be empty")
        if len(value) > self.max_token_length:
            raise UnicodeAssemblyError(f"{label} exceeds max_token_length")
        normalized = unicodedata.normalize("NFC", value)
        if any(char not in _SI_ALLOWED for char in normalized):
            raise UnicodeAssemblyError(f"{label} must contain supported Sinhala Unicode characters")
        return normalized

    def _tokens(self, values: Sequence[str], *, label: str) -> list[str]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{label} must be a sequence of Unicode label strings")
        if len(values) > self.max_tokens:
            raise UnicodeAssemblyError(f"{label} exceeds max_tokens")
        return [self._text(value, label=f"{label}[{index}]") for index, value in enumerate(values)]

    @staticmethod
    def _root_and_marks(base: str) -> tuple[str, str, bool]:
        """Split one base grapheme without treating a terminal virama as a link."""
        if not base or base[0] not in (_SI_CONSONANTS | _SI_VOWELS):
            raise UnicodeAssemblyError("a grapheme must begin with a Sinhala consonant or independent vowel")
        independent = base[0] in _SI_VOWELS
        end = 1
        if not independent:
            while end < len(base) and base[end] == _SI_VIRAMA:
                following = end + 1
                if following < len(base) and base[following] in _SI_JOINERS:
                    following += 1
                if following >= len(base) or base[following] not in _SI_CONSONANTS:
                    break
                end = following + 1
        marks = base[end:]
        if any(char in (_SI_CONSONANTS | _SI_VOWELS) for char in marks):
            raise UnicodeAssemblyError("base must be one grapheme or an explicit virama-linked consonant cluster")
        return base[:end], marks, independent

    @staticmethod
    def _resolve_marks(marks: Sequence[str], *, independent: bool) -> str:
        """Resolve canonical sign components while retaining every detection."""
        finals: list[str] = []
        components: list[str] = []
        joiner = ""
        for mark in marks:
            if any(char not in (_SI_SIGNS | _SI_FINALS | _SI_JOINERS | {_SI_VIRAMA}) for char in mark):
                raise UnicodeAssemblyError("zone marks cannot contain base letters")
            # Retain source sequence for the explicit virama+joiner check, then
            # decompose signs canonically so e+aa and composed o agree.
            previous = ""
            for char in unicodedata.normalize("NFD", mark):
                if char in _SI_FINALS:
                    finals.append(char)
                elif char in _SI_JOINERS:
                    if previous != _SI_VIRAMA or joiner:
                        raise UnicodeAssemblyError("a joiner must follow exactly one explicit virama")
                    joiner = char
                elif char in _SI_SIGNS or char == _SI_VIRAMA:
                    components.append(char)
                else:
                    raise UnicodeAssemblyError("unsupported dependent mark")
                previous = char
        if len(finals) > 1:
            raise UnicodeAssemblyError("duplicate or conflicting final-sign detections")
        if independent and (components or joiner):
            raise UnicodeAssemblyError("an independent vowel cannot carry a dependent vowel sign or virama")
        counts = Counter(components)
        if any(count > 1 for count in counts.values()):
            raise UnicodeAssemblyError("duplicate vowel-component detections")
        component_set = frozenset(components)
        compound = {
            frozenset((_SI_E, _SI_VIRAMA)): "\u0dda",
            frozenset((_SI_E, _SI_AA)): "\u0ddc",
            frozenset((_SI_E, _SI_AA, _SI_VIRAMA)): "\u0ddd",
            frozenset((_SI_E, _SI_GAYA)): "\u0dde",
        }
        if not components:
            vowel = ""
        elif len(components) == 1:
            vowel = components[0]
        elif component_set in compound:
            vowel = compound[component_set]
        else:
            raise UnicodeAssemblyError("contradictory dependent vowel signs")
        if joiner and vowel != _SI_VIRAMA:
            raise UnicodeAssemblyError("a trailing joiner is supported only on an explicit dead consonant")
        return vowel + joiner + "".join(finals)

    def compose(
        self,
        base: str,
        *,
        upper: Sequence[str] = (),
        lower: Sequence[str] = (),
        core_marks: Sequence[str] = (),
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Return NFC ``prefix + cluster + vowel + final`` or raise.

        Prefix/suffix must be the actual Unicode repaya/yansaya/rakaransaya
        sequences declared in config.py, rather than their Roman names.
        Existing marks on a complete base are combined with linked marks;
        contradictory and duplicate detections are rejected, never discarded.
        """
        base = self._text(base, label="base")
        prefix = self._text(prefix, label="prefix", allow_empty=True)
        suffix = self._text(suffix, label="suffix", allow_empty=True)
        if prefix and prefix not in _SI_PREFIXES:
            raise UnicodeAssemblyError("prefix is not a configured Unicode conjunct prefix")
        if suffix and suffix not in _SI_SUFFIXES:
            raise UnicodeAssemblyError("suffix is not a configured Unicode conjunct suffix")
        root, existing, independent = self._root_and_marks(base)
        if independent and (prefix or suffix):
            raise UnicodeAssemblyError("conjuncts require a consonant base")
        zone_tokens = (
            self._tokens(core_marks, label="core_marks")
            + self._tokens(upper, label="upper")
            + self._tokens(lower, label="lower")
        )
        if len(zone_tokens) > self.max_tokens:
            raise UnicodeAssemblyError("combined zone labels exceed max_tokens")
        marks = ([existing] if existing else []) + zone_tokens
        resolved = self._resolve_marks(marks, independent=independent)
        result = unicodedata.normalize("NFC", prefix + root + suffix + resolved)
        # Reparse after adding declared cluster pieces. This catches any
        # invalid configuration override rather than trusting the table name.
        check_root, check_marks, check_independent = self._root_and_marks(result)
        if check_root != prefix + root + suffix:
            raise UnicodeAssemblyError("conjunct pieces do not form one explicit consonant cluster")
        self._resolve_marks([check_marks] if check_marks else [], independent=check_independent)
        return result

    def _validate_logical_word(self, text: str) -> str:
        """Validate consecutive structural graphemes in linear time."""
        output: list[str] = []
        start = 0
        while start < len(text):
            if text[start] not in (_SI_CONSONANTS | _SI_VOWELS):
                raise UnicodeAssemblyError("orphan dependent mark or joiner in visual token sequence")
            end = start + 1
            if text[start] in _SI_CONSONANTS:
                while end < len(text) and text[end] == _SI_VIRAMA:
                    following = end + 1
                    if following < len(text) and text[following] in _SI_JOINERS:
                        following += 1
                    if following >= len(text) or text[following] not in _SI_CONSONANTS:
                        break
                    end = following + 1
            while end < len(text) and text[end] not in (_SI_CONSONANTS | _SI_VOWELS):
                end += 1
            output.append(self.compose(text[start:end]))
            start = end
        return "".join(output)

    def assemble_visual(self, tokens: Sequence[str]) -> str:
        """Assemble bounded visual-order glyph labels into a Unicode word.

        Standalone e/ee/ai signs are pending pre-base signs. Exactly two
        separate e tokens before a base mean the explicitly detected visual
        double-kombuwa (ai); ``compose`` never asserts e+e canonical equality.
        Suffixes and tokenized virama/joiner/consonant sequences retain their
        explicit logical links. Empty input produces an empty word.
        """
        labels = self._tokens(tokens, label="tokens")
        output: list[str] = []
        pending: list[str] = []
        for token in labels:
            if token in _SI_PREBASE_SIGNS:
                pending.append(token)
                if len(pending) > 2 or (len(pending) == 2 and pending != [_SI_E, _SI_E]):
                    raise UnicodeAssemblyError("conflicting pre-base vowel detections")
                continue
            begins_base = token[0] in (_SI_CONSONANTS | _SI_VOWELS)
            if pending:
                if token in _SI_PREFIXES:
                    # Repaya is an explicit prefix fragment; the pending
                    # kombuwa belongs after the following complete cluster.
                    output.append(token)
                    continue
                if not begins_base:
                    raise UnicodeAssemblyError("a visual pre-base sign must be followed by its base")
                mark = _SI_AI if len(pending) == 2 else pending[0]
                output.append(self.compose(token, core_marks=(mark,)))
                pending.clear()
            else:
                if begins_base:
                    # Reject multi-grapheme labels but allow a terminal
                    # virama+joiner to link to the next classifier token.
                    self.compose(token)
                output.append(token)
        if pending:
            raise UnicodeAssemblyError("orphan pre-base vowel sign at end of visual token sequence")
        logical = unicodedata.normalize("NFC", "".join(output))
        return self._validate_logical_word(logical)


def _nfc_word(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("word must be str")
    return unicodedata.normalize("NFC", text)


def _validate_confidence(value: float | None, name: str = "confidence") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return normalized


@dataclass(frozen=True, slots=True)
class WordPath:
    """A complete word hypothesis and its model-generated log probability.

    Log probability is finite and <= 0, or None for nearest-prototype baselines.
    ``confidence`` is optional
    and must be supplied by the recognition/calibration policy. It is never
    guessed from the word's length or lexicon membership.
    """
    text: str
    log_probability: float | None
    confidence: float | None = None
    candidate_ids: tuple[str, ...] = ()
    hypothesis_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _nfc_word(self.text))
        value = self.log_probability
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("log_probability must be a real number or None")
            value = float(value)
            if not math.isfinite(value) or value > 0.0:
                raise ValueError("log_probability must be finite and <= 0")
            object.__setattr__(self, "log_probability", value)
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))

    @property
    def probability(self) -> float | None:
        """May underflow to zero; decisions use the original log probability."""
        return None if self.log_probability is None else math.exp(self.log_probability)


@dataclass(frozen=True, slots=True)
class LexiconDecision:
    raw_text: str
    text: str
    corrected: bool
    bypassed: bool
    reason: str
    edit_distance: int | None
    score: float | None
    selected_path: WordPath
    metadata: Mapping[str, Any] = field(default_factory=dict)


def levenshtein_distance(left: str, right: str) -> int:
    """Exact NFC-normalized code-point Levenshtein distance, linear memory.

    This full-distance helper is for reporting/general comparisons. The gate
    uses its linear-time <=1 specialization rather than quadratic DP.
    """
    left, right = _nfc_word(left), _nfc_word(right)
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, character in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (character != other)))
        previous = current
    return previous[-1]


def _distance_at_most_one(left: str, right: str, limit: int = 1) -> int | None:
    """Return exact distance when <= limit (0/1), otherwise None; O(n)."""
    if left == right:
        return 0
    if limit == 0 or abs(len(left) - len(right)) > 1:
        return None
    if len(left) == len(right):
        mismatches = 0
        for first, second in zip(left, right):
            if first != second:
                mismatches += 1
                if mismatches > 1:
                    return None
        return mismatches
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    first = second = 0
    mismatch_seen = False
    while first < len(shorter) and second < len(longer):
        if shorter[first] == longer[second]:
            first += 1
            second += 1
        elif mismatch_seen:
            return None
        else:
            mismatch_seen = True
            second += 1
    return 1


class LexiconGate:
    """OOV-safe edit-distance gate over at most three model-generated paths.

    Raw output is always the first supplied path. If recognition confidence
    reaches ``tau_oov`` inclusively, it is returned unchanged, even when absent
    from the lexicon. Otherwise a candidate is eligible only when it is BOTH
    a supplied top path and a lexicon entry, and its distance from raw is at
    most ``d_edit_max``. Eligible paths minimize
    ``distance(raw, candidate) - eta * candidate.log_probability``.

    Missing/empty lexicons, missing eligible paths, and empty predictions all
    retain raw text. A missing confidence does not invent a confidence: the
    gate can still choose among scored nearby model paths and records that
    the confidence shield could not be evaluated. Multiword lexicon entries
    are rejected; preserve word boundaries upstream rather than stripping
    whitespace or joiners. Grapheme mode is intentionally unsupported without
    a pinned, full UAX #29 implementation.
    """

    def __init__(
        self,
        lexicon: Iterable[str] | None,
        *,
        tau_oov: float = 0.85,
        d_edit_max: int = 1,
        eta: float = 0.10,
        top_paths: int = 3,
        enabled: bool = True,
        edit_unit: str = "codepoint",
    ) -> None:
        self.tau_oov = _validate_confidence(tau_oov, "tau_oov")
        if self.tau_oov is None:
            raise TypeError("tau_oov cannot be None")
        if type(d_edit_max) is not int or d_edit_max not in (0, 1):
            raise ValueError("OOV-safe lexicon gate supports d_edit_max=0 or 1")
        if type(top_paths) is not int or not 1 <= top_paths <= 3:
            raise ValueError("top_paths must be an integer between one and three")
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if isinstance(eta, bool) or not isinstance(eta, Real):
            raise TypeError("eta must be a real number")
        if not math.isfinite(float(eta)) or eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        if edit_unit != "codepoint":
            raise ValueError("edit_unit='codepoint' is supported; grapheme mode requires "
                             "a pinned full UAX #29 segmenter")
        self.d_edit_max = d_edit_max
        self.eta = float(eta)
        self.top_paths = top_paths
        self.enabled = enabled
        self.edit_unit = edit_unit
        self.available = lexicon is not None
        if isinstance(lexicon, (str, bytes)):
            raise TypeError("lexicon must be an iterable of words, not one string")
        words: set[str] = set()
        for entry in (() if lexicon is None else lexicon):
            normalized = _nfc_word(entry)
            if not normalized or any(character.isspace() for character in normalized):
                raise ValueError("lexicon entries must be nonempty single words without whitespace")
            words.add(normalized)
        self.lexicon = frozenset(words)

    def apply(
        self,
        paths: Sequence[WordPath],
        *,
        confidence: float | None = None,
    ) -> LexiconDecision:
        """Return a deterministic decision, preserving recognition evidence."""
        if not paths:
            raise ValueError("at least one WordPath, including an empty raw path, is required")
        if not all(isinstance(path, WordPath) for path in paths):
            raise TypeError("paths must contain WordPath instances")
        raw = paths[0]
        effective_confidence = _validate_confidence(confidence)
        confidence_source = "argument" if confidence is not None else "raw_path"
        if effective_confidence is None:
            effective_confidence = raw.confidence
            if effective_confidence is None:
                confidence_source = "unavailable"
        metadata: dict[str, Any] = {
            "confidence": effective_confidence,
            "confidence_source": confidence_source,
            "confidence_shield_evaluated": effective_confidence is not None,
            "tau_oov": self.tau_oov,
            "edit_unit": self.edit_unit,
            "lexicon_available": self.available,
            "lexicon_size": len(self.lexicon),
            "raw_in_lexicon": raw.text in self.lexicon,
            "considered_paths": 0,
            "eligible_paths": 0,
            "no_generated_lexicon_neighbours": True,
        }

        def unchanged(reason: str, bypassed: bool = True) -> LexiconDecision:
            return LexiconDecision(raw.text, raw.text, False, bypassed, reason,
                                   0, None, raw, metadata)

        if not self.enabled:
            return unchanged("disabled")
        if not raw.text:
            return unchanged("empty_prediction")
        if effective_confidence is not None and effective_confidence >= self.tau_oov:
            return unchanged("high_confidence_oov_shield")
        if raw.log_probability is None:
            return unchanged("model_probability_unavailable")
        if not self.available:
            return unchanged("lexicon_unavailable")
        if not self.lexicon:
            return unchanged("lexicon_empty")
        eligible: list[tuple[float, int, str, int, WordPath]] = []
        for index, path in enumerate(paths[:self.top_paths]):
            metadata["considered_paths"] += 1
            if path.log_probability is None:
                continue
            if path.text not in self.lexicon:
                continue
            if abs(len(raw.text) - len(path.text)) > self.d_edit_max:
                continue
            distance = _distance_at_most_one(raw.text, path.text, self.d_edit_max)
            if distance is None:
                continue
            score = distance - self.eta * path.log_probability
            if not math.isfinite(score):
                raise ValueError("lexicon score overflow; log probability or eta is too large")
            eligible.append((score, distance, path.text, index, path))
        metadata["eligible_paths"] = len(eligible)
        if not eligible:
            return unchanged("no_eligible_model_path", bypassed=False)
        score, distance, _, _, selected = min(eligible, key=lambda entry: entry[:4])
        changed = selected.text != raw.text
        return LexiconDecision(raw.text, selected.text, changed, False,
                               "eligible_path_selected" if changed else "raw_path_selected",
                               distance, score, selected, metadata)


@dataclass(frozen=True, slots=True)
class DecodedWord:
    word_id: int
    raw_text: str
    text: str
    confidence: float | None
    paths: tuple[WordPath, ...]
    lexicon_decision: LexiconDecision | None
    status: str
    scores: Mapping[str, ScoringResult]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DecodeResult:
    words: tuple[DecodedWord, ...]
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _TokenOption:
    label: str
    evidence: np.longdouble
    log_probability: np.longdouble | None
    confidence: float | None
    scored: bool = True
    max_class_probability: float | None = None


@dataclass(frozen=True, slots=True)
class _BeamState:
    tokens: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    evidence: np.longdouble = np.longdouble(0)
    log_probability: np.longdouble = np.longdouble(0)
    confidence_sum: float = 0.
    scored_tokens: int = 0
    overlap: int = 0
    selected_probability_sum: float = 0.


def _bounded_float(value: np.longdouble) -> float:
    """Finite display/storage value; decisions retain extended precision."""
    maximum = np.longdouble(np.finfo(np.float64).max)
    if not np.isfinite(value):
        raise DecoderError("nonfinite accumulated decoding evidence")
    return float(np.clip(value, -maximum, maximum))


class DecoderPipeline:
    """Bounded candidate-path decoding, explicit Unicode assembly and OOV gate.

    ``models`` is one whole-glyph book or a mapping keyed by each book's view.
    Auto mode prefers a whole-glyph book; multi_zone requires a core book and
    a matching upper/lower book whenever the respective zone contains ink.
    An empty-string class is an explicit learned null-mark class allowed only
    in upper/lower books. Empty images are neutral coverage transitions.

    Per-word failures are reported as status='rejected' with diagnostics;
    malformed batches, schemas and model-view mismatches raise DecoderError.
    A rejected word stays in ``words`` in its original position. ``text`` joins
    successful word texts with spaces; applications should inspect statuses.
    """

    def __init__(self, models: PrototypeBook | Mapping[str, PrototypeBook],
                 lexicon: Iterable[str] | None = None, config: Any = None, *,
                 mode: str | None = None, options: DecoderOptions | None = None) -> None:
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve(config, mode)
        self.options = DecoderOptions() if options is None else options
        if not isinstance(self.options, DecoderOptions):
            raise TypeError("options must be DecoderOptions")
        books = {models.view: models} if isinstance(models, PrototypeBook) else dict(models)
        if not books or any(not isinstance(book, PrototypeBook) or key != book.view
                            for key, book in books.items()):
            raise ModelCompatibilityError("models must be nonempty and keyed by each PrototypeBook.view")
        recognition = self.options.recognition_mode
        if recognition == "auto":
            recognition = "whole_glyph" if "whole_glyph" in books else "multi_zone"
        required = "whole_glyph" if recognition == "whole_glyph" else "zone:core"
        if required not in books:
            raise ModelCompatibilityError(f"{recognition} recognition requires a {required} book")
        self.recognition_mode = recognition
        self.models: Mapping[str, PrototypeBook] = MappingProxyType(books)
        self.scorers = {view: PrototypeScorer(book, self.config, mode=self.mode)
                        for view, book in books.items()}
        self.assembler = SinhalaAssembler(max_tokens=self.options.max_tokens)
        self.lexicon_gate = LexiconGate(
            lexicon, tau_oov=self.settings.tau_oov, d_edit_max=self.settings.d_edit_max,
            eta=self.settings.eta, top_paths=self.settings.lexicon_top_paths,
            enabled=self.flags.lexicon_gate, edit_unit=self.settings.lexicon_edit_unit)

    def _score(self, feature: FeatureResult, view: str, candidate_id: str,
               scores: dict[str, ScoringResult]) -> ScoringResult:
        if view not in self.scorers:
            raise DecoderError(f"nonempty {view} has no matching prototype book")
        result = self.scorers[view].score(feature)
        scores[f"{candidate_id}:{view}"] = result
        return result

    def _options(self, feature: FeatureResult, candidate_id: str,
                 scores: dict[str, ScoringResult]) -> tuple[_TokenOption, ...]:
        if feature.metadata.get("empty_ink", False) or feature.metadata.get("empty_canvas", False):
            return (_TokenOption("", np.longdouble(0), None, None, False),)
        if self.recognition_mode == "whole_glyph":
            result = self._score(feature, "whole_glyph", candidate_id, scores)
            return tuple(_TokenOption(
                prediction.label,
                np.longdouble(prediction.log_probability if self.flags.probability_scoring
                              else -prediction.distance),
                None if prediction.log_probability is None else np.longdouble(prediction.log_probability),
                prediction.probability, max_class_probability=result.confidence)
                for prediction in result.top_candidates)
        core = feature.zones.get("core")
        if core is None or core.metadata.get("empty_ink", False) or core.metadata.get("empty_canvas", False):
            raise DecoderError("nonempty candidate has no nonempty core zone for multi_zone recognition")
        results: list[tuple[str, ScoringResult]] = [
            ("core", self._score(core, "zone:core", candidate_id, scores))]
        for zone in ("upper", "lower"):
            zone_feature = feature.zones.get(zone)
            if zone_feature is None:
                raise DecoderError(f"multi_zone candidate is missing its {zone} feature view")
            if not zone_feature.metadata.get("empty_ink", False) and not zone_feature.metadata.get("empty_canvas", False):
                results.append((zone, self._score(zone_feature, f"zone:{zone}", candidate_id, scores)))
        options: dict[str, _TokenOption] = {}
        # At most configured glyph_top_k ** 3 combinations, never pixel loops.
        # Glyph top-k is bounded again by beam width to limit joint enumeration.
        limit = min(self.settings.glyph_top_k, self.settings.beam_width)
        choices = [result.top_candidates[:limit] for _, result in results]
        maximum_joint_probability = 0.
        for combination in itertools.product(*choices):
            labels = {zone: prediction.label for (zone, _), prediction in zip(results, combination)}
            try:
                if not labels.get("upper") and not labels.get("lower"):
                    label = labels["core"]  # May be an explicitly detected visual kombuwa.
                else:
                    label = self.assembler.compose(
                        labels["core"], upper=(labels["upper"],) if labels.get("upper") else (),
                        lower=(labels["lower"],) if labels.get("lower") else ())
            except UnicodeAssemblyError:
                continue
            log_probability = (sum((np.longdouble(p.log_probability) for p in combination), np.longdouble(0))
                               if self.flags.probability_scoring else None)
            evidence = (log_probability if log_probability is not None else
                        -sum((np.longdouble(p.distance) for p in combination), np.longdouble(0)))
            probability = None if log_probability is None else float(np.exp(log_probability))
            if probability is not None:
                maximum_joint_probability = max(maximum_joint_probability, probability)
            option = _TokenOption(label, evidence, log_probability, probability)
            if label not in options or evidence > options[label].evidence:
                options[label] = option
        if not options:
            raise DecoderError("no structurally valid combination of the linked zone predictions")
        return tuple(_TokenOption(option.label, option.evidence, option.log_probability, option.confidence,
                                  max_class_probability=maximum_joint_probability if self.flags.probability_scoring else None)
                     for option in sorted(options.values(), key=lambda item: (-item.evidence, item.label))[:limit])

    def _rank(self, state: _BeamState, width: int) -> np.longdouble:
        mean = state.evidence / max(1, state.scored_tokens)
        return mean - np.longdouble(self.options.overlap_penalty) * state.overlap / max(1, width)

    def _prune(self, states: Iterable[_BeamState], width: int) -> tuple[_BeamState, ...]:
        return tuple(sorted(states, key=lambda state: (-self._rank(state, width), state.tokens,
                                                       state.candidate_ids))[:self.settings.beam_width])

    def _extend(self, state: _BeamState, candidate_id: str, option: _TokenOption,
                overlap: int = 0) -> _BeamState:
        return _BeamState(
            state.tokens + ((option.label,) if option.label else ()),
            state.candidate_ids + (candidate_id,), state.evidence + option.evidence,
            state.log_probability + (option.log_probability if option.log_probability is not None else 0),
            state.confidence_sum + (option.max_class_probability if option.max_class_probability is not None else 0.),
            state.scored_tokens + int(option.scored), state.overlap + overlap,
            state.selected_probability_sum + (option.confidence if option.confidence is not None else 0.))

    def _partition_states(self, candidate_ids: tuple[str, ...], width: int,
                          by_id: Mapping[str, FeatureResult],
                          options: Callable[[str], tuple[_TokenOption, ...]]) -> tuple[_BeamState, ...]:
        if len(candidate_ids) > self.options.max_tokens:
            raise DecoderError("hypothesis exceeds max_tokens")
        previous_end: int | None = None
        for candidate_id in candidate_ids:
            box = by_id[candidate_id].identity.normalized_bbox
            if previous_end is not None and box[0] < previous_end:
                raise DecoderError("partition hypothesis contains overlapping or unordered candidates")
            previous_end = box[2]
        states = (_BeamState(),)
        for candidate_id in candidate_ids:
            choices = options(candidate_id)
            states = self._prune((self._extend(state, candidate_id, choice)
                                  for state in states for choice in choices), width)
        return states

    def _window_states(self, candidate_ids: tuple[str, ...], width: int,
                       by_id: Mapping[str, FeatureResult],
                       options: Callable[[str], tuple[_TokenOption, ...]]) -> tuple[_BeamState, ...]:
        ordered = sorted(candidate_ids, key=lambda cid: (by_id[cid].identity.normalized_bbox[0],
                                                         by_id[cid].identity.normalized_bbox[2], cid))
        boxes = {cid: by_id[cid].identity.normalized_bbox for cid in ordered}
        start = min(box[0] for box in boxes.values())
        end = max(box[2] for box in boxes.values())
        by_end: dict[str, tuple[_BeamState, ...]] = {}
        for index, candidate_id in enumerate(ordered):
            left, _, right, _ = boxes[candidate_id]
            predecessors: list[tuple[_BeamState, int]] = []
            if left == start:
                predecessors.append((_BeamState(), 0))
            for previous_id in ordered[:index]:
                previous_left, _, previous_right, _ = boxes[previous_id]
                if previous_left < left <= previous_right < right:
                    predecessors.extend((state, previous_right-left) for state in by_end[previous_id]
                                        if len(state.candidate_ids) < self.options.max_tokens)
            if not predecessors:
                by_end[candidate_id] = ()
                continue
            choices = options(candidate_id)
            by_end[candidate_id] = self._prune(
                (self._extend(state, candidate_id, choice, overlap)
                 for state, overlap in predecessors for choice in choices), width)
        states = (state for cid, entries in by_end.items() if boxes[cid][2] == end for state in entries)
        return self._prune(states, width)

    @staticmethod
    def _check_marks(candidate_ids: tuple[str, ...], links: Sequence[Any]) -> None:
        selected = set(candidate_ids)
        for link in links:
            owners = selected.intersection(link.parent_candidate_ids)
            if len(owners) > 1:
                raise DecoderError(f"mark {link.mark_id!r} spans multiple selected candidates; retain intact hypothesis")
            if not owners:
                raise DecoderError(f"mark {link.mark_id!r} has no selected parent candidate")

    def run(self, batch: FeatureBatchResult) -> DecodeResult:
        if not isinstance(batch, FeatureBatchResult):
            raise TypeError("batch must be features.FeatureBatchResult")
        by_id: dict[str, FeatureResult] = {}
        for feature in batch.features:
            if not isinstance(feature, FeatureResult) or feature.identity is None:
                raise DecoderError("all batch features require candidate identities")
            identity = feature.identity
            if identity.candidate_id in by_id:
                raise DecoderError("candidate IDs must be unique")
            box = identity.normalized_bbox
            if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in box):
                raise DecoderError("normalized bounding boxes must contain four integer coordinates")
            if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                raise DecoderError("candidate bounding boxes must have positive area and nonnegative coordinates")
            by_id[identity.candidate_id] = feature
        if set(by_id) != set(batch.by_candidate_id) or any(
                batch.by_candidate_id[cid] is not feature for cid, feature in by_id.items()):
            raise DecoderError("features and by_candidate_id must refer to the same feature objects")
        if any(feature.identity.word_id not in batch.hypotheses for feature in by_id.values()):
            raise DecoderError("candidate references a word without hypotheses")
        decoded_words: list[DecodedWord] = []
        for word_id, hypotheses in batch.hypotheses.items():
            scores: dict[str, ScoringResult] = {}
            cached_options: dict[str, tuple[_TokenOption, ...]] = {}
            rejected: list[dict[str, str]] = []
            paths_and_ranks: list[tuple[WordPath, np.longdouble]] = []
            seen_hypotheses: set[str] = set()

            def get_options(candidate_id: str) -> tuple[_TokenOption, ...]:
                if candidate_id not in cached_options:
                    cached_options[candidate_id] = self._options(by_id[candidate_id], candidate_id, scores)
                return cached_options[candidate_id]

            for hypothesis in hypotheses:
                if hypothesis.hypothesis_id in seen_hypotheses:
                    raise DecoderError("hypothesis IDs must be unique within a word")
                seen_hypotheses.add(hypothesis.hypothesis_id)
                candidate_ids = tuple(hypothesis.candidate_ids)
                if (not candidate_ids or len(set(candidate_ids)) != len(candidate_ids) or
                        any(cid not in by_id or by_id[cid].identity.word_id != word_id for cid in candidate_ids)):
                    raise DecoderError("hypothesis must reference unique candidates from its own word")
                if hypothesis.kind not in ("intact", "split", "components", "sliding_windows"):
                    raise DecoderError("unsupported segmentation hypothesis kind")
                width = max(by_id[cid].identity.normalized_bbox[2] for cid in candidate_ids)
                try:
                    if hypothesis.kind == "sliding_windows":
                        states = self._window_states(candidate_ids, width, by_id, get_options)
                    else:
                        self._check_marks(candidate_ids, batch.diacritic_links.get(word_id, ()))
                        states = self._partition_states(candidate_ids, width, by_id, get_options)
                    if not states:
                        raise DecoderError("no covering candidate path within configured limits")
                    for state in states:
                        try:
                            if hypothesis.kind == "sliding_windows":
                                self._check_marks(state.candidate_ids, batch.diacritic_links.get(word_id, ()))
                            text = (self.assembler.assemble_visual(state.tokens) if self.flags.unicode_assembly else
                                    unicodedata.normalize("NFC", "".join(state.tokens)))
                        except UnicodeAssemblyError as exc:
                            rejected.append({"hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)})
                            continue
                        except DecoderError as exc:
                            rejected.append({"hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)})
                            continue
                        has_scores = state.scored_tokens > 0 and self.flags.probability_scoring
                        confidence = state.confidence_sum/state.scored_tokens if has_scores else None
                        rank = self._rank(state, width)
                        metadata = _immutable({
                            "kind": hypothesis.kind, "ambiguous": bool(hypothesis.ambiguous),
                            "rank_score": _bounded_float(rank), "scored_token_count": state.scored_tokens,
                            "rank_policy": "mean_log_probability" if self.flags.probability_scoring else "mean_negative_distance",
                            "overlap_pixels": state.overlap, "overlap_penalty": self.options.overlap_penalty,
                            "score_is_correctness_probability": False,
                            "segmentation_prior_available": False,
                            "confidence_policy": "mean_max_token_class_probability" if has_scores else "unavailable",
                            "selected_label_confidence": state.selected_probability_sum/state.scored_tokens if has_scores else None,
                            "log_probability_saturated": has_scores and state.log_probability < -np.finfo(np.float64).max,
                        })
                        path = WordPath(text, _bounded_float(state.log_probability) if has_scores else None,
                                        confidence, state.candidate_ids, hypothesis.hypothesis_id, metadata)
                        paths_and_ranks.append((path, rank))
                except ModelCompatibilityError:
                    raise
                except DecoderError as exc:
                    rejected.append({"hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)})
                    LOGGER.debug("decoder path rejected", extra={"word_id": word_id,
                                 "hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)})
            ordered_paths = sorted(paths_and_ranks, key=lambda entry: (-entry[1], entry[0].text,
                                   entry[0].hypothesis_id or "", entry[0].candidate_ids))
            paths = tuple(path for path, _ in ordered_paths[:self.settings.beam_width])
            metadata = _immutable({"recognition_mode": self.recognition_mode,
                                   "hypotheses_considered": len(seen_hypotheses), "rejected_paths": rejected,
                                   "retained_paths": len(paths)})
            if not paths:
                decoded_words.append(DecodedWord(word_id, "", "", None, (), None, "rejected",
                                                  MappingProxyType(scores), metadata))
                continue
            decision = self.lexicon_gate.apply(paths, confidence=paths[0].confidence)
            decoded_words.append(DecodedWord(word_id, paths[0].text, decision.text,
                                              decision.selected_path.confidence, paths, decision,
                                              "decoded" if decision.text else "empty",
                                              MappingProxyType(scores), metadata))
        words = tuple(decoded_words)
        metadata = _immutable({"pipeline_version": PIPELINE_VERSION, "mode": self.mode,
                               "recognition_mode": self.recognition_mode,
                               "fallback_config_fields": self.fallback_fields,
                               "word_count": len(words), "rejected_word_count": sum(w.status == "rejected" for w in words),
                               "score_is_correctness_probability": False,
                               "model_view_keys": {view: book.model_view_key for view, book in self.models.items()}})
        return DecodeResult(words, " ".join(word.text for word in words if word.text), metadata)

    decode = run


__all__ = [
    "CalibrationError", "CalibrationSample", "ClassPrediction", "DecodedWord", "DecodeResult",
    "DecoderError", "DecoderOptions", "DecoderPipeline", "LexiconDecision", "LexiconGate",
    "ModelCompatibilityError", "PrototypeBook", "PrototypeCalibrationManager", "PrototypeScorer",
    "ScoringResult", "SinhalaAssembler", "UnicodeAssemblyError", "WordPath", "levenshtein_distance",
]
