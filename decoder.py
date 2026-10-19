"""Offline prototype decoding and structurally validated Sinhala word assembly.

Python >=3.10; NumPy and the preceding pipeline modules are required. No model,
lexicon, image directory, or network service is opened merely by importing this
module. Calibration is explicit and must use development/calibration data with
actual Unicode labels. Numeric dataset folder IDs are never inferred as labels.

    from decoder import DevelopmentSetCalibrationManager, DecoderPipeline
    calibration = DevelopmentSetCalibrationManager()
    book = calibration.calibrate_from_folders("development", unicode_label_map)
    book.save("artifacts/prototypes.npz")
    # Use calibration.feature_pipeline for query features of the same schema.
    result = DecoderPipeline(book, lexicon=words).run(feature_batch)

The strict calibration profile is R^46: 16 slope angles, 12 topology values,
and nine hole-centroid pairs. DevelopmentSetCalibrationManager uses sample
covariance + cov_reg_eps*I with no shrinkage. PrototypeCalibrationManager is
the generic schema-compatible API retained for other feature ablations.
PrototypeBundle stores all separately calibrated core/modifier books in one
lossless NPZ. Every NPZ is checked to be <2,000,000 bytes BEFORE writing; some
full-covariance class inventories cannot fit and are explicitly rejected.

Models are bound to both FeatureSchema.schema_id AND feature view. Whole-glyph
models already include vowel marks. Zoned models are separate books keyed by
zone:core/upper/lower. A missing model for nonempty ink rejects that path. Marks
spanning multiple selected candidates reject a contaminated split hypothesis;
the preserved intact hypothesis remains available. No mark is silently dropped
or counted twice. Folder calibration defaults to whole-glyph preparation;
custom preparation requires an explicit feature_builder callback. The strict
manager also supports automatic HPP zone extraction from isolated full glyphs.

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
population std. Constant/tiny-variance dimensions use scale1. The generic manager
requires max(configured_minimum,D+1) samples for shrunk class covariance; the
strict manager regularizes unshrunk covariance from at least two samples.
Unavailable/ill-conditioned covariance uses an explicit Euclidean fallback.
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

decode_word_or_line runs two passes over original recognition paths. Pass 1
freezes high-confidence OOV words (or all words when gating is disabled). B4
Pass 2 scores ambiguous words using model likelihood, immediate frozen left
and right bigrams, and an optional document-domain prior. It implements the
additive 1e-12/1e-6 probability floors exactly, with logaddexp for word scores.
ContextProbabilityModel accepts validated offline probability tables or fits
them from development counts. Missing context sources disable their terms.
No eligible one-edit model path means unchanged output with needs_review=True.
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
from collections.abc import Iterator
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
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
PIPELINE_VERSION = "1.2.0"
MODEL_FORMAT_VERSION = 1
DEFAULT_MAX_MODEL_BYTES = 256 * 1024 * 1024
MASTER_FEATURE_DIMENSION = 46
# Decimal MB, strict inequality. This is distinct from expanded array memory.
MAX_PROTOTYPE_ARCHIVE_BYTES = 2_000_000


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
    eta_lex: float | None = None
    beta_left: float = .10
    beta_right: float = .10
    beta_doc: float = .10
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
        if self.eta_lex is not None:
            _number("eta_lex", self.eta_lex)
        for name in ("beta_left", "beta_right", "beta_doc"):
            _number(name, getattr(self, name))
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
    def raw_means(self) -> NDArray[np.float64]:
        """Development class means in the original, unstandardized feature space."""
        return _freeze(self.means * self.scale + self.center)

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
    def __init__(self, config: Any = None, *, mode: str | None = None,
                 expected_dimension: int | None = None,
                 regularized_small_samples: bool = False) -> None:
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve(config, mode)
        if expected_dimension is not None:
            _number("expected_dimension", expected_dimension, 1, 4096, integer=True)
        if type(regularized_small_samples) is not bool:
            raise TypeError("regularized_small_samples must be bool")
        self.expected_dimension = expected_dimension
        self.regularized_small_samples = regularized_small_samples

    def fit(self, samples: Iterable[CalibrationSample]) -> PrototypeBook:
        records = list(itertools.islice(iter(samples), 1_000_001))
        if not records or len(records) > 1_000_000:
            raise CalibrationError("calibration requires 1..1,000,000 labelled samples")
        first = records[0]
        if not isinstance(first, CalibrationSample) or not isinstance(first.feature, FeatureResult):
            raise CalibrationError("samples must contain CalibrationSample and FeatureResult")
        schema = first.feature.schema
        if self.expected_dimension is not None and schema.dimension != self.expected_dimension:
            raise ModelCompatibilityError(
                f"expected {self.expected_dimension} features, received {schema.dimension}; "
                "use make_master_feature_pipeline() for the real 46-dimensional schema; "
                "vectors are never padded or truncated")
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
        minimum = max(self.settings.min_covariance_samples,
                      2 if self.regularized_small_samples else dimension+1)
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
                try:
                    feature = feature_builder(path, view)
                except (OSError, ValueError, RuntimeError) as exc:
                    raise CalibrationError(f"development feature extraction failed for {path}: {exc}") from exc
                samples.append(CalibrationSample(label, feature, split))
        return self.fit(samples)

    @staticmethod
    def save(book: PrototypeBook, archive_path: str | Path, metadata_path: str | Path | None = None,
             *, overwrite: bool = False) -> tuple[Path, Path]:
        return book.save(archive_path, metadata_path, overwrite=overwrite)

    @staticmethod
    def load(archive_path: str | Path, metadata_path: str | Path | None = None,
             *, max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES) -> PrototypeBook:
        return PrototypeBook.load(archive_path, metadata_path, max_model_bytes=max_model_bytes)


def make_master_feature_pipeline(config: Any = None, *, mode: str | None = None) -> FeaturePipeline:
    """Construct/validate the explicit 46-value Master Specification profile.

    v1: 16 K=4 slope angles + 12 topology values + 9 (x,y) hole centroids.
    This is a declared feature-profile override, with its own schema ID; it is
    not padding an old descriptor. With no config, both feature branches are
    enabled even for a B0 distance comparison. Original B0--B4 whole-pipeline
    feature ablations remain available through the generic FeaturePipeline.
    An explicit config is validated without silently changing its settings.
    """
    if config is None:
        selected = "B4" if mode is None else mode
        factory = getattr(_project_config, "get_config", None)
        overrides = {"features": {"micro_windows": 4, "include_occupancy": False,
                                   "max_hole_centroids": 9},
                     "flags": {"slope_features": True, "topology_features": True}}
        config = (factory(selected, overrides=overrides) if callable(factory) else
                  {"mode": selected, **overrides})
    pipeline = FeaturePipeline(config, mode=mode)
    if (pipeline.schema.dimension != MASTER_FEATURE_DIMENSION or pipeline.schema.version != "v1"
            or pipeline.settings.micro_windows != 4 or pipeline.settings.include_occupancy
            or pipeline.settings.max_hole_centroids != 9
            or not pipeline.flags.slope_features or not pipeline.flags.topology_features):
        raise ModelCompatibilityError(
            f"Master Specification requires 46 real features; configured schema has {pipeline.schema.dimension}. "
            "Use v1, K=4, occupancy=False, nine hole-centroid slots and both feature branches.")
    return pipeline


@dataclass(frozen=True, slots=True)
class DevelopmentFolderSpec:
    """Isolated full-character images with explicit labels for one target view.

    For a zone view, label_map names the actual core/modifier labels visible in
    that zone, rather than inferring annotations from dataset folder numbers.
    Custom already-sliced or otherwise prepared images require feature_builder.
    """
    root: str | Path
    label_map: Mapping[str, str]
    feature_builder: Callable[[Path, str], FeatureResult] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label_map, Mapping) or not self.label_map:
            raise CalibrationError("development folder spec needs explicit Unicode labels")
        if self.feature_builder is not None and not callable(self.feature_builder):
            raise TypeError("feature_builder must be callable or None")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "label_map", MappingProxyType(dict(self.label_map)))


class DevelopmentSetCalibrationManager(PrototypeCalibrationManager):
    """Strict R^46, development-only calibration with covariance+epsilon*I.

    Two or more examples permit a sample covariance, including a rank-deficient
    covariance made positive definite by epsilon*I. A single example cannot
    estimate a covariance and explicitly uses standardized Euclidean fallback.
    No diagonal shrinkage is applied on this strict path. The generic manager
    keeps its earlier dimension-aware/shrinkage policy for existing experiments.
    """

    def __init__(self, config: Any = None, *, mode: str | None = None,
                 cov_reg_eps: float | None = None) -> None:
        self.feature_pipeline = make_master_feature_pipeline(config, mode=mode)
        super().__init__(self.feature_pipeline.config, mode=self.feature_pipeline.mode,
                         expected_dimension=MASTER_FEATURE_DIMENSION, regularized_small_samples=True)
        if cov_reg_eps is not None:
            _number("cov_reg_eps", cov_reg_eps, 0, positive=True)
        self.settings = replace(self.settings, covariance_shrinkage=0., min_covariance_samples=2,
                                covariance_regularization=(self.settings.covariance_regularization
                                                           if cov_reg_eps is None else float(cov_reg_eps)))

    def fit(self, samples: Iterable[CalibrationSample]) -> PrototypeBook:
        iterator = iter(samples)
        first = next(iterator, None)
        if first is None:
            raise CalibrationError("strict calibration requires labelled development samples")
        if (not isinstance(first, CalibrationSample) or not isinstance(first.feature, FeatureResult)
                or first.feature.schema.schema_id != self.feature_pipeline.schema.schema_id
                or first.feature.schema.feature_names != self.feature_pipeline.schema.feature_names):
            raise ModelCompatibilityError("strict calibration samples must use the manager's exact 46-dimensional schema")
        return super().fit(itertools.chain((first,), iterator))

    def calibrate_from_folders(self, root: str | Path, label_map: Mapping[str, str],
                               feature_pipeline: FeaturePipeline | None = None, *,
                               split: str = "development", view: str = "whole_glyph",
                               feature_builder: Callable[[Path, str], FeatureResult] | None = None) -> PrototypeBook:
        pipeline = self.feature_pipeline if feature_pipeline is None else feature_pipeline
        if not isinstance(pipeline, FeaturePipeline) or pipeline.schema.dimension != MASTER_FEATURE_DIMENSION:
            raise ModelCompatibilityError("strict folder calibration requires a matching 46-dimensional FeaturePipeline")
        if pipeline.schema.schema_id != self.feature_pipeline.schema.schema_id:
            raise ModelCompatibilityError("folder feature preparation differs from the strict manager's schema")
        if view not in ("whole_glyph", "zone:core", "zone:upper", "zone:lower"):
            raise CalibrationError("invalid development feature view")
        if feature_builder is None and view != "whole_glyph":
            from preprocessing import PreprocessingPipeline
            from segmentation import SegmentationPipeline
            preprocessor = PreprocessingPipeline(self.config, mode=self.mode)
            segmenter = SegmentationPipeline(self.config, mode=self.mode)

            def build_zone(path: Path, declared_view: str) -> FeatureResult:
                mask, metadata = preprocessor.run(path)
                if metadata["status"] != "ok":
                    raise CalibrationError(f"development image has unusable ink: {path}")
                segmentation = segmenter.run(mask)
                if len(segmentation.words) != 1:
                    raise CalibrationError(f"zone calibration requires one isolated character per image: {path}")
                region = segmentation.words[0].zones[declared_view.split(":", 1)[1]]
                if not region.image.size or not np.any(region.image):
                    raise CalibrationError(f"development image has no ink in {declared_view}: {path}")
                return pipeline.extract(region.image, view=declared_view)
            feature_builder = build_zone
        return super().calibrate_from_folders(root, label_map, pipeline, split=split, view=view,
                                             feature_builder=feature_builder)

    def fit_views(self, samples: Mapping[str, Iterable[CalibrationSample]]) -> PrototypeBundle:
        """Fit explicitly annotated core/upper/lower views into one saved model."""
        if not isinstance(samples, Mapping) or not samples:
            raise CalibrationError("view calibration needs a nonempty mapping")
        books: dict[str, PrototypeBook] = {}
        for view, records in samples.items():
            book = self.fit(records)
            if book.view != view:
                raise ModelCompatibilityError("view calibration key disagrees with the samples")
            books[view] = book
        return PrototypeBundle(books)

    def calibrate_bundle_from_folders(self, folders: Mapping[str, DevelopmentFolderSpec], *,
                                      split: str = "development") -> PrototypeBundle:
        """Calibrate all view signatures/covariances with explicit per-view labels."""
        if not isinstance(folders, Mapping) or not folders:
            raise CalibrationError("provide at least one development folder specification")
        books: dict[str, PrototypeBook] = {}
        for view, specification in folders.items():
            if not isinstance(specification, DevelopmentFolderSpec):
                raise TypeError("folder values must be DevelopmentFolderSpec")
            books[view] = self.calibrate_from_folders(
                specification.root, specification.label_map, split=split, view=view,
                feature_builder=specification.feature_builder)
        return PrototypeBundle(books)


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
    if len(payload) >= MAX_PROTOTYPE_ARCHIVE_BYTES:
        raise CalibrationError(
            f"prototype archive is {len(payload):,} bytes; required <{MAX_PROTOTYPE_ARCHIVE_BYTES:,}. "
            f"Model has {len(book.labels)} classes and {book.dimension} features. "
            "Use separately calibrated core/modifier classes or fewer classes; "
            "full covariance matrices cannot always be losslessly compressed below 2 MB.")
    manifest["archive_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False).encode("utf-8")
    if len(manifest_bytes) > 4*1024*1024:
        raise CalibrationError("prototype metadata exceeds the 4 MiB JSON budget")
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
        manifest_limit = min(max_model_bytes, 4*1024*1024)
        with manifest_path.open("rb") as handle:
            raw_manifest = handle.read(manifest_limit+1)
        if len(raw_manifest) > manifest_limit:
            raise DecoderError("prototype manifest exceeds resource budget")
        manifest = _strict_json(raw_manifest.decode("utf-8"))
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
        if archive.stat().st_size >= MAX_PROTOTYPE_ARCHIVE_BYTES:
            raise DecoderError("prototype archive must be strictly smaller than 2,000,000 bytes")
        if archive.stat().st_size > max_model_bytes:
            raise DecoderError("prototype archive exceeds resource budget")
        with archive.open("rb") as handle:
            payload = handle.read(min(max_model_bytes+1, MAX_PROTOTYPE_ARCHIVE_BYTES))
    except OSError as exc:
        raise DecoderError("prototype files could not be read") from exc
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, DecoderError):
            raise
        raise DecoderError("invalid strict JSON prototype manifest") from exc
    if (len(payload) >= MAX_PROTOTYPE_ARCHIVE_BYTES or len(payload) > max_model_bytes
            or hashlib.sha256(payload).hexdigest() != checksum):
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


# Strict master-specification packaging; generic PrototypeBook remains compatible.

REQUIRED_DIMENSION = MASTER_FEATURE_DIMENSION
MAX_BUNDLE_ARCHIVE_BYTES = MAX_PROTOTYPE_ARCHIVE_BYTES
_BUNDLE_EXPANDED_BYTES = 256 * 1024 * 1024
_BUNDLE_MANIFEST_BYTES = 4 * 1024 * 1024
_BUNDLE_FORMAT_VERSION = 1
_BUNDLE_ARRAY_FIELDS = ("means", "center", "scale", "covariances", "covariance_available", "counts")
_BUNDLE_VIEWS = ("whole_glyph", "zone:core", "zone:lower", "zone:upper")


def _bundle_contracts(manifest: Any) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Validate untrusted descriptors before any numerical array is allocated."""
    if (not isinstance(manifest, dict)
            or set(manifest) != {"bundle_format_version", "required_dimension", "books"}
            or type(manifest["bundle_format_version"]) is not int
            or manifest["bundle_format_version"] != _BUNDLE_FORMAT_VERSION
            or type(manifest["required_dimension"]) is not int
            or manifest["required_dimension"] != REQUIRED_DIMENSION):
        raise DecoderError("invalid 46-dimensional prototype bundle manifest")
    entries = manifest["books"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= len(_BUNDLE_VIEWS):
        raise DecoderError("a bundle requires 1..4 uniquely keyed model views")
    required = {"format_version", "labels", "schema_id", "view", "model_view_key", "dimension", "feature_names", "metadata"}
    contracts: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    views: set[str] = set()
    identity: tuple[str, tuple[str, ...]] | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != required:
            raise DecoderError("invalid bundle model manifest fields")
        view, labels, names, schema_id = (entry[key] for key in ("view", "labels", "feature_names", "schema_id"))
        if (type(entry["format_version"]) is not int or entry["format_version"] != MODEL_FORMAT_VERSION
                or type(entry["dimension"]) is not int or entry["dimension"] != REQUIRED_DIMENSION
                or not isinstance(view, str) or view not in _BUNDLE_VIEWS or view in views
                or not isinstance(schema_id, str) or len(schema_id) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in schema_id)
                or entry["model_view_key"] != f"{schema_id}:{view}"
                or not isinstance(entry["metadata"], dict)):
            raise DecoderError("invalid bundle model format, dimensions, schema or view")
        if (not isinstance(names, list) or len(names) != REQUIRED_DIMENSION
                or any(not isinstance(name, str) or not name or len(name) > 512 for name in names)
                or len(set(names)) != REQUIRED_DIMENSION):
            raise DecoderError("bundle feature names must identify all 46 dimensions")
        if not isinstance(labels, list) or not 1 <= len(labels) <= 4096:
            raise DecoderError("invalid bundle class count")
        normalized = tuple(_validate_label(label, allow_empty=view in ("zone:upper", "zone:lower")) for label in labels)
        if normalized != tuple(labels) or len(set(normalized)) != len(normalized):
            raise DecoderError("bundle labels must be unique and normalized")
        current = (schema_id, tuple(names))
        if identity is not None and identity != current:
            raise ModelCompatibilityError("all bundle books must share exactly one 46-dimensional feature schema")
        identity = current
        views.add(view)
        classes, dimension = len(labels), REQUIRED_DIMENSION
        shapes = ((classes, dimension), (dimension,), (dimension,), (classes, dimension, dimension), (classes,), (classes,))
        for field_name, shape in zip(_BUNDLE_ARRAY_FIELDS, shapes):
            dtype = np.dtype(np.bool_ if field_name == "covariance_available" else np.int64 if field_name == "counts" else np.float64)
            contracts[f"b{index}__{field_name}.npy"] = (shape, dtype)
    return contracts


class PrototypeBundle(Mapping[str, PrototypeBook]):
    """Immutable, lossless R^46 model mapping, stored as one checked NPZ + JSON.

    Every archive is strictly smaller than 2,000,000 bytes. Full float64
    covariance matrices are retained; an oversized model fails before writing.
    Compression is data dependent, so arbitrary class inventories cannot be
    promised to fit. The expanded-array allocation budget is separate.
    """

    __slots__ = ("_books",)

    def __init__(self, books: Mapping[str, PrototypeBook]) -> None:
        if not isinstance(books, Mapping) or not 1 <= len(books) <= len(_BUNDLE_VIEWS):
            raise CalibrationError("a bundle requires a mapping of 1..4 prototype books")
        copied: dict[str, PrototypeBook] = {}
        for key, book in books.items():
            if not isinstance(book, PrototypeBook) or key != book.view:
                raise CalibrationError("bundle keys must equal their PrototypeBook.view")
            if book.dimension != REQUIRED_DIMENSION:
                raise ModelCompatibilityError("strict bundles require real 46-dimensional features; recalibrate the matching schema")
            copied[key] = book
        manifest = {"bundle_format_version": _BUNDLE_FORMAT_VERSION, "required_dimension": REQUIRED_DIMENSION,
                    "books": [_book_manifest(copied[key]) for key in sorted(copied)]}
        _bundle_contracts(manifest)
        self._books: Mapping[str, PrototypeBook] = MappingProxyType(copied)

    def __getitem__(self, key: str) -> PrototypeBook:
        return self._books[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._books)

    def __len__(self) -> int:
        return len(self._books)

    @property
    def modifier_signatures(self) -> Mapping[str, Mapping[str, NDArray[np.float64]]]:
        """Upper/lower view → Unicode modifier → read-only raw mean in R^46."""
        return MappingProxyType({view: MappingProxyType({label: _freeze(row)
                                for label, row in zip(book.labels, book.raw_means)})
                                 for view, book in self.items() if view in ("zone:upper", "zone:lower")})

    def save(self, archive_path: str | Path, metadata_path: str | Path | None = None,
             *, overwrite: bool = False) -> tuple[Path, Path]:
        if type(overwrite) is not bool:
            raise TypeError("overwrite must be bool")
        archive = Path(archive_path)
        sidecar = Path(metadata_path) if metadata_path is not None else archive.with_suffix(".json")
        if archive.resolve() == sidecar.resolve():
            raise DecoderError("bundle NPZ and JSON must be separate files")
        if not overwrite and (archive.exists() or sidecar.exists()):
            raise FileExistsError("bundle archive or manifest already exists")
        ordered = [self[key] for key in sorted(self)]
        manifest = {"bundle_format_version": _BUNDLE_FORMAT_VERSION, "required_dimension": REQUIRED_DIMENSION,
                    "books": [_book_manifest(book) for book in ordered]}
        embedded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
        if len(embedded) > _BUNDLE_MANIFEST_BYTES:
            raise DecoderError("bundle manifest exceeds its allocation budget")
        arrays = {f"b{index}__{name}": getattr(book, name)
                  for index, book in enumerate(ordered) for name in _BUNDLE_ARRAY_FIELDS}
        if sum(array.nbytes for array in arrays.values()) + len(embedded) > _BUNDLE_EXPANDED_BYTES:
            raise DecoderError("expanded bundle exceeds its allocation budget")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays, metadata=np.frombuffer(embedded, dtype=np.uint8))
        payload = buffer.getvalue()
        if len(payload) >= MAX_BUNDLE_ARCHIVE_BYTES:
            classes = sum(len(book.labels) for book in ordered)
            raise DecoderError(f"lossless R^46 bundle with {classes} classes is {len(payload)} bytes; "
                               f"archive must be <{MAX_BUNDLE_ARCHIVE_BYTES} bytes; reduce the inventory or use factorized zone books")
        outer = dict(manifest, archive_sha256=hashlib.sha256(payload).hexdigest())
        sidecar_bytes = json.dumps(outer, ensure_ascii=False, sort_keys=True, allow_nan=False,
                                   indent=2).encode("utf-8")
        if len(sidecar_bytes) > _BUNDLE_MANIFEST_BYTES:
            raise DecoderError("bundle JSON exceeds its allocation budget")
        _atomic_write(archive, payload, overwrite=overwrite)
        try:
            _atomic_write(sidecar, sidecar_bytes, overwrite=overwrite)
        except Exception:
            if not overwrite:
                archive.unlink(missing_ok=True)
            raise
        return archive, sidecar

    @classmethod
    def load(cls, archive_path: str | Path, metadata_path: str | Path | None = None,
             *, max_model_bytes: int = _BUNDLE_EXPANDED_BYTES) -> PrototypeBundle:
        """Check compressed size, checksum and every NPY header before np.load."""
        _number("max_model_bytes", max_model_bytes, 1, integer=True)
        budget = min(int(max_model_bytes), _BUNDLE_EXPANDED_BYTES)
        archive = Path(archive_path)
        sidecar = Path(metadata_path) if metadata_path is not None else archive.with_suffix(".json")
        try:
            with sidecar.open("rb") as handle:
                raw_manifest = handle.read(_BUNDLE_MANIFEST_BYTES + 1)
            if len(raw_manifest) > _BUNDLE_MANIFEST_BYTES:
                raise DecoderError("bundle JSON exceeds its allocation budget")
            outer = _strict_json(raw_manifest.decode("utf-8"))
            if not isinstance(outer, dict) or "archive_sha256" not in outer:
                raise DecoderError("bundle JSON is missing its archive checksum")
            manifest = dict(outer)
            checksum = manifest.pop("archive_sha256")
            if (not isinstance(checksum, str) or len(checksum) != 64
                    or any(char not in "0123456789abcdef" for char in checksum)):
                raise DecoderError("invalid bundle SHA256")
            contracts = _bundle_contracts(manifest)
            numeric_bytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in contracts.values())
            if numeric_bytes > budget:
                raise DecoderError("declared bundle tensors exceed allocation budget")
            if archive.stat().st_size >= MAX_BUNDLE_ARCHIVE_BYTES:
                raise DecoderError("bundle NPZ must be strictly smaller than 2,000,000 bytes")
            with archive.open("rb") as handle:
                payload = handle.read(MAX_BUNDLE_ARCHIVE_BYTES)
            if len(payload) >= MAX_BUNDLE_ARCHIVE_BYTES:
                raise DecoderError("bundle NPZ must be strictly smaller than 2,000,000 bytes")
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise DecoderError("bundle archive SHA256 mismatch")
            with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                infos = zipped.infolist()
                expected = set(contracts) | {"metadata.npy"}
                if len(infos) != len(expected) or {info.filename for info in infos} != expected:
                    raise DecoderError("bundle NPZ contains missing, duplicate or unexpected arrays")
                if sum(info.file_size for info in infos) > budget:
                    raise DecoderError("expanded bundle archive exceeds allocation budget")
                for info in infos:
                    with zipped.open(info) as member:
                        version = np.lib.format.read_magic(member)
                        if version == (1, 0):
                            shape, fortran, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=10000)
                        elif version == (2, 0):
                            shape, fortran, dtype = np.lib.format.read_array_header_2_0(member, max_header_size=10000)
                        else:
                            raise DecoderError("unsupported bundle NPY version")
                        if (dtype.hasobject or dtype.kind not in "fbiu" or len(shape) > 3
                                or any(type(side) is not int or side < 0 for side in shape)
                                or math.prod(shape) * dtype.itemsize > budget):
                            raise DecoderError("unsafe bundle array header")
                        if info.filename == "metadata.npy":
                            if dtype != np.dtype(np.uint8) or len(shape) != 1 or not 1 <= shape[0] <= min(budget, _BUNDLE_MANIFEST_BYTES):
                                raise DecoderError("invalid embedded bundle manifest array")
                        else:
                            expected_shape, expected_dtype = contracts[info.filename]
                            if (shape != expected_shape or dtype.kind != expected_dtype.kind
                                    or dtype.itemsize != expected_dtype.itemsize):
                                raise DecoderError(f"bundle array violates {info.filename} shape/dtype contract")
                        if info.file_size - member.tell() != math.prod(shape) * dtype.itemsize:
                            raise DecoderError("bundle NPY payload length disagrees with its header")
            with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
                arrays = {name: np.array(loaded[name[:-4]], copy=True) for name in contracts}
                embedded = _strict_json(loaded["metadata"].tobytes().decode("utf-8"))
            if embedded != manifest:
                raise DecoderError("bundle NPZ and JSON describe different model generations")
            books: dict[str, PrototypeBook] = {}
            for index, entry in enumerate(manifest["books"]):
                values = {name: arrays[f"b{index}__{name}.npy"] for name in _BUNDLE_ARRAY_FIELDS}
                book = PrototypeBook(tuple(entry["labels"]), **values, schema_id=entry["schema_id"],
                                     view=entry["view"], feature_names=tuple(entry["feature_names"]), metadata=entry["metadata"])
                books[book.view] = book
            return cls(books)
        except (OSError, EOFError, UnicodeError, ValueError, RecursionError, zipfile.BadZipFile) as exc:
            if isinstance(exc, DecoderError):
                raise
            raise DecoderError("invalid or unreadable strict prototype bundle") from exc


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
        use_covariance = (book.covariance_available & (self.settings.distance_metric == "regularized_mahalanobis")
                          & (self.mode != "B0"))
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
            "effective_distance_metric": "standardized_euclidean" if self.mode == "B0" else self.settings.distance_metric,
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


def _context_word(value: Any) -> str:
    """Validate one lexical token without silently changing its boundaries."""
    normalized = _nfc_word(value)
    if not normalized or any(char.isspace() for char in normalized):
        raise DecoderError("context words must be nonempty strings without whitespace")
    return normalized


def _context_doc_type(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("document type must be str")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip():
        raise DecoderError("document type must be nonempty")
    return normalized


def _context_row(values: Mapping[Any, float], *, counts: bool) -> dict[Any, float]:
    """Preserve partial probability mass; normalize complete count rows."""
    if counts:
        largest = max(values.values(), default=0.0)
        if largest == 0.0:
            return {}
        scaled = {key: value / largest for key, value in values.items() if value > 0.0}
        scaled_total = math.fsum(scaled.values())
        return {key: value / scaled_total for key, value in scaled.items()}
    try:
        total = math.fsum(values.values())
    except OverflowError as exc:
        raise DecoderError("context row mass is not finite") from exc
    if not math.isfinite(total):
        raise DecoderError("context row mass is not finite")
    # One ulp-scale allowance handles ordinary decimal probability tables.
    if total > 1.0 + 1e-12:
        raise DecoderError("conditional context probability row exceeds one")
    return {key: value / total if total > 1.0 else value for key, value in values.items()}


def _context_tables(
    bigrams: Mapping[tuple[str, str], float] | None,
    domains: Mapping[str, Mapping[str, float]] | None,
    *,
    counts: bool,
) -> tuple[dict[tuple[str, str], float], dict[str, dict[str, float]]]:
    """Validate before combining NFC-equivalent conditioning contexts."""
    if bigrams is not None and not isinstance(bigrams, Mapping):
        raise TypeError("bigrams must be a mapping or None")
    if domains is not None and not isinstance(domains, Mapping):
        raise TypeError("domains must be a mapping or None")
    rows: dict[str, dict[str, float]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    if bigrams is not None:
        for key, raw in bigrams.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise DecoderError("bigram keys must be (previous_word, next_word) tuples")
            previous, following = (_context_word(key[0]), _context_word(key[1]))
            normalized_key = (previous, following)
            if normalized_key in seen_pairs:
                raise DecoderError("duplicate bigram after NFC normalization")
            seen_pairs.add(normalized_key)
            _number("context count" if counts else "context probability", raw,
                    high=math.inf if counts else 1.0)
            rows.setdefault(previous, {})[following] = float(raw)
    prepared_bigrams = {
        (previous, following): value
        for previous, row in rows.items()
        for following, value in _context_row(row, counts=counts).items()
    }
    prepared_domains: dict[str, dict[str, float]] = {}
    if domains is not None:
        for raw_doc_type, raw_row in domains.items():
            doc_type = _context_doc_type(raw_doc_type)
            if doc_type in prepared_domains:
                raise DecoderError("duplicate document type after NFC normalization")
            if not isinstance(raw_row, Mapping):
                raise TypeError("each domain must map words to probabilities or counts")
            row: dict[str, float] = {}
            for raw_word, raw in raw_row.items():
                word = _context_word(raw_word)
                if word in row:
                    raise DecoderError("duplicate domain word after NFC normalization")
                _number("context count" if counts else "context probability", raw,
                        high=math.inf if counts else 1.0)
                row[word] = float(raw)
            prepared_domains[doc_type] = _context_row(row, counts=counts)
    return prepared_bigrams, prepared_domains


@dataclass(frozen=True, slots=True, init=False)
class ContextProbabilityModel:
    """Offline conditional probabilities used only to rank ambiguous OCR words.

    ``bigrams[(previous, next)]`` stores ``P(next | previous)``; each domain
    maps words to ``P(word | doc_type)``. Entries must be finite probabilities
    and each conditioning row may contain at most unit mass. Partial rows are
    retained as supplied, so omitted events have probability zero. Constructor
    tables are already probabilities; use :meth:`from_counts` for counts.

    ``None`` means the corresponding evidence source is unavailable. An
    explicitly empty mapping is available but assigns zero to every event.
    This distinction lets callers disable unavailable score terms without
    confusing missing evidence with an observed zero-probability event.
    Tables are copied, NFC-normalized, and frozen recursively.
    """

    bigrams: Mapping[tuple[str, str], float]
    domains: Mapping[str, Mapping[str, float]]
    _has_bigrams: bool
    _has_domains: bool

    def __init__(
        self,
        bigrams: Mapping[tuple[str, str], float] | None = None,
        domains: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        prepared_bigrams, prepared_domains = _context_tables(bigrams, domains, counts=False)
        object.__setattr__(self, "bigrams", _immutable(prepared_bigrams))
        object.__setattr__(self, "domains", _immutable(prepared_domains))
        object.__setattr__(self, "_has_bigrams", bigrams is not None)
        object.__setattr__(self, "_has_domains", domains is not None)

    @property
    def has_bigrams(self) -> bool:
        return self._has_bigrams

    @property
    def has_domains(self) -> bool:
        return self._has_domains

    def bigram_probability(self, word: str, previous: str | None) -> float:
        """Return ``P(word | previous)``; absent events and neighbours give zero."""
        normalized_word = _context_word(word)
        if previous is None:
            return 0.0
        return self.bigrams.get((_context_word(previous), normalized_word), 0.0)

    def domain_probability(self, word: str, doc_type: str | None) -> float:
        """Return ``P(word | doc_type)`` without inventing smoothing evidence."""
        normalized_word = _context_word(word)
        if doc_type is None:
            return 0.0
        row = self.domains.get(_context_doc_type(doc_type))
        return 0.0 if row is None else row.get(normalized_word, 0.0)

    @classmethod
    def from_counts(
        cls,
        bigrams: Mapping[tuple[str, str], float] | None = None,
        domains: Mapping[str, Mapping[str, float]] | None = None,
        *,
        split: str = "development",
    ) -> ContextProbabilityModel:
        """Normalize finite, nonnegative development counts per context row.

        Zero counts are omitted; an all-zero row becomes empty. No test or
        evaluation data is accepted under an explicit split label. This
        provenance check cannot establish the origin of caller-supplied data.
        """
        if not isinstance(split, str) or split not in {"development", "dev", "calibration"}:
            raise CalibrationError("context counts must come from a development/calibration split")
        prepared_bigrams, prepared_domains = _context_tables(bigrams, domains, counts=True)
        return cls(prepared_bigrams if bigrams is not None else None,
                   prepared_domains if domains is not None else None)


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
class ContextCandidateScore:
    text: str
    edit_distance: int
    score: float
    path: WordPath
    terms: Mapping[str, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ContextWordDecision:
    index: int
    raw_text: str
    text: str
    confidence: float | None
    selected_path: WordPath | None
    anchor: bool
    needs_review: bool
    reason: str
    edit_distance: int | None
    score: float | None
    left_anchor: str | None
    right_anchor: str | None
    candidates: tuple[ContextCandidateScore, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ContextDecodeResult:
    words: tuple[ContextWordDecision, ...]
    text: str
    metadata: Mapping[str, Any]


class TwoPassContextDecoder:
    """Freeze recognition anchors, then collapse ambiguous model paths once.

    Only B4 with enabled gating changes text. A confidence >= tau_oov is
    protected even for an OOV word. Pass 2 uses immediate neighbours frozen in
    Pass 1; newly corrected words never become anchors in this call. Unknown
    confidence is retained and marked for review rather than guessed.

    The candidate pool is the first at most three original model paths that
    exactly occur in the lexicon and differ from raw by at most one NFC code
    point. No dictionary-only path or candidate probability is fabricated.
    Missing context sources disable their weights; an observed probability
    of zero uses the specified epsilon. All input/output probabilities retain
    their original normalization. This is a deterministic CPU computation.
    """

    def __init__(self, lexicon: Iterable[str] | None,
                 context_model: ContextProbabilityModel | None = None, *,
                 mode: str = "B4", enabled: bool = True, tau_oov: float = .85,
                 d_edit_max: int = 1, top_paths: int = 3, eta_lex: float = .10,
                 beta_left: float = .10, beta_right: float = .10, beta_doc: float = .10) -> None:
        if not isinstance(mode, str) or mode.upper() not in ("B0", "B1", "B2", "B3", "B4"):
            raise DecoderError("context decoding mode must be B0--B4")
        if context_model is not None and not isinstance(context_model, ContextProbabilityModel):
            raise TypeError("context_model must be ContextProbabilityModel or None")
        for name, value in (("eta_lex", eta_lex), ("beta_left", beta_left),
                            ("beta_right", beta_right), ("beta_doc", beta_doc)):
            _number(name, value)
        # Reuse strict lexicon, threshold, boolean and edit-radius validation.
        self.gate = LexiconGate(lexicon, tau_oov=tau_oov, d_edit_max=d_edit_max,
                                eta=eta_lex, top_paths=top_paths, enabled=enabled)
        self.mode = mode.upper()
        self.enabled = enabled and self.mode == "B4"
        self.context_model = context_model
        self.eta_lex = float(eta_lex)
        self.beta_left, self.beta_right, self.beta_doc = map(float, (beta_left, beta_right, beta_doc))

    @staticmethod
    def _word_paths(source: Sequence[WordPath] | Sequence[Sequence[WordPath]]) -> tuple[tuple[WordPath, ...], ...]:
        if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
            raise TypeError("supply model WordPaths for one word, or a sequence of word-path sequences")
        if not source:
            return ()
        single_word = isinstance(source[0], WordPath)
        groups = (source,) if single_word else source
        if len(groups) > 4096:
            raise DecoderError("context decoding exceeds the 4096-word line budget")
        result: list[tuple[WordPath, ...]] = []
        for paths in groups:
            if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
                raise TypeError("every word requires a sequence of WordPaths")
            if len(paths) > 128:
                raise DecoderError("word exceeds the 128-path beam budget")
            if any(not isinstance(path, WordPath) for path in paths):
                raise TypeError("every model path must be WordPath")
            if any(any(character.isspace() for character in path.text) for path in paths):
                raise DecoderError("each context path must describe one word without whitespace")
            result.append(tuple(paths))
        return tuple(result)

    def decode_word_or_line(self, word_paths: Sequence[WordPath] | Sequence[Sequence[WordPath]], *,
                            doc_type: str | None = None) -> ContextDecodeResult:
        groups = self._word_paths(word_paths)
        if doc_type is not None:
            if not isinstance(doc_type, str) or not doc_type.strip() or len(doc_type) > 256:
                raise DecoderError("doc_type must be a nonempty string of at most 256 characters or None")
            doc_type = unicodedata.normalize("NFC", doc_type)
        # Pass 1 is computed for the entire line before any context decision.
        anchors = tuple(bool(paths and paths[0].text and (not self.enabled or
                        (paths[0].confidence is not None and paths[0].confidence >= self.gate.tau_oov)))
                        for paths in groups)
        anchor_texts = tuple(paths[0].text if anchor else None for paths, anchor in zip(groups, anchors))
        decisions: list[ContextWordDecision] = []
        for index, paths in enumerate(groups):
            raw = paths[0] if paths else None
            raw_text = raw.text if raw is not None else ""
            confidence = None if raw is None else raw.confidence
            left = anchor_texts[index-1] if index > 0 else None
            right = anchor_texts[index+1] if index+1 < len(groups) else None
            model = self.context_model
            left_weight = self.beta_left if left is not None and model is not None and model.has_bigrams else 0.
            right_weight = self.beta_right if right is not None and model is not None and model.has_bigrams else 0.
            doc_weight = self.beta_doc if doc_type is not None and model is not None and model.has_domains else 0.
            metadata: dict[str, Any] = {
                "confidence_policy": "raw_mean_max_class_probability", "tau_oov": self.gate.tau_oov,
                "pass_one_anchor": anchors[index], "anchor_context_is_frozen": True,
                "doc_type": doc_type, "lexicon_available": self.gate.available,
                "lexicon_size": len(self.gate.lexicon), "edit_unit": "codepoint",
                "effective_weights": {"eta_lex": self.eta_lex, "beta_left": left_weight,
                                      "beta_right": right_weight, "beta_doc": doc_weight},
                "model_probability_epsilon": 1e-12, "context_probability_epsilon": 1e-6,
                "considered_paths": 0, "eligible_paths": 0,
                "no_generated_lexicon_neighbours": True,
            }

            def retain(reason: str, *, review: bool, anchor: bool = False) -> ContextWordDecision:
                return ContextWordDecision(index, raw_text, raw_text, confidence, raw, anchor, review,
                                           reason, 0 if raw_text else None, None, left, right, (), _immutable(metadata))

            if not raw_text:
                decisions.append(retain("empty_prediction" if raw is not None else "no_model_paths", review=True))
                continue
            if anchors[index]:
                reason = "high_confidence_oov_shield" if self.enabled else "disabled"
                decisions.append(retain(reason, review=False, anchor=True))
                continue
            if confidence is None:
                decisions.append(retain("confidence_unavailable", review=True))
                continue
            if raw.log_probability is None:
                decisions.append(retain("model_probability_unavailable", review=True))
                continue
            if not self.gate.available or not self.gate.lexicon:
                decisions.append(retain("lexicon_unavailable" if not self.gate.available else "lexicon_empty", review=True))
                continue
            scored: list[tuple[np.longdouble, int, str, int, ContextCandidateScore]] = []
            for path_index, path in enumerate(paths[:self.gate.top_paths]):
                metadata["considered_paths"] += 1
                if path.log_probability is None or path.text not in self.gate.lexicon:
                    continue
                if abs(len(raw_text)-len(path.text)) > self.gate.d_edit_max:
                    continue
                distance = _distance_at_most_one(raw_text, path.text, self.gate.d_edit_max)
                if distance is None:
                    continue
                # logaddexp implements log(P_cand+1e-12) without first
                # exponentiating a potentially underflowed word probability.
                model_log = float(np.logaddexp(path.log_probability, math.log(1e-12)))
                left_probability = (model.bigram_probability(path.text, left) if left_weight else None)
                right_probability = (model.bigram_probability(right, path.text) if right_weight else None)
                domain_probability = (model.domain_probability(path.text, doc_type) if doc_weight else None)
                exact_terms = {
                    "edit": np.longdouble(distance),
                    "model": -np.longdouble(self.eta_lex)*model_log,
                    "left": -np.longdouble(left_weight)*math.log(left_probability+1e-6) if left_weight else np.longdouble(0),
                    "right": -np.longdouble(right_weight)*math.log(right_probability+1e-6) if right_weight else np.longdouble(0),
                    "domain": -np.longdouble(doc_weight)*math.log(domain_probability+1e-6) if doc_weight else np.longdouble(0),
                }
                exact_score = sum(exact_terms.values(), np.longdouble(0))
                terms = _immutable({name: _bounded_float(value) for name, value in exact_terms.items()})
                candidate_metadata = _immutable({
                    "original_path_index": path_index, "model_log_with_epsilon": model_log,
                    "left_probability": left_probability, "right_probability": right_probability,
                    "domain_probability": domain_probability,
                    "score_saturated": bool(abs(exact_score) > np.finfo(np.float64).max),
                })
                candidate = ContextCandidateScore(path.text, distance, _bounded_float(exact_score),
                                                   path, terms, candidate_metadata)
                scored.append((exact_score, distance, path.text, path_index, candidate))
            metadata["eligible_paths"] = len(scored)
            if not scored:
                decisions.append(retain("no_eligible_model_path", review=True))
                continue
            ordered = sorted(scored, key=lambda entry: entry[:4])
            selected = ordered[0][4]
            decisions.append(ContextWordDecision(
                index, raw_text, selected.text, confidence, selected.path, False, False,
                "eligible_path_selected" if selected.text != raw_text else "raw_path_selected",
                selected.edit_distance, selected.score, left, right,
                tuple(entry[4] for entry in ordered), _immutable(metadata)))
        words = tuple(decisions)
        metadata = _immutable({"mode": self.mode, "gating_enabled": self.enabled,
                               "passes": 2, "doc_type": doc_type,
                               "anchor_indices": tuple(index for index, anchor in enumerate(anchors) if anchor),
                               "review_indices": tuple(word.index for word in words if word.needs_review),
                               "pass_two_outputs_become_anchors": False})
        return ContextDecodeResult(words, " ".join(word.text for word in words if word.text), metadata)


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
    needs_review: bool = False
    is_context_anchor: bool = False


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


def _zone_identity(feature: FeatureResult, *, views: tuple[str, ...]) -> Any:
    """Validate coordinates shared by all regions of one normalized word."""
    if not isinstance(feature, FeatureResult) or feature.identity is None:
        raise DecoderError("zone observations require FeatureResult candidate identities")
    identity = feature.identity
    if not isinstance(identity.candidate_id, str) or not identity.candidate_id:
        raise DecoderError("zone candidate IDs must be nonempty strings")
    if isinstance(identity.word_id, bool) or not isinstance(identity.word_id, (int, np.integer)):
        raise DecoderError("zone word IDs must be integers")
    if feature.metadata.get("view") not in views:
        raise ModelCompatibilityError(f"zone feature view must be one of {views!r}")
    box = identity.normalized_bbox
    if (not isinstance(box, tuple) or len(box) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in box) or
            not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
        raise DecoderError("zone bounding boxes require four nonnegative, positive-area integer coordinates")
    if any(int(value) > np.iinfo(np.int64).max for value in box):
        raise DecoderError("zone coordinates exceed the supported integer range")
    return identity


@dataclass(frozen=True, slots=True)
class ModifierCandidate:
    """An independently extracted upper/lower observation in word coordinates.

    Its FeatureResult must use the same extraction policy as its zone prototype
    book. The identity's candidate_id is the globally unique mark ID; its box
    locates the observation in the normalized word, not in a cropped canvas.
    External observations replace candidate-local upper/lower crops during
    decoding, so the same ink is never scored from both sources.
    """

    feature: FeatureResult

    def __post_init__(self) -> None:
        _zone_identity(self.feature, views=("zone:upper", "zone:lower"))
        if self.feature.metadata.get("empty_ink", False) or self.feature.metadata.get("empty_canvas", False):
            raise DecoderError("modifier observations must contain ink; absence requires no observation")

    @property
    def mark_id(self) -> str:
        return self.feature.identity.candidate_id

    @property
    def word_id(self) -> int:
        return int(self.feature.identity.word_id)

    @property
    def zone(self) -> str:
        return self.feature.metadata["view"].split(":", 1)[1]

    @property
    def normalized_bbox(self) -> tuple[int, int, int, int]:
        return self.feature.identity.normalized_bbox


@dataclass(frozen=True, slots=True)
class ModifierBinding:
    mark_id: str
    parent_candidate_id: str
    zone: str
    overlap_pixels: int
    overlap_fraction: float


class HorizontalZoneBinder:
    """Bind each mark once to the unique maximum horizontal-overlap core.

    Boxes use half-open x intervals: touching edges have zero overlap. Ties and
    observations without a positive overlap fail explicitly; an intact
    segmentation hypothesis can then retain otherwise ambiguous ink. The
    operation is geometric and does not assert a probabilistic dependence model.
    """

    def __init__(self, *, max_candidates: int = 4096, max_modifiers: int = 256) -> None:
        _number("max_candidates", max_candidates, 1, 4096, integer=True)
        _number("max_modifiers", max_modifiers, 1, 4096, integer=True)
        self.max_candidates = max_candidates
        self.max_modifiers = max_modifiers

    def bind(self, cores: Sequence[FeatureResult], modifiers: Sequence[ModifierCandidate]
             ) -> tuple[ModifierBinding, ...]:
        if isinstance(cores, (str, bytes)) or isinstance(modifiers, (str, bytes)):
            raise TypeError("cores and modifiers must be sequences of observations")
        if len(cores) > self.max_candidates or len(modifiers) > self.max_modifiers:
            raise DecoderError("horizontal binding exceeds its configured observation budget")
        identities = tuple(_zone_identity(core, views=("zone:core",)) for core in cores)
        if any(core.metadata.get("empty_ink", False) or core.metadata.get("empty_canvas", False) for core in cores):
            raise DecoderError("parent core observations must contain ink")
        if len({identity.candidate_id for identity in identities}) != len(identities):
            raise DecoderError("core candidate IDs must be unique in a binding hypothesis")
        if any(not isinstance(mark, ModifierCandidate) for mark in modifiers):
            raise TypeError("modifiers must contain ModifierCandidate instances")
        if len({mark.mark_id for mark in modifiers}) != len(modifiers):
            raise DecoderError("modifier candidate IDs must be unique")
        word_ids = {int(identity.word_id) for identity in identities} | {mark.word_id for mark in modifiers}
        if len(word_ids) > 1:
            raise DecoderError("core and modifier observations must share a normalized word coordinate system")
        if not modifiers:
            return ()
        if not identities:
            raise DecoderError("nonempty modifier observations require parent core candidates")
        core_boxes = np.asarray([identity.normalized_bbox for identity in identities], dtype=np.int64)
        mark_boxes = np.asarray([mark.normalized_bbox for mark in modifiers], dtype=np.int64)
        overlaps = np.maximum(0, np.minimum(mark_boxes[:, None, 2], core_boxes[None, :, 2])
                              - np.maximum(mark_boxes[:, None, 0], core_boxes[None, :, 0]))
        maxima = overlaps.max(axis=1)
        positive = maxima > 0
        if not positive.all():
            index = int(np.flatnonzero(~positive)[0])
            raise DecoderError(f"modifier {modifiers[index].mark_id!r} has no horizontally overlapping parent core")
        winners = overlaps == maxima[:, None]
        unique = winners.sum(axis=1) == 1
        if not unique.all():
            index = int(np.flatnonzero(~unique)[0])
            raise DecoderError(f"modifier {modifiers[index].mark_id!r} has ambiguous equal-overlap parents; retain intact hypothesis")
        owners = overlaps.argmax(axis=1)
        return tuple(ModifierBinding(mark.mark_id, identities[int(owner)].candidate_id, mark.zone,
                                     int(overlap), float(overlap) / (mark.normalized_bbox[2] - mark.normalized_bbox[0]))
                     for mark, owner, overlap in zip(modifiers, owners, maxima))


class DecoderPipeline:
    """Bounded candidate-path decoding, explicit Unicode assembly and OOV gate.

    ``models`` is one whole-glyph book or a mapping keyed by each book's view.
    Auto mode prefers a whole-glyph book; multi_zone requires a core book and
    a matching upper/lower book whenever the respective zone contains ink.
    An empty-string class is an explicit learned null-mark class allowed only
    in upper/lower books. Empty images are neutral coverage transitions.

    ``run(batch, modifiers=observations)`` instead binds independently extracted
    ModifierCandidate observations by horizontal overlap. This replaces native
    upper/lower crops; an empty sequence explicitly supplies no modifiers. All
    supplied observations must bind once per retained path. Extraction must
    match the zone books' calibration policy.

    Per-word failures are reported as status='rejected' with diagnostics;
    malformed batches, schemas and model-view mismatches raise DecoderError.
    A rejected word stays in ``words`` in its original position. ``text`` joins
    successful word texts with spaces; applications should inspect statuses.
    """

    def __init__(self, models: PrototypeBook | Mapping[str, PrototypeBook],
                 lexicon: Iterable[str] | None = None, config: Any = None, *,
                 mode: str | None = None, options: DecoderOptions | None = None,
                 context_model: ContextProbabilityModel | None = None) -> None:
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
        self.zone_binder = HorizontalZoneBinder(max_candidates=self.options.max_tokens,
                                               max_modifiers=self.options.max_tokens)
        self.lexicon_gate = LexiconGate(
            lexicon, tau_oov=self.settings.tau_oov, d_edit_max=self.settings.d_edit_max,
            eta=self.settings.eta, top_paths=self.settings.lexicon_top_paths,
            enabled=self.flags.lexicon_gate, edit_unit=self.settings.lexicon_edit_unit)
        self.context_decoder = TwoPassContextDecoder(
            self.lexicon_gate.lexicon if self.lexicon_gate.available else None, context_model,
            mode=self.mode, enabled=self.flags.lexicon_gate, tau_oov=self.settings.tau_oov,
            d_edit_max=self.settings.d_edit_max, top_paths=self.settings.lexicon_top_paths,
            eta_lex=self.settings.eta if self.settings.eta_lex is None else self.settings.eta_lex,
            beta_left=self.settings.beta_left, beta_right=self.settings.beta_right,
            beta_doc=self.settings.beta_doc)

    def _score(self, feature: FeatureResult, view: str, candidate_id: str,
               scores: dict[str, ScoringResult]) -> ScoringResult:
        if view not in self.scorers:
            raise DecoderError(f"nonempty {view} has no matching prototype book")
        key = f"{candidate_id}:{view}"
        if key not in scores:
            scores[key] = self.scorers[view].score(feature)
        return scores[key]

    def _options(self, feature: FeatureResult, candidate_id: str,
                 scores: dict[str, ScoringResult], *,
                 modifiers: tuple[ModifierCandidate, ...] | None = None) -> tuple[_TokenOption, ...]:
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
        core_identity = _zone_identity(core, views=("zone:core",))
        if (feature.identity is None or core_identity.candidate_id != candidate_id
                or core_identity.word_id != feature.identity.word_id):
            raise DecoderError("core zone identity disagrees with its parent glyph")
        results: list[tuple[str, ScoringResult]] = [
            ("core", self._score(core, "zone:core", candidate_id, scores))]
        if modifiers is None:
            for zone in ("upper", "lower"):
                zone_feature = feature.zones.get(zone)
                if zone_feature is None:
                    raise DecoderError(f"multi_zone candidate is missing its {zone} feature view")
                if not zone_feature.metadata.get("empty_ink", False) and not zone_feature.metadata.get("empty_canvas", False):
                    zone_identity = _zone_identity(zone_feature, views=(f"zone:{zone}",))
                    if (zone_identity.candidate_id != candidate_id or zone_identity.word_id != core_identity.word_id
                            or min(zone_identity.normalized_bbox[2], core_identity.normalized_bbox[2])
                            <= max(zone_identity.normalized_bbox[0], core_identity.normalized_bbox[0])):
                        raise DecoderError(f"native {zone} zone has no valid horizontal parent-core link")
                    results.append((zone, self._score(zone_feature, f"zone:{zone}", candidate_id, scores)))
        else:
            for modifier in modifiers:
                results.append((modifier.zone, self._score(modifier.feature, f"zone:{modifier.zone}",
                                                          modifier.mark_id, scores)))
        options: dict[str, _TokenOption] = {}
        # Zone combinations use a bounded beam. Independent modifiers can be
        # numerous, so never enumerate a Cartesian product exponential in marks.
        limit = min(self.settings.glyph_top_k, self.settings.beam_width)
        choices = [result.top_candidates[:limit] for _, result in results]
        if len(choices) <= 3:
            combinations = list(itertools.product(*choices))
        else:
            combinations = [()]
            for predictions in choices:
                combinations = sorted((prefix + (prediction,) for prefix in combinations for prediction in predictions),
                                      key=lambda group: (
                                          -sum(np.longdouble(p.log_probability) for p in group)
                                          if self.flags.probability_scoring else sum(np.longdouble(p.distance) for p in group),
                                          tuple(p.label for p in group)))[:self.settings.beam_width]
        maximum_joint_probability = 0.
        for combination in combinations:
            zone_labels: dict[str, list[str]] = {"core": [], "upper": [], "lower": []}
            for (zone, _), prediction in zip(results, combination):
                if prediction.label:
                    zone_labels[zone].append(prediction.label)
            core_label = combination[0].label
            try:
                if not zone_labels["upper"] and not zone_labels["lower"]:
                    label = core_label  # May be an explicitly detected visual kombuwa.
                else:
                    label = self.assembler.compose(
                        core_label, upper=tuple(zone_labels["upper"]), lower=tuple(zone_labels["lower"]))
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
    def _check_marks(candidate_ids: tuple[str, ...], links: Sequence[Any],
                     by_id: Mapping[str, FeatureResult]) -> tuple[ModifierBinding, ...]:
        """Verify native crop ownership from geometry, never cached parent IDs.

        Every overlapping crop already contains some pixels from that component.
        Selecting only its maximum-overlap owner cannot undo fragmented pixels
        in another crop, so native paths conservatively reject any multi-owner
        component. External observations can instead be assigned intact.
        """
        boxes = np.asarray([by_id[cid].identity.normalized_bbox for cid in candidate_ids], dtype=np.int64)
        bindings: list[ModifierBinding] = []
        seen: set[str] = set()
        for link in links:
            if link.mark_id in seen:
                raise DecoderError("native diacritic mark IDs must be unique")
            seen.add(link.mark_id)
            if link.zone not in ("upper", "lower"):
                raise DecoderError("native diacritic zones must be upper or lower")
            box = link.normalized_bbox.as_tuple()
            if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                raise DecoderError("native diacritic bounding boxes must have positive area")
            overlap = np.maximum(0, np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]))
            owners = np.flatnonzero(overlap > 0)
            if owners.size > 1:
                raise DecoderError(f"mark {link.mark_id!r} spans multiple selected candidates; retain intact hypothesis")
            if not owners.size:
                raise DecoderError(f"mark {link.mark_id!r} has no selected parent candidate")
            owner = int(owners[0])
            bindings.append(ModifierBinding(link.mark_id, candidate_ids[owner], link.zone,
                                            int(overlap[owner]), float(overlap[owner]) / (box[2]-box[0])))
        return tuple(bindings)

    def _recognize(self, batch: FeatureBatchResult, *,
            modifiers: Sequence[ModifierCandidate] | None = None) -> DecodeResult:
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
            if any(int(value) > np.iinfo(np.int64).max for value in box):
                raise DecoderError("candidate coordinates exceed the supported integer range")
            by_id[identity.candidate_id] = feature
        if set(by_id) != set(batch.by_candidate_id) or any(
                batch.by_candidate_id[cid] is not feature for cid, feature in by_id.items()):
            raise DecoderError("features and by_candidate_id must refer to the same feature objects")
        if any(feature.identity.word_id not in batch.hypotheses for feature in by_id.values()):
            raise DecoderError("candidate references a word without hypotheses")
        external_modifiers: dict[int, list[ModifierCandidate]] | None = None
        if modifiers is not None:
            if self.recognition_mode != "multi_zone":
                raise DecoderError("external modifiers require multi_zone recognition to prevent whole-glyph double counting")
            if isinstance(modifiers, (str, bytes)) or not isinstance(modifiers, Sequence):
                raise TypeError("modifiers must be a bounded sequence of ModifierCandidate observations")
            if len(modifiers) > self.options.max_tokens * max(1, len(batch.hypotheses)):
                raise DecoderError("external modifiers exceed the per-word observation budget")
            external_modifiers = {word_id: [] for word_id in batch.hypotheses}
            seen_marks: set[str] = set()
            for mark in modifiers:
                if not isinstance(mark, ModifierCandidate):
                    raise TypeError("modifiers must contain ModifierCandidate instances")
                if mark.mark_id in seen_marks or mark.mark_id in by_id:
                    raise DecoderError("external mark IDs must be unique and distinct from glyph candidate IDs")
                if mark.word_id not in external_modifiers:
                    raise DecoderError("external modifier references a word absent from the feature batch")
                seen_marks.add(mark.mark_id)
                external_modifiers[mark.word_id].append(mark)
                if len(external_modifiers[mark.word_id]) > self.options.max_tokens:
                    raise DecoderError("word exceeds the configured modifier observation budget")
        decoded_words: list[DecodedWord] = []
        for word_id, hypotheses in batch.hypotheses.items():
            scores: dict[str, ScoringResult] = {}
            cached_options: dict[tuple[str, tuple[str, ...] | None], tuple[_TokenOption, ...]] = {}
            rejected: list[dict[str, str]] = []
            paths_and_ranks: list[tuple[WordPath, np.longdouble]] = []
            seen_hypotheses: set[str] = set()
            bindings_by_path: dict[tuple[str, ...], tuple[ModifierBinding, ...]] = {}
            word_marks = () if external_modifiers is None else tuple(external_modifiers[word_id])
            marks_by_id = {mark.mark_id: mark for mark in word_marks}

            def get_options(candidate_id: str) -> tuple[_TokenOption, ...]:
                key = (candidate_id, None)
                if key not in cached_options:
                    cached_options[key] = self._options(by_id[candidate_id], candidate_id, scores)
                return cached_options[key]

            def assigned_options(candidate_ids: tuple[str, ...]) -> Callable[[str], tuple[_TokenOption, ...]]:
                cores: list[FeatureResult] = []
                for candidate_id in candidate_ids:
                    feature = by_id[candidate_id]
                    if feature.metadata.get("empty_ink", False) or feature.metadata.get("empty_canvas", False):
                        continue
                    core = feature.zones.get("core")
                    if core is None:
                        raise DecoderError("external modifier binding requires every nonempty candidate's core feature")
                    core_identity = _zone_identity(core, views=("zone:core",))
                    if core_identity.candidate_id != candidate_id or core_identity.word_id != word_id:
                        raise DecoderError("core feature identity disagrees with its containing glyph candidate")
                    cores.append(core)
                bindings = self.zone_binder.bind(cores, word_marks)
                bindings_by_path[candidate_ids] = bindings
                assigned: dict[str, tuple[ModifierCandidate, ...]] = {
                    cid: tuple(marks_by_id[entry.mark_id] for entry in sorted(bindings, key=lambda item: item.mark_id)
                               if entry.parent_candidate_id == cid) for cid in candidate_ids}

                def get_assigned(candidate_id: str) -> tuple[_TokenOption, ...]:
                    observations = assigned[candidate_id]
                    key = (candidate_id, tuple(mark.mark_id for mark in observations))
                    if key not in cached_options:
                        cached_options[key] = self._options(by_id[candidate_id], candidate_id, scores,
                                                            modifiers=observations)
                    return cached_options[key]
                return get_assigned

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
                        if external_modifiers is None:
                            states = self._window_states(candidate_ids, width, by_id, get_options)
                        else:
                            # Ownership depends on the selected covering path,
                            # not the complete set of overlapping proposals.
                            geometry = self._window_states(candidate_ids, width, by_id,
                                lambda _: (_TokenOption("", np.longdouble(0), None, None, False),))
                            assigned_states: list[_BeamState] = []
                            for path in geometry:
                                try:
                                    path_options = assigned_options(path.candidate_ids)
                                    current = (_BeamState(overlap=path.overlap),)
                                    for cid in path.candidate_ids:
                                        choices = path_options(cid)
                                        current = self._prune((self._extend(state, cid, choice)
                                                              for state in current for choice in choices), width)
                                    assigned_states.extend(current)
                                except ModelCompatibilityError:
                                    raise
                                except DecoderError as exc:
                                    rejected.append({"hypothesis_id": hypothesis.hypothesis_id, "reason": str(exc)})
                            states = self._prune(assigned_states, width)
                    else:
                        if external_modifiers is None:
                            bindings_by_path[candidate_ids] = self._check_marks(
                                candidate_ids, batch.diacritic_links.get(word_id, ()), by_id)
                            path_options = get_options
                        else:
                            path_options = assigned_options(candidate_ids)
                        states = self._partition_states(candidate_ids, width, by_id, path_options)
                    if not states:
                        raise DecoderError("no covering candidate path within configured limits")
                    for state in states:
                        try:
                            if hypothesis.kind == "sliding_windows" and external_modifiers is None:
                                bindings_by_path[state.candidate_ids] = self._check_marks(
                                    state.candidate_ids, batch.diacritic_links.get(word_id, ()), by_id)
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
                            "modifier_source": "native_candidate_zones" if external_modifiers is None else "external_observations",
                            "modifier_bindings": [asdict(binding) for binding in bindings_by_path.get(state.candidate_ids, ())],
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
                                   "retained_paths": len(paths),
                                   "modifier_source": "native_candidate_zones" if external_modifiers is None else "external_observations",
                                   "external_modifier_count": len(word_marks),
                                   "modifier_bindings_by_candidate_path": [
                                       {"candidate_ids": ids, "bindings": [asdict(entry) for entry in entries]}
                                       for ids, entries in bindings_by_path.items()]})
            if not paths:
                decoded_words.append(DecodedWord(word_id, "", "", None, (), None, "rejected",
                                                  MappingProxyType(scores), metadata))
                continue
            decoded_words.append(DecodedWord(word_id, paths[0].text, paths[0].text,
                                              paths[0].confidence, paths, None,
                                              "decoded" if paths[0].text else "empty",
                                              MappingProxyType(scores), metadata))
        words = tuple(decoded_words)
        metadata = _immutable({"pipeline_version": PIPELINE_VERSION, "mode": self.mode,
                               "recognition_mode": self.recognition_mode,
                               "fallback_config_fields": self.fallback_fields,
                               "word_count": len(words), "rejected_word_count": sum(w.status == "rejected" for w in words),
                               "score_is_correctness_probability": False,
                               "modifier_source": "native_candidate_zones" if external_modifiers is None else "external_observations",
                               "external_window_search": "bounded_geometry_paths_then_binding" if external_modifiers is not None else None,
                               "model_view_keys": {view: book.model_view_key for view, book in self.models.items()}})
        return DecodeResult(words, " ".join(word.text for word in words if word.text), metadata)

    def decode_word_or_line(self, batch: FeatureBatchResult, *, doc_type: str | None = None,
                            modifiers: Sequence[ModifierCandidate] | None = None) -> DecodeResult:
        """Recognize all words, freeze raw anchors, then collapse context once.

        Batch word order defines line reading order. An unresolved/empty word
        keeps its position and never exposes an anchor across that gap.
        Recognition confidence stays C(raw) even after context selects another
        word path. Human review is exposed directly on every DecodedWord.
        """
        recognized = self._recognize(batch, modifiers=modifiers)
        context = self.context_decoder.decode_word_or_line(
            tuple(word.paths for word in recognized.words), doc_type=doc_type)
        words: list[DecodedWord] = []
        for word, decision in zip(recognized.words, context.words):
            metadata = _plain(word.metadata)
            metadata["context_decision"] = {
                **_plain(decision.metadata), "reason": decision.reason,
                "left_anchor": decision.left_anchor, "right_anchor": decision.right_anchor,
                "score": decision.score, "edit_distance": decision.edit_distance,
                "needs_review": decision.needs_review, "anchor": decision.anchor,
                "candidate_scores": [{"text": item.text, "score": item.score,
                                      "edit_distance": item.edit_distance,
                                      "terms": _plain(item.terms), **_plain(item.metadata)}
                                     for item in decision.candidates],
            }
            lexicon_decision = (LexiconDecision(
                decision.raw_text, decision.text, decision.text != decision.raw_text,
                decision.anchor, decision.reason, decision.edit_distance,
                decision.score, decision.selected_path, _immutable(metadata["context_decision"]))
                if decision.selected_path is not None else None)
            words.append(replace(word, text=decision.text, confidence=decision.confidence,
                                 lexicon_decision=lexicon_decision, metadata=_immutable(metadata),
                                 needs_review=decision.needs_review, is_context_anchor=decision.anchor))
        result_words = tuple(words)
        metadata = _plain(recognized.metadata)
        metadata["context_decode"] = _plain(context.metadata)
        metadata["review_word_ids"] = [word.word_id for word in words if word.needs_review]
        return DecodeResult(result_words, context.text, _immutable(metadata))

    def run(self, batch: FeatureBatchResult, *, doc_type: str | None = None,
            modifiers: Sequence[ModifierCandidate] | None = None) -> DecodeResult:
        """Run the full recognizer and two-pass word/line decoder."""
        return self.decode_word_or_line(batch, doc_type=doc_type, modifiers=modifiers)

    decode = run


def decode_word_or_line(batch: FeatureBatchResult,
                        models: PrototypeBook | Mapping[str, PrototypeBook],
                        lexicon: Iterable[str] | None = None, config: Any = None, *,
                        doc_type: str | None = None,
                        context_model: ContextProbabilityModel | None = None,
                        mode: str | None = None, options: DecoderOptions | None = None,
                        modifiers: Sequence[ModifierCandidate] | None = None) -> DecodeResult:
    """Convenience API; reuse DecoderPipeline for repeated offline queries."""
    pipeline = DecoderPipeline(models, lexicon, config, mode=mode, options=options,
                               context_model=context_model)
    return pipeline.decode_word_or_line(batch, doc_type=doc_type, modifiers=modifiers)


__all__ = [
    "ContextProbabilityModel", "ContextCandidateScore", "ContextWordDecision", "ContextDecodeResult",
    "TwoPassContextDecoder", "decode_word_or_line",
    "DevelopmentFolderSpec", "DevelopmentSetCalibrationManager", "PrototypeBundle",
    "MASTER_FEATURE_DIMENSION", "MAX_PROTOTYPE_ARCHIVE_BYTES", "make_master_feature_pipeline",
    "HorizontalZoneBinder", "ModifierCandidate", "ModifierBinding",
    "CalibrationError", "CalibrationSample", "ClassPrediction", "DecodedWord", "DecodeResult",
    "DecoderError", "DecoderOptions", "DecoderPipeline", "LexiconDecision", "LexiconGate",
    "ModelCompatibilityError", "PrototypeBook", "PrototypeCalibrationManager", "PrototypeScorer",
    "ScoringResult", "SinhalaAssembler", "UnicodeAssemblyError", "WordPath", "levenshtein_distance",
]
