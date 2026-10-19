"""Configuration contract for the offline Sinhala handwriting research pipeline.

Python >= 3.10; this module itself uses only the standard library. Importing it
does not read datasets, create directories, seed global RNGs, change thread
settings, or import a numerical framework. Later modules receive an immutable
``PipelineConfig`` explicitly rather than mutating global settings.

The brief did not specify the B0--B4 membership. The cumulative, *grouped*
ablations below are an explicit experimental design, not a published benchmark:

* B0: Otsu, unzoned glyph segmentation, slope features, nearest prototype.
* B1: B0 + clean-scan-aware restoration and Sauvola.
* B2: B1 + three zones and ambiguous-candidate-preserving ligature cutting.
* B3: B2 + topology, regularized Mahalanobis distances and amplitude scoring.
* B4: B3 + confidence-gated, edit-distance-bounded lexicon correction.

Validation, deskew, aspect-preserving resizing, glyph segmentation and Unicode
assembly are shared controls. Short-line word localization is OFF by default:
enable it for line images in every compared preset. B0--B2 confidence values
must not be invented from hard nearest-prototype predictions; the lexicon gate
is disabled in those modes. Do not treat these grouped ablations as estimates
of the individual effect of every newly enabled component.

Weights and quality thresholds are initial engineering defaults, NOT calibrated
Sinhala recognition results. Calibrate on a development split, freeze settings
and prototypes, and evaluate on a disjoint test split. There is intentionally
no invented mapping from a dataset's numeric class IDs to Sinhala characters.

Example::

    from config import get_config, PipelineConfig
    config = get_config("B4", overrides={
        "flags": {"word_localization": True},
        "decoder": {"gamma_sim": 0.8},
    })
    restored = PipelineConfig.from_json(config.to_json())
    assert restored.fingerprint == config.fingerprint

Unicode tables use Unicode character identities and canonical NFC composition,
not font glyph IDs. Visual position is distinct from Unicode logical order:
kombuwa is drawn before its base but encoded after it. Zone hints are geometric
priors, never a replacement for retained bounding boxes or learned labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_PYTHON = (3, 10)
UNICODE_DATA_VERSION = unicodedata.unidata_version
UNICODE_TABLE_SOURCE = (
    "https://github.com/unicode-org/unicodetools/blob/main/"
    "unicodetools/data/ucd/17.0.0/NamesList.txt"
)
RUNTIME_DEPENDENCIES = (
    "numpy", "opencv-python", "scikit-image", "scipy", "pandas", "streamlit",
)


class ConfigError(ValueError):
    """A configuration value, combination, or serialized document is invalid."""


class AblationMode(str, Enum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"


class Zone(str, Enum):
    UPPER = "upper"
    CORE = "core"
    LOWER = "lower"


def _integer(name: str, value: Any, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ConfigError(f"{name} must be an integer in [{minimum}, {maximum}]; got {value!r}")


def _real(
    name: str, value: Any, minimum: float, maximum: float | None = None,
    *, strictly_positive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite real number; got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{name} must be finite and in [{minimum}, {maximum}]")
    if strictly_positive and value <= 0:
        raise ConfigError(f"{name} must be greater than zero")


def _choice(name: str, value: Any, options: Sequence[str]) -> None:
    if not isinstance(value, str) or value not in options:
        raise ConfigError(f"{name} must be one of {tuple(options)!r}; got {value!r}")


def _odd(name: str, value: Any, minimum: int = 3) -> None:
    _integer(name, value, minimum)
    if value % 2 == 0:
        raise ConfigError(f"{name} must be odd")


def _pair(name: str, value: Any, minimum: int = 1) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ConfigError(f"{name} must be a two-element tuple")
    for item in value:
        _integer(name, item, minimum)


@dataclass(frozen=True, slots=True)
class PipelineFlags:
    """Stage switches; internal dependencies are checked by PipelineConfig."""

    quality_metrics: bool = True
    clean_scan_bypass: bool = False
    deskew: bool = True
    retinex: bool = False
    clahe: bool = False
    bilateral_denoise: bool = False
    ruled_line_removal: bool = False
    sauvola: bool = False
    word_localization: bool = False
    aspect_normalization: bool = True
    glyph_segmentation: bool = True
    zone_partition: bool = False
    ligature_cutter: bool = False
    retain_ambiguous_segments: bool = False
    slope_features: bool = True
    topology_features: bool = False
    probability_scoring: bool = False
    unicode_assembly: bool = True
    lexicon_gate: bool = False

    def __post_init__(self) -> None:
        for item in fields(self):
            if type(getattr(self, item.name)) is not bool:
                raise ConfigError(f"flags.{item.name} must be bool")

    @property
    def requires_skeleton(self) -> bool:
        return self.ligature_cutter or self.topology_features


@dataclass(frozen=True, slots=True)
class InputSettings:
    """Limits apply to compressed uploads AND decoded image allocation."""

    allowed_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    max_file_bytes: int = 20 * 1024 * 1024
    max_decoded_pixels: int = 40_000_000
    min_image_side: int = 3
    max_image_side: int = 20_000
    alpha_background: int = 255

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_extensions, tuple) or not self.allowed_extensions:
            raise ConfigError("input.allowed_extensions must be a nonempty tuple")
        supported = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        for extension in self.allowed_extensions:
            _choice("input.allowed_extensions", extension, supported)
        if len(set(self.allowed_extensions)) != len(self.allowed_extensions):
            raise ConfigError("input.allowed_extensions contains duplicates")
        _integer("input.max_file_bytes", self.max_file_bytes, 1)
        _integer("input.max_decoded_pixels", self.max_decoded_pixels, 1)
        _integer("input.min_image_side", self.min_image_side, 1)
        _integer("input.max_image_side", self.max_image_side, self.min_image_side)
        if self.max_decoded_pixels < self.min_image_side ** 2:
            raise ConfigError("input pixel limit cannot accommodate the minimum image")
        _integer("input.alpha_background", self.alpha_background, 0, 255)


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    """Image metrics use grayscale [0,255]; foreground ink is always 1.

    Clean-scan bypass requires ALL configured quality conditions. It skips
    Retinex/CLAHE/bilateral filtering, not validation, ruled-line removal,
    binarization or geometry. Uniform/blank images are not clean scans.
    Small images need an odd Sauvola window clipped to their valid extent.
    A near-zero Retinex rescale range must return a safe constant image.
    """

    retinex_sigma: float = 15.0
    log_offset: float = 1.0
    rescale_percentiles: tuple[float, float] = (1.0, 99.0)
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (4, 4)
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 5.0
    ruled_line_min_width_fraction: float = 0.40
    ruled_line_max_thickness: int = 3
    ruled_line_kernel_min_width: int = 15
    sauvola_window: int = 25
    sauvola_k: float = 0.20
    sauvola_dynamic_range: float = 128.0
    deskew_method: str = "hough"
    deskew_max_degrees: float = 15.0
    deskew_angle_step: float = 0.5
    deskew_min_degrees: float = 0.15
    hough_vote_threshold: int = 20
    hough_min_line_fraction: float = 0.25
    hough_max_gap: int = 10
    clean_min_laplacian_variance: float = 100.0
    clean_min_contrast: float = 100.0  # grayscale p95 - p05
    clean_max_background_std: float = 12.0  # brightest quartile
    min_ink_fraction: float = 0.001
    max_ink_fraction: float = 0.65

    def __post_init__(self) -> None:
        for name in ("retinex_sigma", "log_offset", "clahe_clip_limit",
                     "bilateral_sigma_color", "bilateral_sigma_space",
                     "sauvola_dynamic_range", "deskew_angle_step"):
            _real(f"preprocessing.{name}", getattr(self, name), 0, strictly_positive=True)
        if not isinstance(self.rescale_percentiles, tuple) or len(self.rescale_percentiles) != 2:
            raise ConfigError("preprocessing.rescale_percentiles must be a pair")
        for value in self.rescale_percentiles:
            _real("preprocessing.rescale_percentiles", value, 0, 100)
        if self.rescale_percentiles[0] >= self.rescale_percentiles[1]:
            raise ConfigError("Retinex lower percentile must be less than upper percentile")
        _pair("preprocessing.clahe_grid_size", self.clahe_grid_size)
        _odd("preprocessing.bilateral_diameter", self.bilateral_diameter, 1)
        _odd("preprocessing.sauvola_window", self.sauvola_window)
        for name in ("ruled_line_min_width_fraction", "hough_min_line_fraction"):
            _real(f"preprocessing.{name}", getattr(self, name), 0, 1, strictly_positive=True)
        for name in ("ruled_line_max_thickness", "ruled_line_kernel_min_width", "hough_vote_threshold"):
            _integer(f"preprocessing.{name}", getattr(self, name), 1)
        _integer("preprocessing.hough_max_gap", self.hough_max_gap, 0)
        _choice("preprocessing.deskew_method", self.deskew_method, ("hough", "radon"))
        _real("preprocessing.deskew_max_degrees", self.deskew_max_degrees, 0, 45, strictly_positive=True)
        _real("preprocessing.deskew_min_degrees", self.deskew_min_degrees, 0, self.deskew_max_degrees)
        if self.deskew_angle_step > self.deskew_max_degrees:
            raise ConfigError("deskew angle step exceeds search range")
        _real("preprocessing.sauvola_k", self.sauvola_k, 0, 1)
        _real("preprocessing.clean_min_laplacian_variance", self.clean_min_laplacian_variance, 0)
        _real("preprocessing.clean_min_contrast", self.clean_min_contrast, 0, 255)
        _real("preprocessing.clean_max_background_std", self.clean_max_background_std, 0, 255)
        _real("preprocessing.min_ink_fraction", self.min_ink_fraction, 0, 1, strictly_positive=True)
        _real("preprocessing.max_ink_fraction", self.max_ink_fraction, self.min_ink_fraction, 1)
        if self.min_ink_fraction == self.max_ink_fraction:
            raise ConfigError("ink fraction bounds must define a nonempty interval")


@dataclass(frozen=True, slots=True)
class SegmentationSettings:
    """Retain original coordinates through cropping and resize transforms.

    Height H=64; width=max(1,floor(W_original*H/H_original)). Reject crops
    exceeding max_normalized_width rather than silently distorting aspect ratio.
    HPP smoothing is Gaussian; valley searches are relative to the core peak.
    Flat/no-ink profiles must return an explicit failure or documented fallback.
    Curvature uses a smoothed skeleton ordinate; branch ambiguity must retain
    the unsplit candidate. No threshold here guarantees Sinhala loop integrity.
    """

    target_height: int = 64
    max_normalized_width: int = 8192
    word_dilation_kernel: tuple[int, int] = (15, 3)  # OpenCV (width, height)
    min_component_area: int = 2
    crop_padding: int = 2
    hpp_smoothing_sigma: float = 1.0
    upper_valley_search: tuple[float, float] = (0.10, 0.45)
    lower_valley_search: tuple[float, float] = (0.55, 0.90)
    min_core_height: int = 8
    touching_width_factor: float = 1.4
    lambda_curv: float = 0.35
    skeleton_method: str = "zhang"
    skeleton_connectivity: int = 8
    hole_background_connectivity: int = 4
    curvature_smoothing_sigma: float = 1.0
    min_hole_area: int = 2
    min_segment_width: int = 2
    cut_safety_margin: int = 1
    max_candidates_per_word: int = 128

    def __post_init__(self) -> None:
        _integer("segmentation.target_height", self.target_height, 8)
        _integer("segmentation.max_normalized_width", self.max_normalized_width, 1)
        _pair("segmentation.word_dilation_kernel", self.word_dilation_kernel)
        _integer("segmentation.min_component_area", self.min_component_area, 1)
        _integer("segmentation.crop_padding", self.crop_padding, 0)
        _real("segmentation.hpp_smoothing_sigma", self.hpp_smoothing_sigma, 0, strictly_positive=True)
        for name in ("upper_valley_search", "lower_valley_search"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ConfigError(f"segmentation.{name} must be a pair")
            for bound in value:
                _real(f"segmentation.{name}", bound, 0, 1)
            if value[0] >= value[1]:
                raise ConfigError(f"segmentation.{name} must be increasing")
        if self.upper_valley_search[1] >= self.lower_valley_search[0]:
            raise ConfigError("upper and lower valley search intervals must not overlap")
        _integer("segmentation.min_core_height", self.min_core_height, 1, self.target_height)
        _real("segmentation.touching_width_factor", self.touching_width_factor, 1, strictly_positive=True)
        _real("segmentation.lambda_curv", self.lambda_curv, 0)
        _choice("segmentation.skeleton_method", self.skeleton_method, ("zhang",))
        _integer("segmentation.skeleton_connectivity", self.skeleton_connectivity, 8, 8)
        _integer("segmentation.hole_background_connectivity", self.hole_background_connectivity, 4, 4)
        _real("segmentation.curvature_smoothing_sigma", self.curvature_smoothing_sigma, 0, strictly_positive=True)
        for name in ("min_hole_area", "min_segment_width", "max_candidates_per_word"):
            _integer(f"segmentation.{name}", getattr(self, name), 1)
        _integer("segmentation.cut_safety_margin", self.cut_safety_margin, 0)


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    """K x K cells; normalized slope angles are in [-1,1].

    A nondegenerate vertical cell maps to +1; empty/singleton/isotropic cells
    map to empty_window_value. Optional occupancy preserves missing-data
    evidence; it defaults OFF to keep the brief's K*K slope-only branch.
    Cell coordinates are local, normalized to [0,1]. Adjacent junction pixels
    form one 8-connected junction cluster. No holes/junctions produces zeros.

    Stable descriptor schema: row-major K*K angles, then optional K*K occupancy;
    topology = C, H, chi, endpoint/3-way/4+-way counts, their three normalized
    (x,y) centroids, then max_hole_centroids sorted (x,y) pairs with zero padding.
    Counts remain raw until development-only feature standardization. Zero
    centroids are distinguished from missing points by their accompanying counts.
    """

    micro_windows: int = 4
    lse_epsilon: float = 1e-8
    empty_window_value: float = 0.0
    include_occupancy: bool = False
    max_hole_centroids: int = 4
    slope_weight: float = 1.0
    topology_weight: float = 1.0
    standardization_floor: float = 1e-8

    def __post_init__(self) -> None:
        _integer("features.micro_windows", self.micro_windows, 1, 32)
        _real("features.lse_epsilon", self.lse_epsilon, 0, strictly_positive=True)
        _real("features.empty_window_value", self.empty_window_value, -1, 1)
        if type(self.include_occupancy) is not bool:
            raise ConfigError("features.include_occupancy must be bool")
        _integer("features.max_hole_centroids", self.max_hole_centroids, 0)
        for name in ("slope_weight", "topology_weight", "standardization_floor"):
            _real(f"features.{name}", getattr(self, name), 0, strictly_positive=True)

    @property
    def slope_dimension(self) -> int:
        return self.micro_windows ** 2 * (2 if self.include_occupancy else 1)

    @property
    def topology_dimension(self) -> int:
        return 12 + 2 * self.max_hole_centroids


@dataclass(frozen=True, slots=True)
class DecoderSettings:
    """Distances are nonnegative, UNSQUARED distances.

    Covariance: (1-s)*sample_cov + s*diag(sample_cov) + regularization*I.
    Insufficient/ill-conditioned class covariance uses standardized Euclidean.
    Fit all means/scales/covariances on development data only; never test data.
    Normalized probabilities are softmax(-2*gamma_sim*D), evaluated after
    subtracting the maximum log weight. They are not calibrated correctness
    probabilities. Persist class labels, descriptor schema and config hash.

    High-confidence raw predictions bypass correction. Otherwise score at most
    the top three word paths with d_edit - eta*log(P_candidate). Distances and
    the length prefilter use lexicon_edit_unit consistently. Unseen lexicon
    words have no fabricated probability; no eligible candidate retains raw
    output. Ties are resolved deterministically by score, distance, then text.
    """

    distance_metric: str = "regularized_mahalanobis"
    covariance_shrinkage: float = 0.10
    covariance_regularization: float = 1e-4
    covariance_condition_limit: float = 1e10
    min_covariance_samples: int = 3
    gamma_sim: float = 1.0
    glyph_top_k: int = 3
    beam_width: int = 3
    lexicon_top_paths: int = 3
    tau_oov: float = 0.85
    d_edit_max: int = 1
    eta: float = 0.10
    log_probability_floor: float = 1e-300
    lexicon_edit_unit: str = "codepoint"

    def __post_init__(self) -> None:
        _choice("decoder.distance_metric", self.distance_metric,
                ("regularized_mahalanobis", "standardized_euclidean"))
        _real("decoder.covariance_shrinkage", self.covariance_shrinkage, 0, 1)
        _real("decoder.covariance_regularization", self.covariance_regularization, 0, strictly_positive=True)
        _real("decoder.covariance_condition_limit", self.covariance_condition_limit, 1)
        _integer("decoder.min_covariance_samples", self.min_covariance_samples, 2)
        _real("decoder.gamma_sim", self.gamma_sim, 0, strictly_positive=True)
        for name in ("glyph_top_k", "beam_width", "lexicon_top_paths"):
            _integer(f"decoder.{name}", getattr(self, name), 1)
        if self.lexicon_top_paths > self.beam_width:
            raise ConfigError("lexicon_top_paths cannot exceed beam_width")
        _real("decoder.tau_oov", self.tau_oov, 0, 1)
        _integer("decoder.d_edit_max", self.d_edit_max, 0)
        _real("decoder.eta", self.eta, 0)
        _real("decoder.log_probability_floor", self.log_probability_floor, 0, 1, strictly_positive=True)
        _choice("decoder.lexicon_edit_unit", self.lexicon_edit_unit, ("codepoint", "grapheme"))


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Report both CER units separately; do not mix them in ESI.

    Manifest columns are image_path, ground_truth, split (dev/test), and
    is_oov. Determine OOV from a frozen lexicon, not decoder output. Blank
    references require an explicit denominator policy in evaluation.py.
    tracemalloc measures traced Python allocations, not total/native peak RAM;
    report that distinction in every exported result. Seed and single-thread
    settings improve repeatability but do not promise cross-platform bit parity.
    """

    seed: int = 0
    opencv_threads: int = 1
    blas_threads: int = 1
    warmup_words: int = 1
    latency_repetitions: int = 3
    cer_units: tuple[str, ...] = ("codepoint", "grapheme")
    unicode_normalization: str = "NFC"
    tracemalloc_frames: int = 1
    manifest_columns: tuple[str, ...] = ("image_path", "ground_truth", "split", "is_oov")

    def __post_init__(self) -> None:
        _integer("evaluation.seed", self.seed, 0, 2 ** 32 - 1)
        for name in ("opencv_threads", "blas_threads", "latency_repetitions", "tracemalloc_frames"):
            _integer(f"evaluation.{name}", getattr(self, name), 1)
        _integer("evaluation.warmup_words", self.warmup_words, 0)
        if not isinstance(self.cer_units, tuple) or not self.cer_units:
            raise ConfigError("evaluation.cer_units must be a nonempty tuple")
        for unit in self.cer_units:
            _choice("evaluation.cer_units", unit, ("codepoint", "grapheme"))
        if len(set(self.cer_units)) != len(self.cer_units):
            raise ConfigError("evaluation.cer_units contains duplicates")
        _choice("evaluation.unicode_normalization", self.unicode_normalization, ("NFC",))
        if self.manifest_columns != ("image_path", "ground_truth", "split", "is_oov"):
            raise ConfigError("evaluation.manifest_columns must retain the manifest schema")


@dataclass(frozen=True, slots=True)
class UISettings:
    confidence_green: float = 0.85
    confidence_yellow: float = 0.60
    max_audit_text_length: int = 4096

    def __post_init__(self) -> None:
        _real("ui.confidence_yellow", self.confidence_yellow, 0, 1)
        _real("ui.confidence_green", self.confidence_green, 0, 1)
        if self.confidence_yellow >= self.confidence_green:
            raise ConfigError("yellow confidence threshold must be below green")
        _integer("ui.max_audit_text_length", self.max_audit_text_length, 1)


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Resolve relative paths from the repository, never an arbitrary cwd."""

    dataset_manifest: str = "data/manifest.csv"
    calibration_directory: str = "data/characters"
    class_label_map: str = "data/class_labels.json"
    lexicon: str = "data/lexicon.txt"
    prototypes: str = "artifacts/prototypes.npz"
    prototype_metadata: str = "artifacts/prototypes.json"
    evaluation_directory: str = "artifacts/evaluation"
    correction_audit_log: str = "artifacts/corrections.jsonl"

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ConfigError(f"paths.{item.name} must be a nonempty NUL-free string")
        if self.prototypes == self.prototype_metadata:
            raise ConfigError("prototype archive and metadata must be separate files")

    def resolve(self, name: str, root: str | Path = PROJECT_ROOT) -> Path:
        if name not in {item.name for item in fields(self)}:
            raise ConfigError(f"unknown path setting: {name!r}")
        path = Path(getattr(self, name)).expanduser()
        return (path if path.is_absolute() else Path(root) / path).resolve()


_BASE_FLAGS = PipelineFlags()
_RESTORATION_FLAGS = replace(
    _BASE_FLAGS, clean_scan_bypass=True, retinex=True, clahe=True,
    bilateral_denoise=True, ruled_line_removal=True, sauvola=True,
)
_ZONE_FLAGS = replace(
    _RESTORATION_FLAGS, zone_partition=True, ligature_cutter=True,
    retain_ambiguous_segments=True,
)
_TOPOLOGY_FLAGS = replace(_ZONE_FLAGS, topology_features=True, probability_scoring=True)
_PRESET_FLAGS: Mapping[AblationMode, PipelineFlags] = MappingProxyType({
    AblationMode.B0: _BASE_FLAGS,
    AblationMode.B1: _RESTORATION_FLAGS,
    AblationMode.B2: _ZONE_FLAGS,
    AblationMode.B3: _TOPOLOGY_FLAGS,
    AblationMode.B4: replace(_TOPOLOGY_FLAGS, lexicon_gate=True),
})
ABLATION_DESCRIPTIONS: Mapping[AblationMode, str] = MappingProxyType({
    AblationMode.B0: "Otsu + slope features + standardized nearest prototype",
    AblationMode.B1: "B0 + clean-scan-aware restoration and Sauvola",
    AblationMode.B2: "B1 + three zones and skeletal ligature candidate cutting",
    AblationMode.B3: "B2 + topology, Mahalanobis distances and amplitude scoring",
    AblationMode.B4: "B3 + confidence-gated OOV-safe lexicon correction",
})


def _mode(value: str | AblationMode) -> AblationMode:
    if not isinstance(value, str):
        raise ConfigError("mode must be B0, B1, B2, B3 or B4")
    try:
        return AblationMode(value.upper())
    except ValueError as exc:
        raise ConfigError(f"unknown ablation mode: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    mode: AblationMode = AblationMode.B4
    flags: PipelineFlags = field(default_factory=lambda: _PRESET_FLAGS[AblationMode.B4])
    input: InputSettings = field(default_factory=InputSettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    decoder: DecoderSettings = field(default_factory=DecoderSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    ui: UISettings = field(default_factory=UISettings)
    paths: PathSettings = field(default_factory=PathSettings)

    def __post_init__(self) -> None:
        _integer("schema_version", self.schema_version, CONFIG_SCHEMA_VERSION, CONFIG_SCHEMA_VERSION)
        if not isinstance(self.mode, AblationMode):
            raise ConfigError("use get_config(mode) or from_dict to select an ablation")
        for name, setting_type in _SETTING_TYPES.items():
            if not isinstance(getattr(self, name), setting_type):
                raise ConfigError(f"{name} must be {setting_type.__name__}")
        f = self.flags
        if not (f.slope_features or f.topology_features):
            raise ConfigError("at least one feature branch must be enabled")
        if f.clean_scan_bypass and not f.quality_metrics:
            raise ConfigError("clean_scan_bypass requires quality_metrics")
        if f.ligature_cutter and not f.zone_partition:
            raise ConfigError("ligature_cutter requires zone_partition")
        if f.ligature_cutter and not f.glyph_segmentation:
            raise ConfigError("ligature_cutter requires glyph_segmentation")
        if f.lexicon_gate and not (f.probability_scoring and f.unicode_assembly):
            raise ConfigError("lexicon_gate requires probability_scoring and unicode_assembly")
        if self.features.micro_windows > self.segmentation.target_height:
            raise ConfigError("micro_windows exceeds normalized image height")

    @property
    def descriptor_dimension(self) -> int:
        return (
            (self.features.slope_dimension if self.flags.slope_features else 0)
            + (self.features.topology_dimension if self.flags.topology_features else 0)
        )

    @property
    def is_custom(self) -> bool:
        """An override must not silently be reported as an untouched preset."""
        return self != ABLATION_PRESETS[self.mode]

    @property
    def experiment_label(self) -> str:
        return f"{self.mode.value}-custom-{self.fingerprint[:8]}" if self.is_custom else self.mode.value

    def to_dict(self) -> dict[str, Any]:
        # JSON conversion makes tuples into lists and str enums into strings.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          indent=indent, allow_nan=False)

    @property
    def fingerprint(self) -> str:
        """SHA256 of configuration only, not a hash of code, data or prototypes."""
        canonical = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> PipelineConfig:
        if not isinstance(document, Mapping):
            raise ConfigError("configuration must be an object")
        _reject_unknown(document, {item.name for item in fields(cls)}, "config")
        selected = _mode(document.get("mode", AblationMode.B4))
        kwargs: dict[str, Any] = {"mode": selected}
        if "schema_version" in document:
            kwargs["schema_version"] = document["schema_version"]
        for name, setting_type in _SETTING_TYPES.items():
            baseline = _preset_setting(selected, name, setting_type)
            if name not in document:
                kwargs[name] = baseline
                continue
            raw = document[name]
            if not isinstance(raw, Mapping):
                raise ConfigError(f"{name} must be an object")
            _reject_unknown(raw, {item.name for item in fields(setting_type)}, name)
            converted = dict(raw)
            for key, value in converted.items():
                if isinstance(getattr(baseline, key), tuple):
                    if not isinstance(value, (list, tuple)):
                        raise ConfigError(f"{name}.{key} must be an array")
                    converted[key] = tuple(value)
            kwargs[name] = replace(baseline, **converted)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, document: str) -> PipelineConfig:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ConfigError(f"duplicate JSON key: {key!r}")
                result[key] = value
            return result

        try:
            parsed = json.loads(document, object_pairs_hook=unique_object)
        except (json.JSONDecodeError, TypeError, RecursionError) as exc:
            raise ConfigError("invalid configuration JSON") from exc
        return cls.from_dict(parsed)


_SETTING_TYPES: Mapping[str, type] = MappingProxyType({
    "flags": PipelineFlags, "input": InputSettings,
    "preprocessing": PreprocessingSettings, "segmentation": SegmentationSettings,
    "features": FeatureSettings, "decoder": DecoderSettings,
    "evaluation": EvaluationSettings, "ui": UISettings, "paths": PathSettings,
})


def _reject_unknown(document: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(document) - allowed
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise ConfigError(f"unknown {context} fields: {names}")


def _preset_setting(mode: AblationMode, name: str, setting_type: type) -> Any:
    if name == "flags":
        return _PRESET_FLAGS[mode]
    if name == "decoder" and mode in (AblationMode.B0, AblationMode.B1, AblationMode.B2):
        return DecoderSettings(distance_metric="standardized_euclidean")
    return setting_type()


ABLATION_PRESETS: Mapping[AblationMode, PipelineConfig] = MappingProxyType({
    mode: PipelineConfig(mode=mode, **{
        name: _preset_setting(mode, name, setting_type)
        for name, setting_type in _SETTING_TYPES.items()
    })
    for mode in AblationMode
})
DEFAULT_CONFIG = ABLATION_PRESETS[AblationMode.B4]


def get_config(
    mode: str | AblationMode = AblationMode.B4,
    *, overrides: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """Return a frozen preset or strictly validated partial nested overrides.

    To change modes, pass mode explicitly; never relabel an existing flag set.
    Missing paths/assets do not prevent configuration loading. Startup modules
    must check their required files and fail clearly if calibration is absent.
    """
    selected = _mode(mode)
    if overrides is None:
        return ABLATION_PRESETS[selected]
    if not isinstance(overrides, Mapping):
        raise ConfigError("overrides must be an object")
    if "mode" in overrides:
        raise ConfigError("select the ablation with get_config(mode), not overrides['mode']")
    return PipelineConfig.from_dict({"mode": selected.value, **overrides})


# Sinhala identities. Roman tokens are project identifiers, not a complete
# transliteration scheme. All 18 independent vowels and 41 consonants appear.
# Scope: assigned characters in the main Sinhala block U+0D80--U+0DFF;
# archaic Sinhala numerals in U+111E0--U+111FF are not recognition targets here.
INDEPENDENT_VOWELS: Mapping[str, str] = MappingProxyType({
    "a": "\u0d85", "aa": "\u0d86", "ae": "\u0d87", "aae": "\u0d88",
    "i": "\u0d89", "ii": "\u0d8a", "u": "\u0d8b", "uu": "\u0d8c",
    "vocalic_r": "\u0d8d", "vocalic_rr": "\u0d8e",
    "vocalic_l": "\u0d8f", "vocalic_ll": "\u0d90",
    "e": "\u0d91", "ee": "\u0d92", "ai": "\u0d93",
    "o": "\u0d94", "oo": "\u0d95", "au": "\u0d96",
})
CONSONANTS: Mapping[str, str] = MappingProxyType({
    "ka": "\u0d9a", "kha": "\u0d9b", "ga": "\u0d9c", "gha": "\u0d9d",
    "nga": "\u0d9e", "nnga": "\u0d9f", "ca": "\u0da0", "cha": "\u0da1",
    "ja": "\u0da2", "jha": "\u0da3", "nya": "\u0da4", "jnya": "\u0da5",
    "nyja": "\u0da6", "tta": "\u0da7", "ttha": "\u0da8", "dda": "\u0da9",
    "ddha": "\u0daa", "nna": "\u0dab", "nnda": "\u0dac", "ta": "\u0dad",
    "tha": "\u0dae", "da": "\u0daf", "dha": "\u0db0", "na": "\u0db1",
    "nda": "\u0db3", "pa": "\u0db4", "pha": "\u0db5", "ba": "\u0db6",
    "bha": "\u0db7", "ma": "\u0db8", "mba": "\u0db9", "ya": "\u0dba",
    "ra": "\u0dbb", "la": "\u0dbd", "va": "\u0dc0", "sha": "\u0dc1",
    "ssa": "\u0dc2", "sa": "\u0dc3", "ha": "\u0dc4", "lla": "\u0dc5",
    "fa": "\u0dc6",
})
VOWEL_SIGNS: Mapping[str, str] = MappingProxyType({
    "a": "",  # inherent vowel: never append a literal independent අ
    "aa": "\u0dcf", "ae": "\u0dd0", "aae": "\u0dd1",
    "i": "\u0dd2", "ii": "\u0dd3", "u": "\u0dd4", "uu": "\u0dd6",
    "vocalic_r": "\u0dd8", "vocalic_rr": "\u0df2",
    "vocalic_l": "\u0ddf", "vocalic_ll": "\u0df3",
    "e": "\u0dd9", "ee": "\u0dda", "ai": "\u0ddb",
    "o": "\u0ddc", "oo": "\u0ddd", "au": "\u0dde",
})
VIRAMA = "\u0dca"  # al-lakuna; suppresses the inherent vowel
ZWJ = "\u200d"
ZWNJ = "\u200c"
FINAL_SIGNS: Mapping[str, str] = MappingProxyType({
    "candrabindu": "\u0d81", "anusvaraya": "\u0d82", "visargaya": "\u0d83",
})
PUNCTUATION: Mapping[str, str] = MappingProxyType({"kunddaliya": "\u0df4"})
SINHALA_DIGITS: Mapping[str, str] = MappingProxyType({
    str(value): chr(0x0DE6 + value) for value in range(10)
})

# Semantic mark names for classifier label maps. Never infer these labels from
# numeric folder order. Vocalic_l shares U+0DDF with the au right-hand component.
MARK_ALIASES: Mapping[str, str] = MappingProxyType({
    "al_lakuna": VIRAMA, "aela_pilla": "\u0dcf",
    "ketti_aeda_pilla": "\u0dd0", "diga_aeda_pilla": "\u0dd1",
    "ketti_ispilla": "\u0dd2", "diga_ispilla": "\u0dd3",
    "ketti_papilla": "\u0dd4", "diga_papilla": "\u0dd6",
    "gaetta_pilla": "\u0dd8", "diga_gaetta_pilla": "\u0df2",
    "kombuwa": "\u0dd9", "diga_kombuwa": "\u0dda", "kombu_deka": "\u0ddb",
    "kombuwa_aela_pilla": "\u0ddc", "kombuwa_diga_aela_pilla": "\u0ddd",
    "kombuwa_gayanukitta": "\u0dde", "gayanukitta": "\u0ddf",
    "diga_gayanukitta": "\u0df3",
})

# These are Unicode canonical decompositions, including recursive decomposition
# of U+0DDD. Combine these SIGN COMPONENTS, NFC-normalize, then append the result
# AFTER the base/cluster. Do not insert a pre-base kombuwa before the base codepoint.
COMPOSITE_VOWEL_COMPONENTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "\u0dda": ("\u0dd9", VIRAMA),
    "\u0ddc": ("\u0dd9", "\u0dcf"),
    "\u0ddd": ("\u0dd9", "\u0dcf", VIRAMA),
    "\u0dde": ("\u0dd9", "\u0ddf"),
})
COMPONENTS_TO_VOWEL_SIGN: Mapping[tuple[str, ...], str] = MappingProxyType({
    **{components: sign for sign, components in COMPOSITE_VOWEL_COMPONENTS.items()},
    ("\u0ddc", VIRAMA): "\u0ddd",  # partially composed input
})

# Zone membership may vary with writer/style and image bounds. The split e/o/au
# signs include core-zone kombuwa even when their right/upper parts are separate.
SIGN_ZONE_HINTS: Mapping[str, tuple[Zone, ...]] = MappingProxyType({
    VIRAMA: (Zone.UPPER,),
    "\u0dcf": (Zone.CORE,), "\u0dd0": (Zone.CORE,), "\u0dd1": (Zone.CORE,),
    "\u0dd2": (Zone.UPPER,), "\u0dd3": (Zone.UPPER,),
    "\u0dd4": (Zone.LOWER,), "\u0dd6": (Zone.LOWER,),
    "\u0dd8": (Zone.CORE, Zone.LOWER), "\u0df2": (Zone.CORE, Zone.LOWER),
    "\u0dd9": (Zone.CORE,), "\u0dda": (Zone.CORE, Zone.UPPER),
    "\u0ddb": (Zone.CORE,), "\u0ddc": (Zone.CORE,),
    "\u0ddd": (Zone.CORE, Zone.UPPER), "\u0dde": (Zone.CORE,),
    "\u0ddf": (Zone.CORE,), "\u0df3": (Zone.CORE,),
    "\u0d81": (Zone.UPPER,), "\u0d82": (Zone.CORE, Zone.UPPER),
    "\u0d83": (Zone.CORE,),
})

# Explicit ZWJ-bearing rendering conventions. They are not canonically equal
# to the same strings without ZWJ. Preserve joiners in labels, predictions and
# references; do not strip them or silently score them as interchangeable.
CONJUNCT_SUFFIXES: Mapping[str, str] = MappingProxyType({
    "yansaya": VIRAMA + ZWJ + CONSONANTS["ya"],
    "rakaransaya": VIRAMA + ZWJ + CONSONANTS["ra"],
})
CONJUNCT_PREFIXES: Mapping[str, str] = MappingProxyType({
    "repaya": CONSONANTS["ra"] + VIRAMA + ZWJ,
})

SINHALA_LETTERS = frozenset((*INDEPENDENT_VOWELS.values(), *CONSONANTS.values()))
SINHALA_MARKS = frozenset((VIRAMA, *FINAL_SIGNS.values(), *filter(None, VOWEL_SIGNS.values())))
SINHALA_ASSIGNED_CHARACTERS = frozenset((
    *SINHALA_LETTERS, *SINHALA_MARKS, *SINHALA_DIGITS.values(), *PUNCTUATION.values(),
))
UNICODE_NAMES: Mapping[str, str] = MappingProxyType({
    char: unicodedata.name(char) for char in sorted(SINHALA_ASSIGNED_CHARACTERS)
})


def normalize_sinhala(text: str) -> str:
    """Canonical NFC only: retain spaces, joiners, punctuation and other scripts.

    This is not a grapheme splitter, spelling validator or phonetic normalizer.
    Use the same normalization for labels, references, lexicon and predictions.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return unicodedata.normalize("NFC", text)


def compose_consonant_vowel(consonant: str, vowel: str = "a") -> str:
    """Compose a single consonant plus a vowel TOKEN, or 'virama'.

    Accept the consonant's project token (e.g. 'ka') or its Unicode character.
    No visual reordering heuristics or combining-class sorting are applied:
    most Sinhala vowel marks have combining class zero. Multi-zone candidate
    association and full conjunct/grapheme decoding belong in decoder.py.
    """
    if not isinstance(consonant, str) or not isinstance(vowel, str):
        raise ConfigError("consonant and vowel must be strings")
    base = CONSONANTS.get(consonant, consonant)
    if base not in CONSONANTS.values():
        raise ConfigError(f"unknown Sinhala consonant: {consonant!r}")
    if vowel == "virama":
        sign = VIRAMA
    elif vowel in VOWEL_SIGNS:
        sign = VOWEL_SIGNS[vowel]
    else:
        raise ConfigError(f"unknown vowel token: {vowel!r}")
    return normalize_sinhala(base + sign)
