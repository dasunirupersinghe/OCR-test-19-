"""Spatial segmentation for offline Sinhala handwriting; Python >= 3.10.

Dependencies: numpy, opencv-python, scipy, scikit-image. Input is a 2-D mask
with foreground INK=1 or 255 and background=0. Returned masks use uint8 0/1.
No polarity guessing, training, file access, global RNG changes or network calls.

    from config import get_config
    from segmentation import SegmentationPipeline, SegmentationOptions
    result = SegmentationPipeline(get_config("B4")).run(binary_word)
    # A measured development-set width enables calibrated trigger decisions:
    options = SegmentationOptions(calibrated_median_glyph_width=24.0)
    result = SegmentationPipeline(mode="B4", options=options).run(binary_word)

B0/B1 use actual uniform sliding-window proposals, without HPP zone slicing or
skeletonization. B2--B4 enable zones and Zhang--Suen cutting according to flags.
Windows are proposals and can overlap; cut hypotheses are disjoint partitions.
Every candidate retains its FULL vertical span and zone crops, so detached marks
are never discarded just because the cutter operates on the core zone.

All boxes are HALF-OPEN [x_min,x_max) x [y_min,y_max). In particular the user's
inclusive lower-core row is ``core.normalized_bbox.y_max - 1``. Normalized boxes
refer to a word's normalized image. Global boxes/polygons refer to pipeline input
coordinates, or to input_to_global when supplied. Global integer boxes round
outward; polygons preserve the exact continuous pixel-edge geometry. For an
OpenCV affine matrix defined on pixel CENTERS, call pixel_center_affine_to_edges
before passing it as input_to_global (e.g. preprocessing's inverse deskew).

Legacy config constants H_TARGET=64, LAMBDA_CURV=0.5,
LIGATURE_WIDTH_FACTOR=1.4 and ABLATION_PRESET='B4' are supported with safe
fallbacks. Existing structured config values take precedence, including File 1's
lambda_curv=0.35. Explicit SegmentationOptions override both. Missing fields and
the effective configuration are recorded; malformed values raise errors.

The cut objective is EXACTLY VPP + lambda*abs(y[x+1]-2*y[x]+y[x-1]). A column
with multiple skeleton pixels does not define a single-valued y(x); it is not
averaged into a fictional path. Only single-valued three-column neighborhoods
have defined curvature. Branches, missing path samples, uncalibrated widths and
marginal costs retain the intact hypothesis. If no valid curvature exists, no
mathematically unsupported cut is committed. Cut slices do not erase a column.

Cut acceptance uses raw_cost/core_height with a separately documented heuristic
threshold and margin; these are NOT calibrated probabilities. Data-dependent
component/loop width estimates are explicitly provisional, never described as
development calibration. Real Sinhala data is required to calibrate these rules.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
import skimage
from skimage.morphology import skeletonize

try:
    import config as _project_config
except ModuleNotFoundError as exc:
    if exc.name != "config":
        raise
    _project_config = None


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
PIPELINE_VERSION = "1.0.0"
BinaryMask: TypeAlias = NDArray[np.uint8]
Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = tuple[Point, Point, Point, Point]

H_TARGET = getattr(_project_config, "H_TARGET", 64)
LAMBDA_CURV = getattr(_project_config, "LAMBDA_CURV", 0.5)
LIGATURE_WIDTH_FACTOR = getattr(_project_config, "LIGATURE_WIDTH_FACTOR", 1.4)
ABLATION_PRESET = getattr(_project_config, "ABLATION_PRESET", "B4")


class SegmentationError(ValueError):
    """An input or processing invariant prevents segmentation."""


class SegmentationConfigError(ValueError):
    """An explicitly supplied setting is invalid; it was not defaulted away."""


def _number(name: str, value: Any, minimum: float = 0.0, maximum: float = math.inf,
            *, integer: bool = False, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentationConfigError(f"{name} must be a real number")
    if integer and type(value) is not int:
        raise SegmentationConfigError(f"{name} must be an integer")
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise SegmentationConfigError(f"invalid {name}: {value!r}")


def _freeze(array: NDArray[Any], dtype: Any = None) -> NDArray[Any]:
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _binary(image: NDArray[Any], *, max_pixels: int = 40_000_000) -> BinaryMask:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or 0 in image.shape:
        raise SegmentationError("expected a nonempty 2-D binary ndarray")
    if image.size > max_pixels:
        raise SegmentationError("input exceeds the configured pixel limit")
    if not (np.issubdtype(image.dtype, np.number) or image.dtype == np.bool_):
        raise SegmentationError("binary mask must have a numeric or bool dtype")
    if np.issubdtype(image.dtype, np.complexfloating):
        raise SegmentationError("complex masks are unsupported")
    if not (np.all((image == 0) | (image == 1)) or np.all((image == 0) | (image == 255))):
        raise SegmentationError("expected ink/background values {0,1} or {0,255}; polarity is not inferred")
    return np.ascontiguousarray(image != 0, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in self.as_tuple()):
            raise TypeError("bounding coordinates must be Python integers")
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("bounding box endpoints are reversed")

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y_min, self.y_max), slice(self.x_min, self.x_max)


def _affine(matrix: NDArray[Any] | None) -> NDArray[np.float64]:
    if matrix is None:
        return np.eye(3, dtype=np.float64)
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape == (2, 3):
        result = np.vstack((result, (0.0, 0.0, 1.0)))
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise SegmentationError("input_to_global must be a finite 2x3 or 3x3 affine matrix")
    if not np.allclose(result[2], (0, 0, 1), rtol=0, atol=1e-12):
        raise SegmentationError("perspective transforms are not affine coordinate links")
    if abs(float(np.linalg.det(result[:2, :2]))) <= 1e-12:
        raise SegmentationError("input_to_global must be invertible")
    return result.copy()


def pixel_center_affine_to_edges(matrix: NDArray[Any]) -> NDArray[np.float64]:
    """Convert an OpenCV pixel-center affine to this module's pixel-edge frame."""
    transform = _affine(matrix)
    plus = np.array(((1, 0, 0.5), (0, 1, 0.5), (0, 0, 1)), dtype=np.float64)
    minus = np.array(((1, 0, -0.5), (0, 1, -0.5), (0, 0, 1)), dtype=np.float64)
    return plus @ transform @ minus


@dataclass(frozen=True, slots=True)
class _CoordinateMap:
    normalized_to_global: NDArray[np.float64]

    def polygon(self, box: BoundingBox) -> Polygon:
        points = np.array(((box.x_min, box.y_min, 1), (box.x_max, box.y_min, 1),
                           (box.x_max, box.y_max, 1), (box.x_min, box.y_max, 1)), dtype=np.float64)
        transformed = points @ self.normalized_to_global.T
        return tuple((float(x), float(y)) for x, y in transformed[:, :2])  # type: ignore[return-value]

    def box(self, box: BoundingBox) -> BoundingBox:
        points = np.asarray(self.polygon(box))
        # Avoid an extra outward pixel from floating-point noise at exact edges.
        rounded = np.rint(points)
        points = np.where(np.isclose(points, rounded, atol=1e-9, rtol=0), rounded, points)
        lo, hi = points.min(axis=0), points.max(axis=0)
        return BoundingBox(math.floor(lo[0]), math.floor(lo[1]), math.ceil(hi[0]), math.ceil(hi[1]))


@dataclass(frozen=True, slots=True)
class ZoneRegion:
    zone: Literal["upper", "core", "lower"]
    image: BinaryMask
    normalized_bbox: BoundingBox
    global_bbox: BoundingBox
    global_polygon: Polygon
    word_id: int


@dataclass(frozen=True, slots=True)
class LoopRegion:
    loop_id: int
    component_id: int
    normalized_bbox: BoundingBox
    centroid: Point
    global_centroid: Point
    area: int


@dataclass(frozen=True, slots=True)
class CutDecision:
    x: int
    cost: float
    normalized_cost: float
    curvature: float | None
    accepted: bool
    ambiguous: bool
    left_loop_id: int
    right_loop_id: int
    gap_width: int
    reason: str


@dataclass(frozen=True, slots=True)
class DiacriticLink:
    mark_id: str
    zone: str
    normalized_bbox: BoundingBox
    global_bbox: BoundingBox
    parent_candidate_ids: tuple[str, ...]
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class GlyphCandidate:
    candidate_id: str
    word_id: int
    image: BinaryMask
    normalized_bbox: BoundingBox
    global_bbox: BoundingBox
    global_polygon: Polygon
    zones: Mapping[str, ZoneRegion]
    provenance: str
    associated_mark_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SegmentationHypothesis:
    hypothesis_id: str
    candidate_ids: tuple[str, ...]
    kind: Literal["intact", "split", "sliding_windows", "components"]
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class SegmentationDiagnostics:
    hpp: NDArray[np.int64]
    smoothed_hpp: NDArray[np.float64]
    vpp: NDArray[np.int64]
    skeleton: BinaryMask
    skeleton_overlay: NDArray[np.uint8]
    loops: tuple[LoopRegion, ...]
    cuts: tuple[CutDecision, ...]
    curvature: NDArray[np.float64]
    curvature_valid: NDArray[np.bool_]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WordSegmentation:
    word_id: int
    input_bbox: BoundingBox
    normalized_mask: BinaryMask
    zones: Mapping[str, ZoneRegion]
    candidates: tuple[GlyphCandidate, ...]
    hypotheses: tuple[SegmentationHypothesis, ...]
    diagnostics: SegmentationDiagnostics
    diacritic_links: tuple[DiacriticLink, ...]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    input_shape: tuple[int, int]
    words: tuple[WordSegmentation, ...]
    metadata: dict[str, Any]

    @property
    def candidates(self) -> tuple[GlyphCandidate, ...]:
        return tuple(candidate for word in self.words for candidate in word.candidates)


@dataclass(frozen=True, slots=True)
class SegmentationOptions:
    target_height: int | None = None
    lambda_curv: float | None = None
    ligature_width_factor: float | None = None
    calibrated_median_glyph_width: float | None = None
    baseline_window_width: int | None = None
    baseline_stride: int | None = None
    interpolation: Literal["auto", "area", "nearest"] = "auto"
    marginal_cost_threshold: float = 0.12
    marginal_cost_margin: float = 0.06
    max_hypotheses: int = 16
    protect_core_components: bool = True

    def __post_init__(self) -> None:
        for name in ("target_height", "baseline_window_width", "baseline_stride"):
            value = getattr(self, name)
            if value is not None:
                _number(name, value, 1, integer=True)
        if self.lambda_curv is not None:
            _number("lambda_curv", self.lambda_curv)
        for name in ("ligature_width_factor", "calibrated_median_glyph_width"):
            if getattr(self, name) is not None:
                _number(name, getattr(self, name), 0, positive=True)
        for name in ("marginal_cost_threshold", "marginal_cost_margin"):
            _number(name, getattr(self, name))
        _number("max_hypotheses", self.max_hypotheses, 2, 1024, integer=True)
        if self.interpolation not in ("auto", "area", "nearest"):
            raise SegmentationConfigError("interpolation must be auto, area or nearest")
        if type(self.protect_core_components) is not bool:
            raise SegmentationConfigError("protect_core_components must be bool")
        if (self.baseline_stride is not None and self.baseline_window_width is not None
                and self.baseline_stride > self.baseline_window_width):
            raise SegmentationConfigError("baseline stride cannot leave gaps between windows")


@dataclass(frozen=True, slots=True)
class _Settings:
    target_height: int = 64
    max_normalized_width: int = 8192
    max_input_pixels: int = 40_000_000
    word_dilation_kernel: tuple[int, int] = (15, 3)
    min_component_area: int = 2
    crop_padding: int = 2
    hpp_smoothing_sigma: float = 1.0
    upper_valley_search: tuple[float, float] = (0.10, 0.45)
    lower_valley_search: tuple[float, float] = (0.55, 0.90)
    min_core_height: int = 8
    touching_width_factor: float = 1.4
    lambda_curv: float = 0.5
    min_hole_area: int = 2
    min_segment_width: int = 2
    cut_safety_margin: int = 1
    max_candidates_per_word: int = 128

    def __post_init__(self) -> None:
        for name in ("target_height", "max_normalized_width", "max_input_pixels", "min_component_area",
                     "min_core_height", "min_hole_area", "min_segment_width", "max_candidates_per_word"):
            _number(name, getattr(self, name), 1, integer=True)
        for name in ("crop_padding", "cut_safety_margin"):
            _number(name, getattr(self, name), 0, integer=True)
        for name in ("hpp_smoothing_sigma", "touching_width_factor"):
            _number(name, getattr(self, name), 0, positive=True)
        _number("lambda_curv", self.lambda_curv)
        if not isinstance(self.word_dilation_kernel, tuple) or len(self.word_dilation_kernel) != 2:
            raise SegmentationConfigError("word_dilation_kernel must be a pair")
        for size in self.word_dilation_kernel:
            _number("word_dilation_kernel", size, 1, integer=True)
        for name in ("upper_valley_search", "lower_valley_search"):
            interval = getattr(self, name)
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise SegmentationConfigError(f"{name} must be a pair")
            for bound in interval:
                _number(name, bound, 0, 1)
            if interval[0] >= interval[1]:
                raise SegmentationConfigError(f"{name} must be increasing")
        if self.upper_valley_search[1] >= self.lower_valley_search[0]:
            raise SegmentationConfigError("upper/lower valley intervals overlap")


@dataclass(frozen=True, slots=True)
class _Flags:
    word_localization: bool = False
    aspect_normalization: bool = True
    glyph_segmentation: bool = True
    zone_partition: bool = True
    ligature_cutter: bool = True
    retain_ambiguous_segments: bool = True


def _read(source: Any, name: str, missing: list[str], default: Any, section: str = "segmentation") -> Any:
    marker = object()
    group = source.get(section, marker) if isinstance(source, Mapping) else getattr(source, section, marker)
    value = group.get(name, marker) if isinstance(group, Mapping) else getattr(group, name, marker)
    if value is marker:
        missing.append(f"{section}.{name}")
        return default
    return value


def _resolve(config: Any, mode: str | None, options: SegmentationOptions) -> tuple[Any, str, _Settings, _Flags, list[str]]:
    supplied = config.get("mode") if isinstance(config, Mapping) else getattr(config, "mode", None)
    selected = mode if mode is not None else supplied if supplied is not None else ABLATION_PRESET
    if not isinstance(selected, str) or selected.upper() not in ("B0", "B1", "B2", "B3", "B4"):
        raise SegmentationConfigError("ablation mode must be B0, B1, B2, B3 or B4")
    selected = selected.upper()
    if supplied is not None and (not isinstance(supplied, str) or supplied.upper() != selected):
        raise SegmentationConfigError("mode conflicts with config.mode")
    if config is None and _project_config is not None:
        factory = getattr(_project_config, "get_config", None)
        config = factory(selected) if callable(factory) else getattr(_project_config, "DEFAULT_CONFIG", _project_config)
    missing: list[str] = []
    defaults = _Settings(target_height=H_TARGET, lambda_curv=LAMBDA_CURV,
                         touching_width_factor=LIGATURE_WIDTH_FACTOR)
    aliases = {"target_height": "H_TARGET", "lambda_curv": "LAMBDA_CURV",
               "touching_width_factor": "LIGATURE_WIDTH_FACTOR"}
    values: dict[str, Any] = {}
    for item in fields(defaults):
        fallback = getattr(defaults, item.name)
        if item.name in aliases:
            alias = aliases[item.name]
            fallback = config.get(alias, fallback) if isinstance(config, Mapping) else getattr(config, alias, fallback)
        section, name = ("input", "max_decoded_pixels") if item.name == "max_input_pixels" else ("segmentation", item.name)
        value = _read(config, name, missing, fallback, section)
        if isinstance(fallback, tuple) and isinstance(value, list):
            value = tuple(value)
        values[item.name] = value
    for source, destination in (("target_height", "target_height"), ("lambda_curv", "lambda_curv"),
                                ("ligature_width_factor", "touching_width_factor")):
        if getattr(options, source) is not None:
            values[destination] = getattr(options, source)
    settings = _Settings(**values)
    flags: dict[str, bool] = {}
    for item in fields(_Flags):
        default = getattr(_Flags(), item.name)
        if item.name in ("zone_partition", "ligature_cutter", "retain_ambiguous_segments"):
            default = selected in ("B2", "B3", "B4")
        value = _read(config, item.name, missing, default, "flags")
        if type(value) is not bool:
            raise SegmentationConfigError(f"flags.{item.name} must be bool")
        flags[item.name] = value
    resolved = _Flags(**flags)
    if resolved.ligature_cutter and not (resolved.zone_partition and resolved.glyph_segmentation):
        raise SegmentationConfigError("ligature cutting requires zone and glyph segmentation")
    return config, selected, settings, resolved, missing


def _runs(occupied: NDArray[Any]) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(occupied.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


class WordLocalizer:
    """Morphological word localization for a SINGLE short text line.

    Horizontally overlapping components merge to retain detached upper/lower
    marks, including one-pixel marks. Supply multi-line pages one line at a time.
    min_component_area affects width estimation, never mark deletion here.
    """

    def __init__(self, settings: _Settings) -> None:
        self.settings = settings

    def localize(self, binary: BinaryMask) -> tuple[BoundingBox, ...]:
        if not np.any(binary):
            return ()
        dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT,
                                                             self.settings.word_dilation_kernel))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        boxes: list[BoundingBox] = []
        for label in range(1, count):
            x, y, w, h, _ = map(int, stats[label])
            yy, xx = np.nonzero((labels[y:y+h, x:x+w] == label) & (binary[y:y+h, x:x+w] > 0))
            if xx.size:
                boxes.append(BoundingBox(x + int(xx.min()), y + int(yy.min()),
                                         x + int(xx.max()) + 1, y + int(yy.max()) + 1))
        merged: list[BoundingBox] = []
        for box in sorted(boxes, key=lambda box: (box.x_min, box.y_min)):
            if merged and box.x_min < merged[-1].x_max:
                old = merged.pop()
                box = BoundingBox(old.x_min, min(old.y_min, box.y_min), max(old.x_max, box.x_max),
                                  max(old.y_max, box.y_max))
            merged.append(box)
        padding = self.settings.crop_padding
        height, width = binary.shape
        # Custom dilation/padding settings may otherwise let adjacent crops
        # ingest each other's handwriting. Share the gap at its midpoint and
        # clip padding there, without merging two distinct word hypotheses.
        boundaries = [(left.x_max+right.x_min)//2 for left, right in zip(merged, merged[1:])]
        padded: list[BoundingBox] = []
        for index, box in enumerate(merged):
            left_limit = boundaries[index-1] if index else 0
            right_limit = boundaries[index] if index < len(boundaries) else width
            padded.append(BoundingBox(max(left_limit, box.x_min-padding), max(0, box.y_min-padding),
                                      min(right_limit, box.x_max+padding), min(height, box.y_max+padding)))
        return tuple(padded)


class ZoneSegmenter:
    def __init__(self, settings: _Settings, options: SegmentationOptions) -> None:
        self.settings, self.options = settings, options

    def normalize(self, binary: BinaryMask, *, enabled: bool = True) -> tuple[BinaryMask, dict[str, Any]]:
        height, width = binary.shape
        target = self.settings.target_height if enabled else height
        new_width = max(1, (width * target) // height) if enabled else width
        if new_width > self.settings.max_normalized_width:
            raise SegmentationError("normalized word exceeds max_normalized_width; aspect ratio was not distorted")
        if new_width * target > self.settings.max_input_pixels:
            raise SegmentationError("normalized image exceeds pixel limit")
        interpolation = self.options.interpolation
        interpolation = ("area" if target < height else "nearest") if interpolation == "auto" else interpolation
        if (target, new_width) == binary.shape:
            result = binary.copy()
        elif interpolation == "area":
            # uint8 AREA would round away low-coverage strokes; float coverage
            # followed by >0 keeps foreground contributions during reduction.
            result = (cv2.resize(binary.astype(np.float32), (new_width, target),
                                 interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        else:
            result = cv2.resize(binary, (new_width, target), interpolation=cv2.INTER_NEAREST)
        return _freeze(result, np.uint8), {
            "enabled": enabled, "original_shape": [height, width], "normalized_shape": [target, new_width],
            "scale_x": new_width / width, "scale_y": target / height,
            "interpolation": interpolation, "area_threshold": "any_positive_coverage",
        }

    def slice(self, mask: BinaryMask, coordinates: _CoordinateMap, word_id: int,
              *, enabled: bool = True) -> tuple[Mapping[str, ZoneRegion], NDArray[np.int64], NDArray[np.float64], dict[str, Any]]:
        height, width = mask.shape
        hpp = mask.sum(axis=1, dtype=np.int64)
        smooth = gaussian_filter1d(hpp.astype(np.float64), self.settings.hpp_smoothing_sigma, mode="nearest")
        upper, lower = 0, height
        details: dict[str, Any] = {"enabled": enabled, "half_open_bounds": True}
        if enabled and hpp.max(initial=0) > 0 and np.ptp(smooth) > 1e-8 and height >= 3:
            central_lo, central_hi = int(height * .2), max(int(height * .8), int(height * .2) + 1)
            peak = central_lo + int(np.argmax(smooth[central_lo:central_hi]))
            u0, u1 = self.settings.upper_valley_search
            l0, l1 = self.settings.lower_valley_search
            upper_rows = np.arange(max(0, int(height*u0)), min(peak, max(1, int(math.ceil(height*u1)))))
            lower_rows = np.arange(max(peak+1, int(height*l0)), min(height, int(math.ceil(height*l1))))
            if upper_rows.size:
                values = smooth[upper_rows]
                ties = upper_rows[np.isclose(values, values.min(), rtol=1e-7, atol=1e-10)]
                upper = int(ties[-1]) + 1
            if lower_rows.size:
                values = smooth[lower_rows]
                ties = lower_rows[np.isclose(values, values.min(), rtol=1e-7, atol=1e-10)]
                lower = int(ties[0])
            minimum = min(height, self.settings.min_core_height)
            if lower - upper < minimum:
                upper = max(0, min(peak - minimum//2, height-minimum))
                lower = upper + minimum
            proposed = [upper, lower]
            if self.options.protect_core_components:
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                ids = np.unique(labels[max(0, peak-1):min(height, peak+2)])
                ids = ids[ids != 0]
                if ids.size:
                    # HPP valleys can pass through the two sides of a loop.
                    # Keep the connected central body intact rather than open it.
                    upper = min(upper, int(stats[ids, cv2.CC_STAT_TOP].min()))
                    lower = max(lower, int((stats[ids, cv2.CC_STAT_TOP] + stats[ids, cv2.CC_STAT_HEIGHT]).max()))
            details.update({"peak_row": peak, "proposed_core_bounds": proposed,
                            "core_component_protection_applied": proposed != [upper, lower]})
        else:
            details["fallback"] = "zones_disabled" if not enabled else "flat_or_empty_projection"
        details.update({"upper_boundary": upper, "lower_boundary_exclusive": lower,
                        "lower_core_row_inclusive": lower-1})
        regions: dict[str, ZoneRegion] = {}
        for name, y0, y1 in (("upper", 0, upper), ("core", upper, lower), ("lower", lower, height)):
            box = BoundingBox(0, y0, width, y1)
            regions[name] = ZoneRegion(name, _freeze(mask[y0:y1], np.uint8), box,
                                       coordinates.box(box), coordinates.polygon(box), word_id)
        return MappingProxyType(regions), _freeze(hpp), _freeze(smooth), details


@dataclass(frozen=True, slots=True)
class _CoreAnalysis:
    skeleton: BinaryMask
    vpp: NDArray[np.int64]
    curvature: NDArray[np.float64]
    curvature_valid: NDArray[np.bool_]
    loops: tuple[LoopRegion, ...]
    cuts: tuple[CutDecision, ...]
    metadata: dict[str, Any]


class SkeletalLigatureCutter:
    def __init__(self, settings: _Settings, options: SegmentationOptions) -> None:
        self.settings, self.options = settings, options

    @staticmethod
    def calibrate_glyph_width(development_masks: Iterable[NDArray[Any]], *, target_height: int = 64) -> float:
        """Median normalized ink width from ISOLATED development glyphs only.

        Each crop's full height is scaled to target_height, matching word
        normalization. The caller owns split provenance and persists the result
        in SegmentationOptions. No test or recognition input updates this value.
        """
        _number("target_height", target_height, 1, integer=True)
        widths: list[float] = []
        for supplied in development_masks:
            mask = _binary(supplied)
            occupied = np.flatnonzero(mask.any(axis=0))
            if occupied.size:
                widths.append(float((occupied[-1] - occupied[0] + 1) * target_height / mask.shape[0]))
        if not widths:
            raise SegmentationError("development glyph calibration has no nonempty masks")
        return float(np.median(widths))

    def _loops(self, skeleton: BinaryMask, y_offset: int, coordinates: _CoordinateMap,
               component_labels: NDArray[np.int32]) -> tuple[LoopRegion, ...]:
        padded = np.pad(skeleton, 1)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(1 - padded, connectivity=4)
        exterior = int(labels[0, 0])
        result: list[LoopRegion] = []
        height, width = skeleton.shape
        for label in range(1, count):
            if label == exterior or stats[label, cv2.CC_STAT_AREA] < self.settings.min_hole_area:
                continue
            xp, yp, w, h, area = map(int, stats[label])
            x, y = xp-1, yp-1
            # The pixel immediately ABOVE a hole's topmost background row
            # belongs to its enclosing ink boundary. Voting over the entire
            # bounding rectangle could instead select a detached island inside
            # that hole. Padding ensures these enclosing-boundary indices exist.
            top_columns = np.flatnonzero(labels[yp, xp:xp+w] == label)
            boundary_labels = component_labels[yp-2, xp+top_columns-1]
            positive = boundary_labels[boundary_labels > 0]
            if not positive.size:
                continue
            component = int(np.argmax(np.bincount(positive)))
            cx, cy = float(centroids[label, 0]-1), float(centroids[label, 1]-1+y_offset)
            global_point = coordinates.normalized_to_global @ np.array((cx+.5, cy+.5, 1.0))
            result.append(LoopRegion(len(result), component,
                                     BoundingBox(x, y+y_offset, x+w, y+y_offset+h), (cx, cy),
                                     (float(global_point[0]), float(global_point[1])), area))
        return tuple(sorted(result, key=lambda loop: (loop.normalized_bbox.x_min, loop.centroid[1])))

    def _median_width(self, component_stats: NDArray[np.int32], loops: tuple[LoopRegion, ...],
                      core_height: int) -> tuple[float, str, bool]:
        measured = self.options.calibrated_median_glyph_width
        if measured is not None:
            return float(measured), "development_calibrated_argument", True
        widths = component_stats[1:, cv2.CC_STAT_WIDTH]
        heights = component_stats[1:, cv2.CC_STAT_HEIGHT]
        areas = component_stats[1:, cv2.CC_STAT_AREA]
        plausible = ((areas >= self.settings.min_component_area)
                     & (heights >= max(2, core_height*.35)) & (widths >= 2)
                     & (widths <= max(2, core_height*1.5)))
        selected = widths[plausible]
        if selected.size:
            return float(np.median(selected)), "within_image_component_estimate", False
        if loops:
            return float(np.median([loop.normalized_bbox.width for loop in loops])), "within_image_loop_width_proxy", False
        return float(max(1, core_height*.6)), "height_based_proxy_no_independent_glyphs", False

    def analyze(self, core: BinaryMask, coordinates: _CoordinateMap, *, y_offset: int = 0) -> _CoreAnalysis:
        height, width = core.shape
        # White/zero padding makes edge-touching foreground an interior object
        # for thinning; remove padding before linking coordinates.
        skeleton = skeletonize(np.pad(core.astype(bool), 1), method="zhang")[1:-1, 1:-1].astype(np.uint8)
        _, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        loops = self._loops(skeleton, y_offset, coordinates, component_labels)
        vpp = core.sum(axis=0, dtype=np.int64)
        counts = skeleton.sum(axis=0, dtype=np.int64)
        single = counts == 1
        ordinate = np.argmax(skeleton, axis=0).astype(np.float64)
        curvature = np.zeros(width, np.float64)
        valid = np.zeros(width, bool)
        if width >= 3:
            valid[1:-1] = single[:-2] & single[1:-1] & single[2:]
            second = np.abs(ordinate[2:] - 2.0*ordinate[1:-1] + ordinate[:-2])
            curvature[1:-1] = np.where(valid[1:-1], second, 0.0)
        median_width, width_source, calibrated = self._median_width(component_stats, loops, height)
        blocked_delta = np.zeros(width+1, np.int32)
        for loop in loops:
            blocked_delta[loop.normalized_bbox.x_min] += 1
            blocked_delta[loop.normalized_bbox.x_max] -= 1
        blocked = np.cumsum(blocked_delta[:-1]) > 0
        cuts: list[CutDecision] = []
        no_path_intervals: list[list[int]] = []
        same_component_pairs = 0
        # Pair adjacent nonoverlapping holes only within the SAME connected ink
        # component. Separate characters are not evidence of a cursive ligature.
        for _, group in itertools.groupby(sorted(loops, key=lambda loop: (loop.component_id, loop.normalized_bbox.x_min)),
                                           key=lambda loop: loop.component_id):
            group_loops = list(group)
            for left, right in zip(group_loops, group_loops[1:]):
                gap = right.normalized_bbox.x_min - left.normalized_bbox.x_max
                if gap <= self.settings.touching_width_factor * median_width:
                    continue
                same_component_pairs += 1
                component = component_stats[left.component_id]
                component_left = int(component[cv2.CC_STAT_LEFT])
                component_right = component_left + int(component[cv2.CC_STAT_WIDTH])
                lo = max(left.normalized_bbox.x_max + max(1, self.settings.cut_safety_margin),
                         component_left + self.settings.min_segment_width, 1)
                hi = min(right.normalized_bbox.x_min - self.settings.cut_safety_margin,
                         component_right - self.settings.min_segment_width + 1, width-1)
                columns = np.arange(lo, hi, dtype=np.int64)
                if not columns.size:
                    continue
                eligible = columns[valid[columns] & ~blocked[columns]]
                if not eligible.size:
                    no_path_intervals.append([lo, hi])
                    continue
                costs = vpp[eligible].astype(np.float64) + self.settings.lambda_curv * curvature[eligible]
                midpoint = (left.normalized_bbox.x_max + right.normalized_bbox.x_min) / 2.0
                order = np.lexsort((eligible, np.abs(eligible-midpoint), costs))
                best_index = int(order[0])
                x, cost = int(eligible[best_index]), float(costs[best_index])
                normalized_cost = cost / max(1, height)
                threshold, margin = self.options.marginal_cost_threshold, self.options.marginal_cost_margin
                accepted = normalized_cost <= threshold+margin
                marginal = abs(normalized_cost-threshold) <= margin
                uncertain_path = bool(np.any(~valid[columns]))
                ambiguous = marginal or uncertain_path or not calibrated
                reason = ("cost_above_acceptance_band" if not accepted else
                          "marginal_or_uncertain_path" if ambiguous else "low_cost_valid_path")
                cuts.append(CutDecision(x, cost, normalized_cost, float(curvature[x]), accepted, ambiguous,
                                        left.loop_id, right.loop_id, gap, reason))
        details = {
            "skeleton_executed": True, "skeleton_method": "zhang", "foreground_connectivity": 8,
            "background_connectivity": 4, "median_glyph_width": median_width,
            "glyph_width_source": width_source, "width_is_development_calibrated": calibrated,
            "trigger_width": self.settings.touching_width_factor * median_width,
            "candidate_loop_pairs": same_component_pairs, "no_valid_path_intervals": no_path_intervals,
            "single_valued_column_count": int(single.sum()), "valid_curvature_columns": int(valid.sum()),
            "curvature_formula": "abs(y[x+1]-2*y[x]+y[x-1])",
            "cost_formula": "VPP[x]+lambda_curv*curvature[x]",
            "decision_statistic": "raw_cost/core_height", "lambda_curv": self.settings.lambda_curv,
        }
        return _CoreAnalysis(_freeze(skeleton), _freeze(vpp), _freeze(curvature), _freeze(valid),
                             loops, tuple(sorted(cuts, key=lambda cut: cut.x)), details)


class SegmentationPipeline:
    """Run localization, normalization, zones and bounded candidate hypotheses.

    The runner is stateless between calls. Arrays are read-only snapshots.
    If word_localization=False (the default), input is ONE word including its
    margins. For a short line set that flag in get_config(..., overrides=...).
    """

    def __init__(self, config: Any = None, *, mode: str | None = None,
                 options: SegmentationOptions | None = None) -> None:
        self.options = options if options is not None else SegmentationOptions()
        if not isinstance(self.options, SegmentationOptions):
            raise TypeError("options must be SegmentationOptions")
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve(config, mode, self.options)
        self.localizer = WordLocalizer(self.settings)
        self.zone_segmenter = ZoneSegmenter(self.settings, self.options)
        self.cutter = SkeletalLigatureCutter(self.settings, self.options)

    def _windows(self, width: int, height: int) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        window = self.options.baseline_window_width or max(1, int(height*.6))
        window = min(width, window)
        stride = min(window, self.options.baseline_stride or max(1, window//2))
        starts = list(range(0, max(1, width-window+1), stride))
        if starts[-1] != width-window:
            starts.append(width-window)
        adapted = False
        cap = self.settings.max_candidates_per_word
        if len(starts) > cap:
            # Adapt the uniform proposal grid, never silently truncate the word.
            window = max(window, int(math.ceil(width/cap)))
            stride = window
            starts = list(range(0, max(1, width-window+1), stride))
            if starts[-1] != width-window:
                starts.append(width-window)
            if len(starts) > cap:
                window = min(width, int(math.ceil(width/max(1, cap-1))))
                starts = list(range(0, max(1, width-window+1), window))
                if starts[-1] != width-window:
                    starts.append(width-window)
            adapted = True
        spans = [(x, x+window) for x in starts]
        return spans, {"baseline_window_width": window, "baseline_stride": stride,
                       "baseline_grid_adapted_to_budget": adapted, "sliding_windows_may_overlap": stride < window}

    @staticmethod
    def _blocks(mask: BinaryMask, vpp: NDArray[np.int64]) -> list[tuple[int, int]]:
        occupied_runs = _runs(vpp > 0)
        width = mask.shape[1]
        if len(occupied_runs) <= 1:
            return [(0, width)]
        boundaries = [0] + [(left[1]+right[0])//2 for left, right in zip(occupied_runs, occupied_runs[1:])] + [width]
        return list(zip(boundaries, boundaries[1:]))

    def _paths(self, mask: BinaryMask, analysis: _CoreAnalysis) -> tuple[list[tuple[tuple[int, int], ...]], tuple[CutDecision, ...], dict[str, Any]]:
        blocks = self._blocks(mask, analysis.vpp)
        if not self.flags.glyph_segmentation:
            return [((0, mask.shape[1]),)], analysis.cuts, {"proposal_kind": "intact"}
        if len(blocks) > self.settings.max_candidates_per_word:
            return [((0, mask.shape[1]),)], tuple(replace(cut, accepted=False, reason="candidate_budget_retained_intact")
                                                  for cut in analysis.cuts), {"candidate_budget_fallback": True}
        alternatives: list[list[tuple[tuple[int, int], ...]]] = []
        revised = list(analysis.cuts)
        for left, right in blocks:
            block_cuts = [(i, cut) for i, cut in enumerate(revised) if cut.accepted and left < cut.x < right]
            kept: list[tuple[int, CutDecision]] = []
            last = left
            for index, cut in block_cuts:
                if cut.x-last < self.settings.min_segment_width or right-cut.x < self.settings.min_segment_width:
                    revised[index] = replace(cut, accepted=False, ambiguous=True, reason="minimum_segment_width_guard")
                else:
                    kept.append((index, cut))
                    last = cut.x
            intact = ((left, right),)
            if not kept:
                alternatives.append([intact])
                continue
            boundaries = [left, *[cut.x for _, cut in kept], right]
            split = tuple(zip(boundaries, boundaries[1:]))
            ambiguous = any(cut.ambiguous for _, cut in kept)
            if ambiguous and self.flags.retain_ambiguous_segments:
                alternatives.append([intact, split])
            elif ambiguous:
                for index, cut in kept:
                    revised[index] = replace(cut, accepted=False, reason="ambiguity_retention_disabled_kept_intact")
                alternatives.append([intact])
            else:
                alternatives.append([split])
        paths: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        proposals = itertools.chain(
            [tuple(option[0] for option in alternatives), tuple(option[-1] for option in alternatives)],
            itertools.product(*alternatives),
        )
        total_possible = math.prod(len(option) for option in alternatives)
        for choice in proposals:
            path = tuple(span for part in choice for span in part)
            if path not in seen:
                seen.add(path)
                paths.append(path)
            if len(paths) >= self.options.max_hypotheses:
                break
        distinct_spans = set(span for path in paths for span in path)
        if len(distinct_spans) > self.settings.max_candidates_per_word:
            paths = [tuple(blocks)]
            revised = [replace(cut, accepted=False, ambiguous=True, reason="candidate_budget_retained_intact")
                       if cut.accepted else cut for cut in revised]
        return paths, tuple(revised), {
            "hypotheses_possible": total_possible, "hypotheses_retained": len(paths),
            "hypotheses_truncated": total_possible > len(paths), "proposal_kind": "partition",
            "candidate_budget_fallback": len(distinct_spans) > self.settings.max_candidates_per_word,
        }

    @staticmethod
    def _candidate(span: tuple[int, int], mask: BinaryMask, zones: Mapping[str, ZoneRegion],
                   coordinates: _CoordinateMap, word_id: int, index: int, provenance: str) -> GlyphCandidate:
        left, right = span
        box = BoundingBox(left, 0, right, mask.shape[0])
        pieces: dict[str, ZoneRegion] = {}
        for name, zone in zones.items():
            local = BoundingBox(left, zone.normalized_bbox.y_min, right, zone.normalized_bbox.y_max)
            pieces[name] = ZoneRegion(name, _freeze(mask[local.slices()], np.uint8), local,
                                      coordinates.box(local), coordinates.polygon(local), word_id)
        return GlyphCandidate(f"w{word_id}-g{index}", word_id, _freeze(mask[box.slices()], np.uint8), box,
                              coordinates.box(box), coordinates.polygon(box), MappingProxyType(pieces), provenance)

    @staticmethod
    def _marks(zones: Mapping[str, ZoneRegion], candidates: Sequence[GlyphCandidate],
               coordinates: _CoordinateMap, word_id: int) -> tuple[tuple[GlyphCandidate, ...], tuple[DiacriticLink, ...]]:
        links: list[DiacriticLink] = []
        associated: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in candidates}
        for name in ("upper", "lower"):
            zone = zones[name]
            if zone.image.size == 0:
                continue
            count, _, stats, _ = cv2.connectedComponentsWithStats(zone.image, connectivity=8)
            for label in range(1, count):
                x, y, w, h, _ = map(int, stats[label])
                box = BoundingBox(x, y+zone.normalized_bbox.y_min, x+w, y+h+zone.normalized_bbox.y_min)
                parents = tuple(candidate.candidate_id for candidate in candidates
                                if candidate.normalized_bbox.x_min < box.x_max
                                and candidate.normalized_bbox.x_max > box.x_min)
                mark_id = f"w{word_id}-{name}-m{label-1}"
                links.append(DiacriticLink(mark_id, name, box, coordinates.box(box), parents, len(parents) > 1))
                for parent in parents:
                    associated[parent].append(mark_id)
        return tuple(replace(candidate, associated_mark_ids=tuple(associated[candidate.candidate_id]))
                     for candidate in candidates), tuple(links)

    def _word(self, source: BinaryMask, input_box: BoundingBox, global_matrix: NDArray[np.float64],
              word_id: int) -> WordSegmentation:
        started = time.perf_counter()
        crop = source[input_box.slices()]
        mask, normalization = self.zone_segmenter.normalize(crop, enabled=self.flags.aspect_normalization)
        height, width = mask.shape
        to_input = np.array(((input_box.width/width, 0, input_box.x_min),
                             (0, input_box.height/height, input_box.y_min), (0, 0, 1)), dtype=np.float64)
        coordinates = _CoordinateMap(global_matrix @ to_input)
        zones, hpp, smooth, zone_metadata = self.zone_segmenter.slice(mask, coordinates, word_id,
                                                                     enabled=self.flags.zone_partition)
        core = zones["core"]
        if self.flags.ligature_cutter:
            analysis = self.cutter.analyze(core.image, coordinates, y_offset=core.normalized_bbox.y_min)
        else:
            analysis = _CoreAnalysis(_freeze(np.zeros_like(core.image)), _freeze(core.image.sum(axis=0, dtype=np.int64)),
                                      _freeze(np.zeros(width, np.float64)), _freeze(np.zeros(width, bool)), (), (),
                                      {"skeleton_executed": False, "reason": "disabled_by_configuration"})
        if not self.flags.zone_partition and self.flags.glyph_segmentation:
            spans, proposal_metadata = self._windows(width, height)
            paths, cuts, provenance = [tuple(spans)], analysis.cuts, "sliding_windows"
        else:
            paths, cuts, proposal_metadata = self._paths(mask, analysis)
            provenance = "partition"
        spans = sorted(set(span for path in paths for span in path))
        candidates = tuple(self._candidate(span, mask, zones, coordinates, word_id, index, provenance)
                           for index, span in enumerate(spans))
        by_span = {span: candidate.candidate_id for span, candidate in zip(spans, candidates)}
        accepted_x = {cut.x for cut in cuts if cut.accepted}
        hypotheses: list[SegmentationHypothesis] = []
        for index, path in enumerate(paths):
            split = any(right in accepted_x for _, right in path[:-1])
            kind = ("sliding_windows" if provenance == "sliding_windows" else "split" if split
                    else "intact" if len(path) == 1 else "components")
            hypotheses.append(SegmentationHypothesis(f"w{word_id}-h{index}", tuple(by_span[span] for span in path),
                                                      kind, any(cut.ambiguous and cut.accepted for cut in cuts)))
        candidates, marks = self._marks(zones, candidates, coordinates, word_id)
        overlay = cv2.cvtColor((1-mask)*255, cv2.COLOR_GRAY2BGR)
        y0, y1 = core.normalized_bbox.y_min, core.normalized_bbox.y_max
        core_overlay = overlay[y0:y1]
        core_overlay[analysis.skeleton > 0] = (0, 0, 255)
        for cut in cuts:
            if cut.accepted:
                cv2.line(overlay, (cut.x, y0), (cut.x, max(y0, y1-1)), (255, 0, 0), 1)
        metadata = {
            "word_id": word_id, "normalization": normalization, "zones": zone_metadata,
            "cutter": analysis.metadata, "proposals": proposal_metadata,
            "normalized_to_input": to_input.tolist(),
            "normalized_to_global": coordinates.normalized_to_global.tolist(),
            "accepted_cut_coordinates": sorted(accepted_x),
            "elapsed_ms": (time.perf_counter()-started)*1000.0,
            "diagnostic_frames": {"hpp": "normalized_word_rows", "vpp": "core_columns",
                                  "skeleton": "core_local_pixels", "overlay": "normalized_word_pixels",
                                  "loops_and_cuts": "normalized_word_pixels"},
        }
        diagnostics = SegmentationDiagnostics(hpp, smooth, analysis.vpp, analysis.skeleton, _freeze(overlay),
                                              analysis.loops, cuts, analysis.curvature, analysis.curvature_valid, metadata)
        LOGGER.debug("word segmentation completed", extra={"event": "word_completed", "word_id": word_id,
                                                           "candidate_count": len(candidates)})
        return WordSegmentation(word_id, input_box, mask, zones, candidates, tuple(hypotheses), diagnostics, marks)

    def run(self, binary: NDArray[Any], *, input_to_global: NDArray[Any] | None = None) -> SegmentationResult:
        started = time.perf_counter()
        mask = _binary(binary, max_pixels=self.settings.max_input_pixels)
        transform = _affine(input_to_global)
        effective = {"pipeline_version": PIPELINE_VERSION, "mode": self.mode,
                     "settings": asdict(self.settings), "flags": asdict(self.flags), "options": asdict(self.options)}
        fingerprint = hashlib.sha256(json.dumps(effective, sort_keys=True, allow_nan=False,
                                                separators=(",", ":")).encode("utf-8")).hexdigest()
        metadata: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION, "mode": self.mode,
            "config_fingerprint": getattr(self.config, "fingerprint", None),
            "effective_configuration": effective, "effective_fingerprint": fingerprint,
            "fallback_fields": list(self.fallback_fields), "foreground_value": 1,
            "bounding_box_convention": "half_open_pixel_edges",
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "scikit_image": skimage.__version__},
        }
        if not mask.any():
            metadata.update({"status": "empty_foreground", "word_count": 0,
                             "elapsed_ms": (time.perf_counter()-started)*1000.0})
            return SegmentationResult(tuple(mask.shape), (), metadata)
        boxes = (self.localizer.localize(mask) if self.flags.word_localization else
                 (BoundingBox(0, 0, mask.shape[1], mask.shape[0]),))
        words = tuple(self._word(mask, box, transform, index) for index, box in enumerate(boxes))
        metadata.update({"status": "ok", "word_count": len(words), "word_localization": self.flags.word_localization,
                         "elapsed_ms": (time.perf_counter()-started)*1000.0})
        LOGGER.info("segmentation completed", extra={"event": "completed", "mode": self.mode,
                                                      "word_count": len(words), "elapsed_ms": metadata["elapsed_ms"]})
        return SegmentationResult(tuple(mask.shape), words, metadata)


__all__ = [
    "H_TARGET", "LAMBDA_CURV", "LIGATURE_WIDTH_FACTOR", "ABLATION_PRESET",
    "BoundingBox", "ZoneRegion", "LoopRegion", "CutDecision", "DiacriticLink", "GlyphCandidate",
    "SegmentationHypothesis", "SegmentationDiagnostics", "WordSegmentation", "SegmentationResult",
    "SegmentationOptions", "SegmentationError", "SegmentationConfigError",
    "WordLocalizer", "ZoneSegmenter", "SkeletalLigatureCutter", "SegmentationPipeline",
    "pixel_center_affine_to_edges",
]
