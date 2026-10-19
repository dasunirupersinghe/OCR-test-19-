"""Offline, CPU-only preprocessing for Sinhala word and short-line images.

Requires Python >= 3.10, NumPy and OpenCV. No training or network calls occur.
All outputs are uint8 masks with FOREGROUND INK=1 and BACKGROUND=0, not
display-ready white-page images. Use ``(1 - binary) * 255`` for that display.

Typical use::

    from config import get_config
    from preprocessing import PreprocessingPipeline, PreprocessingOptions

    pipeline = PreprocessingPipeline(get_config("B4"))
    binary, metadata = pipeline.run("word.png", debug=True, debug_dir="artifacts/debug")
    # Arrays from OpenCV are BGR/BGRA by default; for RGB/RGBA upload arrays:
    binary, metadata = pipeline.run(rgb_array, color_order="rgb")

Existing config.py flags select B0--B4. Missing attributes use documented
defaults, listed in metadata; malformed values do NOT silently become defaults.
Extra algorithms requested for File 2 live in PreprocessingOptions so config.py
remains unchanged. Border cleanup and loop repair default ON for B1--B4 and OFF
for B0. A forced binarization mode or changed options are recorded in the
effective-settings fingerprint; do not report such a run as an untouched preset.

Contour/minAreaRect estimation replaces config.py's older Hough/Radon selector;
the requested/effective estimators are recorded. Estimates are in [-45,45]. The
configured maximum applied angle (15 degrees by default) remains a safety limit;
set options.max_deskew_degrees=45 to accept the full estimate range. Positive
estimated angles mean a baseline descends rightward in image coordinates. The
same signed number passed to OpenCV's counterclockwise rotation corrects it.

Clean-scan bypass skips ONLY Retinex, CLAHE and bilateral filtering, as defined
in config.py. It applies in B4 and any other preset explicitly enabling that
flag. Line removal, geometry and binarization still run. Thresholds and repair
heuristics need validation on real Sinhala data. Inpainting estimates missing
intensities; it cannot guarantee recovery of strokes obscured by ruled lines.
File decoding is first-frame only. Pixel limits are checked after OpenCV's
native decoder returns; use process isolation for hostile untrusted uploads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import config as _project_config
except ModuleNotFoundError as exc:
    if exc.name != "config":
        raise
    _project_config = None


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
PIPELINE_VERSION = "1.0.0"
GrayImage: TypeAlias = NDArray[np.uint8]
BinaryImage: TypeAlias = NDArray[np.uint8]
ImageSource: TypeAlias = NDArray[Any] | str | Path | bytes | bytearray | memoryview
StageResult: TypeAlias = tuple[GrayImage, dict[str, Any]]


class PreprocessingError(RuntimeError):
    """A stage failed; metadata contains the completed execution trace."""

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata


class ImageValidationError(PreprocessingError):
    """Input cannot be interpreted safely under the configured image limits."""


def _number(name: str, value: Any, low: float, high: float = math.inf,
            *, positive: bool = False, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise ValueError(f"invalid {name}: {value!r}")


def _odd_window(requested: int, shape: tuple[int, ...]) -> int:
    limit = min(shape[:2])
    if limit % 2 == 0:
        limit -= 1
    return max(1, min(requested, limit))


def _uniform(gray: GrayImage) -> bool:
    return int(gray.max()) == int(gray.min())


def _otsu(gray: GrayImage) -> tuple[BinaryImage, float]:
    if _uniform(gray):
        return np.zeros_like(gray), float(gray.flat[0])
    threshold, inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return (inverse > 0).astype(np.uint8), float(threshold)


@dataclass(frozen=True, slots=True)
class PreprocessingOptions:
    """Extensions to File 1; None uses its value or the indicated derivation."""

    retinex_scales: tuple[float, ...] | None = None  # sigma/3, sigma, 3*sigma
    binarization_mode: Literal["auto", "otsu", "sauvola", "hybrid"] = "auto"
    max_deskew_degrees: float | None = None
    hybrid_variance_threshold: float = 225.0  # grayscale intensity squared
    hybrid_min_text_contrast: float = 5.0
    clean_min_illumination_uniformity: float = 0.95
    clean_min_background_brightness: float = 175.0
    inpaint_radius: float = 3.0
    border_shadow_suppression: bool | None = None
    loop_repair: bool | None = None
    border_band_fraction: float = 0.06
    border_min_span_fraction: float = 0.60

    def __post_init__(self) -> None:
        if self.retinex_scales is not None:
            if not isinstance(self.retinex_scales, tuple) or not self.retinex_scales:
                raise ValueError("retinex_scales must be a nonempty tuple")
            for sigma in self.retinex_scales:
                _number("retinex scale", sigma, 0, positive=True)
        if self.binarization_mode not in ("auto", "otsu", "sauvola", "hybrid"):
            raise ValueError("binarization_mode must be auto, otsu, sauvola or hybrid")
        if self.max_deskew_degrees is not None:
            _number("max_deskew_degrees", self.max_deskew_degrees, 0, 45, positive=True)
        _number("hybrid_variance_threshold", self.hybrid_variance_threshold, 0)
        _number("hybrid_min_text_contrast", self.hybrid_min_text_contrast, 0, 255)
        _number("clean_min_illumination_uniformity", self.clean_min_illumination_uniformity, 0, 1)
        _number("clean_min_background_brightness", self.clean_min_background_brightness, 0, 255)
        _number("inpaint_radius", self.inpaint_radius, 0, positive=True)
        _number("border_band_fraction", self.border_band_fraction, 0, 0.25, positive=True)
        _number("border_min_span_fraction", self.border_min_span_fraction, 0.5, 1)
        for name in ("border_shadow_suppression", "loop_repair"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool or None")


@dataclass(frozen=True, slots=True)
class _Settings:
    max_file_bytes: int = 20 * 1024 * 1024
    max_decoded_pixels: int = 40_000_000
    min_image_side: int = 3
    max_image_side: int = 20_000
    alpha_background: int = 255
    allowed_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    retinex_sigma: float = 15.0
    log_offset: float = 1.0
    rescale_percentiles: tuple[float, float] = (1.0, 99.0)
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (4, 4)
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 5.0
    ruled_line_min_width_fraction: float = 0.4
    ruled_line_max_thickness: int = 3
    ruled_line_kernel_min_width: int = 15
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    sauvola_dynamic_range: float = 128.0
    deskew_method: str = "hough"
    deskew_max_degrees: float = 15.0
    deskew_min_degrees: float = 0.15
    clean_min_laplacian_variance: float = 100.0
    clean_min_contrast: float = 100.0
    clean_max_background_std: float = 12.0
    min_ink_fraction: float = 0.001
    max_ink_fraction: float = 0.65

    def __post_init__(self) -> None:
        for name in ("max_file_bytes", "max_decoded_pixels", "min_image_side", "max_image_side",
                     "ruled_line_max_thickness", "ruled_line_kernel_min_width"):
            _number(name, getattr(self, name), 1, integer=True)
        _number("alpha_background", self.alpha_background, 0, 255, integer=True)
        if self.max_image_side < self.min_image_side:
            raise ValueError("max_image_side is below min_image_side")
        if not isinstance(self.allowed_extensions, tuple) or not self.allowed_extensions:
            raise ValueError("allowed_extensions must be a nonempty tuple")
        for ext in self.allowed_extensions:
            if not isinstance(ext, str) or not ext.startswith('.') or ext != ext.lower():
                raise ValueError("allowed_extensions must contain lowercase dotted extensions")
        for name in ("retinex_sigma", "log_offset", "clahe_clip_limit",
                     "bilateral_sigma_color", "bilateral_sigma_space", "sauvola_dynamic_range"):
            _number(name, getattr(self, name), 0, positive=True)
        for name in ("bilateral_diameter", "sauvola_window"):
            value = getattr(self, name)
            _number(name, value, 1 if name == "bilateral_diameter" else 3, integer=True)
            if value % 2 == 0:
                raise ValueError(f"{name} must be odd")
        if not isinstance(self.clahe_grid_size, tuple) or len(self.clahe_grid_size) != 2:
            raise ValueError("clahe_grid_size must be a pair")
        for value in self.clahe_grid_size:
            _number("clahe_grid_size", value, 1, integer=True)
        if not isinstance(self.rescale_percentiles, tuple) or len(self.rescale_percentiles) != 2:
            raise ValueError("rescale_percentiles must be a pair")
        for value in self.rescale_percentiles:
            _number("rescale_percentiles", value, 0, 100)
        if self.rescale_percentiles[0] >= self.rescale_percentiles[1]:
            raise ValueError("rescale_percentiles must be increasing")
        _number("ruled_line_min_width_fraction", self.ruled_line_min_width_fraction, 0, 1, positive=True)
        _number("sauvola_k", self.sauvola_k, 0, 1)
        _number("deskew_max_degrees", self.deskew_max_degrees, 0, 45, positive=True)
        _number("deskew_min_degrees", self.deskew_min_degrees, 0, self.deskew_max_degrees)
        _number("clean_min_laplacian_variance", self.clean_min_laplacian_variance, 0)
        _number("clean_min_contrast", self.clean_min_contrast, 0, 255)
        _number("clean_max_background_std", self.clean_max_background_std, 0, 255)
        _number("min_ink_fraction", self.min_ink_fraction, 0, 1, positive=True)
        _number("max_ink_fraction", self.max_ink_fraction, self.min_ink_fraction, 1)
        if self.min_ink_fraction == self.max_ink_fraction:
            raise ValueError("ink fraction interval is empty")
        if not isinstance(self.deskew_method, str):
            raise ValueError("deskew_method must be str")


@dataclass(frozen=True, slots=True)
class _Flags:
    quality_metrics: bool = True
    clean_scan_bypass: bool = True
    deskew: bool = True
    retinex: bool = True
    clahe: bool = True
    bilateral_denoise: bool = True
    ruled_line_removal: bool = True
    sauvola: bool = True


def _lookup(source: Any, section: str, name: str, default: Any, missing: list[str]) -> Any:
    sentinel = object()
    group = source.get(section, sentinel) if isinstance(source, Mapping) else getattr(source, section, sentinel)
    value = group.get(name, sentinel) if isinstance(group, Mapping) else getattr(group, name, sentinel)
    if value is sentinel:
        missing.append(f"{section}.{name}")
        return default
    return value


def _resolve_config(config: Any, mode: str | None) -> tuple[Any, str, _Settings, _Flags, list[str]]:
    supplied_mode = config.get("mode") if isinstance(config, Mapping) else getattr(config, "mode", None)
    selected = mode if mode is not None else supplied_mode if supplied_mode is not None else "B4"
    if not isinstance(selected, str) or selected.upper() not in ("B0", "B1", "B2", "B3", "B4"):
        raise ValueError("mode must be B0, B1, B2, B3 or B4")
    selected = selected.upper()
    if supplied_mode is not None:
        if not isinstance(supplied_mode, str):
            raise ValueError("config.mode must be a string")
        if mode is not None and supplied_mode.upper() != selected:
            raise ValueError("mode conflicts with config.mode; select a matching preset")
    if config is None and _project_config is not None:
        factory = getattr(_project_config, "get_config", None)
        config = factory(selected) if callable(factory) else getattr(_project_config, "DEFAULT_CONFIG", None)
    missing: list[str] = []
    defaults = _Settings()
    input_names = {"max_file_bytes", "max_decoded_pixels", "min_image_side", "max_image_side",
                   "alpha_background", "allowed_extensions"}
    values: dict[str, Any] = {}
    for item in fields(defaults):
        section = "input" if item.name in input_names else "preprocessing"
        value = _lookup(config, section, item.name, getattr(defaults, item.name), missing)
        if isinstance(getattr(defaults, item.name), tuple) and isinstance(value, list):
            value = tuple(value)
        values[item.name] = value
    settings = _Settings(**values)
    baseline = selected == "B0"
    flag_values: dict[str, bool] = {}
    for item in fields(_Flags):
        fallback = True if item.name in ("quality_metrics", "deskew") else not baseline
        value = _lookup(config, "flags", item.name, fallback, missing)
        if type(value) is not bool:
            raise ValueError(f"flags.{item.name} must be bool")
        flag_values[item.name] = value
    flags = _Flags(**flag_values)
    if flags.clean_scan_bypass and not flags.quality_metrics:
        raise ValueError("clean_scan_bypass requires quality_metrics")
    return config, selected, settings, flags, missing


class ImageIntake:
    def __init__(self, settings: _Settings) -> None:
        self.settings = settings

    def load(self, source: ImageSource, *, color_order: str = "bgr") -> StageResult:
        """Decode first frame; float arrays with max<=1 are unit-range intensities."""
        if color_order not in ("bgr", "rgb"):
            raise ImageValidationError("color_order must be 'bgr' or 'rgb'")
        details: dict[str, Any] = {"source_type": type(source).__name__}
        encoded = False
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() not in self.settings.allowed_extensions:
                raise ImageValidationError(f"unsupported image extension: {path.suffix!r}")
            try:
                size = path.stat().st_size
                if size > self.settings.max_file_bytes:
                    raise ImageValidationError("encoded image exceeds max_file_bytes")
                # Bound the read even if the file grows after stat().
                with path.open('rb') as handle:
                    payload = handle.read(self.settings.max_file_bytes + 1)
            except OSError as exc:
                raise ImageValidationError("image file could not be read") from exc
            encoded = True
        elif isinstance(source, (bytes, bytearray, memoryview)):
            size = source.nbytes if isinstance(source, memoryview) else len(source)
            if size > self.settings.max_file_bytes:
                raise ImageValidationError("encoded image exceeds max_file_bytes")
            payload = bytes(source)
            encoded = True
        elif isinstance(source, np.ndarray):
            image = source
        else:
            raise ImageValidationError("source must be an ndarray, image path or encoded bytes")
        if encoded:
            if not payload or len(payload) > self.settings.max_file_bytes:
                raise ImageValidationError("encoded image is empty or exceeds max_file_bytes")
            try:
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
            except cv2.error as exc:
                raise ImageValidationError("native image decoder rejected the input") from exc
            if image is None:
                raise ImageValidationError("input is not a decodable image")
            details["encoded_bytes"] = len(payload)
            # OpenCV decoders always return BGR/BGRA, independent of array preference.
            effective_order = "bgr"
        else:
            effective_order = color_order
        if image.ndim not in (2, 3) or image.size == 0:
            raise ImageValidationError("expected a nonempty HxW, HxWx1, HxWx3 or HxWx4 image")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ImageValidationError("unsupported channel count")
        h, w = image.shape[:2]
        if min(h, w) < self.settings.min_image_side:
            raise ImageValidationError("image is smaller than min_image_side")
        if max(h, w) > self.settings.max_image_side or h * w > self.settings.max_decoded_pixels:
            raise ImageValidationError("decoded image exceeds configured dimension/pixel limits")
        details.update({"original_shape": list(image.shape), "original_dtype": str(image.dtype),
                        "color_order": effective_order, "alpha_composited": False})
        if image.dtype == np.uint8:
            scaled = image
            details["intensity_conversion"] = "uint8_unchanged"
        elif image.dtype == np.uint16:
            scaled = np.rint(image.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
            details["intensity_conversion"] = "uint16_full_range"
        elif image.dtype == np.bool_:
            scaled = image.astype(np.uint8) * 255
            details["intensity_conversion"] = "bool_intensity"
        elif np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all():
                raise ImageValidationError("floating-point image contains NaN or infinity")
            low, high = float(image.min()), float(image.max())
            if low < 0 or high > 255:
                raise ImageValidationError("floating-point intensities must be in [0,1] or [0,255]")
            factor = 255.0 if high <= 1.0 else 1.0
            scaled = np.rint(image * factor).astype(np.uint8)
            details["intensity_conversion"] = "float_unit_range" if factor == 255 else "float_byte_range"
        else:
            raise ImageValidationError("supported dtypes are uint8, uint16, bool and finite float")
        if scaled.ndim == 3 and scaled.shape[2] == 4:
            alpha = scaled[..., 3:4].astype(np.float32) / 255.0
            scaled = np.rint(scaled[..., :3].astype(np.float32) * alpha
                             + self.settings.alpha_background * (1.0 - alpha)).astype(np.uint8)
            details["alpha_composited"] = True
        if scaled.ndim == 2:
            gray = scaled.copy()
        elif scaled.shape[2] == 1:
            gray = scaled[..., 0].copy()
        else:
            code = cv2.COLOR_BGR2GRAY if effective_order == "bgr" else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(scaled, code)
        return np.ascontiguousarray(gray), details


class QualityAssessor:
    def __init__(self, settings: _Settings, options: PreprocessingOptions) -> None:
        self.settings, self.options = settings, options

    def assess(self, gray: GrayImage) -> dict[str, Any]:
        # Background estimation on a thumbnail bounds quality-assessment cost.
        h, w = gray.shape
        scale = min(1.0, 256.0 / max(h, w))
        thumbnail = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        k = _odd_window(31, thumbnail.shape)
        background = cv2.morphologyEx(thumbnail, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 3.0)
        mean_background = float(background.mean())
        variation = float(background.std()) / max(mean_background, 1e-8)
        uniformity = 1.0 / (1.0 + variation)
        q05, q95 = np.percentile(gray, (5, 95))
        bright = gray[gray >= np.percentile(gray, 75)]
        ink, _ = _otsu(gray)
        return {
            "blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "contrast_rms": float(gray.std(dtype=np.float64)),
            "contrast_percentile": float(q95 - q05),
            "illumination_uniformity": float(uniformity),
            "illumination_coefficient_of_variation": float(variation),
            "estimated_background_brightness": mean_background,
            "brightest_quartile_std": float(bright.std(dtype=np.float64)),
            "otsu_ink_fraction": float(ink.mean()),
            "uniform_image": _uniform(gray),
        }

    def is_clean(self, scores: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
        s, o = self.settings, self.options
        criteria = {
            "not_uniform": not scores["uniform_image"],
            "sharp_enough": scores["blur_laplacian_variance"] >= s.clean_min_laplacian_variance,
            # File 1 explicitly defines clean_min_contrast as p95-p05, not RMS.
            "contrast_enough": scores["contrast_percentile"] >= s.clean_min_contrast,
            "bright_background": scores["estimated_background_brightness"] >= o.clean_min_background_brightness,
            "uniform_illumination": scores["illumination_uniformity"] >= o.clean_min_illumination_uniformity,
            "stable_bright_background": scores["brightest_quartile_std"] <= s.clean_max_background_std,
            "plausible_ink": s.min_ink_fraction <= scores["otsu_ink_fraction"] <= s.max_ink_fraction,
        }
        return all(criteria.values()), criteria


class SkewCorrector:
    def __init__(self, settings: _Settings, options: PreprocessingOptions) -> None:
        self.settings, self.options = settings, options

    @staticmethod
    def _angle(points: NDArray[Any]) -> tuple[float, float]:
        box = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
        vectors = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(vectors, axis=1)
        vector = vectors[int(np.argmax(lengths))]
        angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        angle = (angle + 90.0) % 180.0 - 90.0
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        ratio = float(lengths.max() / max(float(lengths.min()), 1e-8))
        return float(angle), ratio

    def apply(self, gray: GrayImage) -> StageResult:
        ink, _ = _otsu(gray)
        h, w = gray.shape
        details: dict[str, Any] = {
            "requested_estimator": self.settings.deskew_method,
            "effective_estimator": "contour_min_area_rect",
            "estimated_skew_degrees": 0.0, "applied_rotation_degrees": 0.0,
            "affine_original_to_processed": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "affine_processed_to_original": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
        if int(ink.sum()) < 8:
            return gray, {**details, "reason": "insufficient_ink"}
        # Horizontal grouping produces contour boxes for words/lines, rather
        # than treating each nearly circular Sinhala character as a baseline.
        kw = max(3, min(31, round(w * 0.04)))
        grouped = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3)))
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angles: list[float] = []
        weights: list[float] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8 or len(contour) < 3:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            # Skip border-spanning shadows and long thin rules.
            if (cw >= w * 0.75 and ch <= self.settings.ruled_line_max_thickness + 2):
                continue
            if (x == 0 or y == 0 or x + cw == w or y + ch == h) and area > h * w * 0.3:
                continue
            angle, ratio = self._angle(contour)
            if ratio >= 1.5:
                angles.append(angle)
                weights.append(area)
        if angles:
            order = np.argsort(angles)
            ordered_weights = np.asarray(weights)[order]
            index = int(np.searchsorted(np.cumsum(ordered_weights), ordered_weights.sum() * 0.5))
            angle = float(np.asarray(angles)[order[index]])
            details["estimation_source"] = "grouped_contours"
        else:
            yy, xx = np.nonzero(ink)
            angle, ratio = self._angle(np.column_stack((xx, yy)))
            if ratio < 1.5:
                return gray, {**details, "reason": "ambiguous_near_square_geometry"}
            details["estimation_source"] = "foreground_envelope"
        details.update({"estimated_skew_degrees": angle, "candidate_contours": len(angles)})
        maximum = self.options.max_deskew_degrees
        maximum = self.settings.deskew_max_degrees if maximum is None else maximum
        if abs(angle) > maximum:
            return gray, {**details, "reason": "estimate_exceeds_configured_maximum"}
        if abs(angle) < self.settings.deskew_min_degrees:
            return gray, {**details, "reason": "below_minimum_rotation"}
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = int(math.ceil(w * cosine + h * sine))
        new_h = int(math.ceil(h * cosine + w * sine))
        if new_h * new_w > self.settings.max_decoded_pixels or max(new_h, new_w) > self.settings.max_image_side:
            return gray, {**details, "reason": "expanded_canvas_exceeds_image_limits"}
        matrix[0, 2] += (new_w - 1) / 2.0 - center[0]
        matrix[1, 2] += (new_h - 1) / 2.0 - center[1]
        rotated = cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        details.update({"applied_rotation_degrees": angle,
                        "affine_original_to_processed": matrix.tolist(),
                        "affine_processed_to_original": cv2.invertAffineTransform(matrix).tolist(),
                        "expanded_shape": [new_h, new_w]})
        return rotated, details


class IlluminationRestorer:
    def __init__(self, settings: _Settings, options: PreprocessingOptions) -> None:
        self.settings, self.options = settings, options

    def retinex(self, gray: GrayImage) -> StageResult:
        sigma = self.settings.retinex_sigma
        scales = self.options.retinex_scales or (sigma / 3.0, sigma, sigma * 3.0)
        details: dict[str, Any] = {"algorithm": "multi_scale_retinex", "scales": list(scales)}
        if _uniform(gray):
            return gray.copy(), {**details, "reason": "uniform_image"}
        image = gray.astype(np.float32)
        log_image = np.log(image + self.settings.log_offset)
        response = np.zeros_like(image)
        # A scale loop is not a pixel loop: every convolution is vectorized.
        for scale in scales:
            illumination = cv2.GaussianBlur(image, (0, 0), scale, borderType=cv2.BORDER_REFLECT_101)
            response += log_image - np.log(illumination + self.settings.log_offset)
        response /= len(scales)
        low, high = np.percentile(response, self.settings.rescale_percentiles)
        details.update({"response_low": float(low), "response_high": float(high)})
        if high - low <= np.finfo(np.float32).eps:
            return gray.copy(), {**details, "reason": "degenerate_response_range"}
        restored = np.rint(np.clip((response - low) * (255.0 / (high - low)), 0, 255)).astype(np.uint8)
        return restored, details

    def clahe(self, gray: GrayImage) -> StageResult:
        if _uniform(gray):
            return gray.copy(), {"reason": "uniform_image"}
        # OpenCV accepts a grid larger than an image but gains no useful tiles.
        grid = tuple(min(size, extent) for size, extent in zip(self.settings.clahe_grid_size,
                                                              (gray.shape[1], gray.shape[0])))
        operator = cv2.createCLAHE(clipLimit=self.settings.clahe_clip_limit, tileGridSize=grid)
        return operator.apply(gray), {"clip_limit": self.settings.clahe_clip_limit, "grid": list(grid)}

    def bilateral(self, gray: GrayImage) -> StageResult:
        filtered = cv2.bilateralFilter(gray, self.settings.bilateral_diameter,
                                       self.settings.bilateral_sigma_color,
                                       self.settings.bilateral_sigma_space,
                                       borderType=cv2.BORDER_REFLECT_101)
        return filtered, {"diameter": self.settings.bilateral_diameter}


class RuledLineRemover:
    def __init__(self, settings: _Settings, options: PreprocessingOptions) -> None:
        self.settings, self.options = settings, options

    def apply(self, gray: GrayImage) -> StageResult:
        h, w = gray.shape
        minimum_width = max(self.settings.ruled_line_kernel_min_width,
                            int(math.ceil(w * self.settings.ruled_line_min_width_fraction)))
        if minimum_width > w or _uniform(gray):
            return gray.copy(), {"line_pixels": 0, "reason": "image_too_small_or_uniform"}
        window = _odd_window(31, gray.shape)
        ink, _ = _otsu(gray)
        if window >= 3:
            local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, window, 10)
            ink = np.maximum(ink, (local > 0).astype(np.uint8))
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (minimum_width, 1)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        accepted = [label for label in range(1, count)
                    if stats[label, cv2.CC_STAT_WIDTH] >= minimum_width
                    and stats[label, cv2.CC_STAT_HEIGHT] <= self.settings.ruled_line_max_thickness]
        if not accepted:
            return gray.copy(), {"line_pixels": 0, "detected_lines": 0}
        mask = np.isin(labels, accepted).astype(np.uint8)
        # Preserve vertical stroke evidence across the line instead of asking
        # inpainting to invent the entire crossing. Angled crossings may remain
        # uncertain; this is intentionally a conservative repair heuristic.
        vertical_length = 2 * self.settings.ruled_line_max_thickness + 3
        crossing = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)))
        crossing = cv2.dilate(crossing, np.ones((1, 3), np.uint8))
        protected = (mask > 0) & (crossing > 0)
        mask[protected] = 0
        pixels = int(np.count_nonzero(mask))
        if pixels == 0:
            return gray.copy(), {"line_pixels": 0, "protected_crossing_pixels": int(protected.sum())}
        repaired = cv2.inpaint(gray, mask * 255, self.options.inpaint_radius, cv2.INPAINT_TELEA)
        return repaired, {"line_pixels": pixels, "detected_lines": len(accepted),
                          "protected_crossing_pixels": int(protected.sum()), "repair": "telea"}


class Binarizer:
    def __init__(self, settings: _Settings, options: PreprocessingOptions) -> None:
        self.settings, self.options = settings, options

    def apply(self, gray: GrayImage, *, mode: str) -> StageResult:
        if mode not in ("otsu", "sauvola", "hybrid"):
            raise ValueError("binarization mode must be otsu, sauvola or hybrid")
        otsu_mask, threshold = _otsu(gray)
        details: dict[str, Any] = {"mode": mode, "otsu_threshold": threshold}
        if mode == "otsu" or _uniform(gray):
            return otsu_mask, {**details, "uniform_image": _uniform(gray)}
        window = _odd_window(self.settings.sauvola_window, gray.shape)
        if window < 3:
            return otsu_mask, {**details, "effective_mode": "otsu", "reason": "no_valid_local_window"}
        image = gray.astype(np.float64)
        mean = cv2.boxFilter(image, cv2.CV_64F, (window, window), normalize=True,
                             borderType=cv2.BORDER_REFLECT_101)
        second = cv2.boxFilter(image * image, cv2.CV_64F, (window, window), normalize=True,
                               borderType=cv2.BORDER_REFLECT_101)
        variance = np.maximum(second - mean * mean, 0.0)
        local_threshold = mean * (1.0 + self.settings.sauvola_k
                                  * (np.sqrt(variance) / self.settings.sauvola_dynamic_range - 1.0))
        sauvola_mask = (image <= local_threshold).astype(np.uint8)
        details.update({"window": window, "k": self.settings.sauvola_k,
                        "threshold_min": float(local_threshold.min()),
                        "threshold_max": float(local_threshold.max())})
        if mode == "sauvola":
            return sauvola_mask, details
        seed = (image < mean - self.options.hybrid_min_text_contrast).astype(np.uint8)
        text_region = cv2.dilate(seed, np.ones((window, window), np.uint8)) > 0
        use_sauvola = (variance >= self.options.hybrid_variance_threshold) & text_region
        hybrid = np.where(use_sauvola, sauvola_mask, otsu_mask).astype(np.uint8)
        return hybrid, {**details, "sauvola_region_fraction": float(use_sauvola.mean()),
                        "variance_threshold": self.options.hybrid_variance_threshold}


class BinaryPostprocessor:
    def __init__(self, options: PreprocessingOptions) -> None:
        self.options = options

    def suppress_border_shadows(self, binary: BinaryImage) -> StageResult:
        h, w = binary.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        band_x = max(1, int(math.ceil(w * self.options.border_band_fraction)))
        band_y = max(1, int(math.ceil(h * self.options.border_band_fraction)))
        span = self.options.border_min_span_fraction
        rejected: list[int] = []
        for label in range(1, count):
            x, y, cw, ch, _ = map(int, stats[label])
            horizontal = cw >= span * w and ch <= band_y and (y == 0 or y + ch == h)
            vertical = ch >= span * h and cw <= band_x and (x == 0 or x + cw == w)
            if horizontal or vertical:
                rejected.append(label)
        remove = np.isin(labels, rejected)
        result = binary.copy()
        result[remove] = 0
        return result, {"removed_components": len(rejected), "removed_pixels": int(remove.sum()),
                        "policy": "only_narrow_border_connected_components"}

    @staticmethod
    def repair_loops(binary: BinaryImage) -> StageResult:
        if not np.any(binary):
            return binary.copy(), {"added_pixels": 0, "reason": "empty_foreground"}
        _, labels = cv2.connectedComponents(binary, connectivity=8)
        background_count, background_labels = cv2.connectedComponents(1 - binary, connectivity=4)
        border_labels = np.unique(np.concatenate((background_labels[0], background_labels[-1],
                                                  background_labels[:, 0], background_labels[:, -1])))
        outside = np.isin(background_labels, border_labels[border_labels != 0])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        # A 3x3 closing can add TWO adjacent gap pixels. A 3x3 label guard
        # alone lets each pixel see a different component and then join them.
        # Radius-two label evidence also catches those new-to-new connections.
        guard = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        maximum = cv2.dilate(labels.astype(np.float64), guard)
        minimum = cv2.erode(np.where(labels > 0, labels, np.inf).astype(np.float64), guard)
        # A pixel may be added only next to exactly one original ink component
        # and only in exterior background. Existing holes and all original ink
        # survive. This prevents joining a detached ispilla/papilla to the body.
        additions = (closed > binary) & (maximum == minimum) & (maximum > 0) & outside
        result = binary.copy()
        result[additions] = 1
        return result, {"added_pixels": int(additions.sum()),
                        "original_enclosed_background_components": int(background_count - 1
                            - len(border_labels[border_labels != 0])),
                        "policy": "one_component_exterior_gap_closing"}


class _RunTrace:
    def __init__(self, metadata: dict[str, Any], debug: bool, debug_dir: str | Path | None) -> None:
        self.metadata = metadata
        self.directory: Path | None = None
        if debug:
            parent = Path(debug_dir) if debug_dir is not None else None
            if parent is not None:
                parent.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="preprocess-", dir=parent))
            metadata["debug_directory"] = str(self.directory.resolve())

    def record(self, name: str, image: GrayImage, *, binary: bool = False) -> None:
        if self.directory is None:
            return
        display = (1 - image) * 255 if binary else image
        destination = self.directory / f"{len(self.metadata['steps']):02d}_{name}.png"
        ok, encoded = cv2.imencode('.png', display)
        if not ok:
            raise PreprocessingError(f"could not encode debug image for {name}")
        destination.write_bytes(encoded.tobytes())
        self.metadata["debug_images"][name] = str(destination.resolve())

    def stage(self, name: str, image: GrayImage | None, operation: Callable[[], StageResult],
              *, enabled: bool = True, reason: str = "disabled_by_configuration",
              binary: bool = False) -> GrayImage:
        if not enabled:
            if image is None:
                raise PreprocessingError("a skipped stage requires an existing image", metadata=self.metadata)
            self.metadata["steps"].append({"name": name, "status": "skipped", "reason": reason,
                                           "elapsed_ms": 0.0})
            return image
        started = time.perf_counter()
        try:
            result, details = operation()
            self.record(name, result, binary=binary)
            self.metadata["steps"].append({"name": name, "status": "completed",
                                           "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                                           "output_shape": list(result.shape), "details": details})
            LOGGER.debug("preprocessing stage completed", extra={"stage": name, "event": "completed"})
            return result
        except Exception as exc:
            self.metadata.update({"status": "failed", "failed_stage": name})
            self.metadata["steps"].append({"name": name, "status": "failed",
                                           "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                                           "error_type": type(exc).__name__})
            LOGGER.error("preprocessing stage failed", extra={"stage": name, "event": "failed"})
            if isinstance(exc, PreprocessingError):
                exc.metadata = self.metadata
                raise
            raise PreprocessingError(f"preprocessing stage failed: {name}", metadata=self.metadata) from exc


class PreprocessingPipeline:
    """Reusable runner with per-call state, deterministic algorithms and no RNG.

    Partial mappings are supported for older configs. A missing config module
    uses the local B0--B4 preprocessing defaults and records every fallback.
    Logging is library-style: configure handlers/formatters in the application.
    Debug images contain source handwriting; dumping requires explicit opt-in.
    """

    def __init__(self, config: Any = None, *, mode: str | None = None,
                 options: PreprocessingOptions | None = None) -> None:
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve_config(config, mode)
        self.options = options if options is not None else PreprocessingOptions()
        if not isinstance(self.options, PreprocessingOptions):
            raise TypeError("options must be PreprocessingOptions")
        self.intake = ImageIntake(self.settings)
        self.quality = QualityAssessor(self.settings, self.options)
        self.skew = SkewCorrector(self.settings, self.options)
        self.restorer = IlluminationRestorer(self.settings, self.options)
        self.lines = RuledLineRemover(self.settings, self.options)
        self.binarizer = Binarizer(self.settings, self.options)
        self.postprocessor = BinaryPostprocessor(self.options)

    def run(self, source: ImageSource, *, color_order: Literal["bgr", "rgb"] = "bgr",
            debug: bool = False, debug_dir: str | Path | None = None) -> tuple[BinaryImage, dict[str, Any]]:
        if type(debug) is not bool:
            raise TypeError("debug must be bool")
        started = time.perf_counter()
        effective = {"mode": self.mode, "settings": asdict(self.settings),
                     "flags": asdict(self.flags), "options": asdict(self.options),
                     "skew_estimator": "contour_min_area_rect", "pipeline_version": PIPELINE_VERSION}
        fingerprint = hashlib.sha256(json.dumps(effective, sort_keys=True, allow_nan=False,
                                                separators=(',', ':')).encode('utf-8')).hexdigest()
        config_label = getattr(self.config, "experiment_label", self.mode)
        experiment_label = (config_label if self.options == PreprocessingOptions() and not self.fallback_fields
                            else f"{self.mode}-preprocessing-custom-{fingerprint[:8]}")
        metadata: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION, "mode": self.mode,
            "experiment_label": experiment_label,
            "config_fingerprint": getattr(self.config, "fingerprint", None),
            "effective_fingerprint": fingerprint, "effective_configuration": effective,
            "config_fallback_fields": list(self.fallback_fields),
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
            "foreground_value": 1, "background_value": 0,
            "steps": [], "debug_images": {}, "warnings": [],
        }
        try:
            trace = _RunTrace(metadata, debug, debug_dir)
        except OSError as exc:
            raise PreprocessingError("could not create debug directory", metadata=metadata) from exc
        gray = trace.stage("grayscale", None,
                           lambda: self.intake.load(source, color_order=color_order))
        metadata["input"] = metadata["steps"][-1]["details"]
        uniform_input = _uniform(gray)
        gray = trace.stage("quality_assessment", gray, lambda: (gray, self.quality.assess(gray)),
                           enabled=self.flags.quality_metrics)
        scores = metadata["steps"][-1].get("details", {})
        clean, criteria = self.quality.is_clean(scores) if scores else (False, {})
        bypass = self.flags.clean_scan_bypass and clean
        metadata.update({"quality_scores": scores, "clean_scan": clean,
                         "clean_scan_criteria": criteria, "restoration_bypassed": bypass})
        gray = trace.stage("deskew", gray, lambda: self.skew.apply(gray), enabled=self.flags.deskew)
        for name, enabled, operation in (
            ("retinex", self.flags.retinex, self.restorer.retinex),
            ("clahe", self.flags.clahe, self.restorer.clahe),
            ("bilateral", self.flags.bilateral_denoise, self.restorer.bilateral),
        ):
            gray = trace.stage(name, gray, lambda op=operation: op(gray), enabled=enabled and not bypass,
                               reason="clean_scan_bypass" if enabled and bypass else "disabled_by_configuration")
        gray = trace.stage("ruled_line_removal", gray, lambda: self.lines.apply(gray),
                           enabled=self.flags.ruled_line_removal)
        mode = self.options.binarization_mode
        mode = ("sauvola" if self.flags.sauvola else "otsu") if mode == "auto" else mode
        binary = trace.stage("binarization", gray, lambda: self.binarizer.apply(gray, mode=mode), binary=True)
        border_enabled = self.options.border_shadow_suppression
        repair_enabled = self.options.loop_repair
        border_enabled = self.mode != "B0" if border_enabled is None else border_enabled
        repair_enabled = self.mode != "B0" if repair_enabled is None else repair_enabled
        binary = trace.stage("border_shadow_suppression", binary,
                             lambda: self.postprocessor.suppress_border_shadows(binary),
                             enabled=border_enabled, binary=True)
        binary = trace.stage("loop_repair", binary, lambda: self.postprocessor.repair_loops(binary),
                             enabled=repair_enabled, binary=True)
        # Uniform dark inputs are ambiguous, not a page of valid foreground ink.
        if uniform_input:
            binary = np.zeros_like(binary)
            metadata["warnings"].append("Uniform input: no distinguishable handwriting; output is empty.")
        ink_fraction = float(binary.mean())
        usable = self.settings.min_ink_fraction <= ink_fraction <= self.settings.max_ink_fraction
        if not usable and not uniform_input:
            metadata["warnings"].append("Output ink fraction is outside configured quality bounds.")
        skew_record = next((step for step in metadata["steps"] if step["name"] == "deskew"), {})
        skew_details = skew_record.get("details", {})
        metadata.update({"estimated_skew_degrees": skew_details.get("estimated_skew_degrees", 0.0),
                         "skew_angle_degrees": skew_details.get("applied_rotation_degrees", 0.0),
                         "output_shape": list(binary.shape), "output_ink_fraction": ink_fraction,
                         "usable_ink_fraction": usable,
                         "status": "ok" if usable else "blank_or_uniform" if uniform_input else "quality_warning",
                         "elapsed_ms": (time.perf_counter() - started) * 1000.0})
        trace.record("final_binary", binary, binary=True)
        LOGGER.info("preprocessing completed", extra={"event": "completed", "mode": self.mode,
                                                       "status": metadata["status"],
                                                       "elapsed_ms": metadata["elapsed_ms"]})
        return np.ascontiguousarray(binary, dtype=np.uint8), metadata


__all__ = [
    "PreprocessingPipeline", "PreprocessingOptions", "PreprocessingError", "ImageValidationError",
    "ImageIntake", "QualityAssessor", "SkewCorrector", "IlluminationRestorer",
    "RuledLineRemover", "Binarizer", "BinaryPostprocessor",
]
