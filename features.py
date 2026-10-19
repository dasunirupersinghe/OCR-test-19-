"""Deterministic CPU features for Sinhala glyph candidates; Python >= 3.10.

    from features import FeaturePipeline, FeatureOptions
    baseline = FeaturePipeline(mode="B4").extract(binary_glyph)
    improved = FeaturePipeline(mode="B4", options=FeatureOptions(schema_version="v2"))
    batch = improved.run(segmentation_result)

Inputs are 2-D masks with INK=1 or 255 and BACKGROUND=0. There is no implicit
cropping, resizing, denoising, spur pruning, class fitting or network access.
Use IDENTICAL canvas preparation for prototype calibration and inference.
Candidate-local masks are used; word-level core skeletons are never reused for
different candidates. Full glyphs and their zone views remain separate results.

v1 preserves File 1's descriptor layout: K*K LSE angles, optional K*K occupancy,
then C,H,chi, endpoint/3-way/4+-way counts, their three (x,y) centroids, and padded
hole centroids. K=4 and four hole slots give 16+20=36 default B3/B4 values.
v2 explicitly enables occupancy, uses isotropic pixel coordinates for LSE, and
keeps redundant chi=C-H in diagnostics only: 32+19=51 default B3/B4 values.
These schemas must have separate prototype/scaler metadata. Graph junction
counts in BOTH schemas are true branch counts, not raw neighbor-count maxima.

LSE is deliberately the specified y-from-x regression, not PCA/TLS. v1 uses
cell-local coordinates normalized independently to [0,1]; v2 uses pixel units
to preserve geometric slope in rectangular cells. y increases DOWNWARD. Scalar
angles are 2*atan(beta1)/pi; exact vertical degeneracy maps to +1. Empty,
singleton and isotropic windows use the configured fallback. A thick vertical
stroke can still have slope zero: inspect occupancy, validity, coherence and
regression fit diagnostics. Scalar angle wraparound is a known baseline limit.

Counts/holes/Euler come from the ORIGINAL binary mask with 8-connected ink and
4-connected padded background. Endpoints and junctions come from a padded
Zhang--Suen skeleton and a contracted branch graph. Topology disagreement is
reported rather than repaired. No short spur is silently removed. Counts stay
raw; later fit any scaler/prototypes on the calibration split, never queries.

All returned arrays are read-only snapshots. Metadata and named schemas expose
enabled stages and dimensions. Normalized spatial centroids use pixel centers:
((x+0.5)/width, (y+0.5)/height). A zero-padded centroid slot has a separate
presence mask. More holes than slots are counted and explicitly reported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
import skimage
from skimage.morphology import skeletonize

from segmentation import DiacriticLink, GlyphCandidate, SegmentationHypothesis, SegmentationResult

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


class FeatureError(ValueError):
    """Input data violates a feature extraction invariant."""


class FeatureConfigError(ValueError):
    """A supplied feature setting is invalid, rather than simply missing."""


def _number(name: str, value: Any, minimum: float = 0, maximum: float = math.inf,
            *, positive: bool = False, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureConfigError(f"{name} must be a real number")
    if integer and type(value) is not int:
        raise FeatureConfigError(f"{name} must be an integer")
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise FeatureConfigError(f"invalid {name}: {value!r}")


def _freeze(array: Any, dtype: Any = None) -> NDArray[Any]:
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _plain_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_metadata(item) for item in value]
    return value


def _binary(image: NDArray[Any], *, max_pixels: int, allow_empty: bool = False) -> BinaryMask:
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise FeatureError("expected a two-dimensional binary ndarray")
    if image.size > max_pixels:
        raise FeatureError("candidate exceeds max_pixels; supply glyphs rather than page images")
    if not allow_empty and 0 in image.shape:
        raise FeatureError("whole-glyph input must have nonzero height and width")
    if not (np.issubdtype(image.dtype, np.number) or image.dtype == np.bool_):
        raise FeatureError("binary mask dtype must be numeric or bool")
    if np.issubdtype(image.dtype, np.complexfloating):
        raise FeatureError("complex-valued masks are unsupported")
    if not (np.all((image == 0) | (image == 1)) or np.all((image == 0) | (image == 255))):
        raise FeatureError("expected {0,1} or {0,255}; grayscale and guessed polarity are unsupported")
    return np.ascontiguousarray(image != 0, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class FeatureOptions:
    schema_version: Literal["v1", "v2"] = "v1"
    extract_zones: bool = True
    isotropy_tolerance: float = 1e-3
    max_pixels: int = 4_000_000
    max_skeleton_pixels: int = 65_536
    max_graph_cycle_rank: int = 128

    def __post_init__(self) -> None:
        if self.schema_version not in ("v1", "v2"):
            raise FeatureConfigError("schema_version must be v1 or v2")
        if type(self.extract_zones) is not bool:
            raise FeatureConfigError("extract_zones must be bool")
        _number("isotropy_tolerance", self.isotropy_tolerance, 0, 1)
        _number("max_pixels", self.max_pixels, 1, integer=True)
        _number("max_skeleton_pixels", self.max_skeleton_pixels, 1, integer=True)
        _number("max_graph_cycle_rank", self.max_graph_cycle_rank, 0, integer=True)


@dataclass(frozen=True, slots=True)
class _Settings:
    micro_windows: int = 4
    lse_epsilon: float = 1e-8
    empty_window_value: float = 0.0
    include_occupancy: bool = False
    max_hole_centroids: int = 4
    slope_weight: float = 1.0
    topology_weight: float = 1.0
    standardization_floor: float = 1e-8

    def __post_init__(self) -> None:
        _number("micro_windows", self.micro_windows, 1, 32, integer=True)
        _number("lse_epsilon", self.lse_epsilon, 0, positive=True)
        _number("empty_window_value", self.empty_window_value, -1, 1)
        if type(self.include_occupancy) is not bool:
            raise FeatureConfigError("include_occupancy must be bool")
        _number("max_hole_centroids", self.max_hole_centroids, 0, 1024, integer=True)
        for name in ("slope_weight", "topology_weight", "standardization_floor"):
            _number(name, getattr(self, name), 0, positive=True)


@dataclass(frozen=True, slots=True)
class _Flags:
    slope_features: bool = True
    topology_features: bool = True


def _lookup(config: Any, section: str, name: str, default: Any, missing: list[str]) -> Any:
    sentinel = object()
    group = config.get(section, sentinel) if isinstance(config, Mapping) else getattr(config, section, sentinel)
    value = group.get(name, sentinel) if isinstance(group, Mapping) else getattr(group, name, sentinel)
    if value is sentinel:
        missing.append(f"{section}.{name}")
        return default
    return value


def _resolve(config: Any, mode: str | None) -> tuple[Any, str, _Settings, _Flags, list[str]]:
    supplied = config.get("mode") if isinstance(config, Mapping) else getattr(config, "mode", None)
    selected = mode if mode is not None else supplied if supplied is not None else "B4"
    if not isinstance(selected, str) or selected.upper() not in ("B0", "B1", "B2", "B3", "B4"):
        raise FeatureConfigError("mode must be B0, B1, B2, B3 or B4")
    selected = selected.upper()
    if supplied is not None and (not isinstance(supplied, str) or supplied.upper() != selected):
        raise FeatureConfigError("mode conflicts with config.mode")
    if config is None and _project_config is not None:
        factory = getattr(_project_config, "get_config", None)
        config = factory(selected) if callable(factory) else getattr(_project_config, "DEFAULT_CONFIG", _project_config)
    missing: list[str] = []
    defaults = _Settings()
    settings = _Settings(**{item.name: _lookup(config, "features", item.name, getattr(defaults, item.name), missing)
                            for item in fields(defaults)})
    slope = _lookup(config, "flags", "slope_features", True, missing)
    topology = _lookup(config, "flags", "topology_features", selected in ("B3", "B4"), missing)
    if type(slope) is not bool or type(topology) is not bool or not (slope or topology):
        raise FeatureConfigError("feature flags must be booleans and enable at least one branch")
    return config, selected, settings, _Flags(slope, topology), missing


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str
    feature_names: tuple[str, ...]
    slope_dimension: int
    topology_dimension: int
    schema_id: str
    metadata: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready schema for prototype manifests; schema_id binds semantics."""
        return {"version": self.version, "feature_names": list(self.feature_names),
                "dimension": self.dimension, "slope_dimension": self.slope_dimension,
                "topology_dimension": self.topology_dimension, "schema_id": self.schema_id,
                "metadata": _plain_metadata(self.metadata)}


@dataclass(frozen=True, slots=True)
class SlopeFeatures:
    vector: NDArray[np.float64]
    angles: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    validity: NDArray[np.bool_]
    counts: NDArray[np.int64]
    beta0: NDArray[np.float64]
    beta1: NDArray[np.float64]
    coherence: NDArray[np.float64]
    regression_r2: NDArray[np.float64]
    vertical: NDArray[np.bool_]
    isotropic: NDArray[np.bool_]
    cell_bounds: NDArray[np.int64]
    cell_centroids: NDArray[np.float64]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TopologyFeatures:
    vector: NDArray[np.float64]
    component_count: int
    hole_count: int
    euler_characteristic: int
    endpoint_count: int
    three_way_junction_count: int
    four_way_junction_count: int
    endpoint_centroid: tuple[float, float]
    three_way_centroid: tuple[float, float]
    four_way_centroid: tuple[float, float]
    hole_centroids: NDArray[np.float64]
    all_hole_centroids: NDArray[np.float64]
    hole_presence: NDArray[np.bool_]
    hole_areas: NDArray[np.int64]
    skeleton: BinaryMask
    raw_neighbor_counts: NDArray[np.int16]
    graph: SkeletonGraph
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    candidate_id: str
    word_id: int
    normalized_bbox: tuple[int, int, int, int]
    global_bbox: tuple[int, int, int, int]
    global_polygon: tuple[tuple[float, float], ...]
    associated_mark_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureResult:
    vector: NDArray[np.float64]
    schema: FeatureSchema
    image_shape: tuple[int, int]
    slope: SlopeFeatures | None
    topology: TopologyFeatures | None
    identity: CandidateIdentity | None
    zones: Mapping[str, FeatureResult]
    metadata: dict[str, Any]

    @property
    def v_si(self) -> NDArray[np.float64]:
        return self.slope.vector if self.slope is not None else _freeze(np.empty(0, np.float64))

    @property
    def v_topo(self) -> NDArray[np.float64]:
        return self.topology.vector if self.topology is not None else _freeze(np.empty(0, np.float64))


@dataclass(frozen=True, slots=True)
class FeatureBatchResult:
    features: tuple[FeatureResult, ...]
    by_candidate_id: Mapping[str, FeatureResult]
    hypotheses: Mapping[int, tuple[SegmentationHypothesis, ...]]
    diacritic_links: Mapping[int, tuple[DiacriticLink, ...]]
    metadata: dict[str, Any]


def _feature_names(settings: _Settings, flags: _Flags, options: FeatureOptions) -> tuple[tuple[str, ...], int, int]:
    k = settings.micro_windows
    slope_names: list[str] = []
    if flags.slope_features:
        slope_names.extend(f"si.angle.r{row:02d}.c{column:02d}" for row in range(k) for column in range(k))
        if settings.include_occupancy or options.schema_version == "v2":
            slope_names.extend(f"si.occupancy.r{row:02d}.c{column:02d}" for row in range(k) for column in range(k))
    topo_names: list[str] = []
    if flags.topology_features:
        topo_names.extend(("topo.components", "topo.holes"))
        if options.schema_version == "v1":
            topo_names.append("topo.euler")
        topo_names.extend(("topo.endpoints", "topo.junctions3", "topo.junctions4plus"))
        topo_names.extend(f"topo.{kind}.centroid_{axis}" for kind in ("endpoints", "junctions3", "junctions4plus")
                          for axis in ("x", "y"))
        topo_names.extend(f"topo.hole{index:03d}.centroid_{axis}" for index in range(settings.max_hole_centroids)
                          for axis in ("x", "y"))
    return tuple(slope_names+topo_names), len(slope_names), len(topo_names)


class MicroWindowLSEExtractor:
    """Vectorized foreground moment regression with row-major K x K windows."""

    def __init__(self, settings: _Settings, options: FeatureOptions) -> None:
        self.settings, self.options = settings, options

    def extract(self, image: NDArray[Any], *, allow_empty: bool = False) -> SlopeFeatures:
        mask = _binary(image, max_pixels=self.options.max_pixels, allow_empty=allow_empty)
        height, width = mask.shape
        k, epsilon = self.settings.micro_windows, self.settings.lse_epsilon
        size = k*k
        y_edges = np.arange(k+1, dtype=np.int64)*height//k
        x_edges = np.arange(k+1, dtype=np.int64)*width//k
        cell_widths = np.tile(np.diff(x_edges), k)
        cell_heights = np.repeat(np.diff(y_edges), k)
        areas = cell_widths*cell_heights
        yy, xx = np.nonzero(mask)
        rows = np.searchsorted(y_edges[1:], yy, side="right")
        columns = np.searchsorted(x_edges[1:], xx, side="right")
        ids = rows*k+columns
        x = xx.astype(np.float64)-x_edges[columns]
        y = yy.astype(np.float64)-y_edges[rows]
        if self.options.schema_version == "v1":
            x /= np.maximum(cell_widths[ids]-1, 1)
            y /= np.maximum(cell_heights[ids]-1, 1)
        count = np.bincount(ids, minlength=size).astype(np.int64)
        denominator = np.maximum(count, 1)
        sum_x = np.bincount(ids, weights=x, minlength=size)
        sum_y = np.bincount(ids, weights=y, minlength=size)
        mean_x, mean_y = sum_x/denominator, sum_y/denominator
        sxx = np.maximum(np.bincount(ids, weights=x*x, minlength=size)-sum_x*mean_x, 0.0)
        syy = np.maximum(np.bincount(ids, weights=y*y, minlength=size)-sum_y*mean_y, 0.0)
        sxy = np.bincount(ids, weights=x*y, minlength=size)-sum_x*mean_y
        beta1 = sxy/(sxx+epsilon)
        beta0 = mean_y-beta1*mean_x
        coherence = np.clip(np.hypot(sxx-syy, 2*sxy)/(sxx+syy+epsilon), 0.0, 1.0)
        supported = count >= 2
        vertical = supported & (sxx <= epsilon) & (syy > epsilon)
        isotropic = supported & (coherence <= self.options.isotropy_tolerance)
        valid = supported & ((sxx+syy) > epsilon) & ~isotropic
        angles = np.full(size, self.settings.empty_window_value, np.float64)
        angles[valid] = 2.0*np.arctan(beta1[valid])/np.pi
        angles[vertical] = 1.0
        occupancy = np.divide(count, areas, out=np.zeros(size, np.float64), where=areas > 0)
        regression_r2 = np.clip(sxy*sxy/(sxx*syy+epsilon), 0.0, 1.0)
        horizontal = supported & (syy <= epsilon) & (sxx > epsilon)
        regression_r2[horizontal] = 1.0
        cell_centroids = np.zeros((size, 2), np.float64)
        pixel_x = np.bincount(ids, weights=xx-x_edges[columns]+.5, minlength=size)
        pixel_y = np.bincount(ids, weights=yy-y_edges[rows]+.5, minlength=size)
        cell_centroids[:, 0] = np.divide(pixel_x, denominator*np.maximum(cell_widths, 1),
                                        out=np.zeros(size, np.float64), where=count > 0)
        cell_centroids[:, 1] = np.divide(pixel_y, denominator*np.maximum(cell_heights, 1),
                                        out=np.zeros(size, np.float64), where=count > 0)
        bounds = np.column_stack((np.tile(x_edges[:-1], k), np.repeat(y_edges[:-1], k),
                                   np.tile(x_edges[1:], k), np.repeat(y_edges[1:], k)))
        include_occupancy = self.settings.include_occupancy or self.options.schema_version == "v2"
        vector = np.concatenate((angles, occupancy)) if include_occupancy else angles
        shape = (k, k)
        metadata = {
            "source": "original_candidate_ink", "cell_order": "row_major", "k": k,
            "coordinate_policy": "cell_local_independently_normalized" if self.options.schema_version == "v1" else "isotropic_pixel_units",
            "angle_formula": "2*atan(beta1)/pi", "y_axis": "downward", "include_occupancy": include_occupancy,
            "occupied_cells": int(np.count_nonzero(count)), "valid_cells": int(valid.sum()),
            "vertical_degenerate_cells": int(vertical.sum()), "isotropic_cells": int(isotropic.sum()),
            "zero_area_cells": int(np.count_nonzero(areas == 0)), "empty_canvas": mask.size == 0,
            "limitation": "OLS is directional regression; high coherence alone does not guarantee a faithful stroke angle",
        }
        return SlopeFeatures(
            _freeze(vector), _freeze(angles.reshape(shape)), _freeze(occupancy.reshape(shape)),
            _freeze(valid.reshape(shape)), _freeze(count.reshape(shape)), _freeze(beta0.reshape(shape)),
            _freeze(beta1.reshape(shape)), _freeze(coherence.reshape(shape)), _freeze(regression_r2.reshape(shape)),
            _freeze(vertical.reshape(shape)), _freeze(isotropic.reshape(shape)), _freeze(bounds.reshape(k, k, 4)),
            _freeze(cell_centroids.reshape(k, k, 2)), metadata,
        )


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A contracted skeleton region; coordinates are (x, y), boxes half-open.

    ``degree`` counts incident graph edges, with a returning/self edge counted
    twice.  It is not a maximum or sum of eight-neighbour pixel counts.
    ``centroid`` uses pixel centers in the [0, 1] image-edge coordinate frame:
    ((x + 0.5) / width, (y + 0.5) / height).
    """

    node_id: int
    kind: str
    degree: int
    centroid: tuple[float, float]
    centroid_xy: tuple[float, float]
    pixels: NDArray[np.int32]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A maximal branch between node boundaries, or a retained graph cycle.

    Pixel arrays are coordinate sets, not a promised traversal order.  Length
    is the sum of unit orthogonal and sqrt(2) diagonal pixel-edge lengths.
    Junction-region interior travel is excluded from inter-node branch length.
    """

    edge_id: int
    source_id: int
    target_id: int
    length: float
    pixel_count: int
    pixels: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class SkeletonGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    shape: tuple[int, int]
    component_count: int
    pixel_count: int
    cycle_rank: int
    adjacency_convention: str = "8-connected; diagonal shortcuts removed; filled 2x2 faces contracted"


def _graph_group_indices(labels: NDArray[np.int32], count: int) -> list[NDArray[np.int64]]:
    """Group integer indices with one sort, retaining empty groups if needed."""
    if count == 0:
        return []
    order = np.argsort(labels, kind="stable")
    bounds = np.cumsum(np.bincount(labels, minlength=count))[:-1]
    return list(np.split(order, bounds))


def _graph_internal_cycles(
    vertices: NDArray[np.int64],
    edge_u: NDArray[np.int64],
    edge_v: NDArray[np.int64],
    edge_w: NDArray[np.float64],
) -> list[tuple[NDArray[np.int64], float]]:
    """A fundamental cycle basis for a contracted junction's interior.

    Junction contraction can itself swallow a small genuine loop.  Each
    non-tree edge is retained as one self-edge, so contraction preserves the
    original graph's cycle rank.  Only these usually tiny exceptional regions
    need traversal; ordinary branches use connected-component operations.
    """
    if edge_u.size - vertices.size + 1 <= 0:
        return []
    local_u = np.searchsorted(vertices, edge_u)
    local_v = np.searchsorted(vertices, edge_v)
    size = vertices.size
    adjacency = csr_matrix(
        (np.ones(2 * edge_u.size, dtype=np.uint8),
         (np.concatenate((local_u, local_v)), np.concatenate((local_v, local_u)))),
        shape=(size, size),
    )
    _, predecessors = breadth_first_order(adjacency, 0, directed=False, return_predecessors=True)
    tree_edge = (predecessors[local_u] == local_v) | (predecessors[local_v] == local_u)
    edge_weight = {(min(int(a), int(b)), max(int(a), int(b))): float(w)
                   for a, b, w in zip(local_u, local_v, edge_w, strict=True)}
    cycles: list[tuple[NDArray[np.int64], float]] = []
    for edge_index in np.flatnonzero(~tree_edge):
        first, second = int(local_u[edge_index]), int(local_v[edge_index])
        ancestors: list[int] = []
        cursor = first
        while cursor >= 0:
            ancestors.append(cursor)
            cursor = int(predecessors[cursor])
        ancestor_positions = {value: index for index, value in enumerate(ancestors)}
        second_path: list[int] = []
        cursor = second
        while cursor not in ancestor_positions:
            second_path.append(cursor)
            cursor = int(predecessors[cursor])
        path = ancestors[:ancestor_positions[cursor] + 1] + second_path[::-1]
        length = float(edge_w[edge_index])
        length += sum(edge_weight[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:], strict=True))
        cycles.append((vertices[np.asarray(path, dtype=np.int64)], length))
    return cycles


def build_skeleton_graph(skeleton: BinaryMask, *, max_skeleton_pixels: int = 65_536,
                         max_graph_cycle_rank: int = 128) -> SkeletonGraph:
    """Extract actual branches from a two-dimensional one-pixel skeleton.

    Nonzero pixels are foreground.  Endpoints and isolated pixels remain
    separate nodes.  Adjacent branching pixels form a junction region; every
    intervening degree-two component becomes one edge.  Disconnected pure
    rings become cycle nodes with a self-edge.  Cycles inside junction regions
    are also retained, and no short branches/spurs are removed.

    Main work is vectorized O(P + E), plus component/group sorting.  P and E are
    the skeleton pixel/adjacency counts.  In particular, the implementation
    never traces all branches using nested Python pixel-neighbour loops.

    Two true junctions directly touching without an intervening degree-two
    chain are intentionally represented as one junction region.  Its degree
    counts the outgoing branches of that region, including returning loops.
    A filled 2x2 pixel square is a filled face, not an enclosed background hole.
    Remove its bottom horizontal edge and contract its four vertices into one
    region. Repeated upper detours preserve connectivity, while deleting exactly
    one edge per filled face removes digital face cycles. Remaining graph cycle
    rank matches four-connected background holes; raster topology is still
    independently measured on the original candidate.
    """
    source = np.asarray(skeleton)
    _number("max_skeleton_pixels", max_skeleton_pixels, 1, integer=True)
    _number("max_graph_cycle_rank", max_graph_cycle_rank, 0, integer=True)
    if source.ndim != 2:
        raise ValueError("skeleton must be a two-dimensional mask")
    if source.dtype.kind not in "buif":
        raise TypeError("skeleton must have a numeric or boolean dtype")
    if source.dtype.kind == "f" and not np.isfinite(source).all():
        raise ValueError("skeleton contains non-finite values")
    mask = np.asarray(source != 0, dtype=np.uint8)
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    count = xs.size
    if count == 0:
        return SkeletonGraph((), (), (height, width), 0, 0, 0)
    if count > max_skeleton_pixels:
        raise FeatureError("skeleton exceeds max_skeleton_pixels; reduce candidate size or raise the explicit budget")
    pixel_xy = np.column_stack((xs, ys)).astype(np.int32)
    pixel_index = np.full(mask.shape, -1, dtype=np.int64)
    pixel_index[ys, xs] = np.arange(count, dtype=np.int64)
    filled_faces = ((mask[:-1, :-1] > 0) & (mask[:-1, 1:] > 0)
                    & (mask[1:, :-1] > 0) & (mask[1:, 1:] > 0))
    face_vertices = np.zeros(mask.shape, bool)
    face_vertices[:-1, :-1] |= filled_faces
    face_vertices[:-1, 1:] |= filled_faces
    face_vertices[1:, :-1] |= filled_faces
    face_vertices[1:, 1:] |= filled_faces
    all_u: list[NDArray[np.int64]] = []
    all_v: list[NDArray[np.int64]] = []
    all_w: list[NDArray[np.float64]] = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y_start, y_end = 0, height - dy
        x_start, x_end = max(0, -dx), min(width, width - dx)
        if y_end <= y_start or x_end <= x_start:
            continue
        a = mask[y_start:y_end, x_start:x_end]
        b = mask[y_start + dy:y_end + dy, x_start + dx:x_end + dx]
        connected = (a != 0) & (b != 0)
        if dy == 0 and dx == 1:
            connected[1:] &= ~filled_faces
        if dy and dx:
            # A shared ink pixel supplies an orthogonal two-edge replacement.
            bridge_a = mask[y_start:y_end, x_start + dx:x_end + dx]
            bridge_b = mask[y_start + dy:y_end + dy, x_start:x_end]
            connected &= (bridge_a == 0) & (bridge_b == 0)
        local_y, local_x = np.nonzero(connected)
        start_y, start_x = local_y + y_start, local_x + x_start
        all_u.append(pixel_index[start_y, start_x])
        all_v.append(pixel_index[start_y + dy, start_x + dx])
        all_w.append(np.full(local_x.size, np.hypot(dx, dy), dtype=np.float64))
    edge_u = np.concatenate(all_u) if all_u else np.empty(0, dtype=np.int64)
    edge_v = np.concatenate(all_v) if all_v else np.empty(0, dtype=np.int64)
    edge_w = np.concatenate(all_w) if all_w else np.empty(0, dtype=np.float64)
    degrees = np.bincount(np.concatenate((edge_u, edge_v)), minlength=count)
    adjacency = csr_matrix(
        (np.ones(2 * edge_u.size, dtype=np.uint8),
         (np.concatenate((edge_u, edge_v)), np.concatenate((edge_v, edge_u)))),
        shape=(count, count),
    )
    component_count = int(connected_components(adjacency, directed=False, return_labels=False))
    if int(edge_u.size-count+component_count) > max_graph_cycle_rank:
        raise FeatureError("skeleton exceeds max_graph_cycle_rank; dense/noisy candidates require an explicit larger budget")

    branch_mask = (degrees >= 3) | face_vertices[ys, xs]
    regular_mask = (degrees == 2) & ~branch_mask
    branch_pixels = np.flatnonzero(branch_mask)
    endpoint_pixels = np.flatnonzero((degrees <= 1) & ~branch_mask)
    branch_map = np.full(count, -1, dtype=np.int64)
    branch_map[branch_pixels] = np.arange(branch_pixels.size)
    branch_edges = branch_mask[edge_u] & branch_mask[edge_v]
    branch_u = branch_map[edge_u[branch_edges]]
    branch_v = branch_map[edge_v[branch_edges]]
    branch_adjacency = csr_matrix(
        (np.ones(2 * branch_u.size, dtype=np.uint8),
         (np.concatenate((branch_u, branch_v)), np.concatenate((branch_v, branch_u)))),
        shape=(branch_pixels.size, branch_pixels.size),
    )
    branch_count, branch_labels = connected_components(branch_adjacency, directed=False)
    groups = [branch_pixels[index] for index in _graph_group_indices(branch_labels, branch_count)]
    groups.extend(np.asarray([index], dtype=np.int64) for index in endpoint_pixels)
    groups.sort(key=lambda group: int(group[0]))
    node_groups = groups
    node_for_pixel = np.full(count, -1, dtype=np.int64)
    for node_id, vertices in enumerate(node_groups):
        node_for_pixel[vertices] = node_id

    edge_records: list[tuple[int, int, float, NDArray[np.int64]]] = []
    # Direct adjacency of non-degree-two pixels can connect endpoint nodes.
    node_u = node_for_pixel[edge_u]
    node_v = node_for_pixel[edge_v]
    direct = (node_u >= 0) & (node_v >= 0) & (node_u != node_v)
    for index in np.flatnonzero(direct):
        edge_records.append((int(node_u[index]), int(node_v[index]), float(edge_w[index]),
                             np.asarray([edge_u[index], edge_v[index]], dtype=np.int64)))

    # A genuine cycle may be entirely inside one junction region.
    internal = (node_u >= 0) & (node_u == node_v)
    internal_indices = np.flatnonzero(internal)
    internal_groups = _graph_group_indices(node_u[internal_indices].astype(np.int32), len(node_groups))
    for node_id, local_indices in enumerate(internal_groups):
        indices = internal_indices[local_indices]
        vertices = node_groups[node_id]
        if indices.size - vertices.size + 1 > 0:
            for cycle_pixels, length in _graph_internal_cycles(
                vertices, edge_u[indices], edge_v[indices], edge_w[indices]
            ):
                edge_records.append((node_id, node_id, length, cycle_pixels))

    regular = np.flatnonzero(regular_mask)
    regular_map = np.full(count, -1, dtype=np.int64)
    regular_map[regular] = np.arange(regular.size)
    regular_edges = regular_mask[edge_u] & regular_mask[edge_v]
    regular_u = regular_map[edge_u[regular_edges]]
    regular_v = regular_map[edge_v[regular_edges]]
    regular_adjacency = csr_matrix(
        (np.ones(2 * regular_u.size, dtype=np.uint8),
         (np.concatenate((regular_u, regular_v)), np.concatenate((regular_v, regular_u)))),
        shape=(regular.size, regular.size),
    )
    chain_count, chain_labels = connected_components(regular_adjacency, directed=False)
    chain_for_pixel = np.full(count, -1, dtype=np.int32)
    chain_for_pixel[regular] = chain_labels
    chain_groups = _graph_group_indices(chain_labels, chain_count)
    inner_indices = np.flatnonzero(regular_edges)
    chain_lengths = np.bincount(chain_for_pixel[edge_u[inner_indices]], weights=edge_w[inner_indices], minlength=chain_count)
    attachments = (regular_mask[edge_u] & (node_v >= 0)) | (regular_mask[edge_v] & (node_u >= 0))
    attachment_indices = np.flatnonzero(attachments)
    regular_first = regular_mask[edge_u[attachment_indices]]
    attachment_regular = np.where(regular_first, edge_u[attachment_indices], edge_v[attachment_indices])
    attachment_node = np.where(regular_first, edge_v[attachment_indices], edge_u[attachment_indices])
    attachment_groups = _graph_group_indices(chain_for_pixel[attachment_regular], chain_count)
    cycle_node_ids: set[int] = set()
    for chain_id, local_pixels in enumerate(chain_groups):
        vertices = regular[local_pixels]
        attachment_local = attachment_groups[chain_id]
        if attachment_local.size == 0:
            node_id = len(node_groups)
            node_groups.append(vertices)
            cycle_node_ids.add(node_id)
            edge_records.append((node_id, node_id, float(chain_lengths[chain_id]), vertices))
        elif attachment_local.size == 2:
            attached_vertices = attachment_node[attachment_local]
            attached_ids = node_for_pixel[attached_vertices]
            vertices_with_boundaries = np.unique(np.concatenate((vertices, attached_vertices)))
            length = chain_lengths[chain_id] + edge_w[attachment_indices[attachment_local]].sum()
            edge_records.append((int(attached_ids[0]), int(attached_ids[1]), float(length), vertices_with_boundaries))
        else:
            # A degree-two connected component is a cycle or has two exits.
            raise RuntimeError("invalid skeleton graph: degree-two chain does not have exactly two boundary edges")

    final_degrees = np.zeros(len(node_groups), dtype=np.int64)
    edges: list[GraphEdge] = []
    for edge_id, (first, second, length, vertices) in enumerate(edge_records):
        final_degrees[first] += 1
        final_degrees[second] += 1
        pixels = _freeze(pixel_xy[vertices], np.int32)
        edges.append(GraphEdge(edge_id, first, second, length, int(vertices.size), pixels))
    nodes: list[GraphNode] = []
    for node_id, vertices in enumerate(node_groups):
        pixels = pixel_xy[vertices]
        centroid_x, centroid_y = pixels.mean(axis=0)
        degree = int(final_degrees[node_id])
        kind = ("cycle" if node_id in cycle_node_ids else
                "isolated" if degree == 0 else "endpoint" if degree == 1 else
                "transit" if degree == 2 else "junction")
        minimum = pixels.min(axis=0)
        maximum = pixels.max(axis=0) + 1
        nodes.append(GraphNode(
            node_id=node_id, kind=kind, degree=degree,
            centroid=(float((centroid_x + 0.5) / width), float((centroid_y + 0.5) / height)),
            centroid_xy=(float(centroid_x), float(centroid_y)), pixels=_freeze(pixels, np.int32),
            bbox=(int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])),
        ))
    cycle_rank = len(edges) - len(nodes) + component_count
    expected_cycle_rank = int(edge_u.size - count + component_count)
    if cycle_rank != expected_cycle_rank:
        raise RuntimeError("skeleton graph contraction changed its cycle rank")
    return SkeletonGraph(tuple(nodes), tuple(edges), (height, width), component_count, int(count), cycle_rank)


def _mask_topology(mask: BinaryMask) -> tuple[int, int, NDArray[np.float64], NDArray[np.int64]]:
    """Authoritative raster topology with dual foreground/background connectivity."""
    height, width = mask.shape
    if mask.size == 0:
        return 0, 0, np.empty((0, 2), np.float64), np.empty(0, np.int64)
    component_count = int(cv2.connectedComponents(mask, connectivity=8)[0]-1)
    padded = np.pad(mask, 1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(1-padded, connectivity=4)
    ids = np.arange(1, count, dtype=np.int64)
    ids = ids[ids != int(labels[0, 0])]
    normalized = (centroids[ids]-.5)/np.array((width, height), np.float64)
    areas = stats[ids, cv2.CC_STAT_AREA].astype(np.int64)
    order = np.lexsort((normalized[:, 1], normalized[:, 0])) if ids.size else np.empty(0, np.int64)
    return component_count, int(ids.size), normalized[order], areas[order]


class EulerTopologyExtractor:
    """Raster Euler measurements plus actual skeleton-graph branch features."""

    def __init__(self, settings: _Settings, options: FeatureOptions) -> None:
        self.settings, self.options = settings, options

    def extract(self, image: NDArray[Any], *, allow_empty: bool = False) -> TopologyFeatures:
        mask = _binary(image, max_pixels=self.options.max_pixels, allow_empty=allow_empty)
        components, holes, centroids, areas = _mask_topology(mask)
        if mask.size == 0 or not np.any(mask):
            skeleton = np.zeros_like(mask)
        else:
            skeleton = skeletonize(np.pad(mask.astype(bool), 1), method="zhang")[1:-1, 1:-1].astype(np.uint8)
        graph = build_skeleton_graph(skeleton, max_skeleton_pixels=self.options.max_skeleton_pixels,
                                     max_graph_cycle_rank=self.options.max_graph_cycle_rank)
        endpoints = [node for node in graph.nodes if node.degree == 1]
        three = [node for node in graph.nodes if node.degree == 3]
        four = [node for node in graph.nodes if node.degree >= 4]

        def centroid(nodes: list[GraphNode]) -> tuple[float, float]:
            if not nodes:
                return 0.0, 0.0
            mean = np.mean([node.centroid for node in nodes], axis=0)
            return float(mean[0]), float(mean[1])

        endpoint_centroid, three_centroid, four_centroid = centroid(endpoints), centroid(three), centroid(four)
        cap = self.settings.max_hole_centroids
        padded_centroids = np.zeros((cap, 2), np.float64)
        presence = np.zeros(cap, bool)
        retained = min(cap, holes)
        padded_centroids[:retained] = centroids[:retained]
        presence[:retained] = True
        values: list[float] = [float(components), float(holes)]
        if self.options.schema_version == "v1":
            values.append(float(components-holes))
        values.extend((float(len(endpoints)), float(len(three)), float(len(four)),
                       *endpoint_centroid, *three_centroid, *four_centroid))
        vector = np.concatenate((np.asarray(values, np.float64), padded_centroids.ravel()))
        skel_components, skel_holes, _, _ = _mask_topology(skeleton)
        if skeleton.size:
            kernel = np.ones((3, 3), np.uint8)
            kernel[1, 1] = 0
            neighbors = cv2.filter2D(skeleton, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
        else:
            neighbors = np.zeros(skeleton.shape, np.int16)
        metadata = {
            "raster_foreground_connectivity": 8, "raster_background_connectivity": 4,
            "raster_source": "original_candidate_ink", "skeleton_method": "zhang",
            "skeleton_components": skel_components, "skeleton_holes": skel_holes,
            "skeleton_euler": skel_components-skel_holes,
            "thinning_preserved_topology": (components, holes) == (skel_components, skel_holes),
            "graph_cycle_rank": graph.cycle_rank, "graph_cycle_rank_matches_raster_holes": graph.cycle_rank == holes,
            "raw_pixel_junction_candidates": int(np.count_nonzero((skeleton > 0) & (neighbors >= 3))),
            "junction_measurement": "contracted_graph_incident_branch_degree",
            "isolated_graph_nodes": sum(node.degree == 0 for node in graph.nodes),
            "hole_slots": cap, "omitted_hole_centroids": max(0, holes-cap),
            "centroid_normalization": "pixel_centers_divided_by_canvas_extent",
            "centroid_sort_order": "x_then_y", "spur_pruning": False,
            "euler_in_classifier_vector": self.options.schema_version == "v1",
        }
        return TopologyFeatures(
            _freeze(vector), components, holes, components-holes, len(endpoints), len(three), len(four),
            endpoint_centroid, three_centroid, four_centroid, _freeze(padded_centroids), _freeze(centroids),
            _freeze(presence), _freeze(areas), _freeze(skeleton), _freeze(neighbors), graph, metadata,
        )


class FeaturePipeline:
    """Extract raw descriptors and preserve segmentation identity/alternatives.

    ``extract`` handles one binary mask. ``run`` additionally accepts a
    GlyphCandidate or SegmentationResult. Per-zone results use the SAME schema
    as whole-glyph results but have a separate view label; never mix their
    prototypes without an explicit per-view model. No scaler is fitted here.
    """

    def __init__(self, config: Any = None, *, mode: str | None = None,
                 options: FeatureOptions | None = None) -> None:
        self.options = options if options is not None else FeatureOptions()
        if not isinstance(self.options, FeatureOptions):
            raise TypeError("options must be FeatureOptions")
        self.config, self.mode, self.settings, self.flags, self.fallback_fields = _resolve(config, mode)
        self.slope_extractor = MicroWindowLSEExtractor(self.settings, self.options)
        self.topology_extractor = EulerTopologyExtractor(self.settings, self.options)
        names, slope_dimension, topology_dimension = _feature_names(self.settings, self.flags, self.options)
        semantics = {
            "pipeline_version": PIPELINE_VERSION, "schema_version": self.options.schema_version,
            "feature_settings": asdict(self.settings), "flags": asdict(self.flags),
            "options": asdict(self.options), "feature_names": names,
            "canvas_policy": "preserve_no_crop_or_resize", "source": "candidate_local_ink",
            "raster_connectivity": [8, 4], "skeleton_method": "zhang",
            "graph_policy": "branch_graph_diagonal_shortcuts_removed_and_filled_faces_contracted",
            "centroid_policy": "pixel_center_divided_by_canvas_extent", "standardization": "none",
        }
        signature = hashlib.sha256(json.dumps(semantics, sort_keys=True, allow_nan=False,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
        self.schema = FeatureSchema(self.options.schema_version, names, slope_dimension, topology_dimension,
                                    signature, _freeze_metadata(semantics))

    def _extract(self, image: NDArray[Any], *, allow_empty: bool = False,
                 identity: CandidateIdentity | None = None, view: str = "whole_glyph") -> FeatureResult:
        started = time.perf_counter()
        mask = _binary(image, max_pixels=self.options.max_pixels, allow_empty=allow_empty)
        slope = self.slope_extractor.extract(mask, allow_empty=allow_empty) if self.flags.slope_features else None
        topology = self.topology_extractor.extract(mask, allow_empty=allow_empty) if self.flags.topology_features else None
        blocks: list[NDArray[np.float64]] = []
        if slope is not None:
            blocks.append(slope.vector*self.settings.slope_weight)
        if topology is not None:
            blocks.append(topology.vector*self.settings.topology_weight)
        vector = np.concatenate(blocks)
        if vector.shape != (self.schema.dimension,) or not np.isfinite(vector).all():
            raise FeatureError("feature extraction violated its fixed-dimension/finite-vector contract")
        metadata = {
            "pipeline_version": PIPELINE_VERSION, "mode": self.mode,
            "schema_id": self.schema.schema_id, "schema_version": self.schema.version,
            "config_fingerprint": getattr(self.config, "fingerprint", None),
            "fallback_fields": list(self.fallback_fields), "view": view,
            "model_view_key": f"{self.schema.schema_id}:{view}",
            "canvas_policy": "preserve_no_crop_or_resize", "empty_ink": not bool(mask.any()),
            "empty_canvas": mask.size == 0, "foreground_pixels": int(mask.sum()),
            "aspect_ratio": mask.shape[1]/mask.shape[0] if mask.shape[0] else 0.0,
            "enabled_stages": {"slope": self.flags.slope_features, "topology": self.flags.topology_features},
            "block_weights": {"slope": self.settings.slope_weight, "topology": self.settings.topology_weight},
            "standardization": "none_fit_scaler_on_calibration_data_only",
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "scikit_image": skimage.__version__},
            "elapsed_ms": (time.perf_counter()-started)*1000.0,
        }
        LOGGER.debug("glyph features extracted", extra={"event": "features_extracted", "view": view,
                                                        "dimension": self.schema.dimension})
        return FeatureResult(_freeze(vector), self.schema, tuple(mask.shape), slope, topology, identity,
                             MappingProxyType({}), metadata)

    def extract(self, image: NDArray[Any], *, view: str = "whole_glyph") -> FeatureResult:
        """Declare the calibration/model view explicitly; no geometry is changed."""
        if view not in ("whole_glyph", "zone:upper", "zone:core", "zone:lower"):
            raise FeatureError("view must be whole_glyph or zone:upper/core/lower")
        return self._extract(image, view=view)

    def extract_candidate(self, candidate: GlyphCandidate) -> FeatureResult:
        if not isinstance(candidate, GlyphCandidate):
            raise TypeError("candidate must be segmentation.GlyphCandidate")
        identity = CandidateIdentity(candidate.candidate_id, candidate.word_id,
                                     candidate.normalized_bbox.as_tuple(), candidate.global_bbox.as_tuple(),
                                     candidate.global_polygon, tuple(candidate.associated_mark_ids))
        result = self._extract(candidate.image, identity=identity)
        if not self.options.extract_zones:
            return result
        zones: dict[str, FeatureResult] = {}
        for name, region in candidate.zones.items():
            zone_identity = CandidateIdentity(candidate.candidate_id, candidate.word_id,
                                              region.normalized_bbox.as_tuple(), region.global_bbox.as_tuple(),
                                              region.global_polygon, tuple(candidate.associated_mark_ids))
            zones[name] = self._extract(region.image, allow_empty=True, identity=zone_identity, view=f"zone:{name}")
        return FeatureResult(result.vector, result.schema, result.image_shape, result.slope, result.topology,
                             identity, MappingProxyType(zones), result.metadata)

    def extract_segmentation(self, segmentation: SegmentationResult) -> FeatureBatchResult:
        if not isinstance(segmentation, SegmentationResult):
            raise TypeError("segmentation must be segmentation.SegmentationResult")
        started = time.perf_counter()
        word_ids: set[int] = set()
        candidate_ids: set[str] = set()
        for word in segmentation.words:
            if word.word_id in word_ids:
                raise FeatureError("segmentation word IDs must be unique")
            word_ids.add(word.word_id)
            local_ids: set[str] = set()
            for candidate in word.candidates:
                if candidate.word_id != word.word_id:
                    raise FeatureError("candidate word_id disagrees with its containing word")
                if candidate.candidate_id in candidate_ids:
                    raise FeatureError("segmentation candidate IDs must be unique")
                candidate_ids.add(candidate.candidate_id)
                local_ids.add(candidate.candidate_id)
            for hypothesis in word.hypotheses:
                if not hypothesis.candidate_ids or any(candidate_id not in local_ids for candidate_id in hypothesis.candidate_ids):
                    raise FeatureError("hypothesis must reference nonempty candidates from its own word")
        features = tuple(self.extract_candidate(candidate) for candidate in segmentation.candidates)
        by_id: dict[str, FeatureResult] = {}
        for result in features:
            if result.identity is None or result.identity.candidate_id in by_id:
                raise FeatureError("segmentation candidates must have unique, present identities")
            by_id[result.identity.candidate_id] = result
        hypotheses: dict[int, tuple[SegmentationHypothesis, ...]] = {}
        diacritic_links: dict[int, tuple[DiacriticLink, ...]] = {}
        for word in segmentation.words:
            if word.word_id in hypotheses:
                raise FeatureError("segmentation word IDs must be unique")
            hypotheses[word.word_id] = word.hypotheses
            diacritic_links[word.word_id] = word.diacritic_links
            for hypothesis in word.hypotheses:
                if any(candidate_id not in by_id for candidate_id in hypothesis.candidate_ids):
                    raise FeatureError("segmentation hypothesis references a missing candidate")
        metadata = {
            "pipeline_version": PIPELINE_VERSION, "mode": self.mode, "schema_id": self.schema.schema_id,
            "schema_version": self.schema.version, "dimension": self.schema.dimension,
            "candidate_count": len(features), "word_count": len(segmentation.words),
            "segmentation_metadata": dict(segmentation.metadata),
            "elapsed_ms": (time.perf_counter()-started)*1000.0,
        }
        return FeatureBatchResult(features, MappingProxyType(by_id), MappingProxyType(hypotheses),
                                  MappingProxyType(diacritic_links), metadata)

    def run(self, source: NDArray[Any] | GlyphCandidate | SegmentationResult) -> FeatureResult | FeatureBatchResult:
        if isinstance(source, np.ndarray):
            return self.extract(source)
        if isinstance(source, GlyphCandidate):
            return self.extract_candidate(source)
        if isinstance(source, SegmentationResult):
            return self.extract_segmentation(source)
        raise TypeError("source must be a binary ndarray, GlyphCandidate or SegmentationResult")


FeatureExtractor = FeaturePipeline
TopologicalGraphExtractor = EulerTopologyExtractor

__all__ = [
    "FeaturePipeline", "FeatureExtractor", "FeatureOptions", "FeatureSchema", "FeatureResult", "FeatureBatchResult",
    "FeatureError", "FeatureConfigError", "CandidateIdentity", "SlopeFeatures", "TopologyFeatures",
    "MicroWindowLSEExtractor", "EulerTopologyExtractor", "TopologicalGraphExtractor",
    "GraphNode", "GraphEdge", "SkeletonGraph", "build_skeleton_graph",
]
