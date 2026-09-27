"""Stage 1: process before/after pairs into a cache folder."""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .align import Alignment, downsample, estimate_shift, overlap_slices
from .colour import delta_e76, srgb_to_lab
from .colourmap import Colourmap, default_colourmap, load_csv, rgb_to_height
from .crop import crop_to_frame, find_plot_frame
from .difference import Plane, fit_plane, histogram, subtract_plane
from .images import read_rgb
from .masks import Polygon, polygons_to_mask

TYPES = {"pseudo-colour": "pseudo-colour-image", "true-colour": "true-colour"}
REGION_ORDER = ("all", "glaze", "encrustation")


@dataclass
class Settings:
    type: str
    pixel_um: float = 1.35
    scan_length_um: float = 10674.0
    thresholds: tuple[float, ...] | None = None
    plane_correction: bool = True
    tolerance: float = 25.0
    colourmap: str = "default"
    max_shift_um: float = 1000.0
    min_confidence: float = 20.0
    min_overlap_pct: float = 80.0
    unstable_threshold: float | None = None
    display_max: int = 1600

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown type {self.type!r}; choose from {sorted(TYPES)}")
        if self.thresholds is not None:
            self.thresholds = tuple(float(t) for t in self.thresholds)

    @property
    def signed(self) -> bool:
        return self.type == "pseudo-colour"

    @property
    def thresholds_resolved(self) -> tuple[float, ...]:
        if self.thresholds:
            return self.thresholds
        return (1.0, 2.0, 5.0, 10.0) if self.signed else (2.0, 5.0, 10.0)

    @property
    def unstable(self) -> float:
        if self.unstable_threshold is not None:
            return self.unstable_threshold
        return 5.0 if self.signed else 10.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["thresholds"] = list(self.thresholds_resolved)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        return cls(**{**d, "thresholds": tuple(d["thresholds"]) if d.get("thresholds") else None})

    def __eq__(self, other):
        return isinstance(other, Settings) and self.to_dict() == other.to_dict()


def load_colourmap(settings: Settings) -> Colourmap | None:
    if not settings.signed:
        return None
    return default_colourmap() if settings.colourmap == "default" else load_csv(settings.colourmap)


@dataclass
class Loaded:
    align: np.ndarray      # 2-D float32 used for alignment (height or L*)
    values: np.ndarray     # height (2-D) or Lab (3-D)
    valid: np.ndarray      # bool
    saturated: np.ndarray  # bool
    rgb: np.ndarray        # uint8, for display
    pixel_um: float


def load_image(path: str | Path, settings: Settings, cmap: Colourmap | None) -> Loaded:
    rgb = read_rgb(path)
    if settings.signed:
        hm = rgb_to_height(rgb, cmap, settings.tolerance)
        return Loaded(hm.height, hm.height, ~hm.excluded, hm.saturated, rgb, settings.pixel_um)
    try:
        rgb = np.ascontiguousarray(crop_to_frame(rgb, find_plot_frame(rgb)))
    except ValueError:
        pass  # image-only export without axes: use as is
    lab = srgb_to_lab(rgb)
    shape = rgb.shape[:2]
    return Loaded(lab[..., 0], lab, np.ones(shape, bool), np.zeros(shape, bool), rgb,
                  settings.scan_length_um / rgb.shape[1])


def difference_values(before: Loaded, after: Loaded, bs, as_, settings: Settings, enc: np.ndarray,
                      plane: Plane | None = None, plane_origin=(0, 0)):
    """Signed height difference (plane-corrected) or ΔE on the overlap; NaN where invalid."""
    valid = before.valid[bs] & after.valid[as_]
    if settings.signed:
        d = after.values[as_] - before.values[bs]
        if plane is None:
            plane = fit_plane(d, valid & ~enc, before.pixel_um) if settings.plane_correction else Plane()
        d = subtract_plane(d, plane, before.pixel_um, plane_origin)
    else:
        d = delta_e76(before.values[bs], after.values[as_])
        plane = Plane()
    return np.where(valid, d, np.nan).astype(np.float32), valid, plane


@dataclass
class PairResult:
    sample: str
    meta: dict
    hists: dict[str, np.ndarray]
    before_small: np.ndarray
    after_small: np.ndarray
    diff_small: np.ndarray


def _shape(sl) -> tuple[int, int]:
    return sl[0].stop - sl[0].start, sl[1].stop - sl[1].start


def process_pair(before_path, after_path, settings: Settings, cmap: Colourmap | None = None,
                 polygons: list[Polygon] = (), manual: tuple[int, int] | None = None,
                 sample: str | None = None) -> PairResult:
    before_path, after_path = Path(before_path), Path(after_path)
    before = load_image(before_path, settings, cmap)
    after = load_image(after_path, settings, cmap)
    polygons = list(polygons)

    def enc_mask(bs):
        if not polygons:
            return np.zeros(_shape(bs), bool)
        return polygons_to_mask(polygons, _shape(bs), before.pixel_um, (bs[0].start, bs[1].start))

    if manual is not None:
        al = Alignment(int(manual[0]), int(manual[1]), float("nan"), "manual")
    else:
        # Pass 1: all pixels. Pass 2: only pixels that did not change much and are not encrustation.
        first = estimate_shift(before.align, after.align)
        bs, as_ = overlap_slices(before.align.shape, after.align.shape, first.dy, first.dx)
        enc = enc_mask(bs)
        d, valid, _ = difference_values(before, after, bs, as_, settings, enc)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            deviation = np.abs(d - np.nanmedian(d)) if settings.signed else d
            stable = np.zeros(before.align.shape, bool)
            stable[bs] = valid & ~enc & (deviation <= settings.unstable)
        al = estimate_shift(before.align, after.align, stable=stable)

    bs, as_ = overlap_slices(before.align.shape, after.align.shape, al.dy, al.dx)
    enc = enc_mask(bs)
    d, valid, plane = difference_values(before, after, bs, as_, settings, enc)

    regions = {"all": valid, "glaze": valid & ~enc}
    if polygons:
        regions["encrustation"] = valid & enc
    hists = {name: histogram(d[mask], settings.signed) for name, mask in regions.items()}

    h, w = _shape(bs)
    px = before.pixel_um
    overlap_pct = 100 * h * w / before.align.size
    warnings_list = []
    if al.method == "auto":
        if al.confidence < settings.min_confidence:
            warnings_list.append("low confidence")
        if math.hypot(al.dy, al.dx) * px > settings.max_shift_um:
            warnings_list.append("large shift")
    if overlap_pct < settings.min_overlap_pct:
        warnings_list.append("low overlap")

    mean_dl = None
    if not settings.signed:
        mean_dl = float(np.mean(after.values[as_][..., 0][valid] - before.values[bs][..., 0][valid]))

    n_valid = max(int(valid.sum()), 1)
    step = max(1, math.ceil(max(h, w) / settings.display_max))
    hs, ws = h // step * step, w // step * step
    meta = {
        "sample": sample or before_path.stem,
        "file": before_path.name,
        "type": settings.type,
        "shift_x_px": al.dx,
        "shift_y_px": al.dy,
        "shift_x_um": round(al.dx * px, 2),
        "shift_y_um": round(al.dy * px, 2),
        "align_method": al.method,
        "align_confidence": None if math.isnan(al.confidence) else round(al.confidence, 1),
        "align_warning": "; ".join(warnings_list),
        "overlap_pct": round(overlap_pct, 3),
        "plane_offset_um": round(plane.offset, 4),
        "plane_tilt_x": round(plane.tilt_x, 5),
        "plane_tilt_y": round(plane.tilt_y, 5),
        "mean_dL": None if mean_dl is None else round(mean_dl, 3),
        "excluded_pct": round(100 * (1 - valid.mean()), 4),
        "saturated_pct": round(100 * (before.saturated[bs] | after.saturated[as_]).mean(), 4),
        "region_area_pct": {name: round(100 * int(m.sum()) / n_valid, 3) for name, m in regions.items()},
        "pixel_um": px,
        "before_origin_px": [bs[0].start, bs[1].start],
        "overlap_shape": [h, w],
        "display_step": step,
        "mask_changed_at": None,
    }
    return PairResult(
        sample=meta["sample"],
        meta=meta,
        hists=hists,
        before_small=np.ascontiguousarray(before.rgb[bs][:hs:step, :ws:step]),
        after_small=np.ascontiguousarray(after.rgb[as_][:hs:step, :ws:step]),
        diff_small=downsample(d, step).astype(np.float16),
    )
