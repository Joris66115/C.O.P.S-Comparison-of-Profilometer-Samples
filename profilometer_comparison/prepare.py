"""Stage 1: process before/after pairs into a cache folder."""
from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import NAME, __version__
from .align import Alignment, downsample, estimate_shift, overlap_slices
from .colour import delta_e76, srgb_to_lab
from .colourmap import Colourmap, default_colourmap, load_csv, rgb_to_height
from .crop import crop_to_frame, find_plot_frame
from .difference import Plane, fit_plane, histogram, percentages, subtract_plane
from .images import read_rgb
from .masks import Polygon, load_masks, polygons_to_mask, save_masks
from .pairing import natural_key, pair_folders, sample_id

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


class SettingsMismatch(Exception):
    """The cache was made with different settings or input folders."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def detect_type(folder: str | Path) -> str | None:
    """The type whose filename suffix most files in the folder carry."""
    stems = [p.stem for p in Path(folder).iterdir() if p.is_file()]
    counts = {t: sum(s.endswith("-" + suffix) for s in stems) for t, suffix in TYPES.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else None


def _pair_dir(cache: Path, sample: str) -> Path:
    return Path(cache) / "pairs" / sample


def write_pair(cache: Path, result: PairResult) -> None:
    d = _pair_dir(cache, result.sample)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.before_small).save(d / "before.png")
    Image.fromarray(result.after_small).save(d / "after.png")
    np.save(d / "diff.npy", result.diff_small)
    np.savez_compressed(d / "hist.npz", **result.hists)
    (d / "meta.json").write_text(json.dumps(result.meta, indent=2), encoding="utf-8")  # last: marks completion


def read_meta(cache: Path, sample: str) -> dict:
    return json.loads((_pair_dir(cache, sample) / "meta.json").read_text(encoding="utf-8"))


def read_hists(cache: Path, sample: str) -> dict[str, np.ndarray]:
    with np.load(_pair_dir(cache, sample) / "hist.npz") as z:
        return {k: z[k] for k in z.files}


def processed_samples(cache: Path) -> list[str]:
    pairs = Path(cache) / "pairs"
    if not pairs.exists():
        return []
    return sorted((p.name for p in pairs.iterdir() if (p / "meta.json").exists()), key=natural_key)


def read_manifest(cache: Path) -> dict:
    return json.loads((Path(cache) / "manifest.json").read_text(encoding="utf-8"))


def _check_or_write_manifest(cache: Path, settings: Settings, before_dir: Path, after_dir: Path) -> None:
    path = cache / "manifest.json"
    current = {"before_dir": str(Path(before_dir).resolve()), "after_dir": str(Path(after_dir).resolve()),
               "settings": settings.to_dict()}
    if path.exists():
        old = read_manifest(cache)
        diffs = [k for k in ("before_dir", "after_dir") if old[k] != current[k]]
        diffs += [k for k in current["settings"] if old["settings"].get(k) != current["settings"][k]]
        if diffs:
            raise SettingsMismatch(
                f"{cache} was made with different {', '.join(diffs)}. "
                "Use a new output folder (--out) or the original settings.")
        return
    path.write_text(json.dumps({"tool": NAME, "version": __version__,
                                "created": _now(), **current}, indent=2), encoding="utf-8")


def summary_columns(settings: Settings) -> list[str]:
    cols = ["sample", "file", "type", "region", "region_area_pct", "shift_x_um", "shift_y_um", "align_method",
            "overlap_pct", "align_confidence", "align_warning", "plane_offset_um", "plane_tilt_x", "plane_tilt_y",
            "mean_dL", "excluded_pct", "saturated_pct", "mask_changed_at"]
    for t in settings.thresholds_resolved:
        cols.append(f"pct_diff_{t:g}")
        if settings.signed:
            cols += [f"pct_lower_{t:g}", f"pct_higher_{t:g}"]
    return cols


def write_summary(cache: Path) -> Path:
    cache = Path(cache)
    settings = Settings.from_dict(read_manifest(cache)["settings"])
    cols = summary_columns(settings)
    path = cache / "summary.csv"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for sample in processed_samples(cache):
            meta, hists = read_meta(cache, sample), read_hists(cache, sample)
            for region in (r for r in REGION_ORDER if r in hists):
                row = {**meta, "region": region, "region_area_pct": meta["region_area_pct"][region]}
                for t in settings.thresholds_resolved:
                    p = percentages(hists[region], t, settings.signed)
                    row[f"pct_diff_{t:g}"] = round(p["diff"], 4)
                    if settings.signed:
                        row[f"pct_lower_{t:g}"] = round(p["lower"], 4)
                        row[f"pct_higher_{t:g}"] = round(p["higher"], 4)
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    tmp.replace(path)
    return path


def run_prepare(before_dir, after_dir, out_dir, settings: Settings, masks_file=None, log=print) -> Path:
    before_dir, after_dir, cache = Path(before_dir), Path(after_dir), Path(out_dir)
    cache.mkdir(parents=True, exist_ok=True)
    _check_or_write_manifest(cache, settings, before_dir, after_dir)

    masks = load_masks(cache / "masks.json")
    if masks_file:
        masks.update(load_masks(masks_file))
        save_masks(cache / "masks.json", masks)

    log_lines = [f"=== prepare {_now()} ({NAME} {__version__})"]

    def note(message: str) -> None:
        log_lines.append(message)
        log(message)

    pairing = pair_folders(before_dir, after_dir)
    suffix = TYPES[settings.type]
    typed = [p for p in pairing.pairs if Path(p.name).stem.endswith("-" + suffix)]
    pairs = typed or pairing.pairs
    for p in pairing.pairs:
        if typed and p not in typed:
            note(f"ignored (not a {settings.type} image): {p.name}")

    cmap = load_colourmap(settings)
    for i, pair in enumerate(pairs, 1):
        sample = sample_id(pair.name, suffix)
        if (_pair_dir(cache, sample) / "meta.json").exists():
            continue
        try:
            result = process_pair(pair.before, pair.after, settings, cmap, masks.get(sample, []), sample=sample)
            write_pair(cache, result)
        except Exception as exc:  # one bad file must not stop the run
            note(f"[{i}/{len(pairs)}] {sample}: ERROR {type(exc).__name__}: {exc}")
            continue
        m = result.meta
        warn = f"  WARNING: {m['align_warning']}" if m["align_warning"] else ""
        note(f"[{i}/{len(pairs)}] {sample}: shift {m['shift_x_um']:+.1f}/{m['shift_y_um']:+.1f} µm{warn}")

    for name in pairing.before_only:
        note(f"only in before folder: {name}")
    for name in pairing.after_only:
        note(f"only in after folder: {name}")
    write_summary(cache)
    with open(cache / "prepare-log.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return cache


def recompute_pair(cache, sample: str, manual="keep", mask_changed: bool = False) -> None:
    cache = Path(cache)
    manifest = read_manifest(cache)
    settings = Settings.from_dict(manifest["settings"])
    old = read_meta(cache, sample)
    if manual == "keep":
        manual = (old["shift_y_px"], old["shift_x_px"]) if old["align_method"] == "manual" else None
    polygons = load_masks(cache / "masks.json").get(sample, [])
    result = process_pair(Path(manifest["before_dir"]) / old["file"], Path(manifest["after_dir"]) / old["file"],
                          settings, load_colourmap(settings), polygons, manual=manual, sample=sample)
    result.meta["mask_changed_at"] = _now() if mask_changed else old.get("mask_changed_at")
    write_pair(cache, result)
    write_summary(cache)


ZOOM_SIZES = (150, 300, 600, 1200, 2400)  # zoom field of view in full-resolution pixels


class ZoomSource:
    """Full-resolution before/after of one sample, loaded once, for repeated zoom crops."""

    def __init__(self, cache, sample: str):
        cache = Path(cache)
        manifest = read_manifest(cache)
        self.settings = Settings.from_dict(manifest["settings"])
        self.meta = read_meta(cache, sample)
        cmap = load_colourmap(self.settings)
        self.before = load_image(Path(manifest["before_dir"]) / self.meta["file"], self.settings, cmap)
        self.after = load_image(Path(manifest["after_dir"]) / self.meta["file"], self.settings, cmap)
        self.dy, self.dx = self.meta["shift_y_px"], self.meta["shift_x_px"]
        self.overlap, _ = overlap_slices(self.before.align.shape, self.after.align.shape, self.dy, self.dx)
        self.plane = Plane(self.meta["plane_offset_um"], self.meta["plane_tilt_x"], self.meta["plane_tilt_y"])
        self.pixel_um = self.before.pixel_um

    def region(self, centre_before_px, size: int):
        """Crop of `size` x `size` px around (row, col) of the before image, kept inside the overlap.

        Returns (before_rgb, after_rgb, difference, origin (row, col), centre (row, col) after clamping).
        """
        def window(centre, sl):
            start = int(np.clip(centre - size // 2, sl.start, max(sl.start, sl.stop - size)))
            return slice(start, min(start + size, sl.stop))

        rows, cols = window(centre_before_px[0], self.overlap[0]), window(centre_before_px[1], self.overlap[1])
        zb = (rows, cols)
        za = (slice(rows.start + self.dy, rows.stop + self.dy), slice(cols.start + self.dx, cols.stop + self.dx))
        origin = (rows.start - self.overlap[0].start, cols.start - self.overlap[1].start)
        enc = np.zeros(_shape(zb), bool)
        d, _, _ = difference_values(self.before, self.after, zb, za, self.settings, enc,
                                    plane=self.plane, plane_origin=origin)
        centre = (rows.start + (rows.stop - rows.start) // 2, cols.start + (cols.stop - cols.start) // 2)
        return self.before.rgb[zb], self.after.rgb[za], d, (rows.start, cols.start), centre


def zoom_region(cache, sample: str, centre_before_px, size: int = 600):
    """Full-resolution before/after/difference around a point (row, col) of the before image.

    Returns (before_rgb, after_rgb, difference, (row, col) of the region's top-left in the before image).
    """
    return ZoomSource(cache, sample).region(centre_before_px, size)[:4]
