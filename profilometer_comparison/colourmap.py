"""Convert MountainsMap pseudo-colour images back to heights.

The colour scale is read once from the colour bar of a '-studiable' export and
stored as a lookup table (RGB -> height in µm). Each image pixel gets the height
of the nearest lookup colour.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np


@dataclass
class Colourmap:
    rgb: np.ndarray     # (N, 3) uint8, ordered by height
    height: np.ndarray  # (N,) float64, ascending, µm


@dataclass
class HeightMap:
    height: np.ndarray     # float32, NaN where excluded
    excluded: np.ndarray   # bool: colour too far from the scale (e.g. not-measured points)
    saturated: np.ndarray  # bool: lowest or highest colour of the scale (true height unknown)


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    return (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)


def _unpack(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    return np.stack([(codes >> 16) & 255, (codes >> 8) & 255, codes & 255], axis=-1).astype(np.uint8)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each contiguous True run in a 1-D boolean array."""
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def find_colour_bar(img: np.ndarray, search_from_frac: float = 0.75) -> tuple[int, int, int]:
    """Locate the colour bar interior in a pseudo-colour export.

    Returns (top, bottom, column): interior rows top..bottom (inclusive) and a
    column through the middle of the bar. The bar is the first block of mostly
    non-white columns right of `search_from_frac` of the image width, drawn with
    a black frame.
    """
    s = img.astype(np.int32).sum(axis=2)
    nonwhite = s < 700
    black = s < 30
    x_start = int(img.shape[1] * search_from_frac)
    col_runs = _runs(nonwhite[:, x_start:].mean(axis=0) > 0.5)
    if not col_runs:
        raise ValueError("no colour bar found")
    x0, x1 = col_runs[0][0] + x_start, col_runs[0][1] + x_start
    # The left frame line gives the outer top and bottom of the bar.
    y0, y1 = max(_runs(black[:, x0 + 1]), key=lambda r: r[1] - r[0])
    ym = (y0 + y1) // 2
    thickness = _runs(black[ym, x0:x1])[0][1]
    right = x0 + thickness + _runs(black[ym, x0 + thickness:x1])[0][0]
    return y0 + thickness, y1 - 1 - thickness, (x0 + thickness + right) // 2


def build_from_studiable(img: np.ndarray, z_min: float = 0.0, z_max: float = 350.0,
                         search_from_frac: float = 0.75) -> Colourmap:
    top, bottom, col = find_colour_bar(img, search_from_frac)
    rows = np.arange(top, bottom + 1)
    heights = z_max - (rows - top) / (bottom - top) * (z_max - z_min)
    codes = _pack(img[top:bottom + 1, col])
    uniq, inverse = np.unique(codes, return_inverse=True)
    mean_height = np.bincount(inverse, weights=heights) / np.bincount(inverse)
    order = np.argsort(mean_height)
    return Colourmap(rgb=_unpack(uniq[order]), height=mean_height[order])


def save_csv(cmap: Colourmap, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["r", "g", "b", "height_um"])
        for (r, g, b), h in zip(cmap.rgb.tolist(), cmap.height.tolist()):
            writer.writerow([r, g, b, f"{h:.4f}"])


def _read_csv(fh) -> Colourmap:
    rows = list(csv.DictReader(fh))
    rgb = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows], np.uint8)
    height = np.array([float(r["height_um"]) for r in rows])
    return Colourmap(rgb=rgb, height=height)


def load_csv(path: str | Path) -> Colourmap:
    with open(path, newline="", encoding="utf-8") as fh:
        return _read_csv(fh)


def default_colourmap() -> Colourmap:
    """The MarSurf 0-350 µm scale shipped with the package."""
    ref = resources.files("profilometer_comparison") / "colourmaps" / "marsurf_default.csv"
    with ref.open("r", encoding="utf-8", newline="") as fh:
        return _read_csv(fh)


def rgb_to_height(img: np.ndarray, cmap: Colourmap, tolerance: float = 25.0) -> HeightMap:
    """Map each pixel to the height of the nearest scale colour (Euclidean RGB).

    Pixels further than `tolerance` from every scale colour are excluded (NaN).
    Works on the distinct colours of the image only, so it is fast for large images.
    """
    codes = _pack(img)
    used = np.flatnonzero(np.bincount(codes.ravel(), minlength=1 << 24))
    used_rgb = _unpack(used).astype(np.int32)
    lut = cmap.rgb.astype(np.int32)
    nearest = np.empty(len(used), np.int64)
    distance = np.empty(len(used))
    for start in range(0, len(used), 4096):
        d2 = ((used_rgb[start:start + 4096, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        nearest[start:start + 4096] = d2.argmin(axis=1)
        distance[start:start + 4096] = np.sqrt(d2.min(axis=1))

    ok = distance <= tolerance
    height_table = np.full(1 << 24, np.nan, np.float32)
    height_table[used] = np.where(ok, cmap.height[nearest], np.nan)
    saturated_table = np.zeros(1 << 24, bool)
    saturated_table[used] = ok & ((nearest == 0) | (nearest == len(lut) - 1))

    height = height_table[codes]
    return HeightMap(height=height, excluded=np.isnan(height), saturated=saturated_table[codes])
