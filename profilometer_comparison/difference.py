"""Plane correction, difference histograms and threshold percentages.

Percentages come from histograms of the full-resolution difference (bin width
0.05 µm or 0.05 ΔE), so any threshold on that grid is evaluated exactly without
keeping the full-resolution arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_WIDTH = 0.05
SIGNED_RANGE = 400.0    # µm, histogram covers -400..+400
UNSIGNED_RANGE = 200.0  # ΔE, histogram covers 0..200
_N_SIGNED = int(round(SIGNED_RANGE / BIN_WIDTH))
_N_UNSIGNED = int(round(UNSIGNED_RANGE / BIN_WIDTH))
_CHUNK = 8_000_000


@dataclass(frozen=True)
class Plane:
    offset: float = 0.0  # µm at the array's top-left
    tilt_x: float = 0.0  # µm per mm
    tilt_y: float = 0.0  # µm per mm


def fit_plane(d: np.ndarray, valid: np.ndarray, pixel_um: float, step: int = 8,
              k: float = 3.0, min_cutoff: float = 1.0, max_iter: int = 10) -> Plane:
    """Robust plane through every `step`-th valid pixel.

    A plain least-squares plane is pulled towards large changed areas (a lowered
    spot drags the offset down, so the unchanged rest looks raised). So after each
    fit, only pixels within k robust standard deviations (1.4826 * MAD, at least
    `min_cutoff` µm because heights come in ~0.9 µm steps) of the median residual
    are kept, and the plane is refitted until that set stops changing. The plane
    is thus set by the unchanged surface, as long as that is the majority.
    """
    rows, cols = np.nonzero(valid[::step, ::step])
    if rows.size < 3:
        return Plane()
    z = d[::step, ::step][rows, cols].astype(np.float64)
    y_mm = rows * step * pixel_um / 1000
    x_mm = cols * step * pixel_um / 1000
    design = np.column_stack([np.ones_like(x_mm), x_mm, y_mm])
    keep = np.ones(z.size, bool)
    for _ in range(max_iter):
        coef, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        res = z - design @ coef
        centre = np.median(res[keep])
        spread = 1.4826 * np.median(np.abs(res[keep] - centre))
        new_keep = np.abs(res - centre) <= max(k * spread, min_cutoff)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return Plane(float(coef[0]), float(coef[1]), float(coef[2]))


def subtract_plane(d: np.ndarray, plane: Plane, pixel_um: float, origin_px=(0, 0)) -> np.ndarray:
    h, w = d.shape
    y_mm = ((np.arange(h) + origin_px[0]) * pixel_um / 1000).astype(np.float32)[:, None]
    x_mm = ((np.arange(w) + origin_px[1]) * pixel_um / 1000).astype(np.float32)[None, :]
    return (d - (plane.offset + plane.tilt_x * x_mm + plane.tilt_y * y_mm)).astype(np.float32)


def histogram(values: np.ndarray, signed: bool) -> np.ndarray:
    values = np.asarray(values).ravel()
    n = 2 * _N_SIGNED if signed else _N_UNSIGNED
    hist = np.zeros(n, np.int64)
    for start in range(0, values.size, _CHUNK):
        idx = np.floor(values[start:start + _CHUNK].astype(np.float64) / BIN_WIDTH).astype(np.int64)
        if signed:
            idx += _N_SIGNED
        np.clip(idx, 0, n - 1, out=idx)
        hist += np.bincount(idx, minlength=n)
    return hist


def percentages(hist: np.ndarray, threshold: float, signed: bool) -> dict:
    """% of pixels beyond the threshold: |d| >= t (to within one bin)."""
    total = int(hist.sum())
    if total == 0:
        return {"diff": float("nan"), "lower": float("nan"), "higher": float("nan")} if signed \
            else {"diff": float("nan")}
    k = int(round(threshold / BIN_WIDTH))
    if signed:
        higher = int(hist[_N_SIGNED + k:].sum())
        lower = int(hist[:max(_N_SIGNED - k, 0)].sum())
        return {"diff": 100 * (higher + lower) / total, "lower": 100 * lower / total, "higher": 100 * higher / total}
    return {"diff": 100 * int(hist[k:].sum()) / total}
