"""Translation-only alignment by phase correlation.

Convention: a shift (dy, dx) means a feature at before[y, x] appears at
after[y + dy, x + dx].
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Alignment:
    dy: int
    dx: int
    confidence: float  # peak-to-sidelobe ratio of the coarse correlation (NaN for manual)
    method: str = "auto"


def downsample(a: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware block mean by an integer factor (remainder rows/cols are dropped)."""
    a = np.asarray(a, np.float32)
    if f == 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    blocks = a[:h, :w].reshape(h // f, f, w // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks stay NaN
        return np.nanmean(blocks, axis=(1, 3))


def phase_correlation(a: np.ndarray, b: np.ndarray, exclude: int = 5) -> tuple[int, int, float]:
    """Shift of b relative to a, and the peak-to-sidelobe ratio (PSR) of the correlation peak."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a = np.asarray(a[:h, :w], np.float64)
    b = np.asarray(b[:h, :w], np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = np.nan_to_num(a - np.nanmean(a))
        b = np.nan_to_num(b - np.nanmean(b))
    window = np.outer(np.hanning(h), np.hanning(w))
    cross = np.fft.rfft2(b * window) * np.conj(np.fft.rfft2(a * window))
    cross /= np.abs(cross) + 1e-12
    r = np.fft.irfft2(cross, s=(h, w))
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    sidelobe = np.ones(r.shape, bool)
    sidelobe[np.ix_([(py + k) % h for k in range(-exclude, exclude + 1)],
                    [(px + k) % w for k in range(-exclude, exclude + 1)])] = False
    psr = float((r[py, px] - r[sidelobe].mean()) / (r[sidelobe].std() + 1e-12))
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return int(dy), int(dx), psr


def overlap_slices(shape_before, shape_after, dy: int, dx: int):
    """Slices (before_slices, after_slices) of the region both images cover."""
    y0, y1 = max(0, -dy), min(shape_before[0], shape_after[0] - dy)
    x0, x1 = max(0, -dx), min(shape_before[1], shape_after[1] - dx)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("images do not overlap at this shift")
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))


def estimate_shift(before: np.ndarray, after: np.ndarray, stable: np.ndarray | None = None,
                   factor: int = 4, window: int = 2048) -> Alignment:
    """Coarse phase correlation on downsampled images, refined at full resolution.

    If `stable` (bool, before coordinates) is given, unstable pixels are ignored and
    the refinement window is the one of 3 x 3 candidates with the most stable pixels.
    """
    b = np.array(before, np.float32)
    if stable is not None:
        b[~stable] = np.nan
    cy, cx, confidence = phase_correlation(downsample(b, factor), downsample(after, factor))
    dy, dx = cy * factor, cx * factor

    bs, _ = overlap_slices(before.shape, after.shape, dy, dx)
    n_y = min(window, bs[0].stop - bs[0].start)
    n_x = min(window, bs[1].stop - bs[1].start)
    candidates = [(int(ty), int(tx))
                  for ty in np.linspace(bs[0].start, bs[0].stop - n_y, 3)
                  for tx in np.linspace(bs[1].start, bs[1].stop - n_x, 3)]
    if stable is None:
        ty, tx = candidates[4]  # centre
    else:
        ty, tx = max(candidates, key=lambda c: stable[c[0]:c[0] + n_y, c[1]:c[1] + n_x].mean())
    fy, fx, _ = phase_correlation(b[ty:ty + n_y, tx:tx + n_x],
                                  after[ty + dy:ty + dy + n_y, tx + dx:tx + dx + n_x])
    if abs(fy) <= 2 * factor and abs(fx) <= 2 * factor:
        dy, dx = dy + fy, dx + fx
    return Alignment(int(dy), int(dx), confidence)
