"""Alignment by phase correlation: rotation around Z (search) and X/Y shift.

Conventions: a shift (dy, dx) means a feature at before[y, x] appears at
after[y + dy, x + dx]. Angles are in degrees, counter-clockwise positive as seen in
the image (y pointing down), as in PIL's Image.rotate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image


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


def overlap_slices(shape_before, shape_after, dy: int, dx: int, after_bounds=None):
    """Slices (before_slices, after_slices) of the region both images cover.

    `after_bounds` = (y0, y1, x0, x1) limits the usable part of the after image (e.g. the
    rectangle that stays covered after rotating it); default: the whole after image.
    """
    ay0, ay1, ax0, ax1 = after_bounds or (0, shape_after[0], 0, shape_after[1])
    y0, y1 = max(0, ay0 - dy), min(shape_before[0], ay1 - dy)
    x0, x1 = max(0, ax0 - dx), min(shape_before[1], ax1 - dx)
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


def rotate_array(a: np.ndarray, angle_deg: float, nearest: bool = False) -> np.ndarray:
    """Rotate a 2-D array about its centre (same shape); pixels outside the source become NaN."""
    a = np.asarray(a, np.float32)
    resample = Image.NEAREST if nearest else Image.BILINEAR
    data = np.asarray(Image.fromarray(a, mode="F").rotate(angle_deg, resample=resample, fillcolor=0.0))
    cover = np.asarray(Image.fromarray(np.ones(a.shape, np.uint8)).rotate(angle_deg, resample=Image.NEAREST,
                                                                           fillcolor=0))
    return np.where(cover > 0, data, np.nan).astype(np.float32)


def inner_rect(shape, angle_deg: float) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): axis-aligned rectangle fully covered after rotating an image of `shape`."""
    h, w = shape
    if angle_deg == 0:
        return 0, h, 0, w
    t = np.radians(angle_deg)
    cx, cy = w / 2, h / 2

    def turn(x, y):
        u, v = x - cx, y - cy
        return cx + u * np.cos(t) + v * np.sin(t), cy - u * np.sin(t) + v * np.cos(t)

    (tlx, tly), (trx, try_), (blx, bly), (brx, bry) = turn(0, 0), turn(w, 0), turn(0, h), turn(w, h)
    x0 = int(np.ceil(max(tlx, blx))) + 1
    x1 = int(np.floor(min(trx, brx))) - 1
    y0 = int(np.ceil(max(tly, try_))) + 1
    y1 = int(np.floor(min(bly, bry))) - 1
    return max(0, y0), min(h, y1), max(0, x0), min(w, x1)


def estimate_rotation(before: np.ndarray, after: np.ndarray, max_deg: float = 5.0, coarse_step: float = 0.25,
                      fine_step: float = 0.05, fine_span: float = 0.3) -> tuple[float, float]:
    """Correction angle (degrees) to apply to `after` so it matches `before`, and its PSR.

    Coarse search over -max_deg..+max_deg on images of about 1000 px, then a finer search
    around the best angle at twice the resolution, refined with a parabola through the peak.
    Both are scored by the peak-to-sidelobe ratio of the phase correlation, which responds to
    the fine surface texture. (The plain correlation coefficient was tried for the fine step:
    on real glaze it is dominated by the large-scale waviness and gives no clear peak.)
    """
    f_coarse = max(1, round(min(before.shape) / 1000))
    f_fine = max(1, f_coarse // 2)

    def scores(angles, f):
        da, db = downsample(before, f), downsample(after, f)
        return np.array([phase_correlation(da, rotate_array(db, a))[2] for a in angles])

    coarse = np.arange(-max_deg, max_deg + 1e-9, coarse_step)
    best = coarse[int(np.argmax(scores(coarse, f_coarse)))]
    fine = best + np.arange(-fine_span, fine_span + 1e-9, fine_step)
    psr = scores(fine, f_fine)
    i = int(np.argmax(psr))
    angle = fine[i]
    if 0 < i < len(psr) - 1:
        y0, y1, y2 = psr[i - 1], psr[i], psr[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            angle += fine_step * 0.5 * (y0 - y2) / denom
    return float(np.clip(angle, -max_deg, max_deg)), float(psr[i])
