"""sRGB -> CIELAB (D65) and CIE76 colour difference."""
from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], np.float32)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], np.float32)
_EPS = (6 / 29) ** 3


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    c = img.astype(np.float32) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.sqrt(((lab1 - lab2) ** 2).sum(axis=-1)).astype(np.float32)
