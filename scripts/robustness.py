"""Alignment robustness on a full-resolution real scan.

Shifts the scan by a known amount, replaces a growing fraction of it by a changed
surface (patches 8 µm lower, smoothed), and reports whether the shift is recovered.

Usage: python scripts/robustness.py path/to/M35-pseudo-colour-image.jpg
"""
import sys

import numpy as np

from profilometer_comparison.align import estimate_shift
from profilometer_comparison.prepare import Settings, load_colourmap, load_image

TILE = 500
SHIFT = (150, -230)


def main(path: str) -> None:
    settings = Settings(type="pseudo-colour")
    height = load_image(path, settings, load_colourmap(settings)).align
    m = 400
    before = height[m:-m, m:-m]
    after = height[m - SHIFT[0]:height.shape[0] - m - SHIFT[0], m - SHIFT[1]:height.shape[1] - m - SHIFT[1]].copy()
    rng = np.random.default_rng(0)
    ny, nx = after.shape[0] // TILE, after.shape[1] // TILE
    print(f"true shift {SHIFT}; tiles of {TILE} px altered")
    print("fraction altered | recovered shift | correct | confidence")
    for fraction in (0.0, 0.1, 0.3, 0.5, 0.7, 0.85):
        altered = after.copy()
        level = np.nanmean(after)
        for k in rng.choice(ny * nx, int(round(fraction * ny * nx)), replace=False):
            i, j = divmod(int(k), nx)
            altered[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE] = level - 8 + rng.normal(0, 1, (TILE, TILE))
        al = estimate_shift(before, altered)
        ok = (al.dy, al.dx) == SHIFT
        print(f"{fraction:16.0%} | {str((al.dy, al.dx)):15s} | {'yes' if ok else 'NO':7s} | {al.confidence:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
