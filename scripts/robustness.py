"""Alignment robustness on a full-resolution real scan (shift with altered surface, and rotation).

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


def rotation_table(path: str) -> None:
    """Rotate the full scan by known angles and check that the rotation search recovers them."""
    from profilometer_comparison.align import estimate_rotation, rotate_array
    import time

    settings = Settings(type="pseudo-colour")
    height = load_image(path, settings, load_colourmap(settings)).align
    m = 600
    before = height[m:-m, m:-m]
    print("\nrotation | recovered | error | time")
    for angle in (0.37, -1.6, 2.5, -4.9):
        after = rotate_array(height, angle)[m:-m, m:-m]  # the surface turned counter-clockwise by `angle`
        t = time.time()
        correction, _ = estimate_rotation(before, after)
        print(f"{angle:+8.2f}° | {-correction:+8.3f}° | {abs(-correction - angle):.3f}° | {time.time() - t:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1])
    rotation_table(sys.argv[1])
