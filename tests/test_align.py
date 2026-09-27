import numpy as np
import pytest

from profilometer_comparison.align import estimate_shift, overlap_slices, phase_correlation
from profilometer_comparison.colourmap import default_colourmap, rgb_to_height


def texture(shape, seed=0, sigma=2.0):
    """Smooth random surface (Gaussian-filtered noise, sigma in px)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=shape))
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    return np.fft.ifft2(f * np.exp(-2 * (np.pi * sigma) ** 2 * (kx ** 2 + ky ** 2))).real.astype(np.float32)


def shifted_pair(big, margin, size, dy, dx):
    """before = big[m:m+size]; after such that after[y+dy, x+dx] == before[y, x]."""
    before = big[margin:margin + size, margin:margin + size]
    after = big[margin - dy:margin - dy + size, margin - dx:margin - dx + size].copy()
    return before, after


def test_phase_correlation_recovers_shift():
    before, after = shifted_pair(texture((300, 300)), 50, 200, 7, -12)
    dy, dx, psr = phase_correlation(before, after)
    assert (dy, dx) == (7, -12)
    assert psr > 20


def test_phase_correlation_handles_unequal_shapes():
    before, after = shifted_pair(texture((300, 300)), 50, 200, 3, 4)
    dy, dx, _ = phase_correlation(before, after[:-2, :-1])
    assert (dy, dx) == (3, 4)


def test_overlap_slices():
    bs, as_ = overlap_slices((100, 100), (100, 102), 5, -3)
    assert bs == (slice(0, 95), slice(3, 100))
    assert as_ == (slice(5, 100), slice(0, 97))
    with pytest.raises(ValueError):
        overlap_slices((10, 10), (10, 10), 20, 0)


def test_estimate_shift_large_shift():
    before, after = shifted_pair(texture((1400, 1400), seed=1), 250, 900, 150, -230)
    al = estimate_shift(before, after, window=512)
    assert (al.dy, al.dx) == (150, -230)
    assert al.confidence > 20
    assert al.method == "auto"


def test_unrelated_images_have_low_confidence():
    al = estimate_shift(texture((800, 800), seed=2), texture((800, 800), seed=3), window=256)
    assert al.confidence < 15


@pytest.fixture(scope="module")
def real_heights(pc_before_rgb, pc_highres_rgb):
    cmap = default_colourmap()
    return rgb_to_height(pc_before_rgb, cmap).height, rgb_to_height(pc_highres_rgb, cmap).height


def test_real_rescan_alignment(real_heights):
    before, after = real_heights
    al = estimate_shift(before, after, window=512)
    assert abs(al.dy - 2) <= 1 and abs(al.dx) <= 1
    assert al.confidence > 20


def alter_tiles(after, fraction, seed=0, tile=64):
    """Replace `fraction` of the tiles by a smooth patch 8 µm lower (simulated change). Returns the altered mask."""
    rng = np.random.default_rng(seed)
    ny, nx = after.shape[0] // tile, after.shape[1] // tile
    chosen = rng.choice(ny * nx, int(round(fraction * ny * nx)), replace=False)
    altered = np.zeros(after.shape, bool)
    level = np.nanmean(after)
    for k in chosen:
        i, j = divmod(int(k), nx)
        block = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
        after[block] = level - 8 + rng.normal(0, 1, (tile, tile))
        altered[block] = True
    return altered


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5])
def test_alignment_survives_altered_surface(real_heights, fraction):
    before, after = shifted_pair(real_heights[0], 64, 640, 23, -41)
    alter_tiles(after, fraction)
    al = estimate_shift(before, after, window=512)
    assert (al.dy, al.dx) == (23, -41)


def test_stable_mask_rescues_heavily_altered_surface(real_heights):
    dy, dx = 23, -41
    before, after = shifted_pair(real_heights[0], 64, 640, dy, dx)
    altered = alter_tiles(after, 0.7, seed=1)
    # Stable pixels in before coordinates: those whose partner in `after` was not altered.
    bs, as_ = overlap_slices(before.shape, after.shape, dy, dx)
    stable = np.zeros(before.shape, bool)
    stable[bs] = ~altered[as_]
    al = estimate_shift(before, after, stable=stable, window=256)
    assert (al.dy, al.dx) == (dy, dx)
