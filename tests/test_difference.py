import numpy as np

from profilometer_comparison.difference import (
    Plane, fit_plane, histogram, percentages, subtract_plane,
)


def test_identical_is_zero_percent():
    hist = histogram(np.zeros(1000), signed=True)
    assert percentages(hist, 2.0, signed=True) == {"diff": 0.0, "lower": 0.0, "higher": 0.0}


def test_lowered_patch():
    d = np.zeros((100, 100), np.float32)
    d[:20, :50] = -5.0  # 10 % of the area 5 µm lower
    hist = histogram(d.ravel(), signed=True)
    p = percentages(hist, 2.0, signed=True)
    assert p["lower"] == 10.0 and p["higher"] == 0.0 and p["diff"] == 10.0
    assert percentages(hist, 6.0, signed=True)["diff"] == 0.0


def test_raised_patch_counts_as_higher():
    d = np.zeros(1000)
    d[:30] = 3.0
    p = percentages(histogram(d, signed=True), 2.0, signed=True)
    assert p["higher"] == 3.0 and p["lower"] == 0.0


def test_unsigned_histogram():
    de = np.array([0.5, 1.0, 6.0, 12.0])
    hist = histogram(de, signed=False)
    assert percentages(hist, 5.0, signed=False) == {"diff": 50.0}


def test_values_beyond_range_are_clipped_not_lost():
    hist = histogram(np.array([-1000.0, 1000.0]), signed=True)
    assert hist.sum() == 2
    assert percentages(hist, 2.0, signed=True)["diff"] == 100.0


def test_empty_histogram_gives_nan():
    p = percentages(histogram(np.array([]), signed=False), 2.0, signed=False)
    assert np.isnan(p["diff"])


def test_plane_fit_and_subtract():
    pixel_um = 1.35
    h, w = 400, 300
    y_mm = np.arange(h)[:, None] * pixel_um / 1000
    x_mm = np.arange(w)[None, :] * pixel_um / 1000
    d = (3.0 + 0.5 * x_mm - 0.2 * y_mm).astype(np.float32)
    plane = fit_plane(d, np.ones(d.shape, bool), pixel_um)
    assert abs(plane.offset - 3.0) < 1e-3
    assert abs(plane.tilt_x - 0.5) < 1e-3 and abs(plane.tilt_y + 0.2) < 1e-3
    assert np.abs(subtract_plane(d, plane, pixel_um)).max() < 1e-3


def test_subtract_plane_with_origin():
    plane = Plane(offset=0.0, tilt_x=1.0, tilt_y=0.0)  # 1 µm per mm in x
    d = np.zeros((2, 2), np.float32)
    out = subtract_plane(d, plane, pixel_um=1000.0, origin_px=(0, 5))  # d[0,0] is 5 mm to the right
    np.testing.assert_allclose(out, [[-5, -6], [-5, -6]])


def _surface(h=1200, w=1000, pixel_um=1.35, seed=0):
    rng = np.random.default_rng(seed)
    y_mm = np.arange(h)[:, None] * pixel_um / 1000
    x_mm = np.arange(w)[None, :] * pixel_um / 1000
    noise = rng.normal(0, 0.5, (h, w))
    return (1.5 + 0.8 * x_mm - 0.4 * y_mm + noise).astype(np.float32), y_mm, x_mm


def test_plane_fit_ignores_large_changed_area():
    """A large lowered spot (≈30 % of the surface) must not pull the plane down."""
    pixel_um = 1.35
    d, _, _ = _surface(pixel_um=pixel_um)
    yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
    spot = (yy - 500) ** 2 + (xx - 400) ** 2 < 350 ** 2
    d[spot] -= 40.0
    plane = fit_plane(d, np.ones(d.shape, bool), pixel_um)
    assert abs(plane.offset - 1.5) < 0.1
    assert abs(plane.tilt_x - 0.8) < 0.2 and abs(plane.tilt_y + 0.4) < 0.2
    rest = subtract_plane(d, plane, pixel_um)[~spot]
    assert np.mean(np.abs(rest) >= 2.0) < 0.001  # unchanged glaze stays unchanged


def test_plane_fit_on_unchanged_surface_matches_least_squares():
    pixel_um = 1.35
    d, _, _ = _surface(pixel_um=pixel_um, seed=1)
    plane = fit_plane(d, np.ones(d.shape, bool), pixel_um)
    assert abs(plane.offset - 1.5) < 0.02
    assert abs(plane.tilt_x - 0.8) < 0.05 and abs(plane.tilt_y + 0.4) < 0.05
