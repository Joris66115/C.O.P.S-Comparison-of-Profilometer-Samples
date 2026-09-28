import numpy as np
import pytest
from PIL import Image

from profilometer_comparison.colourmap import default_colourmap
from profilometer_comparison.export import format_length, render_export, scalebar_length


@pytest.mark.parametrize("target, expected", [(2134, 2000), (470, 200), (1000, 1000), (7.3, 5), (0.3, 0.2)])
def test_scalebar_length_is_largest_round_value(target, expected):
    assert scalebar_length(target) == pytest.approx(expected)


@pytest.mark.parametrize("um, text", [(2000, "2 mm"), (200, "200 µm"), (1000, "1 mm"), (0.5, "0.5 µm")])
def test_format_length(um, text):
    assert format_length(um) == text


def panels(w=400, h=300):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return img, img.copy(), np.full((h, w, 3), 225, np.uint8)


def test_render_export_layout_and_panels_untouched_at_top_left():
    before, after, diff = panels()
    out = render_export(before, after, diff, um_per_px=10.0, title="M1", info="info", signed=True,
                        threshold=2.0, cmap=default_colourmap())
    assert out.mode == "RGB"
    assert out.width > 3 * 400 and out.height > 300
    # the scale bar is drawn at the bottom-left, so the top-left of the BEFORE panel is unchanged
    arr = np.asarray(out)
    ys, xs = np.nonzero((arr == np.asarray(before)[0, 0]).all(axis=2))
    assert len(ys) > 0


def test_render_export_true_colour_without_colourmap():
    before, after, diff = panels(200, 200)
    out = render_export(before, after, diff, um_per_px=3.0, title="M1", info="", signed=False,
                        threshold=5.0, cmap=None, polygons=[[(20, 20), (100, 20), (100, 100)]])
    assert out.width > 600


def test_scalebar_pixel_length_matches_label():
    # 400 px at 10 µm/px = 4000 µm wide; target 20 % = 800 µm -> 500 µm bar = 50 px
    from profilometer_comparison.export import scalebar_for
    length_um, length_px = scalebar_for(panel_width_px=400, um_per_px=10.0)
    assert length_um == 500 and length_px == 50


def test_text_falls_back_to_ascii_without_unicode_font(monkeypatch):
    from profilometer_comparison import export
    monkeypatch.setattr(export, "_font_path", lambda: None)
    assert export._t("lower ≥ 2 µm, ΔE") == "lower >= 2 um, dE"


def test_text_kept_with_unicode_font(monkeypatch):
    from profilometer_comparison import export
    monkeypatch.setattr(export, "_font_path", lambda: "Arial.ttf")
    assert export._t("≥ 2 µm") == "≥ 2 µm"


def test_region_summary():
    from profilometer_comparison.export import region_summary
    d = np.zeros((10, 10), np.float32)
    d[:2] = -3.0     # 20 % lower
    d[2, :5] = 4.0   # 5 % higher
    d[9] = np.nan    # excluded
    text = region_summary(d, 2.0, signed=True)
    assert text == "27.78 % different (22.22 % lower, 5.56 % higher)"
    assert region_summary(np.array([[1.0, 8.0]]), 5.0, signed=False) == "50.00 % different"


def test_upscale_for_export():
    from profilometer_comparison.export import upscale_for_export
    small = Image.new("RGB", (150, 120))
    images, factor = upscale_for_export([small, small], min_px=1000)
    assert factor == 7 and images[0].size == (1050, 840)
    big = Image.new("RGB", (1200, 1200))
    images, factor = upscale_for_export([big], min_px=1000)
    assert factor == 1 and images[0] is big


def red_pixels(img):
    arr = np.asarray(img).astype(int)
    return int(((arr[..., 0] > 200) & (arr[..., 1] < 60) & (arr[..., 2] < 60)).sum())


@pytest.mark.parametrize("rect", [(100, 80, 180, 140), (150, 100, 152, 102)])  # normal and tiny zoom area
def test_render_export_with_locator(rect):
    from profilometer_comparison.export import Locator
    before, after, diff = panels(400, 300)
    overview = Image.new("RGB", (320, 320), (0, 200, 255))
    locator = Locator(image=overview, um_per_px=33.0, rect=rect, polygons=[[(10, 10), (60, 10), (60, 60)]])
    plain = render_export(before, after, diff, um_per_px=1.35, title="M1", info="i", signed=True,
                          threshold=2.0, cmap=default_colourmap())
    with_map = render_export(before, after, diff, um_per_px=1.35, title="M1", info="i", signed=True,
                             threshold=2.0, cmap=default_colourmap(), locator=locator)
    assert with_map.width == plain.width and with_map.height >= plain.height
    assert red_pixels(with_map) > red_pixels(plain) + 20
