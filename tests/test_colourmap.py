import numpy as np

from profilometer_comparison.colourmap import (
    build_from_studiable, default_colourmap, find_colour_bar, load_csv, rgb_to_height, save_csv,
)


def test_find_colour_bar_on_real_crop(colourbar_img):
    # Crop origin is (x 8330, y 560) of the full image; interior rows 607-5537, centre column 8450.
    assert find_colour_bar(colourbar_img, search_from_frac=0.0) == (47, 4977, 120)


def test_build_from_studiable(colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    assert 350 <= len(cmap.height) <= 450
    assert np.all(np.diff(cmap.height) >= 0)
    assert cmap.rgb[0].tolist() == [0, 0, 0]
    assert cmap.rgb[-1].tolist() == [255, 248, 248]  # the top of the MarSurf scale is near-white
    assert cmap.height[0] < 5 and cmap.height[-1] > 345


def test_csv_roundtrip(tmp_path, colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    save_csv(cmap, tmp_path / "cm.csv")
    back = load_csv(tmp_path / "cm.csv")
    np.testing.assert_array_equal(back.rgb, cmap.rgb)
    np.testing.assert_allclose(back.height, cmap.height, atol=1e-4)


def test_rgb_to_height_maps_bar_colours_back(colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    top, bottom, col = find_colour_bar(colourbar_img, search_from_frac=0.0)
    column = colourbar_img[top:bottom + 1, col][None, :, :]
    hm = rgb_to_height(column, cmap)
    expected = 350 - (np.arange(top, bottom + 1) - top) / (bottom - top) * 350
    inner = (expected > 10) & (expected < 340)  # the ends are saturated black/white bands
    assert np.abs(hm.height[0] - expected)[inner].max() < 1.5
    assert not hm.excluded.any()


def test_off_scale_colour_is_excluded_and_ends_are_saturated(colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    img = np.array([[[128, 128, 128], [0, 0, 0], [255, 255, 255]]], np.uint8)
    hm = rgb_to_height(img, cmap)
    assert hm.excluded.tolist() == [[True, False, False]]
    assert np.isnan(hm.height[0, 0])
    assert hm.saturated.tolist() == [[False, True, True]]


def test_default_colourmap_is_packaged():
    cmap = default_colourmap()
    assert len(cmap.height) > 350
    assert cmap.height[-1] > 345
