import pytest

from profilometer_comparison.masks import (
    load_masks, point_in_polygon, polygons_to_mask, save_masks, validate_polygon,
)

SQUARE = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]


def test_validate_polygon():
    validate_polygon(SQUARE)
    validate_polygon([(0, 0), (5, 0), (0, 5)])
    with pytest.raises(ValueError, match="at least 3"):
        validate_polygon([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="cross"):
        validate_polygon([(0, 0), (10, 10), (10, 0), (0, 10)])  # bow tie


def test_point_in_polygon():
    assert point_in_polygon((15, 15), SQUARE)
    assert not point_in_polygon((5, 15), SQUARE)


def test_polygons_to_mask():
    mask = polygons_to_mask([SQUARE], (40, 40), pixel_um=1.0)
    assert mask[15, 15] and not mask[5, 5]
    assert 100 <= mask.sum() <= 121


def test_polygons_to_mask_uses_pixel_size_and_origin():
    mask = polygons_to_mask([SQUARE], (40, 40), pixel_um=2.0, origin_px=(2, 2))
    # 10-20 µm = 5-10 px in the before image = 3-8 px in this array
    assert mask[5, 5] and not mask[1, 1] and not mask[9, 9]


def test_masks_json_roundtrip(tmp_path):
    path = tmp_path / "masks.json"
    assert load_masks(path) == {}
    save_masks(path, {"M1": [SQUARE]})
    assert load_masks(path) == {"M1": [SQUARE]}
