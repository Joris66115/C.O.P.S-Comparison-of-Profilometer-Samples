import numpy as np
import pytest

from profilometer_comparison.crop import Frame, crop_to_frame, find_plot_frame


def framed_image():
    img = np.full((400, 500, 3), 255, np.uint8)
    img[50:350, 60:460] = 0                 # outer edge of a 3 px black frame
    img[53:347, 63:457] = (120, 110, 100)   # interior (dark-ish tile colour, but not black)
    img[10:20, 100:300] = 0                 # a title line: too short to count as frame
    return img


def test_find_plot_frame():
    assert find_plot_frame(framed_image()) == Frame(top=53, bottom=347, left=63, right=457)


def test_crop_to_frame():
    out = crop_to_frame(framed_image(), Frame(53, 347, 63, 457))
    assert out.shape == (294, 394, 3)
    assert (out == (120, 110, 100)).all()


def test_no_frame_raises():
    with pytest.raises(ValueError):
        find_plot_frame(np.full((50, 50, 3), 200, np.uint8))
