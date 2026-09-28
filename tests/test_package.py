import numpy as np
from PIL import Image

import profilometer_comparison
from profilometer_comparison.images import read_rgb


def test_version():
    assert profilometer_comparison.__version__ == "1.0.2"


def test_read_rgb_converts_to_uint8_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (4, 3), 128).save(path)
    img = read_rgb(path)
    assert img.shape == (3, 4, 3)
    assert img.dtype == np.uint8
    assert img[0, 0].tolist() == [128, 128, 128]
