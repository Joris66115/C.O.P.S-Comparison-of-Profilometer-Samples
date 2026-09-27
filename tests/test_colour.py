import numpy as np

from profilometer_comparison.colour import delta_e76, srgb_to_lab


def test_reference_colours():
    img = np.array([[[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 255]]], np.uint8)
    lab = srgb_to_lab(img)[0]
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)
    np.testing.assert_allclose(lab[3], [32.30, 79.19, -107.86], atol=0.05)


def test_delta_e76():
    a = np.array([[[100, 0, 0], [50, 10, 10]]], np.float32)
    b = np.array([[[50, 0, 0], [50, 10, 10]]], np.float32)
    np.testing.assert_allclose(delta_e76(a, b), [[50, 0]], atol=1e-5)
