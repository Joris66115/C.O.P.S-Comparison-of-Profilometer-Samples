"""Image reading that works for very large profilometer exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Full-resolution exports reach ~72 megapixels; Pillow's default guard is not needed here.
Image.MAX_IMAGE_PIXELS = None


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
