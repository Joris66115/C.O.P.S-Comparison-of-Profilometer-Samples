"""Regenerate the small real-data test fixtures from a MOPA export folder.

Usage: python tests/data/make_test_data.py /path/to/MOPA
"""
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = Path(__file__).parent


def main(mopa: Path) -> None:
    studiable = Image.open(mopa / "pseudo-colour-view/with-scalebar/M1-pseudo-colour-studiable.png").convert("RGB")
    # Colour bar with its black frame and some white margin (full image: 9212 x 5904).
    studiable.crop((8330, 560, 8600, 5600)).save(OUT / "colourbar.png")

    box = (3000, 3000, 3768, 3768)  # same 768 x 768 region in both scans
    for src, dst in [("M35-pseudo-colour-image.jpg", "pc_M35.jpg"),
                     ("M35-high-res-pseudo-colour-image.jpg", "pc_M35_highres.jpg")]:
        img = Image.open(mopa / "pseudo-colour-view/image-only" / src).convert("RGB")
        img.crop(box).save(OUT / dst, quality=95, subsampling=0)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
