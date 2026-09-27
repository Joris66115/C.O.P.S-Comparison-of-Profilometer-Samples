from pathlib import Path

import numpy as np
import pytest

from profilometer_comparison.images import read_rgb

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def colourbar_img() -> np.ndarray:
    return read_rgb(DATA / "colourbar.png")


@pytest.fixture(scope="session")
def pc_before_rgb() -> np.ndarray:
    return read_rgb(DATA / "pc_M35.jpg")


@pytest.fixture(scope="session")
def pc_highres_rgb() -> np.ndarray:
    return read_rgb(DATA / "pc_M35_highres.jpg")
