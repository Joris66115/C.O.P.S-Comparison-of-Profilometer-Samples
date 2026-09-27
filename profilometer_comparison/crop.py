"""Find and crop the plot frame of a MountainsMap image export with axes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colourmap import _runs


@dataclass(frozen=True)
class Frame:
    top: int
    bottom: int  # exclusive
    left: int
    right: int   # exclusive


def _inner_edges(line_mask: np.ndarray) -> tuple[int, int]:
    runs = _runs(line_mask)
    if len(runs) < 2:
        raise ValueError("no plot frame found")
    return runs[0][1], runs[-1][0]


def find_plot_frame(img: np.ndarray, dark: int = 60, line_frac: float = 0.4) -> Frame:
    """Interior of the black rectangular frame drawn around the image.

    Frame lines are rows/columns in which more than `line_frac` of the pixels are
    black (RGB sum < `dark`).
    """
    black = img.astype(np.int32).sum(axis=2) < dark
    top, bottom = _inner_edges(black.mean(axis=1) > line_frac)
    left, right = _inner_edges(black.mean(axis=0) > line_frac)
    if bottom <= top or right <= left:
        raise ValueError("no plot frame found")
    return Frame(top, bottom, left, right)


def crop_to_frame(img: np.ndarray, frame: Frame) -> np.ndarray:
    return img[frame.top:frame.bottom, frame.left:frame.right]
