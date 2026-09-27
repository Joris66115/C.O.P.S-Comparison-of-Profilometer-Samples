"""Encrustation masks: polygons in µm of the before image (x right, y down)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Polygon = list[tuple[float, float]]


def _orient(p, q, r) -> int:
    v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (v > 0) - (v < 0)


def _segments_cross(p1, p2, p3, p4) -> bool:
    return (_orient(p1, p2, p3) * _orient(p1, p2, p4) < 0
            and _orient(p3, p4, p1) * _orient(p3, p4, p2) < 0)


def validate_polygon(points: Polygon) -> None:
    if len(points) < 3:
        raise ValueError("a polygon needs at least 3 points")
    n = len(points)
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # first and last edge share a point
            if _segments_cross(*edges[i], *edges[j]):
                raise ValueError("polygon edges cross each other")


def point_in_polygon(point, polygon: Polygon) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def polygons_to_mask(polygons: list[Polygon], shape, pixel_um: float, origin_px=(0, 0)) -> np.ndarray:
    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    row0, col0 = origin_px
    for poly in polygons:
        draw.polygon([(x / pixel_um - col0, y / pixel_um - row0) for x, y in poly], fill=1)
    return np.asarray(img, dtype=bool)


def load_masks(path: str | Path) -> dict[str, list[Polygon]]:
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {sample: [[(float(x), float(y)) for x, y in poly] for poly in polys] for sample, polys in raw.items()}


def save_masks(path: str | Path, masks: dict[str, list[Polygon]]) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({s: [[list(p) for p in poly] for poly in polys] for s, polys in masks.items()},
                              indent=2), encoding="utf-8")
    tmp.replace(path)
