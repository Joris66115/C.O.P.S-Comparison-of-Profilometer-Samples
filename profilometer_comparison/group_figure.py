"""Export a figure of a sample group: before / after / difference per sample, side by side."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from . import NAME, __version__
from .export import GroupColumn, render_group_export
from .masks import load_masks
from .prepare import load_colourmap
from .viewer_state import ViewerState, render_difference, rotation_text

PANEL_MAX = 1000  # px, longest side of each panel in a group figure


def _column_lines(state: ViewerState, sample: str) -> list[str]:
    t, unit = state.threshold, "µm" if state.signed else "ΔE"
    regions = ["glaze", "encrustation"] if state.percentages("encrustation", sample) is not None else ["glaze"]
    lines = []
    for region in regions:
        p = state.percentages(region, sample)
        prefix = f"{region}: " if len(regions) > 1 else ""
        lines.append(f"{prefix}{p['diff']:.2f} % changed by ≥ {t:g} {unit}")
        if state.signed:
            lines.append(f"({p['lower']:.2f} % lowered, {p['higher']:.2f} % raised)")
    m = state.meta(sample)
    lines.append(f"{rotation_text(m)} · shift {m['shift_x_um']:+.1f}/{m['shift_y_um']:+.1f} µm")
    return lines


def _column(state: ViewerState, sample: str, label: str, masks: dict) -> GroupColumn:
    folder = state.cache / "pairs" / sample
    m = state.meta(sample)
    before = Image.open(folder / "before.png").convert("RGB")
    after = Image.open(folder / "after.png").convert("RGB")
    diff = np.load(folder / "diff.npy")
    diff_rgb = render_difference(diff, state.threshold, state.signed, np.asarray(after))
    scale = min(1.0, PANEL_MAX / max(before.size))
    size = (max(1, round(before.width * scale)), max(1, round(before.height * scale)))
    step, px = m["display_step"], m["pixel_um"]
    r0, c0 = m["before_origin_px"]
    polygons = [[((x / px - c0) / step * scale, (y / px - r0) / step * scale) for x, y in poly]
                for poly in masks.get(sample, [])]
    return GroupColumn(
        before=before.resize(size, Image.BILINEAR),
        after=after.resize(size, Image.BILINEAR),
        diff_rgb=np.asarray(Image.fromarray(diff_rgb).resize(size, Image.NEAREST)),
        um_per_px=px * step / scale,
        heading=f"{sample} · {label}" if label else sample,
        lines=_column_lines(state, sample),
        polygons=polygons,
    )


def export_group(state: ViewerState, group: str) -> tuple[Path, list[tuple[str, str]]]:
    """Write exports/group-<name>-<threshold>.jpg; returns its path and the (sample, label) members used."""
    members = [(s, label) for s, label in state.groups.members(group) if s in state.samples]
    if not members:
        raise ValueError(f"group {group!r} has no compared samples")
    masks = load_masks(state.cache / "masks.json")
    columns = [_column(state, s, label, masks) for s, label in members]
    unit = "µm" if state.signed else "ΔE"
    title = f"Group {group} · {state.settings.type} · threshold {state.threshold:g} {unit}"
    info = f"{NAME} {__version__} · {date.today():%Y-%m-%d}"
    figure = render_group_export(columns, title, info, state.signed, state.threshold, load_colourmap(state.settings))
    safe = re.sub(r"[^A-Za-z0-9]+", "-", group).strip("-") or "group"
    tag = f"{state.threshold:g}um" if state.signed else f"dE{state.threshold:g}"
    folder = state.cache / "exports"
    folder.mkdir(exist_ok=True)
    path = folder / f"group-{safe}-{tag}.jpg"
    figure.save(path, quality=95)
    return path, members
