"""Figure export: before / after / difference with scale bars, legend and info lines."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colourmap import Colourmap
from .viewer_state import HIGHER, LOWER, MARKED, NO_DATA, UNCHANGED

MASK_COLOUR = (255, 220, 0)
TEXT = (20, 20, 20)
BACKGROUND = (255, 255, 255)
# Fonts with µ, ≥ and Δ, tried in order (Linux, macOS, Windows, ...); Pillow searches the system font folders.
FONT_CANDIDATES = ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttc")
ASCII_FALLBACK = {"µ": "u", "≥": ">=", "Δ": "d"}


def scalebar_length(target_um: float) -> float:
    """Largest 1-2-5 value (e.g. 200, 500, 1000 µm) not exceeding the target."""
    exponent = math.floor(math.log10(target_um))
    for mantissa in (5, 2, 1):
        value = mantissa * 10 ** exponent
        if value <= target_um * (1 + 1e-9):
            return value
    return 10 ** exponent


def format_length(um: float) -> str:
    if um >= 1000:
        return f"{um / 1000:g} mm"
    return f"{um:g} µm"


def scalebar_for(panel_width_px: int, um_per_px: float, fraction: float = 0.2) -> tuple[float, int]:
    """(length in µm, length in px) of a round scale bar of about `fraction` of the panel width."""
    length_um = scalebar_length(panel_width_px * um_per_px * fraction)
    return length_um, round(length_um / um_per_px)


def region_summary(diff: np.ndarray, threshold: float, signed: bool) -> str:
    """Percentages of changed pixels in a difference array (NaN = no data)."""
    d = np.asarray(diff, np.float64)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return "no data"
    if not signed:
        return f"{100 * np.mean(d >= threshold):.2f} % different"
    lower, higher = 100 * np.mean(d <= -threshold), 100 * np.mean(d >= threshold)
    return f"{lower + higher:.2f} % different ({lower:.2f} % lower, {higher:.2f} % higher)"


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for name in FONT_CANDIDATES:
        try:
            ImageFont.truetype(name, 12)
            return name
        except OSError:
            continue
    return None


def _font(size: int) -> ImageFont.ImageFont:
    path = _font_path()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def _t(text: str) -> str:
    """Text as drawn: unchanged with a Unicode font, ASCII replacements otherwise."""
    if _font_path():
        return text
    for symbol, replacement in ASCII_FALLBACK.items():
        text = text.replace(symbol, replacement)
    return text


def _text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=TEXT) -> None:
    draw.text(xy, _t(text), fill=fill, font=font)


def _length(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    return draw.textlength(_t(text), font=font)


def _draw_scalebar(img: Image.Image, um_per_px: float, font_size: int) -> None:
    length_um, length_px = scalebar_for(img.width, um_per_px)
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    label = format_length(length_um)
    thickness = max(4, img.width // 150)
    pad = font_size // 2
    box_w = max(length_px, _length(draw, label, font)) + 2 * pad
    box_h = font_size + thickness + 3 * pad
    x0, y1 = pad * 2, img.height - pad * 2
    y0 = y1 - box_h
    draw.rectangle([x0, y0, x0 + box_w, y1], fill=BACKGROUND)
    bar_x = x0 + pad
    bar_y = y1 - pad - thickness
    draw.rectangle([bar_x, bar_y, bar_x + length_px - 1, bar_y + thickness - 1], fill=(0, 0, 0))
    _text(draw, (bar_x, y0 + pad), label, font)


def _draw_polygons(img: Image.Image, polygons, width: int) -> None:
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.line(list(poly) + [poly[0]], fill=MASK_COLOUR, width=width, joint="curve")


def _height_bar(cmap: Colourmap, width: int, height: int) -> tuple[Image.Image, float, float]:
    z_min = math.floor(cmap.height[0] / 10) * 10
    z_max = math.ceil(cmap.height[-1] / 10) * 10
    z = np.linspace(z_min, z_max, width)
    idx = np.clip(np.searchsorted(cmap.height, z), 0, len(cmap.height) - 1)
    row = cmap.rgb[idx]
    return Image.fromarray(np.repeat(row[None, :, :], height, axis=0)), z_min, z_max


def render_export(before: Image.Image, after: Image.Image, diff_rgb: np.ndarray, um_per_px: float,
                  title: str, info: str, signed: bool, threshold: float, cmap: Colourmap | None,
                  polygons=()) -> Image.Image:
    """Compose the export figure.

    `before`, `after` and `diff_rgb` have the same size; `um_per_px` is the size of one of
    their pixels; `polygons` are mask outlines in their pixel coordinates.
    """
    w, h = before.size
    fs = max(14, w // 45)
    m = fs
    unit = "µm" if signed else "ΔE"

    panels = []
    for img in (before.convert("RGB"), after.convert("RGB"), Image.fromarray(diff_rgb).convert("RGB")):
        img = img.copy()
        if polygons:
            _draw_polygons(img, polygons, max(2, w // 400))
        _draw_scalebar(img, um_per_px, fs)
        panels.append(img)

    title_lines = title.split("\n")
    title_h = round(fs * 1.5) * len(title_lines) + fs // 3
    label_h, legend_h, info_h = round(fs * 1.6), round(fs * 2.2), round(fs * 1.6)
    width = 3 * w + 4 * m
    height = m + title_h + label_h + h + m + 2 * legend_h + info_h + m
    out = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(out)
    font, bold, small = _font(fs), _font(round(fs * 1.15)), _font(round(fs * 0.8))

    y = m
    for line in title_lines:
        _text(draw, (m, y), line, bold)
        y += round(fs * 1.5)
    y += fs // 3
    for i, (label, img) in enumerate(zip(("BEFORE", "AFTER (aligned)", "DIFFERENCE"), panels)):
        x = m + i * (w + m)
        _text(draw, (x + (w - _length(draw, label, font)) / 2, y), label, font)
        out.paste(img, (x, y + label_h))
    y += label_h + h + m

    # Legend row 1: height colour scale (pseudo-colour only)
    if signed and cmap is not None:
        label = "Height (µm)"
        _text(draw, (m, y), label, font)
        bar_x = m + round(_length(draw, label, font)) + fs
        bar_w = min(w, width - bar_x - 4 * fs)
        bar, z_min, z_max = _height_bar(cmap, bar_w, round(fs * 0.8))
        out.paste(bar, (bar_x, y))
        draw.rectangle([bar_x, y, bar_x + bar_w - 1, y + bar.height - 1], outline=TEXT)
        step = scalebar_length((z_max - z_min) / 5)
        tick = math.ceil(z_min / step) * step
        while tick <= z_max + 1e-9:
            tx = bar_x + (tick - z_min) / (z_max - z_min) * (bar_w - 1)
            draw.line([tx, y + bar.height, tx, y + bar.height + fs // 4], fill=TEXT)
            text = f"{tick:g}"
            _text(draw, (tx - _length(draw, text, small) / 2, y + bar.height + fs // 4), text, small)
            tick += step
    y += legend_h

    # Legend row 2: difference colours and mask outline
    if signed:
        entries = [(LOWER, f"lower ≥ {threshold:g} {unit}"), (HIGHER, f"higher ≥ {threshold:g} {unit}"),
                   (UNCHANGED, f"|Δh| < {threshold:g} {unit}"), (NO_DATA, "no data")]
    else:
        entries = [(MARKED, f"ΔE ≥ {threshold:g}"), ((150, 150, 150), "below threshold (after image, grey)")]
    x = m
    for colour, text in entries:
        draw.rectangle([x, y, x + fs, y + fs], fill=colour, outline=TEXT)
        _text(draw, (x + fs * 1.4, y), text, font)
        x += round(fs * 2.4 + _length(draw, text, font))
    if polygons:
        draw.rectangle([x, y, x + fs, y + fs], outline=MASK_COLOUR, width=max(2, fs // 6))
        _text(draw, (x + fs * 1.4, y), "encrustation mask", font)
    y += legend_h

    _text(draw, (m, y), info, _font(round(fs * 0.85)), fill=(90, 90, 90))
    return out
