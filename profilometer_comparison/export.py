"""Figure export: before / after / difference with scale bars, legend and info lines."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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
LOCATOR_COLOUR = (255, 0, 0)
LOCATOR_MIN_BOX = 12  # px: smaller zoom areas get a box of this size plus crosshair lines


@dataclass
class Locator:
    """Overview image for the 'location of zoom' inset."""
    image: Image.Image       # whole-sample overview (before image)
    um_per_px: float         # size of one overview pixel
    rect: tuple              # zoom area (x0, y0, x1, y1) in overview pixels
    polygons: list = field(default_factory=list)  # mask outlines in overview pixels


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


def upscale_for_export(images: list, min_px: int = 1000) -> tuple[list, int]:
    """Enlarge small images by an integer factor (no smoothing) so the longest side is at least `min_px`."""
    longest = max(max(img.size) for img in images)
    factor = max(1, math.ceil(min_px / longest))
    if factor == 1:
        return images, 1
    return [img.resize((img.width * factor, img.height * factor), Image.NEAREST) for img in images], factor


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


def _render_locator(loc: Locator, width: int, font_size: int) -> Image.Image:
    scale = width / loc.image.width
    img = loc.image.convert("RGB").resize((width, max(1, round(loc.image.height * scale))), Image.BILINEAR)
    if loc.polygons:
        _draw_polygons(img, [[(x * scale, y * scale) for x, y in poly] for poly in loc.polygons], 2)
    _draw_scalebar(img, loc.um_per_px / scale, font_size)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = (v * scale for v in loc.rect)
    if x1 - x0 < LOCATOR_MIN_BOX or y1 - y0 < LOCATOR_MIN_BOX:
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, LOCATOR_MIN_BOX / 2
        draw.line([0, cy, cx - half - 3, cy], fill=LOCATOR_COLOUR, width=1)
        draw.line([cx + half + 3, cy, img.width, cy], fill=LOCATOR_COLOUR, width=1)
        draw.line([cx, 0, cx, cy - half - 3], fill=LOCATOR_COLOUR, width=1)
        draw.line([cx, cy + half + 3, cx, img.height], fill=LOCATOR_COLOUR, width=1)
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=BACKGROUND, width=5)
    draw.rectangle([x0, y0, x1, y1], outline=LOCATOR_COLOUR, width=2)
    draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=TEXT)
    return img


def _height_bar(cmap: Colourmap, width: int, height: int) -> tuple[Image.Image, float, float]:
    z_min = math.floor(cmap.height[0] / 10) * 10
    z_max = math.ceil(cmap.height[-1] / 10) * 10
    z = np.linspace(z_min, z_max, width)
    idx = np.clip(np.searchsorted(cmap.height, z), 0, len(cmap.height) - 1)
    row = cmap.rgb[idx]
    return Image.fromarray(np.repeat(row[None, :, :], height, axis=0)), z_min, z_max


def render_export(before: Image.Image, after: Image.Image, diff_rgb: np.ndarray, um_per_px: float,
                  title: str, info: str, signed: bool, threshold: float, cmap: Colourmap | None,
                  polygons=(), locator: Locator | None = None) -> Image.Image:
    """Compose the export figure.

    `before`, `after` and `diff_rgb` have the same size; `um_per_px` is the size of one of
    their pixels; `polygons` are mask outlines in their pixel coordinates. With a `locator`,
    a small overview showing where the panels lie is added below the difference panel.
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

    font, bold, small = _font(fs), _font(round(fs * 1.15)), _font(round(fs * 0.8))
    title_lines = title.split("\n")
    title_h = round(fs * 1.5) * len(title_lines) + fs // 3
    label_h, legend_h, info_h = round(fs * 1.6), round(fs * 2.2), round(fs * 1.6)
    width = 3 * w + 4 * m

    # Difference legend, wrapped to the width of the difference panel
    if signed:
        entries = [(LOWER, f"lower ≥ {threshold:g} {unit}"), (HIGHER, f"higher ≥ {threshold:g} {unit}"),
                   (UNCHANGED, f"|Δh| < {threshold:g} {unit}"), (NO_DATA, "no data")]
    else:
        entries = [(MARKED, f"ΔE ≥ {threshold:g}"), ((150, 150, 150), "below threshold (after image, grey)")]
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    positions, x, row = [], 0, 0
    for colour, text in entries:
        item_w = round(fs * 1.4 + _length(measure, text, font))
        if x > 0 and x + item_w > w:
            x, row = 0, row + 1
        positions.append((x, row, colour, text))
        x += item_w + fs
    row_h = round(fs * 1.6)
    diff_legend_h = (row + 1) * row_h

    inset = _render_locator(locator, max(160, w // 3), max(10, fs * 2 // 3)) if locator else None
    inset_label_h = round(fs * 1.5)
    left_h = (legend_h if signed and cmap is not None else 0) + (legend_h if polygons else 0)
    right_h = diff_legend_h + (fs // 2 + inset_label_h + inset.height if inset is not None else 0)
    bottom_h = max(left_h, right_h) + fs // 2 + info_h
    height = m + title_h + label_h + h + m + bottom_h + m
    out = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(out)

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
    bottom_y = y

    # Under the difference panel: its colour legend, then (zoom exports) the location map
    diff_x = m + 2 * (w + m)
    for dx, r, colour, text in positions:
        yy = y + r * row_h
        draw.rectangle([diff_x + dx, yy, diff_x + dx + fs, yy + fs], fill=colour, outline=TEXT)
        _text(draw, (diff_x + dx + fs * 1.4, yy), text, font)
    if inset is not None:
        x = diff_x + w - inset.width
        iy = y + diff_legend_h + fs // 2
        _text(draw, (x, iy), "Location of zoom", font)
        out.paste(inset, (x, iy + inset_label_h))

    # Under the before and after panels: height colour scale (pseudo-colour only), spanning both
    if signed and cmap is not None:
        label = "Height (µm)"
        _text(draw, (m, y), label, font)
        bar_x = m + round(_length(draw, label, font)) + fs
        bar_w = m + 2 * w + m - bar_x - fs
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

    # Mask outlines appear on all three panels, so their legend entry stays on the left
    if polygons:
        draw.rectangle([m, y, m + fs, y + fs], outline=MASK_COLOUR, width=max(2, fs // 6))
        _text(draw, (m + fs * 1.4, y), "encrustation mask", font)

    _text(draw, (m, bottom_y + bottom_h - info_h), info, _font(round(fs * 0.85)), fill=(90, 90, 90))
    return out
