# Rotation correction: design

Date: 2026-10-05 · Version 1.2.0 · Status: approved in conversation

## Problem

Samples are put back into the holder for the after-measurement and can end up slightly rotated around the Z axis. Alignment so far corrected only X/Y shift (phase correlation) and Z offset/tilt (plane fit). A rotation of 0.2° already moves the corners of a 10.7 mm field by about 19 µm (14 px), which shows up as false change near the edges.

## Method

- **Search:** the correction angle is searched in −5…+5° (setting `max_rotation_deg`). Coarse: 0.25° steps on 8× downsampled heights (or L\*). Fine: ±0.3° around the best angle in 0.05° steps on 4× downsampled images, then a parabola through the three best points. Score = peak-to-sidelobe ratio (PSR) of the phase correlation, the same measure as the shift confidence. Prototype on M35 (after rotated by known angles 0, +0.37, −1.60, +4.20°): error ≤ 0.001°, about 7 s per pair.
- **Apply:** the **after** image is rotated (PIL, bilinear for heights/Lab/RGB, nearest for masks); the **before** image stays the reference, so encrustation masks (before coordinates) stay valid. Rotations below 0.01° (`min_rotation_deg`) are not applied, to avoid needless interpolation.
- **Crop:** after rotation the after image covers a rotated rectangle. The comparison area is the axis-aligned rectangle inside it (from the rotated corner positions), intersected with the shifted before image. There are no empty corners, and scale bars stay correct because pixel size is unchanged.
- **Then** the existing two-pass shift estimation and plane correction run on the rotated after image.

## Reporting

- `meta.json`: `rotation_deg` (rotation of the after measurement relative to before, counter-clockwise positive as seen in the images = −correction), `rotation_correction_deg`, `rotation_method` (`auto` / `manual` / `off`), `rotation_confidence`.
- `summary.csv` gets the columns `rotation_deg` and `rotation_method`. Warning `rotation near search limit` when |rotation| > max − 0.25°.
- Viewer status line, exported figure info line: `rotation +0.42° (auto)`.

## Manual correction

Viewer keys **[** / **]** turn the after image by −/+0.05° (live preview, applied after a short pause, marked `manual`). **a** resets both rotation and shift to automatic. A manual shift is kept when the rotation is nudged.

## Settings / compatibility

New settings: `rotation_correction` (default on; CLI `--no-rotation-correction`), `max_rotation_deg` = 5, `min_rotation_deg` = 0.01. Caches made with earlier versions have different settings, so resuming them stops with the usual settings-mismatch message (use a new output folder).

## Validation

Unit tests with synthetic and real rotated surfaces (recovered within 0.02°); README gets a rotation table from `scripts/robustness.py` on full-resolution M35.
