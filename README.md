# profilometer-comparison

Compare **before and after** surface measurements of the same samples, exported from MarSurf MfM / MountainsMap. Developed to assess whether laser cleaning (MOPA fibre laser) of glazed ceramic tiles damages the glaze.

For every sample, the tool aligns the before and after measurement, computes where and how much the surface changed, and reports the percentage of changed pixels for the **glaze** and, where you mark it, for **encrustation** separately. A keyboard-driven viewer lets you browse all samples, blink between before and after, draw encrustation masks, correct the alignment by hand, and flag samples.

## Quick start

Requires Python 3.10 or newer with tkinter (included in the python.org installers), numpy and Pillow (`python3 -m pip install numpy Pillow`). Then, from any folder:

```bash
python3 path/to/profilometer-comparison/run.py
```

A menu offers the three steps below, and folders are chosen with dialogs. `run.py` also accepts all commands and options shown under Usage, for example `python3 run.py view RESULTS`.

## Installation (optional)

Installing makes the `python -m profilometer_comparison` commands and the tests available.

```bash
git clone https://github.com/Joris66115/profilometer-comparison.git
cd profilometer-comparison
python3 -m venv ~/.venvs/profilometer-comparison
~/.venvs/profilometer-comparison/bin/pip install -e .
```

The commands below use `python` for `~/.venvs/profilometer-comparison/bin/python`.

**macOS tip:** keep the virtual environment **outside** folders synced by iCloud (such as Desktop and Documents). iCloud can mark files there as hidden, and Python 3.13 then skips the file that makes the package importable (`ModuleNotFoundError: No module named 'profilometer_comparison'`).

## Input

Two folders, **before** and **after**, in which the same sample has **exactly the same filename**. Two image types are quantified:

| Type | Filename ends with | What is compared |
|---|---|---|
| `pseudo-colour` | `-pseudo-colour-image.jpg` (full-resolution pseudo-colour export, no axes) | height, in µm |
| `true-colour` | `-true-colour.png` (true-colour view) | colour, as ΔE\*ab |

Any other export (3D views, profiles, histograms, parameter tables) can be browsed side by side in plain view mode.

For true colour, a full-resolution export **without axes** is recommended. Exports with axes are cropped automatically to the plot frame.

## Usage

### 1. Prepare (once per image type)

```bash
python -m profilometer_comparison prepare
```

Folder dialogs ask for the before and after folders. Or give everything on the command line:

```bash
python -m profilometer_comparison prepare --before BEFORE --after AFTER --out RESULTS
```

This writes a results folder with `summary.csv` (one row per sample and region), `prepare-log.txt`, and a cache for the viewer. A full-resolution pseudo-colour pair takes about 5 s, so 120 samples take about 10 minutes. An interrupted run continues where it stopped.

### 2. View

```bash
python -m profilometer_comparison view RESULTS
```

| Key | Action |
|---|---|
| ← / → | previous / next sample |
| s | flag / unflag sample (saved to `flagged.csv`) |
| ↑ / ↓ or slider | change the threshold; percentages update live |
| f | show flagged samples only |
| b | blink between before and after |
| m | mask mode: click points around encrustation on the BEFORE panel, Enter closes the polygon, Backspace undoes a point, Delete removes the polygon under the cursor, m or Esc finishes |
| Shift + arrows | move the after image by 1 pixel to correct the alignment (Shift + Alt/Option: 10 pixels) |
| a | back to automatic alignment |
| click | open a full-resolution zoom window at that spot |
| e | export the current sample as a JPG figure (also in the zoom window, at full resolution) |

In the difference panel, **blue** means the surface was lowered by at least the threshold (material lost), **red** that it was raised by at least the threshold (material added), **grey** that it changed by less than the threshold (no significant change), and white that there is no data. Δh = height after − height before.

The percentages read, for example, *"Whole surface: 3.21 % of the area changed by ≥ 5 µm (2.90 % lowered, 0.31 % raised)"*: 3.21 % of the measured area changed in height by 5 µm or more, of which 2.90 % became lower and 0.31 % higher. For samples with encrustation masks, the glaze (outside the masks) and the encrustation (inside the masks) are reported separately.

### Zoom window

Click a panel to open a zoom window at that spot. The full-resolution images are loaded once (about 2 s); after that everything is instant.

| Key | Action |
|---|---|
| + / − or scroll wheel | zoom in / out: field of view 0.2, 0.4, 0.8, 1.6 or 3.2 mm (150–2400 measured points) |
| arrows | move the view by half its width |
| e | export the current view as a JPG |
| Esc | close |

The window shows the field of view, its position and the percentage of changed pixels in this region.

### Exported figures

Press **e** to save a JPG of the before, after and difference panels in `RESULTS/exports/`, for example `M17-overview-2um.jpg`. In a zoom window, **e** (or the Export button) saves the current view, for example `M17-zoom-x4620um-y4620um-600px-2um.jpg`. Large views are exported at full resolution; small views are enlarged without smoothing to at least 1000 px per panel, so individual measurement points stay visible. The figure lists the percentages of that region and of the whole sample. A small **location map** of the whole sample, with the zoomed area outlined in red (very small areas also get crosshair lines), is added below the difference panel. Every figure has a scale bar on each panel (calculated from the pixel size, so correct at any zoom), a legend (height scale, difference colours, mask outline) and a line with the alignment shift, file name, tool version and date.

Masks are stored in µm, so a `masks.json` made for pseudo-colour can be reused for true colour: `prepare --masks RESULTS/masks.json ...`.

### Plain view (any image type)

```bash
python -m profilometer_comparison view --before BEFORE --after AFTER
```

## Method

1. **Height from pseudo-colour.** The colour scale of the export (0–350 µm) is stored as a lookup table of 393 colours, about 0.9 µm per step. Each pixel gets the height of the nearest scale colour. Pixels far from any scale colour are excluded (`excluded_pct`), and pixels at the ends of the scale are counted as saturated (`saturated_pct`). Heights are therefore quantised to about 0.9 µm, and **thresholds below about 1 µm are not meaningful**. For another scale, run `make-colourmap` on a `-studiable` export.
2. **Colour.** sRGB is converted to CIELAB (D65) and compared as ΔE\*ab (CIE76). `mean_dL` shows a global brightness change, for example from lighting.
3. **Alignment.** Translation only, since samples sit in a fixture. Phase correlation runs on 4× downsampled images and is then refined at full resolution. It is repeated using only pixels that did not change much and are not masked as encrustation, so cleaned areas do not pull the alignment. `align_confidence` is the peak-to-sidelobe ratio of the correlation. Unrelated surfaces give 7–8, and below 20 the sample is marked "low confidence". Alignment matters: shifting an identical surface by only 3 pixels (4 µm) already makes 3.5 % of the pixels exceed 2 µm.
4. **Offset and tilt.** Each scan is levelled on its own. A plane (offset + tilt), fitted on the glaze pixels, is therefore subtracted from the height difference. **Consequence:** a uniform loss of material over the whole field cannot be detected, only local changes. This can be switched off with `--no-plane-correction`.
5. **Percentages.** A pixel counts as changed when |after − before| ≥ threshold. Height changes are split into *lowered* (material lost) and *raised* (material added). Percentages are calculated from full-resolution histograms (bin 0.05) of the overlap area, per region: `all`, `glaze` (outside masks) and `encrustation` (inside masks).

## Validation

**Alignment robustness** (`scripts/robustness.py`, sample M35): the scan was shifted by (150, −230) px and a growing fraction of it was replaced by a changed surface (tiles of 500 px lowered by 8 µm).

| Surface altered | Recovered shift | Correct | Confidence |
|---|---|---|---|
| 0 % | (150, −230) | yes | 956 |
| 10 % | (150, −230) | yes | 729 |
| 30 % | (150, −230) | yes | 556 |
| 50 % | (150, −230) | yes | 432 |
| 70 % | (150, −230) | yes | 197 |
| 85 % | (150, −230) | yes | 170 |

**Noise floor** (`scripts/noise_floor.py`): four samples were measured twice without treatment (normal scan and high-res rescan). The percentages below are what the method reports when **nothing changed**. Thresholds should be chosen well above them.

| Sample | Shift x / y (µm) | % > 1 µm | % > 2 µm | % > 5 µm | % > 10 µm |
|---|---|---|---|---|---|
| M35 | 0.0 / 2.7 | 9.12 | 0.23 | 0.012 | 0.006 |
| M45 | 2.7 / 0.0 | 2.92 | 0.14 | 0.013 | 0.005 |
| M53 | −1.35 / 0.0 | 14.94 | 1.52 | 0.019 | 0.006 |
| M105 | 2.7 / −2.7 | 14.42 | 0.14 | 0.014 | 0.004 |

At 2 µm the noise floor is 0.1–1.5 %; at 5 µm it is 0.01–0.02 %.

## Limitations

- Translation-only alignment; rotation is assumed negligible.
- Heights are reconstructed from exported colours (≈ 0.9 µm steps), not read from the native measurement file.
- Uniform material loss over the whole field is removed by the plane correction (see Method 4).
- Encrustation masks are drawn by hand.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use small crops of real measurements in `tests/data/`.

## How to cite

If you use this software, please cite it. GitHub shows the citation under "Cite this repository", generated from `CITATION.cff`. Please also cite the related paper (reference added on publication).

## Licence

[PolyForm Noncommercial License 1.0.0](LICENSE). Research, teaching and other non-commercial use is allowed; keep the copyright notice. Commercial use requires permission from the author. This is a source-available licence, not an OSI-approved open-source licence.
