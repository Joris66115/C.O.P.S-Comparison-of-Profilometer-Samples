# profilometer-comparison: design

Date: 2026-09-27
Status: draft, awaiting review

## 1. Purpose

Compare surface measurements of the same samples before and after a treatment (here: cleaning of glazed ceramic tiles), exported from MarSurf MfM / MountainsMap. The tool must:

1. Show the before and after image of one sample side by side, plus a difference map.
2. Let the user browse all samples quickly with the keyboard and flag samples of interest.
3. Quantify the difference as the percentage of pixels that changed more than a threshold.
4. Produce a reproducible table of these percentages for all samples, suitable for publication.

The tool will be published on GitHub for reproducibility and open science. Nothing in the code is specific to one dataset; dataset-specific values are settings with documented defaults.

## 2. Inputs

The user selects a **before** folder and an **after** folder. Files are paired by **identical filename**. The sample ID shown in the interface is the part of the filename before the image-type suffix (e.g. `M17`, `M35-high-res`).

Two image types are quantified:

| Type | Example file | Pixel meaning | Difference measure |
|---|---|---|---|
| `pseudo-colour` | `M1-pseudo-colour-image.jpg` | one measured point (default 1.35 µm) | height difference in µm |
| `true-colour` | `M1-true-colour.png` | optical colour | colour difference ΔE\*ab (CIELAB, D65) |

Every other image type (3D views, profile curves, histograms, Abbott curves, parameter tables, `-studiable` pseudo-colour) can be **viewed** side by side, without alignment or percentage.

The type is detected from the filename suffix and can be overridden.

**True-colour caveat:** the current true-colour exports are plots with axes and a title. The tool detects the plot frame, crops to it, and derives the pixel size from the frame width and the scan length (default 10674 µm, configurable). A full-resolution, image-only export from MountainsMap, like the pseudo-colour one, is preferred and will be documented as the recommended input.

## 3. Method

### 3.1 Pseudo-colour to height

- The colour scale (black → blue → cyan → green → yellow → red → white over 0–350 µm) is read once from the colour bar of a `-studiable` image and stored in the repo as a lookup table (`colourmaps/marsurf_default.csv`: RGB, height). A command regenerates it from any studiable image, so other colour scales and ranges are supported.
- Each pixel is mapped to the height of the nearest lookup colour (nearest neighbour in RGB).
- Pixels whose distance to the nearest lookup colour exceeds a tolerance (e.g. the NM "not measured" colour, JPG artefacts at edges) are **excluded** and counted as `excluded_pct`.
- Pixels at the ends of the scale (0 or 350 µm) are counted as `saturated_pct`, because their true height is unknown.
- Expected height accuracy is about 1–2 µm due to 8-bit colour and JPG compression. The README states this.

### 3.2 True-colour to CIELAB

sRGB → linear RGB → XYZ (D65) → CIELAB, implemented in numpy. The difference is ΔE\*ab (CIE76). The summary also reports the mean ΔL\*, so a global brightness change between sessions (e.g. lighting) is visible.

### 3.3 Alignment

- Only translation (x, y). Rotation is assumed negligible because samples sit in a fixture.
- **Phase correlation** (FFT, numpy): coarse on a 4× downsampled image, then refined at full resolution in a small window around the coarse result. Integer-pixel precision.
- Output: shift in pixels and µm, overlap area (%), and a confidence score (peak-to-sidelobe ratio of the correlation peak).
- All differences are calculated on the overlap area only.

### 3.4 Height offset and tilt correction (pseudo-colour only)

Before and after scans are levelled and form-removed independently, so their absolute height reference differs. Before thresholding, the tool fits and subtracts a **plane** (offset + tilt) from the height difference over the overlap area. The fitted offset and tilt are reported. This can be switched off (`--no-plane-correction`). Consequence, stated in the README: a uniform removal of material over the whole field cannot be detected, only local changes.

### 3.5 Difference and percentage

- Signed difference `d = after − before` (µm, after correction) or ΔE (true-colour, unsigned).
- A pixel counts as different when `|d| > threshold` (or ΔE > threshold).
- Reported: `% different` and, for height, split into `% lower` (d < −t, material removed) and `% higher` (d > +t).
- Percentages are calculated at **full resolution** from a stored histogram of `d` (0.05 µm / 0.05 ΔE bins), so the viewer slider gives exact full-resolution percentages instantly without keeping full-resolution arrays in memory.

## 4. Components

```
profilometer-comparison/
├── profilometer_comparison/
│   ├── __main__.py      CLI: `prepare`, `view`, `make-colourmap`
│   ├── pairing.py       match files by name; report unmatched
│   ├── colourmap.py     lookup table load/build; RGB → height
│   ├── colour.py        sRGB → CIELAB, ΔE
│   ├── crop.py          detect and crop the plot frame (true-colour)
│   ├── align.py         phase correlation → shift, overlap, confidence
│   ├── difference.py    plane correction, signed difference, histogram, % at threshold
│   ├── prepare.py       batch pipeline → cache + summary.csv
│   └── viewer.py        tkinter app
├── colourmaps/marsurf_default.csv
├── tests/
├── README.md, LICENSE, CITATION.cff, requirements.txt, pyproject.toml
└── .github/workflows/tests.yml
```

Each module except `prepare.py` and `viewer.py` works on arrays or paths only and has no knowledge of the GUI or cache layout.

Dependencies: Python ≥ 3.10, numpy, Pillow. tkinter from the standard library. pytest for tests.

## 5. Stage 1: `prepare`

```
python -m profilometer_comparison prepare [--before DIR] [--after DIR] [--type pseudo-colour|true-colour] [--out DIR]
```

Folders not given on the command line are asked for with folder pickers. Default output: a new folder `comparison-<type>-<date>` next to the after folder.

For each pair: load → convert (height or Lab) → align → difference → store. Progress is shown. Pairs already in the cache are skipped, so an interrupted run can resume.

Cache layout:

```
comparison-pseudo-colour-2026-10-15/
├── manifest.json        tool version, settings, input folders, date
├── summary.csv          one row per sample (below)
├── prepare-log.txt      unmatched files, unreadable files, warnings
├── flagged.csv          written by the viewer
└── pairs/M17/
    ├── before.png       aligned overlap, downsampled for display (max 1600 px)
    ├── after.png
    ├── diff.npy         signed difference, same downsampled size (float16)
    ├── hist.npy         full-resolution histogram of the difference
    └── meta.json        shift, confidence, overlap, offsets, excluded/saturated %
```

`summary.csv` columns: `sample, file, type, shift_x_um, shift_y_um, overlap_pct, align_confidence, align_warning, plane_offset_um, plane_tilt_x, plane_tilt_y, mean_dL, excluded_pct, saturated_pct`, followed by `pct_diff_<t>` (and for height `pct_lower_<t>`, `pct_higher_<t>`) at standard thresholds: 1, 2, 5, 10 µm for height; 2, 5, 10 ΔE for colour. Thresholds are configurable (`--thresholds`).

## 6. Stage 2: `view`

```
python -m profilometer_comparison view [CACHE_DIR]
```

Three panels: **before**, **after (aligned)**, **difference map**. For height, the difference map uses a diverging colour scale (blue = lower, red = higher), with pixels under the threshold shown grey. For colour, pixels over the ΔE threshold are highlighted on a greyscale of the after image.

Status bar: sample ID, position (17/124), flag state, shift in µm, % different (and % lower / % higher), alignment warning if any.

| Input | Action |
|---|---|
| ← / → | previous / next sample |
| s | flag / unflag current sample |
| ↑ / ↓ or slider | change threshold (step 0.5 µm or 0.5 ΔE); % updates live |
| f | show flagged samples only (toggle) |
| click on a panel | zoom window: the same region from the full-resolution originals, re-aligned, in all three panels |

`flagged.csv` is rewritten on every flag change: `sample, file, type, pct_diff, threshold, shift_x_um, shift_y_um, flagged_at`. On opening, existing flags are loaded and the viewer resumes at the last viewed sample.

**Plain view mode:** for image types without quantification, `view --before DIR --after DIR` shows the two images side by side with the same browsing and flagging, without a cache and without difference panel.

## 7. Error handling

- **Alignment uncertain:** confidence below a threshold, or shift > 1 mm (configurable) → `align_warning` set in the summary and shown in the viewer. The values are still reported, never silently dropped.
- **Overlap < 80 %:** warning.
- **Unmatched or unreadable files:** listed in `prepare-log.txt` and at the end of the run. They never abort the run.
- **Excluded / saturated pixels:** reported per sample; excluded pixels are left out of the percentage denominator.
- **Settings mismatch:** resuming a cache with different settings than in `manifest.json` stops with an explanation instead of mixing results.

## 8. Testing

Automated (pytest, run on GitHub Actions for every push):

- `align`: a real image shifted by a known amount → exact shift recovered; low-texture input → low confidence.
- `colourmap`: colours sampled from the colour bar → correct heights within 1 step; NM colour → excluded.
- `colour`: known sRGB → Lab reference values; identical colours → ΔE 0.
- `difference`: identical images → 0 %; synthetic plane tilt → 0 % after plane correction; a patch lowered by 5 µm → exactly the expected % lower.
- `pairing`: matched, before-only and after-only files.
- `crop`: frame detection on a real true-colour export.

A small set of downsampled real test images is stored in `tests/data/` (a few hundred kB).

**Validation on real data (documented in README):** the four high-res rescans (M35, M45, M53, M105) against their normal scans measure the same untreated surface twice. Their percentages give the **noise floor** of the method and support the choice of threshold for publication.

## 9. Publication

- `README.md`: installation, usage, method (sections 3.1–3.5 in plain language), limitations, how to cite.
- `LICENSE`: MIT (to be confirmed by the author).
- `CITATION.cff` with author details; optionally a Zenodo DOI per release.
- Every CSV output contains the tool version (in `manifest.json`), so results can be linked to the exact code.
- No measurement data in the repo except the small test images.

## 10. Out of scope

- Rotation or non-rigid alignment.
- Reading native MountainsMap files (`.mnt`, `.sur`) or raw height data.
- Comparing the ISO 25178 parameter values from the reports (a possible separate addition).
- Automatic classification of change types.
