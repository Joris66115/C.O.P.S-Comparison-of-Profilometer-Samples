# profilometer-comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python tool that aligns before/after profilometer exports, quantifies height (µm) or colour (ΔE) change per region (glaze / encrustation), and lets the user browse, blink, mask and flag samples with the keyboard.

**Architecture:** Two stages. `prepare` batch-processes every before/after pair (colour→height or colour→CIELAB, two-pass phase-correlation alignment, plane correction, per-region full-resolution histograms) into a cache folder with a `summary.csv`. `view` is a tkinter app that reads only the cache (instant browsing) and calls back into `prepare` to recompute a single pair when the user nudges alignment or edits a mask. All numerical modules are pure functions on numpy arrays, independently tested.

**Tech Stack:** Python ≥ 3.10, numpy, Pillow, tkinter (standard library), pytest, GitHub Actions.

**Spec:** `docs/specs/2026-09-27-profilometer-comparison-design.md`

## Global Constraints

- Python ≥ 3.10; runtime dependencies **only** numpy and Pillow; tkinter from the standard library; pytest for tests.
- Package name `profilometer_comparison`; repo/distribution name `profilometer-comparison`; version `0.1.0` defined once in `profilometer_comparison/__init__.py`.
- No module except `viewer.py`, `plain_viewer.py` and `__main__.py` may import tkinter (tests run headless on CI).
- Licence: PolyForm Noncommercial License 1.0.0, unmodified text, with `Required Notice: Copyright (c) 2026 Joris Nagtegaal`.
- No measurement data in the repo except `tests/data/` (small crops, < 1 MB total).
- Coordinate conventions (used everywhere):
  - Arrays are indexed `[row, col]` = `[y, x]`; y points **down**; origin is the top-left of the **before** image.
  - A shift `(dy, dx)` means: a feature at `before[y, x]` appears at `after[y + dy, x + dx]`.
  - Mask polygons are lists of `(x_um, y_um)` in the before image's coordinates (µm = pixel × pixel size).
- Defaults (copied from the spec): pixel size 1.35 µm (pseudo-colour); true-colour scan length 10674 µm; thresholds 1, 2, 5, 10 µm (height) and 2, 5, 10 ΔE (colour); max shift 1000 µm; min overlap 80 %; display max 1600 px; histogram bin 0.05; threshold step 0.5.
- Default minimum alignment confidence: **20** (peak-to-sidelobe ratio of the coarse correlation). Measured on the real data during planning: unrelated samples 7–8; same surface 1230; with 85 % of the surface altered still 128.
- Facts about the real exports, measured during planning (MOPA before-measurements):
  - Pseudo-colour full-resolution JPGs are 7920–7918 × 7928–7930 px — **before and after can differ by a pixel or two**; every function must handle unequal shapes.
  - Colour bar in `*-pseudo-colour-studiable.png` (9212 × 5904): outer black frame x 8344–8556, y 594–5550, border 13 px; interior rows 607–5537; 393 unique colours; white = 350 µm (top), black = 0 µm (bottom). JPG pixels are ≤ 3.2 RGB units from the nearest bar colour (99.9th percentile).
  - True-colour PNGs (4488 × 5196): plot frame lines 13 px thick at rows 897–909 / 4513–4525 and cols 494–506 / 4107–4119; interior 3600 × 3603 px.
  - Noise floor (M35 vs its high-res rescan, same surface): 9.4 % > 1 µm, 0.22 % > 2 µm, 0.01 % > 5 µm.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

Paths used below:
- `REPO` = `/Users/jorisnagtegaal/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/profilometer-comparison`
- `MOPA` = `/Users/jorisnagtegaal/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA`

All commands run from `REPO`.

## File Structure

```
profilometer-comparison/
├── profilometer_comparison/
│   ├── __init__.py          version
│   ├── __main__.py          CLI: prepare / view / make-colourmap
│   ├── images.py            read_rgb (large-image safe)
│   ├── pairing.py           pair files by name, natural sort, sample IDs
│   ├── colourmap.py         colour bar → lookup table; RGB → height
│   ├── colourmaps/marsurf_default.csv
│   ├── colour.py            sRGB → CIELAB, ΔE76
│   ├── crop.py              detect/crop plot frame (true-colour)
│   ├── align.py             phase correlation, overlap, two-window refinement
│   ├── masks.py             polygons (µm) ↔ pixel masks, validation, JSON
│   ├── difference.py        plane fit, histograms, percentages
│   ├── prepare.py           Settings, load/process pair, cache, summary, recompute, zoom
│   ├── flags.py             FlagStore (flagged.csv)
│   ├── viewer_state.py      non-GUI viewer logic + render_difference
│   ├── viewer.py            tkinter comparison viewer
│   └── plain_viewer.py      tkinter side-by-side viewer for any image type
├── scripts/
│   ├── robustness.py        alignment robustness on full-resolution real data
│   └── noise_floor.py       same-surface rescans → noise floor
├── tests/
│   ├── conftest.py
│   ├── data/                small real crops + make_test_data.py
│   └── test_*.py
├── docs/specs/…, docs/plans/…
├── README.md, LICENSE, CITATION.cff, pyproject.toml, requirements.txt, .gitignore
└── .github/workflows/tests.yml
```

(The spec lists `colourmaps/` at the top level; it lives inside the package so it is installed with it. `flags.py`, `viewer_state.py`, `plain_viewer.py` and `images.py` split the spec's viewer/IO responsibilities so the logic is testable without a display.)

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `LICENSE`, `CITATION.cff`, `.github/workflows/tests.yml`
- Create: `profilometer_comparison/__init__.py`, `profilometer_comparison/images.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: `profilometer_comparison.__version__ == "0.1.0"`; `images.read_rgb(path) -> np.ndarray (H, W, 3) uint8`.

- [ ] **Step 1: Create the virtual environment**

```bash
python3 -m venv .venv
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "profilometer-comparison"
dynamic = ["version"]
description = "Compare before/after surface profilometry exports: alignment, height and colour difference maps, and a keyboard-driven viewer."
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE"}
authors = [{name = "Joris Nagtegaal"}]
dependencies = ["numpy>=1.24", "Pillow>=10"]

[project.optional-dependencies]
dev = ["pytest>=7"]

[project.scripts]
profilometer-comparison = "profilometer_comparison.__main__:main"

[tool.setuptools]
packages = ["profilometer_comparison"]

[tool.setuptools.dynamic]
version = {attr = "profilometer_comparison.__version__"}

[tool.setuptools.package-data]
profilometer_comparison = ["colourmaps/*.csv"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `requirements.txt` and `.gitignore`**

`requirements.txt`:
```
numpy>=1.24
Pillow>=10
```

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.DS_Store
comparison-*/
```

- [ ] **Step 4: Write the failing test `tests/test_package.py`**

```python
import numpy as np
from PIL import Image

import profilometer_comparison
from profilometer_comparison.images import read_rgb


def test_version():
    assert profilometer_comparison.__version__ == "0.1.0"


def test_read_rgb_converts_to_uint8_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (4, 3), 128).save(path)
    img = read_rgb(path)
    assert img.shape == (3, 4, 3)
    assert img.dtype == np.uint8
    assert img[0, 0].tolist() == [128, 128, 128]
```

- [ ] **Step 5: Write `profilometer_comparison/__init__.py` and `images.py`**

`profilometer_comparison/__init__.py`:
```python
"""Compare before/after surface profilometry exports."""

__version__ = "0.1.0"
```

`profilometer_comparison/images.py`:
```python
"""Image reading that works for very large profilometer exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Full-resolution exports reach ~72 megapixels; Pillow's default guard is not needed here.
Image.MAX_IMAGE_PIXELS = None


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
```

- [ ] **Step 6: Install and run the tests**

Run:
```bash
.venv/bin/pip install -e ".[dev]" && .venv/bin/pytest -v
```
Expected: 2 passed.

- [ ] **Step 7: Add LICENSE (PolyForm Noncommercial 1.0.0, unmodified) with the required notice**

```bash
{ printf 'Required Notice: Copyright (c) 2026 Joris Nagtegaal\n\n'; curl -fsSL https://raw.githubusercontent.com/polyformproject/polyform-licenses/1.0.0/PolyForm-Noncommercial-1.0.0.md; } > LICENSE
head -5 LICENSE
```
Expected: line 1 is the Required Notice, line 3 starts with `# PolyForm Noncommercial License 1.0.0`. If the URL fails, download the text from https://polyformproject.org/licenses/noncommercial/1.0.0 and paste it unmodified below the notice.

- [ ] **Step 8: Add `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "profilometer-comparison"
version: 0.1.0
date-released: 2026-09-27
authors:
  - family-names: Nagtegaal
    given-names: Joris
license-url: "https://polyformproject.org/licenses/noncommercial/1.0.0"
abstract: "Compare before/after surface profilometry exports of cultural heritage objects: alignment, height and colour difference maps per region, and a keyboard-driven viewer for flagging samples."
keywords:
  - profilometry
  - conservation
  - glazed ceramics
  - laser cleaning
  - surface metrology
```

- [ ] **Step 9: Add `.github/workflows/tests.yml`**

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest -v
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore LICENSE CITATION.cff .github profilometer_comparison tests
git commit -m "Scaffold package, licence, citation and CI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Real test fixtures

**Files:**
- Create: `tests/data/make_test_data.py`, `tests/data/README.md`, `tests/conftest.py`
- Create (generated): `tests/data/colourbar.png`, `tests/data/pc_M35.jpg`, `tests/data/pc_M35_highres.jpg`

**Interfaces:**
- Produces pytest fixtures (in `tests/conftest.py`): `data_dir -> Path`, `colourbar_img -> np.ndarray`, `pc_before_rgb -> np.ndarray (768, 768, 3)`, `pc_highres_rgb -> np.ndarray (768, 768, 3)`.

- [ ] **Step 1: Write `tests/data/make_test_data.py`**

```python
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
```

- [ ] **Step 2: Run it and check sizes**

```bash
.venv/bin/python tests/data/make_test_data.py "/Users/jorisnagtegaal/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA"
du -ch tests/data/*.png tests/data/*.jpg
```
Expected: three files, total < 1 MB.

- [ ] **Step 3: Write `tests/data/README.md`**

```markdown
# Test data

Small crops of real MarSurf MfM exports (glazed ceramic tile, sample M35, MOPA series), made with `make_test_data.py`:

- `colourbar.png`: the pseudo-colour colour bar (0–350 µm) of `M1-pseudo-colour-studiable.png`, crop x 8330–8600, y 560–5600.
- `pc_M35.jpg`, `pc_M35_highres.jpg`: the same 768 × 768 px region (x, y 3000–3768) of the full-resolution pseudo-colour image of M35 and of its high-res rescan. Same surface, measured twice (true shift ≈ 2 px in y).
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 5: Run the tests (fixtures load without error)**

Run: `.venv/bin/pytest -v`
Expected: 2 passed (no new tests yet; conftest imports cleanly).

- [ ] **Step 6: Commit**

```bash
git add tests
git commit -m "Add small real-data test fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pairing

**Files:**
- Create: `profilometer_comparison/pairing.py`
- Test: `tests/test_pairing.py`

**Interfaces:**
- Produces:
  - `natural_key(name: str) -> list` — sort key, `M2` < `M10`.
  - `@dataclass(frozen=True) Pair(name: str, before: Path, after: Path)`
  - `@dataclass PairingResult(pairs: list[Pair], before_only: list[str], after_only: list[str])`
  - `pair_folders(before_dir, after_dir) -> PairingResult`
  - `sample_id(filename: str, type_suffix: str | None = None) -> str`

- [ ] **Step 1: Write the failing tests `tests/test_pairing.py`**

```python
from profilometer_comparison.pairing import natural_key, pair_folders, sample_id


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_natural_key_orders_numbers():
    assert sorted(["M10", "M2", "M1", "M35-high-res"], key=natural_key) == ["M1", "M2", "M10", "M35-high-res"]


def test_pair_folders_matches_identical_names(tmp_path):
    for name in ["M1-a.png", "M2-a.png", "M10-a.png"]:
        touch(tmp_path / "before" / name)
    for name in ["M1-a.png", "M10-a.png", "M3-a.png"]:
        touch(tmp_path / "after" / name)
    touch(tmp_path / "before" / ".DS_Store")
    touch(tmp_path / "before" / "rename-log.csv")

    result = pair_folders(tmp_path / "before", tmp_path / "after")

    assert [p.name for p in result.pairs] == ["M1-a.png", "M10-a.png"]
    assert result.pairs[0].before == tmp_path / "before" / "M1-a.png"
    assert result.pairs[0].after == tmp_path / "after" / "M1-a.png"
    assert result.before_only == ["M2-a.png"]
    assert result.after_only == ["M3-a.png"]


def test_sample_id():
    assert sample_id("M17-pseudo-colour-image.jpg", "pseudo-colour-image") == "M17"
    assert sample_id("M35-high-res-true-colour.png", "true-colour") == "M35-high-res"
    assert sample_id("M17-3D-view.png", "true-colour") == "M17-3D-view"
    assert sample_id("M17-3D-view.png") == "M17-3D-view"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_pairing.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'profilometer_comparison.pairing'`.

- [ ] **Step 3: Implement `profilometer_comparison/pairing.py`**

```python
"""Match before/after files by identical filename."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Pair:
    name: str
    before: Path
    after: Path


@dataclass
class PairingResult:
    pairs: list[Pair] = field(default_factory=list)
    before_only: list[str] = field(default_factory=list)
    after_only: list[str] = field(default_factory=list)


def natural_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically (M2 before M10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_images(folder: str | Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in Path(folder).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def pair_folders(before_dir: str | Path, after_dir: str | Path) -> PairingResult:
    before = list_images(before_dir)
    after = list_images(after_dir)
    common = sorted(before.keys() & after.keys(), key=natural_key)
    return PairingResult(
        pairs=[Pair(name, before[name], after[name]) for name in common],
        before_only=sorted(before.keys() - after.keys(), key=natural_key),
        after_only=sorted(after.keys() - before.keys(), key=natural_key),
    )


def sample_id(filename: str, type_suffix: str | None = None) -> str:
    """'M17-pseudo-colour-image.jpg' -> 'M17' (strip the image-type suffix if present)."""
    stem = Path(filename).stem
    if type_suffix and stem.endswith("-" + type_suffix):
        return stem[: -len(type_suffix) - 1]
    return stem
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_pairing.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add profilometer_comparison/pairing.py tests/test_pairing.py
git commit -m "Add before/after file pairing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Colour map (pseudo-colour → height)

**Files:**
- Create: `profilometer_comparison/colourmap.py`
- Create (generated): `profilometer_comparison/colourmaps/marsurf_default.csv`
- Test: `tests/test_colourmap.py`

**Interfaces:**
- Consumes: `images.read_rgb`.
- Produces:
  - `@dataclass Colourmap(rgb: np.ndarray (N,3) uint8, height: np.ndarray (N,) float64 ascending)`
  - `@dataclass HeightMap(height: float32 (NaN = excluded), excluded: bool, saturated: bool)`
  - `find_colour_bar(img, search_from_frac=0.75) -> tuple[int, int, int]` (top, bottom inclusive interior rows; sample column)
  - `build_from_studiable(img, z_min=0.0, z_max=350.0, search_from_frac=0.75) -> Colourmap`
  - `save_csv(cmap, path)`, `load_csv(path) -> Colourmap`, `default_colourmap() -> Colourmap`
  - `rgb_to_height(img, cmap, tolerance=25.0) -> HeightMap`

- [ ] **Step 1: Write the failing tests `tests/test_colourmap.py`**

```python
import numpy as np
import pytest

from profilometer_comparison.colourmap import (
    build_from_studiable, default_colourmap, find_colour_bar, load_csv, rgb_to_height, save_csv,
)


def test_find_colour_bar_on_real_crop(colourbar_img):
    # Crop origin is (x 8330, y 560) of the full image; interior rows 607-5537, centre column 8450.
    assert find_colour_bar(colourbar_img, search_from_frac=0.0) == (47, 4977, 120)


def test_build_from_studiable(colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    assert 350 <= len(cmap.height) <= 450
    assert np.all(np.diff(cmap.height) >= 0)
    assert cmap.rgb[0].tolist() == [0, 0, 0]
    assert cmap.rgb[-1].tolist() == [255, 255, 255]
    assert cmap.height[0] < 5 and cmap.height[-1] > 345


def test_csv_roundtrip(tmp_path, colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    save_csv(cmap, tmp_path / "cm.csv")
    back = load_csv(tmp_path / "cm.csv")
    np.testing.assert_array_equal(back.rgb, cmap.rgb)
    np.testing.assert_allclose(back.height, cmap.height, atol=1e-4)


def test_rgb_to_height_maps_bar_colours_back(colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    top, bottom, col = find_colour_bar(colourbar_img, search_from_frac=0.0)
    column = colourbar_img[top:bottom + 1, col][None, :, :]
    hm = rgb_to_height(column, cmap)
    expected = 350 - (np.arange(top, bottom + 1) - top) / (bottom - top) * 350
    inner = (expected > 10) & (expected < 340)  # the ends are saturated black/white bands
    assert np.abs(hm.height[0] - expected)[inner].max() < 1.5
    assert not hm.excluded.any()


def test_off_scale_colour_is_excluded_and_ends_are_saturated(colourbar_img):
    cmap = build_from_studiable(colourbar_img, search_from_frac=0.0)
    img = np.array([[[128, 128, 128], [0, 0, 0], [255, 255, 255]]], np.uint8)
    hm = rgb_to_height(img, cmap)
    assert hm.excluded.tolist() == [[True, False, False]]
    assert np.isnan(hm.height[0, 0])
    assert hm.saturated.tolist() == [[False, True, True]]


def test_default_colourmap_is_packaged():
    cmap = default_colourmap()
    assert len(cmap.height) > 350
    assert cmap.height[-1] > 345
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_colourmap.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/colourmap.py`**

```python
"""Convert MountainsMap pseudo-colour images back to heights.

The colour scale is read once from the colour bar of a '-studiable' export and
stored as a lookup table (RGB -> height in µm). Each image pixel gets the height
of the nearest lookup colour.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np


@dataclass
class Colourmap:
    rgb: np.ndarray     # (N, 3) uint8, ordered by height
    height: np.ndarray  # (N,) float64, ascending, µm


@dataclass
class HeightMap:
    height: np.ndarray     # float32, NaN where excluded
    excluded: np.ndarray   # bool: colour too far from the scale (e.g. not-measured points)
    saturated: np.ndarray  # bool: lowest or highest colour of the scale (true height unknown)


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    return (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)


def _unpack(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    return np.stack([(codes >> 16) & 255, (codes >> 8) & 255, codes & 255], axis=-1).astype(np.uint8)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each contiguous True run in a 1-D boolean array."""
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def find_colour_bar(img: np.ndarray, search_from_frac: float = 0.75) -> tuple[int, int, int]:
    """Locate the colour bar interior in a pseudo-colour export.

    Returns (top, bottom, column): interior rows top..bottom (inclusive) and a
    column through the middle of the bar. The bar is the first block of mostly
    non-white columns right of `search_from_frac` of the image width, drawn with
    a black frame.
    """
    s = img.astype(np.int32).sum(axis=2)
    nonwhite = s < 700
    black = s < 30
    x_start = int(img.shape[1] * search_from_frac)
    col_runs = _runs(nonwhite[:, x_start:].mean(axis=0) > 0.5)
    if not col_runs:
        raise ValueError("no colour bar found")
    x0, x1 = col_runs[0][0] + x_start, col_runs[0][1] + x_start
    # The left frame line gives the outer top and bottom of the bar.
    y0, y1 = max(_runs(black[:, x0 + 1]), key=lambda r: r[1] - r[0])
    ym = (y0 + y1) // 2
    thickness = _runs(black[ym, x0:x1])[0][1]
    right = x0 + thickness + _runs(black[ym, x0 + thickness:x1])[0][0]
    return y0 + thickness, y1 - 1 - thickness, (x0 + thickness + right) // 2


def build_from_studiable(img: np.ndarray, z_min: float = 0.0, z_max: float = 350.0,
                         search_from_frac: float = 0.75) -> Colourmap:
    top, bottom, col = find_colour_bar(img, search_from_frac)
    rows = np.arange(top, bottom + 1)
    heights = z_max - (rows - top) / (bottom - top) * (z_max - z_min)
    codes = _pack(img[top:bottom + 1, col])
    uniq, inverse = np.unique(codes, return_inverse=True)
    mean_height = np.bincount(inverse, weights=heights) / np.bincount(inverse)
    order = np.argsort(mean_height)
    return Colourmap(rgb=_unpack(uniq[order]), height=mean_height[order])


def save_csv(cmap: Colourmap, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["r", "g", "b", "height_um"])
        for (r, g, b), h in zip(cmap.rgb.tolist(), cmap.height.tolist()):
            writer.writerow([r, g, b, f"{h:.4f}"])


def _read_csv(fh) -> Colourmap:
    rows = list(csv.DictReader(fh))
    rgb = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows], np.uint8)
    height = np.array([float(r["height_um"]) for r in rows])
    return Colourmap(rgb=rgb, height=height)


def load_csv(path: str | Path) -> Colourmap:
    with open(path, newline="", encoding="utf-8") as fh:
        return _read_csv(fh)


def default_colourmap() -> Colourmap:
    """The MarSurf 0-350 µm scale shipped with the package."""
    ref = resources.files("profilometer_comparison") / "colourmaps" / "marsurf_default.csv"
    with ref.open("r", encoding="utf-8", newline="") as fh:
        return _read_csv(fh)


def rgb_to_height(img: np.ndarray, cmap: Colourmap, tolerance: float = 25.0) -> HeightMap:
    """Map each pixel to the height of the nearest scale colour (Euclidean RGB).

    Pixels further than `tolerance` from every scale colour are excluded (NaN).
    Works on the distinct colours of the image only, so it is fast for large images.
    """
    codes = _pack(img)
    used = np.flatnonzero(np.bincount(codes.ravel(), minlength=1 << 24))
    used_rgb = _unpack(used).astype(np.int32)
    lut = cmap.rgb.astype(np.int32)
    nearest = np.empty(len(used), np.int64)
    distance = np.empty(len(used))
    for start in range(0, len(used), 4096):
        d2 = ((used_rgb[start:start + 4096, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        nearest[start:start + 4096] = d2.argmin(axis=1)
        distance[start:start + 4096] = np.sqrt(d2.min(axis=1))

    ok = distance <= tolerance
    height_table = np.full(1 << 24, np.nan, np.float32)
    height_table[used] = np.where(ok, cmap.height[nearest], np.nan)
    saturated_table = np.zeros(1 << 24, bool)
    saturated_table[used] = ok & ((nearest == 0) | (nearest == len(lut) - 1))

    height = height_table[codes]
    return HeightMap(height=height, excluded=np.isnan(height), saturated=saturated_table[codes])
```

- [ ] **Step 4: Generate the packaged default colour map from the test crop**

```bash
mkdir -p profilometer_comparison/colourmaps
.venv/bin/python -c "
from profilometer_comparison.colourmap import build_from_studiable, save_csv
from profilometer_comparison.images import read_rgb
cmap = build_from_studiable(read_rgb('tests/data/colourbar.png'), search_from_frac=0.0)
save_csv(cmap, 'profilometer_comparison/colourmaps/marsurf_default.csv')
print(len(cmap.height), cmap.height.min(), cmap.height.max())
"
```
Expected: about `393 3.x 349.x`.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pip install -e ".[dev]" -q && .venv/bin/pytest tests/test_colourmap.py -v`
Expected: 6 passed. (Reinstall makes the package-data CSV visible to `importlib.resources`.)

- [ ] **Step 6: Sanity check on a full real image**

```bash
.venv/bin/python -c "
import time
from profilometer_comparison.colourmap import default_colourmap, rgb_to_height
from profilometer_comparison.images import read_rgb
t = time.time()
hm = rgb_to_height(read_rgb('$HOME/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA/pseudo-colour-view/image-only/M1-pseudo-colour-image.jpg'), default_colourmap())
print(hm.height.shape, round(time.time() - t, 1), 's; excluded %', 100 * hm.excluded.mean(), '; saturated %', 100 * hm.saturated.mean())
"
```
Expected: `(7928, 7920)`, under ~3 s, excluded ≈ 0 %, saturated < 0.01 %.

- [ ] **Step 7: Commit**

```bash
git add profilometer_comparison/colourmap.py profilometer_comparison/colourmaps tests/test_colourmap.py
git commit -m "Add pseudo-colour to height conversion with MarSurf default scale

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Colour science (sRGB → CIELAB, ΔE)

**Files:**
- Create: `profilometer_comparison/colour.py`
- Test: `tests/test_colour.py`

**Interfaces:**
- Produces: `srgb_to_lab(img uint8 (H,W,3)) -> float32 (H,W,3)`; `delta_e76(lab1, lab2) -> float32 (H,W)`.

- [ ] **Step 1: Write the failing tests `tests/test_colour.py`**

```python
import numpy as np

from profilometer_comparison.colour import delta_e76, srgb_to_lab


def test_reference_colours():
    img = np.array([[[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 255]]], np.uint8)
    lab = srgb_to_lab(img)[0]
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)
    np.testing.assert_allclose(lab[3], [32.30, 79.19, -107.86], atol=0.05)


def test_delta_e76():
    a = np.array([[[100, 0, 0], [50, 10, 10]]], np.float32)
    b = np.array([[[50, 0, 0], [50, 10, 10]]], np.float32)
    np.testing.assert_allclose(delta_e76(a, b), [[50, 0]], atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_colour.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/colour.py`**

```python
"""sRGB -> CIELAB (D65) and CIE76 colour difference."""
from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], np.float32)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], np.float32)
_EPS = (6 / 29) ** 3


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    c = img.astype(np.float32) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.sqrt(((lab1 - lab2) ** 2).sum(axis=-1)).astype(np.float32)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_colour.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add profilometer_comparison/colour.py tests/test_colour.py
git commit -m "Add sRGB to CIELAB conversion and delta E

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Plot-frame cropping (true-colour)

**Files:**
- Create: `profilometer_comparison/crop.py`
- Test: `tests/test_crop.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) Frame(top, bottom, left, right)` (interior; bottom/right exclusive); `find_plot_frame(img, dark=60, line_frac=0.4) -> Frame` (raises `ValueError` if none); `crop_to_frame(img, frame) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests `tests/test_crop.py`**

```python
import numpy as np
import pytest

from profilometer_comparison.crop import Frame, crop_to_frame, find_plot_frame


def framed_image():
    img = np.full((400, 500, 3), 255, np.uint8)
    img[50:350, 60:460] = 0                 # outer edge of a 3 px black frame
    img[53:347, 63:457] = (120, 110, 100)   # interior (dark-ish tile colour, but not black)
    img[10:20, 100:300] = 0                 # a title line: too short to count as frame
    return img


def test_find_plot_frame():
    assert find_plot_frame(framed_image()) == Frame(top=53, bottom=347, left=63, right=457)


def test_crop_to_frame():
    out = crop_to_frame(framed_image(), Frame(53, 347, 63, 457))
    assert out.shape == (294, 394, 3)
    assert (out == (120, 110, 100)).all()


def test_no_frame_raises():
    with pytest.raises(ValueError):
        find_plot_frame(np.full((50, 50, 3), 200, np.uint8))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_crop.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/crop.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_crop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Check on the real true-colour export**

```bash
.venv/bin/python -c "
from profilometer_comparison.crop import find_plot_frame
from profilometer_comparison.images import read_rgb
print(find_plot_frame(read_rgb('$HOME/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA/true-colour/M1-true-colour.png')))
"
```
Expected: `Frame(top=910, bottom=4513, left=507, right=4107)`.

- [ ] **Step 6: Commit**

```bash
git add profilometer_comparison/crop.py tests/test_crop.py
git commit -m "Add plot-frame detection for true-colour exports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Alignment

**Files:**
- Create: `profilometer_comparison/align.py`
- Test: `tests/test_align.py`

**Interfaces:**
- Consumes: `colourmap.default_colourmap`, `colourmap.rgb_to_height` (tests only).
- Produces:
  - `@dataclass(frozen=True) Alignment(dy: int, dx: int, confidence: float, method: str = "auto")`
  - `downsample(a, f) -> float32` (NaN-aware block mean; drops the remainder rows/cols)
  - `phase_correlation(a, b, exclude=5) -> tuple[int, int, float]` — (dy, dx, peak-to-sidelobe ratio); crops both to their common shape
  - `overlap_slices(shape_before, shape_after, dy, dx) -> tuple[tuple[slice, slice], tuple[slice, slice]]` — raises `ValueError` if no overlap
  - `estimate_shift(before, after, stable=None, factor=4, window=2048) -> Alignment` — `stable` is a bool array in before coordinates; unstable pixels are ignored

- [ ] **Step 1: Write the failing tests `tests/test_align.py`**

```python
import numpy as np
import pytest

from profilometer_comparison.align import estimate_shift, overlap_slices, phase_correlation
from profilometer_comparison.colourmap import default_colourmap, rgb_to_height


def texture(shape, seed=0, sigma=2.0):
    """Smooth random surface (Gaussian-filtered noise, sigma in px)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=shape))
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    return np.fft.ifft2(f * np.exp(-2 * (np.pi * sigma) ** 2 * (kx ** 2 + ky ** 2))).real.astype(np.float32)


def shifted_pair(big, margin, size, dy, dx):
    """before = big[m:m+size]; after such that after[y+dy, x+dx] == before[y, x]."""
    before = big[margin:margin + size, margin:margin + size]
    after = big[margin - dy:margin - dy + size, margin - dx:margin - dx + size].copy()
    return before, after


def test_phase_correlation_recovers_shift():
    before, after = shifted_pair(texture((300, 300)), 50, 200, 7, -12)
    dy, dx, psr = phase_correlation(before, after)
    assert (dy, dx) == (7, -12)
    assert psr > 20


def test_phase_correlation_handles_unequal_shapes():
    before, after = shifted_pair(texture((300, 300)), 50, 200, 3, 4)
    dy, dx, _ = phase_correlation(before, after[:-2, :-1])
    assert (dy, dx) == (3, 4)


def test_overlap_slices():
    bs, as_ = overlap_slices((100, 100), (100, 102), 5, -3)
    assert bs == (slice(0, 95), slice(3, 100))
    assert as_ == (slice(5, 100), slice(0, 97))
    with pytest.raises(ValueError):
        overlap_slices((10, 10), (10, 10), 20, 0)


def test_estimate_shift_large_shift():
    before, after = shifted_pair(texture((1400, 1400), seed=1), 250, 900, 150, -230)
    al = estimate_shift(before, after, window=512)
    assert (al.dy, al.dx) == (150, -230)
    assert al.confidence > 20
    assert al.method == "auto"


def test_unrelated_images_have_low_confidence():
    al = estimate_shift(texture((800, 800), seed=2), texture((800, 800), seed=3), window=256)
    assert al.confidence < 15


@pytest.fixture(scope="module")
def real_heights(pc_before_rgb, pc_highres_rgb):
    cmap = default_colourmap()
    return rgb_to_height(pc_before_rgb, cmap).height, rgb_to_height(pc_highres_rgb, cmap).height


def test_real_rescan_alignment(real_heights):
    before, after = real_heights
    al = estimate_shift(before, after, window=512)
    assert abs(al.dy - 2) <= 1 and abs(al.dx) <= 1
    assert al.confidence > 20


def alter_tiles(after, fraction, seed=0, tile=64):
    """Replace `fraction` of the tiles by a smooth patch 8 µm lower (simulated change). Returns the altered mask."""
    rng = np.random.default_rng(seed)
    ny, nx = after.shape[0] // tile, after.shape[1] // tile
    chosen = rng.choice(ny * nx, int(round(fraction * ny * nx)), replace=False)
    altered = np.zeros(after.shape, bool)
    level = np.nanmean(after)
    for k in chosen:
        i, j = divmod(int(k), nx)
        block = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
        after[block] = level - 8 + rng.normal(0, 1, (tile, tile))
        altered[block] = True
    return altered


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5])
def test_alignment_survives_altered_surface(real_heights, fraction):
    before, after = shifted_pair(real_heights[0], 64, 640, 23, -41)
    alter_tiles(after, fraction)
    al = estimate_shift(before, after, window=512)
    assert (al.dy, al.dx) == (23, -41)


def test_stable_mask_rescues_heavily_altered_surface(real_heights):
    dy, dx = 23, -41
    before, after = shifted_pair(real_heights[0], 64, 640, dy, dx)
    altered = alter_tiles(after, 0.7, seed=1)
    # Stable pixels in before coordinates: those whose partner in `after` was not altered.
    bs, as_ = overlap_slices(before.shape, after.shape, dy, dx)
    stable = np.zeros(before.shape, bool)
    stable[bs] = ~altered[as_]
    al = estimate_shift(before, after, stable=stable, window=256)
    assert (al.dy, al.dx) == (dy, dx)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_align.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/align.py`**

```python
"""Translation-only alignment by phase correlation.

Convention: a shift (dy, dx) means a feature at before[y, x] appears at
after[y + dy, x + dx].
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Alignment:
    dy: int
    dx: int
    confidence: float  # peak-to-sidelobe ratio of the coarse correlation (NaN for manual)
    method: str = "auto"


def downsample(a: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware block mean by an integer factor (remainder rows/cols are dropped)."""
    a = np.asarray(a, np.float32)
    if f == 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    blocks = a[:h, :w].reshape(h // f, f, w // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks stay NaN
        return np.nanmean(blocks, axis=(1, 3))


def phase_correlation(a: np.ndarray, b: np.ndarray, exclude: int = 5) -> tuple[int, int, float]:
    """Shift of b relative to a, and the peak-to-sidelobe ratio (PSR) of the correlation peak."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a = np.asarray(a[:h, :w], np.float64)
    b = np.asarray(b[:h, :w], np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = np.nan_to_num(a - np.nanmean(a))
        b = np.nan_to_num(b - np.nanmean(b))
    window = np.outer(np.hanning(h), np.hanning(w))
    cross = np.fft.rfft2(b * window) * np.conj(np.fft.rfft2(a * window))
    cross /= np.abs(cross) + 1e-12
    r = np.fft.irfft2(cross, s=(h, w))
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    sidelobe = np.ones(r.shape, bool)
    sidelobe[np.ix_([(py + k) % h for k in range(-exclude, exclude + 1)],
                    [(px + k) % w for k in range(-exclude, exclude + 1)])] = False
    psr = float((r[py, px] - r[sidelobe].mean()) / (r[sidelobe].std() + 1e-12))
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return int(dy), int(dx), psr


def overlap_slices(shape_before, shape_after, dy: int, dx: int):
    """Slices (before_slices, after_slices) of the region both images cover."""
    y0, y1 = max(0, -dy), min(shape_before[0], shape_after[0] - dy)
    x0, x1 = max(0, -dx), min(shape_before[1], shape_after[1] - dx)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("images do not overlap at this shift")
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))


def estimate_shift(before: np.ndarray, after: np.ndarray, stable: np.ndarray | None = None,
                   factor: int = 4, window: int = 2048) -> Alignment:
    """Coarse phase correlation on downsampled images, refined at full resolution.

    If `stable` (bool, before coordinates) is given, unstable pixels are ignored and
    the refinement window is the one of 3 x 3 candidates with the most stable pixels.
    """
    b = np.array(before, np.float32)
    if stable is not None:
        b[~stable] = np.nan
    cy, cx, confidence = phase_correlation(downsample(b, factor), downsample(after, factor))
    dy, dx = cy * factor, cx * factor

    bs, _ = overlap_slices(before.shape, after.shape, dy, dx)
    n_y = min(window, bs[0].stop - bs[0].start)
    n_x = min(window, bs[1].stop - bs[1].start)
    candidates = [(int(ty), int(tx))
                  for ty in np.linspace(bs[0].start, bs[0].stop - n_y, 3)
                  for tx in np.linspace(bs[1].start, bs[1].stop - n_x, 3)]
    if stable is None:
        ty, tx = candidates[4]  # centre
    else:
        ty, tx = max(candidates, key=lambda c: stable[c[0]:c[0] + n_y, c[1]:c[1] + n_x].mean())
    fy, fx, _ = phase_correlation(b[ty:ty + n_y, tx:tx + n_x],
                                  after[ty + dy:ty + dy + n_y, tx + dx:tx + dx + n_x])
    if abs(fy) <= 2 * factor and abs(fx) <= 2 * factor:
        dy, dx = dy + fy, dx + fx
    return Alignment(int(dy), int(dx), confidence)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_align.py -v`
Expected: 10 passed. If a robustness case fails, debug with superpowers:systematic-debugging before changing any assertion; do not weaken the test to pass.

- [ ] **Step 5: Commit**

```bash
git add profilometer_comparison/align.py tests/test_align.py
git commit -m "Add phase-correlation alignment with stable-pixel refinement

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Masks

**Files:**
- Create: `profilometer_comparison/masks.py`
- Test: `tests/test_masks.py`

**Interfaces:**
- Produces:
  - `Polygon = list[tuple[float, float]]` (x_um, y_um)
  - `validate_polygon(points) -> None` (raises `ValueError`)
  - `point_in_polygon(point, polygon) -> bool`
  - `polygons_to_mask(polygons, shape, pixel_um, origin_px=(0, 0)) -> np.ndarray bool` — `origin_px` = (row, col) of the array's top-left in before-image pixels
  - `load_masks(path) -> dict[str, list[Polygon]]` (missing file → `{}`), `save_masks(path, masks)`

- [ ] **Step 1: Write the failing tests `tests/test_masks.py`**

```python
import pytest

from profilometer_comparison.masks import (
    load_masks, point_in_polygon, polygons_to_mask, save_masks, validate_polygon,
)

SQUARE = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]


def test_validate_polygon():
    validate_polygon(SQUARE)
    validate_polygon([(0, 0), (5, 0), (0, 5)])
    with pytest.raises(ValueError, match="at least 3"):
        validate_polygon([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="cross"):
        validate_polygon([(0, 0), (10, 10), (10, 0), (0, 10)])  # bow tie


def test_point_in_polygon():
    assert point_in_polygon((15, 15), SQUARE)
    assert not point_in_polygon((5, 15), SQUARE)


def test_polygons_to_mask():
    mask = polygons_to_mask([SQUARE], (40, 40), pixel_um=1.0)
    assert mask[15, 15] and not mask[5, 5]
    assert 100 <= mask.sum() <= 121


def test_polygons_to_mask_uses_pixel_size_and_origin():
    mask = polygons_to_mask([SQUARE], (40, 40), pixel_um=2.0, origin_px=(2, 2))
    # 10-20 µm = 5-10 px in the before image = 3-8 px in this array
    assert mask[5, 5] and not mask[1, 1] and not mask[9, 9]


def test_masks_json_roundtrip(tmp_path):
    path = tmp_path / "masks.json"
    assert load_masks(path) == {}
    save_masks(path, {"M1": [SQUARE]})
    assert load_masks(path) == {"M1": [SQUARE]}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_masks.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/masks.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_masks.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add profilometer_comparison/masks.py tests/test_masks.py
git commit -m "Add encrustation mask polygons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Difference, plane correction, histograms

**Files:**
- Create: `profilometer_comparison/difference.py`
- Test: `tests/test_difference.py`

**Interfaces:**
- Produces:
  - `BIN_WIDTH = 0.05`
  - `@dataclass(frozen=True) Plane(offset=0.0, tilt_x=0.0, tilt_y=0.0)` — µm, µm/mm, µm/mm; coordinates relative to the array's top-left
  - `fit_plane(d, valid, pixel_um, step=8) -> Plane`
  - `subtract_plane(d, plane, pixel_um, origin_px=(0, 0)) -> float32` — `origin_px` = position of `d[0, 0]` in the plane's coordinate frame
  - `histogram(values_1d, signed: bool) -> np.ndarray int64`
  - `percentages(hist, threshold, signed) -> dict` — keys `diff` (+ `lower`, `higher` if signed); NaN if the histogram is empty

- [ ] **Step 1: Write the failing tests `tests/test_difference.py`**

```python
import numpy as np

from profilometer_comparison.difference import (
    Plane, fit_plane, histogram, percentages, subtract_plane,
)


def test_identical_is_zero_percent():
    hist = histogram(np.zeros(1000), signed=True)
    assert percentages(hist, 2.0, signed=True) == {"diff": 0.0, "lower": 0.0, "higher": 0.0}


def test_lowered_patch():
    d = np.zeros((100, 100), np.float32)
    d[:20, :50] = -5.0  # 10 % of the area 5 µm lower
    hist = histogram(d.ravel(), signed=True)
    p = percentages(hist, 2.0, signed=True)
    assert p["lower"] == 10.0 and p["higher"] == 0.0 and p["diff"] == 10.0
    assert percentages(hist, 6.0, signed=True)["diff"] == 0.0


def test_raised_patch_counts_as_higher():
    d = np.zeros(1000)
    d[:30] = 3.0
    p = percentages(histogram(d, signed=True), 2.0, signed=True)
    assert p["higher"] == 3.0 and p["lower"] == 0.0


def test_unsigned_histogram():
    de = np.array([0.5, 1.0, 6.0, 12.0])
    hist = histogram(de, signed=False)
    assert percentages(hist, 5.0, signed=False) == {"diff": 50.0}


def test_values_beyond_range_are_clipped_not_lost():
    hist = histogram(np.array([-1000.0, 1000.0]), signed=True)
    assert hist.sum() == 2
    assert percentages(hist, 2.0, signed=True)["diff"] == 100.0


def test_empty_histogram_gives_nan():
    p = percentages(histogram(np.array([]), signed=False), 2.0, signed=False)
    assert np.isnan(p["diff"])


def test_plane_fit_and_subtract():
    pixel_um = 1.35
    h, w = 400, 300
    y_mm = np.arange(h)[:, None] * pixel_um / 1000
    x_mm = np.arange(w)[None, :] * pixel_um / 1000
    d = (3.0 + 0.5 * x_mm - 0.2 * y_mm).astype(np.float32)
    plane = fit_plane(d, np.ones(d.shape, bool), pixel_um)
    assert abs(plane.offset - 3.0) < 1e-3
    assert abs(plane.tilt_x - 0.5) < 1e-3 and abs(plane.tilt_y + 0.2) < 1e-3
    assert np.abs(subtract_plane(d, plane, pixel_um)).max() < 1e-3


def test_subtract_plane_with_origin():
    plane = Plane(offset=0.0, tilt_x=1.0, tilt_y=0.0)  # 1 µm per mm in x
    d = np.zeros((2, 2), np.float32)
    out = subtract_plane(d, plane, pixel_um=1000.0, origin_px=(0, 5))  # d[0,0] is 5 mm to the right
    np.testing.assert_allclose(out, [[-5, -6], [-5, -6]])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_difference.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/difference.py`**

```python
"""Plane correction, difference histograms and threshold percentages.

Percentages come from histograms of the full-resolution difference (bin width
0.05 µm or 0.05 ΔE), so any threshold on that grid is evaluated exactly without
keeping the full-resolution arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_WIDTH = 0.05
SIGNED_RANGE = 400.0    # µm, histogram covers -400..+400
UNSIGNED_RANGE = 200.0  # ΔE, histogram covers 0..200
_N_SIGNED = int(round(SIGNED_RANGE / BIN_WIDTH))
_N_UNSIGNED = int(round(UNSIGNED_RANGE / BIN_WIDTH))
_CHUNK = 8_000_000


@dataclass(frozen=True)
class Plane:
    offset: float = 0.0  # µm at the array's top-left
    tilt_x: float = 0.0  # µm per mm
    tilt_y: float = 0.0  # µm per mm


def fit_plane(d: np.ndarray, valid: np.ndarray, pixel_um: float, step: int = 8) -> Plane:
    """Least-squares plane through every `step`-th valid pixel."""
    rows, cols = np.nonzero(valid[::step, ::step])
    if rows.size < 3:
        return Plane()
    z = d[::step, ::step][rows, cols].astype(np.float64)
    y_mm = rows * step * pixel_um / 1000
    x_mm = cols * step * pixel_um / 1000
    design = np.column_stack([np.ones_like(x_mm), x_mm, y_mm])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    return Plane(float(coef[0]), float(coef[1]), float(coef[2]))


def subtract_plane(d: np.ndarray, plane: Plane, pixel_um: float, origin_px=(0, 0)) -> np.ndarray:
    h, w = d.shape
    y_mm = ((np.arange(h) + origin_px[0]) * pixel_um / 1000).astype(np.float32)[:, None]
    x_mm = ((np.arange(w) + origin_px[1]) * pixel_um / 1000).astype(np.float32)[None, :]
    return (d - (plane.offset + plane.tilt_x * x_mm + plane.tilt_y * y_mm)).astype(np.float32)


def histogram(values: np.ndarray, signed: bool) -> np.ndarray:
    values = np.asarray(values).ravel()
    n = 2 * _N_SIGNED if signed else _N_UNSIGNED
    hist = np.zeros(n, np.int64)
    for start in range(0, values.size, _CHUNK):
        idx = np.floor(values[start:start + _CHUNK].astype(np.float64) / BIN_WIDTH).astype(np.int64)
        if signed:
            idx += _N_SIGNED
        np.clip(idx, 0, n - 1, out=idx)
        hist += np.bincount(idx, minlength=n)
    return hist


def percentages(hist: np.ndarray, threshold: float, signed: bool) -> dict:
    """% of pixels beyond the threshold: |d| >= t (to within one bin)."""
    total = int(hist.sum())
    if total == 0:
        return {"diff": float("nan"), "lower": float("nan"), "higher": float("nan")} if signed \
            else {"diff": float("nan")}
    k = int(round(threshold / BIN_WIDTH))
    if signed:
        higher = int(hist[_N_SIGNED + k:].sum())
        lower = int(hist[:max(_N_SIGNED - k, 0)].sum())
        return {"diff": 100 * (higher + lower) / total, "lower": 100 * lower / total, "higher": 100 * higher / total}
    return {"diff": 100 * int(hist[k:].sum()) / total}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_difference.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add profilometer_comparison/difference.py tests/test_difference.py
git commit -m "Add plane correction and threshold histograms

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Processing one pair

**Files:**
- Create: `profilometer_comparison/prepare.py`
- Test: `tests/test_prepare_pair.py`

**Interfaces:**
- Consumes: everything from Tasks 3–9.
- Produces:
  - `TYPES = {"pseudo-colour": "pseudo-colour-image", "true-colour": "true-colour"}` (type → filename suffix)
  - `@dataclass Settings(type, pixel_um=1.35, scan_length_um=10674.0, thresholds=None, plane_correction=True, tolerance=25.0, colourmap="default", max_shift_um=1000.0, min_confidence=20.0, min_overlap_pct=80.0, unstable_threshold=None, display_max=1600)` with `.signed`, `.thresholds_resolved`, `.unstable`, `.to_dict()`, `Settings.from_dict(d)`
  - `load_colourmap(settings) -> Colourmap | None`
  - `@dataclass Loaded(align, values, valid, saturated, rgb, pixel_um)`
  - `load_image(path, settings, cmap) -> Loaded`
  - `difference_values(before, after, bs, as_, settings, enc, plane=None, plane_origin=(0, 0)) -> (d float32 with NaN, valid bool, Plane)`
  - `@dataclass PairResult(sample, meta: dict, hists: dict[str, np.ndarray], before_small, after_small, diff_small)`
  - `process_pair(before_path, after_path, settings, cmap=None, polygons=(), manual=None, sample=None) -> PairResult`
  - `meta` keys: `sample, file, type, shift_x_px, shift_y_px, shift_x_um, shift_y_um, align_method, align_confidence, align_warning, overlap_pct, plane_offset_um, plane_tilt_x, plane_tilt_y, mean_dL, excluded_pct, saturated_pct, region_area_pct (dict), pixel_um, before_origin_px [row, col], overlap_shape [h, w], display_step, mask_changed_at`

- [ ] **Step 1: Write the failing tests `tests/test_prepare_pair.py`**

```python
import numpy as np
import pytest
from PIL import Image

from profilometer_comparison.prepare import Settings, load_colourmap, process_pair


@pytest.fixture
def pc_settings():
    return Settings(type="pseudo-colour")


def test_settings_defaults_and_roundtrip():
    s = Settings(type="pseudo-colour")
    assert s.signed and s.thresholds_resolved == (1.0, 2.0, 5.0, 10.0) and s.unstable == 5.0
    t = Settings(type="true-colour")
    assert not t.signed and t.thresholds_resolved == (2.0, 5.0, 10.0) and t.unstable == 10.0
    assert Settings.from_dict(s.to_dict()) == s


def test_pseudo_colour_pair_same_surface(data_dir, pc_settings):
    r = process_pair(data_dir / "pc_M35.jpg", data_dir / "pc_M35_highres.jpg", pc_settings,
                     load_colourmap(pc_settings), sample="M35")
    m = r.meta
    assert abs(m["shift_y_px"] - 2) <= 1 and abs(m["shift_x_px"]) <= 1
    assert m["align_method"] == "auto" and m["align_warning"] == ""
    assert m["overlap_pct"] > 95
    assert set(r.hists) == {"all", "glaze"}
    from profilometer_comparison.difference import percentages
    assert percentages(r.hists["glaze"], 5.0, signed=True)["diff"] < 1.0
    assert r.before_small.shape[:2] == r.after_small.shape[:2] == r.diff_small.shape
    assert r.diff_small.dtype == np.float16


def test_encrustation_region(data_dir, pc_settings):
    square = [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)]
    r = process_pair(data_dir / "pc_M35.jpg", data_dir / "pc_M35_highres.jpg", pc_settings,
                     load_colourmap(pc_settings), polygons=[square], sample="M35")
    assert set(r.hists) == {"all", "glaze", "encrustation"}
    assert 5 < r.meta["region_area_pct"]["encrustation"] < 12
    assert r.hists["glaze"].sum() + r.hists["encrustation"].sum() == r.hists["all"].sum()


def test_manual_alignment(data_dir, pc_settings):
    r = process_pair(data_dir / "pc_M35.jpg", data_dir / "pc_M35_highres.jpg", pc_settings,
                     load_colourmap(pc_settings), manual=(10, -5), sample="M35")
    assert (r.meta["shift_y_px"], r.meta["shift_x_px"]) == (10, -5)
    assert r.meta["align_method"] == "manual"
    assert r.meta["align_confidence"] is None


def framed(texture_rgb):
    page = np.full((texture_rgb.shape[0] + 80, texture_rgb.shape[1] + 80, 3), 255, np.uint8)
    page[37:-37, 37:-37] = 0
    page[40:-40, 40:-40] = texture_rgb
    return page


def test_true_colour_pair(tmp_path):
    rng = np.random.default_rng(0)
    f = np.fft.fft2(rng.normal(size=(420, 420)))
    k = np.fft.fftfreq(420)
    smooth = np.fft.ifft2(f * np.exp(-2 * (np.pi * 2) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))).real
    grey = np.clip(128 + 60 * smooth / smooth.std(), 70, 230).astype(np.uint8)
    big = np.stack([grey, (grey * 0.9).astype(np.uint8), (grey * 0.8).astype(np.uint8)], axis=2)
    dy, dx = 4, -6
    before = big[50:370, 50:370]
    after = big[50 - dy:370 - dy, 50 - dx:370 - dx]
    Image.fromarray(framed(before)).save(tmp_path / "b.png")
    Image.fromarray(framed(after)).save(tmp_path / "a.png")

    settings = Settings(type="true-colour", scan_length_um=3200.0)
    r = process_pair(tmp_path / "b.png", tmp_path / "a.png", settings, None, sample="T1")
    assert (r.meta["shift_y_px"], r.meta["shift_x_px"]) == (dy, dx)
    assert r.meta["pixel_um"] == pytest.approx(10.0)
    from profilometer_comparison.difference import percentages
    assert percentages(r.hists["all"], 2.0, signed=False)["diff"] == 0.0
    assert r.meta["mean_dL"] == pytest.approx(0.0, abs=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_prepare_pair.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/prepare.py` (pair processing)**

```python
"""Stage 1: process before/after pairs into a cache folder."""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .align import Alignment, downsample, estimate_shift, overlap_slices
from .colour import delta_e76, srgb_to_lab
from .colourmap import Colourmap, default_colourmap, load_csv, rgb_to_height
from .crop import crop_to_frame, find_plot_frame
from .difference import Plane, fit_plane, histogram, subtract_plane
from .images import read_rgb
from .masks import Polygon, polygons_to_mask

TYPES = {"pseudo-colour": "pseudo-colour-image", "true-colour": "true-colour"}
REGION_ORDER = ("all", "glaze", "encrustation")


@dataclass
class Settings:
    type: str
    pixel_um: float = 1.35
    scan_length_um: float = 10674.0
    thresholds: tuple[float, ...] | None = None
    plane_correction: bool = True
    tolerance: float = 25.0
    colourmap: str = "default"
    max_shift_um: float = 1000.0
    min_confidence: float = 20.0
    min_overlap_pct: float = 80.0
    unstable_threshold: float | None = None
    display_max: int = 1600

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown type {self.type!r}; choose from {sorted(TYPES)}")
        if self.thresholds is not None:
            self.thresholds = tuple(float(t) for t in self.thresholds)

    @property
    def signed(self) -> bool:
        return self.type == "pseudo-colour"

    @property
    def thresholds_resolved(self) -> tuple[float, ...]:
        if self.thresholds:
            return self.thresholds
        return (1.0, 2.0, 5.0, 10.0) if self.signed else (2.0, 5.0, 10.0)

    @property
    def unstable(self) -> float:
        if self.unstable_threshold is not None:
            return self.unstable_threshold
        return 5.0 if self.signed else 10.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["thresholds"] = list(self.thresholds_resolved)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        return cls(**{**d, "thresholds": tuple(d["thresholds"]) if d.get("thresholds") else None})

    def __eq__(self, other):
        return isinstance(other, Settings) and self.to_dict() == other.to_dict()


def load_colourmap(settings: Settings) -> Colourmap | None:
    if not settings.signed:
        return None
    return default_colourmap() if settings.colourmap == "default" else load_csv(settings.colourmap)


@dataclass
class Loaded:
    align: np.ndarray      # 2-D float32 used for alignment (height or L*)
    values: np.ndarray     # height (2-D) or Lab (3-D)
    valid: np.ndarray      # bool
    saturated: np.ndarray  # bool
    rgb: np.ndarray        # uint8, for display
    pixel_um: float


def load_image(path: str | Path, settings: Settings, cmap: Colourmap | None) -> Loaded:
    rgb = read_rgb(path)
    if settings.signed:
        hm = rgb_to_height(rgb, cmap, settings.tolerance)
        return Loaded(hm.height, hm.height, ~hm.excluded, hm.saturated, rgb, settings.pixel_um)
    try:
        rgb = np.ascontiguousarray(crop_to_frame(rgb, find_plot_frame(rgb)))
    except ValueError:
        pass  # image-only export without axes: use as is
    lab = srgb_to_lab(rgb)
    shape = rgb.shape[:2]
    return Loaded(lab[..., 0], lab, np.ones(shape, bool), np.zeros(shape, bool), rgb,
                  settings.scan_length_um / rgb.shape[1])


def difference_values(before: Loaded, after: Loaded, bs, as_, settings: Settings, enc: np.ndarray,
                      plane: Plane | None = None, plane_origin=(0, 0)):
    """Signed height difference (plane-corrected) or ΔE on the overlap; NaN where invalid."""
    valid = before.valid[bs] & after.valid[as_]
    if settings.signed:
        d = after.values[as_] - before.values[bs]
        if plane is None:
            plane = fit_plane(d, valid & ~enc, before.pixel_um) if settings.plane_correction else Plane()
        d = subtract_plane(d, plane, before.pixel_um, plane_origin)
    else:
        d = delta_e76(before.values[bs], after.values[as_])
        plane = Plane()
    return np.where(valid, d, np.nan).astype(np.float32), valid, plane


@dataclass
class PairResult:
    sample: str
    meta: dict
    hists: dict[str, np.ndarray]
    before_small: np.ndarray
    after_small: np.ndarray
    diff_small: np.ndarray


def _shape(sl) -> tuple[int, int]:
    return sl[0].stop - sl[0].start, sl[1].stop - sl[1].start


def process_pair(before_path, after_path, settings: Settings, cmap: Colourmap | None = None,
                 polygons: list[Polygon] = (), manual: tuple[int, int] | None = None,
                 sample: str | None = None) -> PairResult:
    before_path, after_path = Path(before_path), Path(after_path)
    before = load_image(before_path, settings, cmap)
    after = load_image(after_path, settings, cmap)
    polygons = list(polygons)

    def enc_mask(bs):
        if not polygons:
            return np.zeros(_shape(bs), bool)
        return polygons_to_mask(polygons, _shape(bs), before.pixel_um, (bs[0].start, bs[1].start))

    if manual is not None:
        al = Alignment(int(manual[0]), int(manual[1]), float("nan"), "manual")
    else:
        # Pass 1: all pixels. Pass 2: only pixels that did not change much and are not encrustation.
        first = estimate_shift(before.align, after.align)
        bs, as_ = overlap_slices(before.align.shape, after.align.shape, first.dy, first.dx)
        enc = enc_mask(bs)
        d, valid, _ = difference_values(before, after, bs, as_, settings, enc)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            deviation = np.abs(d - np.nanmedian(d)) if settings.signed else d
            stable = np.zeros(before.align.shape, bool)
            stable[bs] = valid & ~enc & (deviation <= settings.unstable)
        al = estimate_shift(before.align, after.align, stable=stable)

    bs, as_ = overlap_slices(before.align.shape, after.align.shape, al.dy, al.dx)
    enc = enc_mask(bs)
    d, valid, plane = difference_values(before, after, bs, as_, settings, enc)

    regions = {"all": valid, "glaze": valid & ~enc}
    if polygons:
        regions["encrustation"] = valid & enc
    hists = {name: histogram(d[mask], settings.signed) for name, mask in regions.items()}

    h, w = _shape(bs)
    px = before.pixel_um
    overlap_pct = 100 * h * w / before.align.size
    warnings_list = []
    if al.method == "auto":
        if al.confidence < settings.min_confidence:
            warnings_list.append("low confidence")
        if math.hypot(al.dy, al.dx) * px > settings.max_shift_um:
            warnings_list.append("large shift")
    if overlap_pct < settings.min_overlap_pct:
        warnings_list.append("low overlap")

    mean_dl = None
    if not settings.signed:
        mean_dl = float(np.mean(after.values[as_][..., 0][valid] - before.values[bs][..., 0][valid]))

    n_valid = max(int(valid.sum()), 1)
    step = max(1, math.ceil(max(h, w) / settings.display_max))
    hs, ws = h // step * step, w // step * step
    meta = {
        "sample": sample or before_path.stem,
        "file": before_path.name,
        "type": settings.type,
        "shift_x_px": al.dx,
        "shift_y_px": al.dy,
        "shift_x_um": round(al.dx * px, 2),
        "shift_y_um": round(al.dy * px, 2),
        "align_method": al.method,
        "align_confidence": None if math.isnan(al.confidence) else round(al.confidence, 1),
        "align_warning": "; ".join(warnings_list),
        "overlap_pct": round(overlap_pct, 3),
        "plane_offset_um": round(plane.offset, 4),
        "plane_tilt_x": round(plane.tilt_x, 5),
        "plane_tilt_y": round(plane.tilt_y, 5),
        "mean_dL": None if mean_dl is None else round(mean_dl, 3),
        "excluded_pct": round(100 * (1 - valid.mean()), 4),
        "saturated_pct": round(100 * (before.saturated[bs] | after.saturated[as_]).mean(), 4),
        "region_area_pct": {name: round(100 * int(m.sum()) / n_valid, 3) for name, m in regions.items()},
        "pixel_um": px,
        "before_origin_px": [bs[0].start, bs[1].start],
        "overlap_shape": [h, w],
        "display_step": step,
        "mask_changed_at": None,
    }
    return PairResult(
        sample=meta["sample"],
        meta=meta,
        hists=hists,
        before_small=np.ascontiguousarray(before.rgb[bs][:hs:step, :ws:step]),
        after_small=np.ascontiguousarray(after.rgb[as_][:hs:step, :ws:step]),
        diff_small=downsample(d, step).astype(np.float16),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_prepare_pair.py -v`
Expected: 5 passed.

- [ ] **Step 5: Time one full-resolution real pair**

```bash
.venv/bin/python -c "
import time
from profilometer_comparison.prepare import Settings, load_colourmap, process_pair
from profilometer_comparison.difference import percentages
D = '$HOME/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA/pseudo-colour-view/image-only/'
s = Settings(type='pseudo-colour'); t = time.time()
r = process_pair(D + 'M35-pseudo-colour-image.jpg', D + 'M35-high-res-pseudo-colour-image.jpg', s, load_colourmap(s), sample='M35')
print(round(time.time() - t, 1), 's', r.meta['shift_y_px'], r.meta['shift_x_px'], r.meta['align_confidence'])
print({t: round(percentages(r.hists['glaze'], t, True)['diff'], 3) for t in (1, 2, 5, 10)})
"
```
Expected: under ~15 s; shift ≈ (2, 0); confidence > 100; percentages close to the planning noise floor (≈ 9.4, 0.22, 0.01, 0.006).

- [ ] **Step 6: Commit**

```bash
git add profilometer_comparison/prepare.py tests/test_prepare_pair.py
git commit -m "Add single-pair processing: conversion, two-pass alignment, regions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Cache, summary, resume, recompute, zoom

**Files:**
- Modify: `profilometer_comparison/prepare.py` (append)
- Test: `tests/test_prepare_cache.py`

**Interfaces:**
- Consumes: Task 10.
- Produces:
  - `class SettingsMismatch(Exception)`
  - `detect_type(folder) -> str | None`
  - `write_pair(cache, result)`, `read_meta(cache, sample) -> dict`, `read_hists(cache, sample) -> dict[str, np.ndarray]`, `processed_samples(cache) -> list[str]`
  - `read_manifest(cache) -> dict`, `summary_columns(settings) -> list[str]`, `write_summary(cache) -> Path`
  - `run_prepare(before_dir, after_dir, out_dir, settings, masks_file=None, log=print) -> Path`
  - `recompute_pair(cache, sample, manual="keep", mask_changed=False) -> None` — `manual`: `"keep"` (keep a previous manual shift), `None` (automatic) or `(dy, dx)`
  - `zoom_region(cache, sample, centre_before_px, size=600) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (before RGB, after RGB, difference)
  - Cache files: `manifest.json`, `summary.csv`, `masks.json`, `prepare-log.txt`, `pairs/<sample>/{before.png, after.png, diff.npy, hist.npz, meta.json}`

- [ ] **Step 1: Write the failing tests `tests/test_prepare_cache.py`**

```python
import csv
import json
import shutil

import pytest

from profilometer_comparison import prepare
from profilometer_comparison.prepare import (
    Settings, SettingsMismatch, detect_type, read_meta, recompute_pair, run_prepare, zoom_region,
)

NAME = "{}-pseudo-colour-image.jpg"


@pytest.fixture
def folders(tmp_path, data_dir):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    for sample in ("M2", "M35"):
        shutil.copy(data_dir / "pc_M35.jpg", before / NAME.format(sample))
        shutil.copy(data_dir / "pc_M35_highres.jpg", after / NAME.format(sample))
    shutil.copy(data_dir / "pc_M35.jpg", before / NAME.format("M9"))  # before only
    return before, after


def read_summary(cache):
    with open(cache / "summary.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_detect_type(folders):
    assert detect_type(folders[0]) == "pseudo-colour"


def test_run_prepare_writes_cache(folders, tmp_path):
    cache = run_prepare(*folders, tmp_path / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)
    assert sorted(p.name for p in (cache / "pairs").iterdir()) == ["M2", "M35"]
    for f in ("before.png", "after.png", "diff.npy", "hist.npz", "meta.json"):
        assert (cache / "pairs" / "M35" / f).exists()
    rows = read_summary(cache)
    assert [(r["sample"], r["region"]) for r in rows] == [("M2", "all"), ("M2", "glaze"), ("M35", "all"), ("M35", "glaze")]
    assert {"pct_diff_2", "pct_lower_2", "pct_higher_2", "align_method", "region_area_pct"} <= set(rows[0])
    assert "M9" in (cache / "prepare-log.txt").read_text()
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["version"] == "0.1.0" and manifest["settings"]["type"] == "pseudo-colour"


def test_resume_skips_done_pairs(folders, tmp_path, monkeypatch):
    settings = Settings(type="pseudo-colour")
    cache = run_prepare(*folders, tmp_path / "cache", settings, log=lambda *_: None)
    calls = []
    monkeypatch.setattr(prepare, "process_pair", lambda *a, **k: calls.append(1))
    run_prepare(*folders, cache, settings, log=lambda *_: None)
    assert calls == []


def test_settings_mismatch_stops(folders, tmp_path):
    cache = run_prepare(*folders, tmp_path / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)
    with pytest.raises(SettingsMismatch, match="thresholds"):
        run_prepare(*folders, cache, Settings(type="pseudo-colour", thresholds=(3,)), log=lambda *_: None)


def test_masks_file_adds_encrustation_rows(folders, tmp_path):
    masks = tmp_path / "m.json"
    masks.write_text(json.dumps({"M35": [[[100, 100], [400, 100], [400, 400], [100, 400]]]}))
    cache = run_prepare(*folders, tmp_path / "cache", Settings(type="pseudo-colour"), masks_file=masks,
                        log=lambda *_: None)
    regions = [r["region"] for r in read_summary(cache) if r["sample"] == "M35"]
    assert regions == ["all", "glaze", "encrustation"]
    assert json.loads((cache / "masks.json").read_text())["M35"]


def test_recompute_manual_and_back(folders, tmp_path):
    cache = run_prepare(*folders, tmp_path / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)
    recompute_pair(cache, "M35", manual=(12, -3))
    assert read_meta(cache, "M35")["align_method"] == "manual"
    recompute_pair(cache, "M35")  # "keep"
    assert (read_meta(cache, "M35")["shift_y_px"], read_meta(cache, "M35")["shift_x_px"]) == (12, -3)
    assert [r["align_method"] for r in read_summary(cache) if r["sample"] == "M35"][0] == "manual"
    recompute_pair(cache, "M35", manual=None, mask_changed=True)
    meta = read_meta(cache, "M35")
    assert meta["align_method"] == "auto" and meta["mask_changed_at"]


def test_zoom_region(folders, tmp_path):
    cache = run_prepare(*folders, tmp_path / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)
    b, a, d = zoom_region(cache, "M35", (380, 380), size=200)
    assert b.shape == a.shape == (200, 200, 3)
    assert d.shape == (200, 200)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_prepare_cache.py -v`
Expected: FAIL, `ImportError: cannot import name 'SettingsMismatch'`.

- [ ] **Step 3: Append to `profilometer_comparison/prepare.py`**

Add these imports at the top of the file (merge with the existing import block):
```python
import csv
import json
from datetime import datetime

from PIL import Image

from . import __version__
from .difference import percentages
from .masks import load_masks, save_masks
from .pairing import natural_key, pair_folders, sample_id
```

Append:
```python
class SettingsMismatch(Exception):
    """The cache was made with different settings or input folders."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def detect_type(folder: str | Path) -> str | None:
    """The type whose filename suffix most files in the folder carry."""
    stems = [p.stem for p in Path(folder).iterdir() if p.is_file()]
    counts = {t: sum(s.endswith("-" + suffix) for s in stems) for t, suffix in TYPES.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else None


def _pair_dir(cache: Path, sample: str) -> Path:
    return Path(cache) / "pairs" / sample


def write_pair(cache: Path, result: PairResult) -> None:
    d = _pair_dir(cache, result.sample)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.before_small).save(d / "before.png")
    Image.fromarray(result.after_small).save(d / "after.png")
    np.save(d / "diff.npy", result.diff_small)
    np.savez_compressed(d / "hist.npz", **result.hists)
    (d / "meta.json").write_text(json.dumps(result.meta, indent=2), encoding="utf-8")  # last: marks completion


def read_meta(cache: Path, sample: str) -> dict:
    return json.loads((_pair_dir(cache, sample) / "meta.json").read_text(encoding="utf-8"))


def read_hists(cache: Path, sample: str) -> dict[str, np.ndarray]:
    with np.load(_pair_dir(cache, sample) / "hist.npz") as z:
        return {k: z[k] for k in z.files}


def processed_samples(cache: Path) -> list[str]:
    pairs = Path(cache) / "pairs"
    if not pairs.exists():
        return []
    return sorted((p.name for p in pairs.iterdir() if (p / "meta.json").exists()), key=natural_key)


def read_manifest(cache: Path) -> dict:
    return json.loads((Path(cache) / "manifest.json").read_text(encoding="utf-8"))


def _check_or_write_manifest(cache: Path, settings: Settings, before_dir: Path, after_dir: Path) -> None:
    path = cache / "manifest.json"
    current = {"before_dir": str(Path(before_dir).resolve()), "after_dir": str(Path(after_dir).resolve()),
               "settings": settings.to_dict()}
    if path.exists():
        old = read_manifest(cache)
        diffs = [k for k in ("before_dir", "after_dir") if old[k] != current[k]]
        diffs += [k for k in current["settings"] if old["settings"].get(k) != current["settings"][k]]
        if diffs:
            raise SettingsMismatch(
                f"{cache} was made with different {', '.join(diffs)}. "
                "Use a new output folder (--out) or the original settings.")
        return
    path.write_text(json.dumps({"tool": "profilometer-comparison", "version": __version__,
                                "created": _now(), **current}, indent=2), encoding="utf-8")


def summary_columns(settings: Settings) -> list[str]:
    cols = ["sample", "file", "type", "region", "region_area_pct", "shift_x_um", "shift_y_um", "align_method",
            "overlap_pct", "align_confidence", "align_warning", "plane_offset_um", "plane_tilt_x", "plane_tilt_y",
            "mean_dL", "excluded_pct", "saturated_pct", "mask_changed_at"]
    for t in settings.thresholds_resolved:
        cols.append(f"pct_diff_{t:g}")
        if settings.signed:
            cols += [f"pct_lower_{t:g}", f"pct_higher_{t:g}"]
    return cols


def write_summary(cache: Path) -> Path:
    cache = Path(cache)
    settings = Settings.from_dict(read_manifest(cache)["settings"])
    cols = summary_columns(settings)
    path = cache / "summary.csv"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for sample in processed_samples(cache):
            meta, hists = read_meta(cache, sample), read_hists(cache, sample)
            for region in (r for r in REGION_ORDER if r in hists):
                row = {**meta, "region": region, "region_area_pct": meta["region_area_pct"][region]}
                for t in settings.thresholds_resolved:
                    p = percentages(hists[region], t, settings.signed)
                    row[f"pct_diff_{t:g}"] = round(p["diff"], 4)
                    if settings.signed:
                        row[f"pct_lower_{t:g}"] = round(p["lower"], 4)
                        row[f"pct_higher_{t:g}"] = round(p["higher"], 4)
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    tmp.replace(path)
    return path


def run_prepare(before_dir, after_dir, out_dir, settings: Settings, masks_file=None, log=print) -> Path:
    before_dir, after_dir, cache = Path(before_dir), Path(after_dir), Path(out_dir)
    cache.mkdir(parents=True, exist_ok=True)
    _check_or_write_manifest(cache, settings, before_dir, after_dir)

    masks = load_masks(cache / "masks.json")
    if masks_file:
        masks.update(load_masks(masks_file))
        save_masks(cache / "masks.json", masks)

    log_lines = [f"=== prepare {_now()} (profilometer-comparison {__version__})"]

    def note(message: str) -> None:
        log_lines.append(message)
        log(message)

    pairing = pair_folders(before_dir, after_dir)
    suffix = TYPES[settings.type]
    typed = [p for p in pairing.pairs if Path(p.name).stem.endswith("-" + suffix)]
    pairs = typed or pairing.pairs
    for p in pairing.pairs:
        if typed and p not in typed:
            note(f"ignored (not a {settings.type} image): {p.name}")

    cmap = load_colourmap(settings)
    for i, pair in enumerate(pairs, 1):
        sample = sample_id(pair.name, suffix)
        if (_pair_dir(cache, sample) / "meta.json").exists():
            continue
        try:
            result = process_pair(pair.before, pair.after, settings, cmap, masks.get(sample, []), sample=sample)
            write_pair(cache, result)
        except Exception as exc:  # one bad file must not stop the run
            note(f"[{i}/{len(pairs)}] {sample}: ERROR {type(exc).__name__}: {exc}")
            continue
        m = result.meta
        warn = f"  WARNING: {m['align_warning']}" if m["align_warning"] else ""
        note(f"[{i}/{len(pairs)}] {sample}: shift {m['shift_x_um']:+.1f}/{m['shift_y_um']:+.1f} µm{warn}")

    for name in pairing.before_only:
        note(f"only in before folder: {name}")
    for name in pairing.after_only:
        note(f"only in after folder: {name}")
    write_summary(cache)
    with open(cache / "prepare-log.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    return cache


def recompute_pair(cache, sample: str, manual="keep", mask_changed: bool = False) -> None:
    cache = Path(cache)
    manifest = read_manifest(cache)
    settings = Settings.from_dict(manifest["settings"])
    old = read_meta(cache, sample)
    if manual == "keep":
        manual = (old["shift_y_px"], old["shift_x_px"]) if old["align_method"] == "manual" else None
    polygons = load_masks(cache / "masks.json").get(sample, [])
    result = process_pair(Path(manifest["before_dir"]) / old["file"], Path(manifest["after_dir"]) / old["file"],
                          settings, load_colourmap(settings), polygons, manual=manual, sample=sample)
    result.meta["mask_changed_at"] = _now() if mask_changed else old.get("mask_changed_at")
    write_pair(cache, result)
    write_summary(cache)


def zoom_region(cache, sample: str, centre_before_px, size: int = 600):
    """Full-resolution before/after/difference around a point (row, col) of the before image."""
    cache = Path(cache)
    manifest = read_manifest(cache)
    settings = Settings.from_dict(manifest["settings"])
    meta = read_meta(cache, sample)
    cmap = load_colourmap(settings)
    before = load_image(Path(manifest["before_dir"]) / meta["file"], settings, cmap)
    after = load_image(Path(manifest["after_dir"]) / meta["file"], settings, cmap)
    dy, dx = meta["shift_y_px"], meta["shift_x_px"]
    bs, _ = overlap_slices(before.align.shape, after.align.shape, dy, dx)

    def window(centre, sl):
        start = int(np.clip(centre - size // 2, sl.start, max(sl.start, sl.stop - size)))
        return slice(start, min(start + size, sl.stop))

    rows, cols = window(centre_before_px[0], bs[0]), window(centre_before_px[1], bs[1])
    zb = (rows, cols)
    za = (slice(rows.start + dy, rows.stop + dy), slice(cols.start + dx, cols.stop + dx))
    plane = Plane(meta["plane_offset_um"], meta["plane_tilt_x"], meta["plane_tilt_y"])
    origin = (rows.start - bs[0].start, cols.start - bs[1].start)
    enc = np.zeros(_shape(zb), bool)
    d, _, _ = difference_values(before, after, zb, za, settings, enc, plane=plane, plane_origin=origin)
    return before.rgb[zb], after.rgb[za], d
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (7 new).

- [ ] **Step 5: Commit**

```bash
git add profilometer_comparison/prepare.py tests/test_prepare_cache.py
git commit -m "Add cache, summary CSV, resume, recompute and zoom

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Flags and viewer state (no GUI)

**Files:**
- Create: `profilometer_comparison/flags.py`, `profilometer_comparison/viewer_state.py`
- Test: `tests/test_viewer_state.py`

**Interfaces:**
- Consumes: `prepare.read_manifest, read_meta, read_hists, processed_samples, Settings`; `difference.percentages`; `pairing.natural_key`.
- Produces:
  - `FlagStore(path, fields)`: `__contains__`, `__len__`, `keys() -> list[str]`, `toggle(key, row) -> bool` (rewrites the CSV; adds `flagged_at`)
  - `render_difference(diff, threshold, signed, after_small=None) -> np.ndarray uint8 (H, W, 3)`
  - `ViewerState(cache)`: attributes `cache, settings, signed, samples, flags, threshold, flagged_only, current`; methods `visible_samples()`, `go(step: int)`, `toggle_flag() -> bool`, `toggle_flagged_only() -> bool`, `step_threshold(direction: int)`, `set_threshold(t)`, `meta(sample=None) -> dict`, `percentages(region="glaze", sample=None) -> dict | None`, `reload(sample)`, `save_position()`
  - `FLAG_FIELDS = ["sample", "file", "type", "pct_diff_glaze", "threshold", "shift_x_um", "shift_y_um", "align_method", "flagged_at"]`

- [ ] **Step 1: Write the failing tests `tests/test_viewer_state.py`**

```python
import csv
import shutil

import numpy as np
import pytest

from profilometer_comparison.flags import FlagStore
from profilometer_comparison.prepare import Settings, run_prepare
from profilometer_comparison.viewer_state import ViewerState, render_difference


@pytest.fixture(scope="module")
def cache(tmp_path_factory, data_dir):
    root = tmp_path_factory.mktemp("viewer")
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    for sample in ("M1", "M2", "M10"):
        shutil.copy(data_dir / "pc_M35.jpg", before / f"{sample}-pseudo-colour-image.jpg")
        shutil.copy(data_dir / "pc_M35_highres.jpg", after / f"{sample}-pseudo-colour-image.jpg")
    return run_prepare(before, after, root / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)


@pytest.fixture
def state(cache):
    for f in ("flagged.csv", "state.json"):
        (cache / f).unlink(missing_ok=True)
    return ViewerState(cache)


def test_flag_store(tmp_path):
    store = FlagStore(tmp_path / "f.csv", ["sample", "note", "flagged_at"])
    assert store.toggle("M2", {"note": "x"}) is True
    assert store.toggle("M10", {"note": "y"}) is True
    assert store.keys() == ["M2", "M10"]
    assert store.toggle("M2", {}) is False
    reopened = FlagStore(tmp_path / "f.csv", ["sample", "note", "flagged_at"])
    assert "M10" in reopened and "M2" not in reopened and len(reopened) == 1


def test_navigation_is_natural_and_bounded(state):
    assert state.samples == ["M1", "M2", "M10"]
    assert state.current == "M1"
    state.go(-1)
    assert state.current == "M1"
    state.go(1)
    state.go(1)
    state.go(1)
    assert state.current == "M10"


def test_flagging_writes_csv_and_filter(state, cache):
    state.go(1)
    assert state.toggle_flag() is True
    rows = list(csv.DictReader(open(cache / "flagged.csv", encoding="utf-8")))
    assert rows[0]["sample"] == "M2" and rows[0]["threshold"] == "2.0" and rows[0]["align_method"] == "auto"
    assert state.toggle_flagged_only() is True
    assert state.visible_samples() == ["M2"]
    state.go(1)
    assert state.current == "M2"


def test_flagged_only_refuses_when_nothing_flagged(state):
    assert state.toggle_flagged_only() is False


def test_threshold_and_percentages(state):
    assert state.threshold == 2.0
    state.step_threshold(-1)
    state.step_threshold(-1)
    state.step_threshold(-1)
    assert state.threshold == 0.5
    low = state.percentages()["diff"]
    state.set_threshold(10)
    assert state.percentages()["diff"] < low
    assert state.percentages(region="encrustation") is None


def test_position_is_restored(cache, state):
    state.go(1)
    state.save_position()
    assert ViewerState(cache).current == "M2"


def test_render_difference_signed():
    diff = np.array([[-3, 0, 3, np.nan]], np.float16)
    out = render_difference(diff, 2.0, signed=True)
    assert out.shape == (1, 4, 3)
    assert out[0, 0].tolist() == [40, 90, 220]    # lower
    assert out[0, 1].tolist() == [225, 225, 225]  # unchanged
    assert out[0, 2].tolist() == [220, 50, 40]    # higher
    assert out[0, 3].tolist() == [255, 255, 255]  # no data


def test_render_difference_unsigned():
    diff = np.array([[1, 8]], np.float16)
    after = np.full((1, 2, 3), 100, np.uint8)
    out = render_difference(diff, 5.0, signed=False, after_small=after)
    assert out[0, 0].tolist() == [100, 100, 100]
    assert out[0, 1].tolist() == [255, 0, 200]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_viewer_state.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `profilometer_comparison/flags.py`**

```python
"""Flagged samples, stored as a CSV that is rewritten on every change."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .pairing import natural_key


class FlagStore:
    def __init__(self, path: str | Path, fields: list[str]):
        self.path = Path(path)
        self.fields = list(fields)
        self.key_field = self.fields[0]
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.rows[row[self.key_field]] = row

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def __len__(self) -> int:
        return len(self.rows)

    def keys(self) -> list[str]:
        return sorted(self.rows, key=natural_key)

    def toggle(self, key: str, row: dict) -> bool:
        if key in self.rows:
            del self.rows[key]
            flagged = False
        else:
            self.rows[key] = {**row, self.key_field: key,
                              "flagged_at": datetime.now().isoformat(timespec="seconds")}
            flagged = True
        self._save()
        return flagged

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore")
            writer.writeheader()
            for key in self.keys():
                writer.writerow(self.rows[key])
        tmp.replace(self.path)
```

- [ ] **Step 4: Implement `profilometer_comparison/viewer_state.py`**

```python
"""Viewer logic without any GUI, so it can be tested headless."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .difference import percentages
from .flags import FlagStore
from .prepare import Settings, processed_samples, read_hists, read_manifest, read_meta

FLAG_FIELDS = ["sample", "file", "type", "pct_diff_glaze", "threshold", "shift_x_um", "shift_y_um",
               "align_method", "flagged_at"]
LOWER = (40, 90, 220)
HIGHER = (220, 50, 40)
UNCHANGED = (225, 225, 225)
NO_DATA = (255, 255, 255)
MARKED = (255, 0, 200)


def render_difference(diff: np.ndarray, threshold: float, signed: bool,
                      after_small: np.ndarray | None = None) -> np.ndarray:
    """Colour image of the difference: blue lower / red higher (height) or magenta over a grey after image (colour)."""
    d = diff.astype(np.float32)
    with np.errstate(invalid="ignore"):
        if signed:
            out = np.empty(d.shape + (3,), np.uint8)
            out[:] = UNCHANGED
            out[d <= -threshold] = LOWER
            out[d >= threshold] = HIGHER
        else:
            grey = after_small.astype(np.float32).mean(axis=2).astype(np.uint8)
            out = np.repeat(grey[..., None], 3, axis=2)
            out[d >= threshold] = MARKED
    out[np.isnan(d)] = NO_DATA
    return out


class ViewerState:
    def __init__(self, cache: str | Path):
        self.cache = Path(cache)
        self.manifest = read_manifest(self.cache)
        self.settings = Settings.from_dict(self.manifest["settings"])
        self.signed = self.settings.signed
        self.samples = processed_samples(self.cache)
        if not self.samples:
            raise ValueError(f"no processed samples in {self.cache}")
        self.flags = FlagStore(self.cache / "flagged.csv", FLAG_FIELDS)
        self.threshold = 2.0 if self.signed else 5.0
        self.flagged_only = False
        self.current = self.samples[0]
        self._meta: dict[str, dict] = {}
        self._hists: dict[str, dict] = {}
        state_file = self.cache / "state.json"
        if state_file.exists():
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            if saved.get("last_sample") in self.samples:
                self.current = saved["last_sample"]
            self.threshold = float(saved.get("threshold", self.threshold))

    def visible_samples(self) -> list[str]:
        if self.flagged_only:
            return [s for s in self.samples if s in self.flags]
        return self.samples

    def go(self, step: int) -> None:
        visible = self.visible_samples()
        if self.current in visible:
            i = visible.index(self.current) + step
        else:  # current sample was hidden by the filter: move to the nearest visible one
            order = self.samples.index(self.current)
            i = next((k for k, s in enumerate(visible) if self.samples.index(s) > order), len(visible) - 1)
        self.current = visible[max(0, min(i, len(visible) - 1))]

    def meta(self, sample: str | None = None) -> dict:
        sample = sample or self.current
        if sample not in self._meta:
            self._meta[sample] = read_meta(self.cache, sample)
        return self._meta[sample]

    def percentages(self, region: str = "glaze", sample: str | None = None) -> dict | None:
        sample = sample or self.current
        if sample not in self._hists:
            self._hists[sample] = read_hists(self.cache, sample)
        hist = self._hists[sample].get(region)
        return None if hist is None else percentages(hist, self.threshold, self.signed)

    def reload(self, sample: str) -> None:
        self._meta.pop(sample, None)
        self._hists.pop(sample, None)

    def toggle_flag(self) -> bool:
        m = self.meta()
        row = {"file": m["file"], "type": m["type"], "pct_diff_glaze": round(self.percentages()["diff"], 4),
               "threshold": self.threshold, "shift_x_um": m["shift_x_um"], "shift_y_um": m["shift_y_um"],
               "align_method": m["align_method"]}
        return self.flags.toggle(self.current, row)

    def toggle_flagged_only(self) -> bool:
        if not self.flagged_only and len(self.flags) == 0:
            return False
        self.flagged_only = not self.flagged_only
        if self.flagged_only and self.current not in self.flags:
            self.go(0)
        return self.flagged_only

    def set_threshold(self, value: float) -> None:
        self.threshold = max(0.5, round(float(value) * 2) / 2)

    def step_threshold(self, direction: int) -> None:
        self.set_threshold(self.threshold + 0.5 * direction)

    def save_position(self) -> None:
        (self.cache / "state.json").write_text(
            json.dumps({"last_sample": self.current, "threshold": self.threshold}), encoding="utf-8")
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_viewer_state.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add profilometer_comparison/flags.py profilometer_comparison/viewer_state.py tests/test_viewer_state.py
git commit -m "Add flag store and headless viewer state

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Comparison viewer (tkinter): panels, browsing, flags, threshold, filter

**Files:**
- Create: `profilometer_comparison/viewer.py`

**Interfaces:**
- Consumes: `ViewerState`, `render_difference` (Task 12).
- Produces: `class Viewer(root: tk.Tk, state: ViewerState)`; `run_viewer(cache) -> None`. Task 14 adds methods to this class; the attribute names below are relied on there: `self.root, self.state, self.panel, self.canvases (dict name→Canvas), self.panels (Frame), self.controls (Frame), self.status (Label), self.meta (dict), self.before_img, self.after_img (PIL Images), self.diff (ndarray), self.scale (float), self.masks (list of polygons), self.nudge ([dy, dx]), self.mask_mode (bool), self.blinking (bool)`.

GUI code is verified manually (Step 3), because tkinter needs a display.

- [ ] **Step 1: Implement `profilometer_comparison/viewer.py`**

```python
"""Stage 2: keyboard-driven before/after/difference viewer."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from .masks import load_masks
from .viewer_state import ViewerState, render_difference

HELP = ("← → sample   s flag   ↑ ↓ threshold   f flagged only   b blink   m mask   "
        "Shift+arrows nudge (Shift+Alt: ×10)   a auto-align   click: zoom")


class Viewer:
    def __init__(self, root: tk.Tk, state: ViewerState):
        self.root, self.state = root, state
        self.panel = max(300, min(700, (root.winfo_screenwidth() - 80) // 3))
        self.blink_size = max(400, min(2 * self.panel, root.winfo_screenheight() - 220))
        self.blinking = False
        self._blink_job = None
        self._blink_phase = 0
        self.mask_mode = False
        self.points: list[tuple[float, float]] = []
        self.masks_dirty = False
        self.nudge = [0, 0]
        self._nudge_job = None
        self._photos: list = []
        self._build()
        self._bind_keys()
        self.show()

    # ---------- layout ----------
    def _build(self) -> None:
        self.root.title(f"profilometer-comparison: {self.state.cache.name}")
        self.panels = tk.Frame(self.root)
        self.panels.pack(side=tk.TOP)
        self.canvases: dict[str, tk.Canvas] = {}
        for col, name in enumerate(("before", "after", "difference")):
            tk.Label(self.panels, text=name.upper(), font=("TkDefaultFont", 12, "bold")).grid(row=0, column=col)
            canvas = tk.Canvas(self.panels, width=self.panel, height=self.panel, bg="white", highlightthickness=0)
            canvas.grid(row=1, column=col, padx=4, pady=4)
            canvas.bind("<Button-1>", lambda e, n=name: self._on_click(n, e))
            self.canvases[name] = canvas
        self.blink_canvas = tk.Canvas(self.root, width=self.blink_size, height=self.blink_size, bg="black",
                                      highlightthickness=0)

        self.controls = tk.Frame(self.root)
        self.controls.pack(side=tk.TOP, fill=tk.X, padx=8)
        unit = "µm" if self.state.signed else "ΔE"
        tk.Label(self.controls, text=f"Threshold ({unit})").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.state.threshold)
        tk.Scale(self.controls, from_=0.5, to=50, resolution=0.5, orient=tk.HORIZONTAL, length=320,
                 variable=self.threshold_var, command=self._on_slider, takefocus=0).pack(side=tk.LEFT)
        self.status = tk.Label(self.root, anchor="w", justify=tk.LEFT, font=("TkFixedFont", 12))
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8)
        tk.Label(self.root, anchor="w", fg="grey40", text=HELP).pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<Left>", lambda e: self._go(-1))
        r.bind("<Right>", lambda e: self._go(1))
        r.bind("<Up>", lambda e: self._step_threshold(1))
        r.bind("<Down>", lambda e: self._step_threshold(-1))
        r.bind("s", lambda e: self._toggle_flag())
        r.bind("f", lambda e: self._toggle_flagged_only())
        r.protocol("WM_DELETE_WINDOW", self._quit)

    # ---------- drawing ----------
    def show(self) -> None:
        s = self.state
        folder = s.cache / "pairs" / s.current
        self.meta = s.meta()
        self.before_img = Image.open(folder / "before.png").convert("RGB")
        self.after_img = Image.open(folder / "after.png").convert("RGB")
        self.diff = np.load(folder / "diff.npy")
        self.scale = self.panel / max(self.before_img.size)
        self.masks = load_masks(s.cache / "masks.json").get(s.current, [])
        self._render_panels()
        if self.blinking:
            self._render_blink()
        self._update_status()

    def _nudge_offset(self, scale: float) -> tuple[float, float]:
        step = self.meta["display_step"]
        return -self.nudge[1] / step * scale, -self.nudge[0] / step * scale

    def _render_panels(self) -> None:
        s = self.state
        w, h = self.before_img.size
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        diff_rgb = render_difference(self.diff, s.threshold, s.signed, np.asarray(self.after_img))
        images = {"before": (self.before_img, Image.BILINEAR),
                  "after": (self.after_img, Image.BILINEAR),
                  "difference": (Image.fromarray(diff_rgb), Image.NEAREST)}
        self._photos = []
        for name, (img, resample) in images.items():
            canvas = self.canvases[name]
            canvas.delete("all")
            photo = ImageTk.PhotoImage(img.resize(size, resample))
            self._photos.append(photo)
            ox, oy = self._nudge_offset(self.scale) if name == "after" else (0, 0)
            canvas.create_image(ox, oy, image=photo, anchor="nw")
            self._draw_polygons(canvas)

    def _draw_polygons(self, canvas: tk.Canvas) -> None:
        for poly in self.masks:
            coords = [c for p in poly for c in self._um_to_canvas(*p)]
            canvas.create_polygon(*coords, outline="yellow", fill="", width=2)
        if self.points:
            coords = [c for p in self.points for c in self._um_to_canvas(*p)]
            if len(self.points) > 1:
                canvas.create_line(*coords, fill="yellow", width=2)
            for x, y in zip(coords[::2], coords[1::2]):
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="yellow", outline="")

    def _update_status(self, extra: str = "") -> None:
        s, m = self.state, self.meta
        visible = s.visible_samples()
        position = f"{visible.index(s.current) + 1}/{len(visible)}" if s.current in visible else "-"
        flag = "  ★ FLAGGED" if s.current in s.flags else ""
        filt = "  [flagged only]" if s.flagged_only else ""
        manual = " (manual)" if m["align_method"] == "manual" else ""
        lines = [f"{s.current}  ({position}){flag}{filt}    shift {m['shift_x_um']:+.1f} / "
                 f"{m['shift_y_um']:+.1f} µm{manual}    threshold {s.threshold:g} {'µm' if s.signed else 'ΔE'}"]
        for region in ("glaze", "encrustation"):
            p = s.percentages(region=region)
            if p is None:
                continue
            if s.signed:
                lines.append(f"{region:13s} {p['diff']:6.2f} % different   "
                             f"({p['lower']:.2f} % lower, {p['higher']:.2f} % higher)")
            else:
                lines.append(f"{region:13s} {p['diff']:6.2f} % with ΔE ≥ {s.threshold:g}")
        if m["align_warning"]:
            lines.append(f"⚠ alignment: {m['align_warning']}")
        if self.mask_mode:
            lines.append("MASK MODE: click points on BEFORE · Enter close polygon · Backspace undo point · "
                         "Delete remove polygon under cursor · m/Esc done")
        if self.nudge != [0, 0]:
            lines.append(f"nudge {self.nudge[1]:+d} / {self.nudge[0]:+d} px (applied after a short pause)")
        if extra:
            lines.append(extra)
        self.status.config(text="\n".join(lines))

    # ---------- coordinates ----------
    def _canvas_to_um(self, x: float, y: float) -> tuple[float, float]:
        m = self.meta
        row = y / self.scale * m["display_step"] + m["before_origin_px"][0]
        col = x / self.scale * m["display_step"] + m["before_origin_px"][1]
        return col * m["pixel_um"], row * m["pixel_um"]

    def _um_to_canvas(self, x_um: float, y_um: float) -> tuple[float, float]:
        m = self.meta
        col = x_um / m["pixel_um"] - m["before_origin_px"][1]
        row = y_um / m["pixel_um"] - m["before_origin_px"][0]
        return col / m["display_step"] * self.scale, row / m["display_step"] * self.scale

    # ---------- actions ----------
    def _go(self, step: int) -> None:
        if self.mask_mode:
            self._update_status("Leave mask mode (m) before changing sample.")
            return
        self._flush_nudge()
        self.state.go(step)
        self.show()

    def _step_threshold(self, direction: int) -> None:
        self.state.step_threshold(direction)
        self.threshold_var.set(self.state.threshold)
        self._render_panels()
        self._update_status()

    def _on_slider(self, value) -> None:
        self.state.set_threshold(float(value))
        self._render_panels()
        self._update_status()
        self.root.focus_set()  # keep arrow keys for navigation

    def _toggle_flag(self) -> None:
        self.state.toggle_flag()
        self._update_status()

    def _toggle_flagged_only(self) -> None:
        if not self.state.flagged_only and len(self.state.flags) == 0:
            self._update_status("No flagged samples yet.")
            return
        self.state.toggle_flagged_only()
        self.show()

    def _on_click(self, name: str, event) -> None:
        pass  # replaced in Task 14 (mask points / zoom)

    def _flush_nudge(self) -> None:
        pass  # replaced in Task 14

    def _quit(self) -> None:
        self._flush_nudge()
        self.state.save_position()
        self.root.destroy()


def run_viewer(cache: str | Path) -> None:
    root = tk.Tk()
    try:
        state = ViewerState(cache)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Cannot open comparison", str(exc))
        root.destroy()
        return
    Viewer(root, state)
    root.mainloop()
```

- [ ] **Step 2: Build a real cache for manual testing (M1–M4 before vs the same files, and M35 vs its rescan)**

```bash
MOPA="$HOME/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA"
mkdir -p /tmp/pcmp/before /tmp/pcmp/after
for s in M1 M2 M3 M4 M35; do cp "$MOPA/pseudo-colour-view/image-only/$s-pseudo-colour-image.jpg" /tmp/pcmp/before/; done
for s in M1 M2 M3 M4; do cp "$MOPA/pseudo-colour-view/image-only/$s-pseudo-colour-image.jpg" /tmp/pcmp/after/; done
cp "$MOPA/pseudo-colour-view/image-only/M35-high-res-pseudo-colour-image.jpg" /tmp/pcmp/after/M35-pseudo-colour-image.jpg
.venv/bin/python -c "
from profilometer_comparison.prepare import Settings, run_prepare
run_prepare('/tmp/pcmp/before', '/tmp/pcmp/after', '/tmp/pcmp/cache', Settings(type='pseudo-colour'))
"
```
Expected: 5 progress lines; M1–M4 shift +0.0/+0.0 µm; M35 shift ≈ 0/+2.7 µm; no warnings.

- [ ] **Step 3: Manual check**

```bash
.venv/bin/python -c "from profilometer_comparison.viewer import run_viewer; run_viewer('/tmp/pcmp/cache')"
```
Check each and note the result:
1. Three panels appear; M1 difference panel is all grey (identical files).
2. → / ← move through M1, M2, M3, M4, M35 and stop at the ends; the position counter updates.
3. ↑ / ↓ and the slider change the threshold; percentages update; after using the slider, ← / → still change sample.
4. At M35 with threshold 0.5 the difference panel shows scattered blue/red; at 2 µm it is almost all grey.
5. `s` shows ★ FLAGGED and writes `/tmp/pcmp/cache/flagged.csv`; `s` again removes it.
6. `f` with one flagged sample shows only that sample; `f` again shows all. `f` with no flags shows "No flagged samples yet."
7. Close the window and reopen: it resumes at the last sample and threshold.

- [ ] **Step 4: Run all tests (nothing broken) and commit**

```bash
.venv/bin/pytest -q
git add profilometer_comparison/viewer.py
git commit -m "Add comparison viewer with browsing, flags, threshold and filter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Viewer: blink, nudge, masks, zoom

**Files:**
- Modify: `profilometer_comparison/viewer.py`

**Interfaces:**
- Consumes: `prepare.recompute_pair(cache, sample, manual="keep", mask_changed=False)`, `prepare.zoom_region(cache, sample, centre_before_px, size)`, `masks.load_masks/save_masks/validate_polygon/point_in_polygon`, attributes from Task 13.
- Produces: keys `b`, `m`, `a`, `Return`, `BackSpace`, `Delete`, `Escape`, `Shift+arrows`, `Shift+Alt/Option+arrows`; click-to-zoom.

- [ ] **Step 1: Extend imports in `viewer.py`**

Replace the import lines for `masks` and `viewer_state` with:
```python
from . import prepare
from .masks import load_masks, point_in_polygon, save_masks, validate_polygon
from .viewer_state import ViewerState, render_difference
```

- [ ] **Step 2: Add the key bindings at the end of `_bind_keys`**

```python
        r.bind("b", lambda e: self._toggle_blink())
        r.bind("m", lambda e: self._toggle_mask_mode())
        r.bind("a", lambda e: self._reset_alignment())
        r.bind("<Return>", lambda e: self._close_polygon())
        r.bind("<BackSpace>", lambda e: self._undo_point())
        r.bind("<Delete>", lambda e: self._delete_polygon())
        r.bind("<Escape>", lambda e: self._leave_mask_mode() if self.mask_mode else None)
        # Arrow = direction the AFTER image moves on screen.
        moves = {"Left": (0, 1), "Right": (0, -1), "Up": (1, 0), "Down": (-1, 0)}
        for key, delta in moves.items():
            r.bind(f"<Shift-{key}>", lambda e, d=delta: self._nudge(d, 1))
            for modifier in ("Alt", "Option"):
                try:
                    r.bind(f"<Shift-{modifier}-{key}>", lambda e, d=delta: self._nudge(d, 10))
                except tk.TclError:
                    pass  # modifier name not known on this platform
```

- [ ] **Step 3: Replace the `_on_click` and `_flush_nudge` placeholders and add the new methods to the class**

```python
    # ---------- blink ----------
    def _toggle_blink(self) -> None:
        if self.mask_mode:
            return
        self.blinking = not self.blinking
        if self.blinking:
            self.panels.pack_forget()
            self.blink_canvas.pack(side=tk.TOP, before=self.controls)
            self._blink_tick()
        else:
            if self._blink_job:
                self.root.after_cancel(self._blink_job)
                self._blink_job = None
            self.blink_canvas.pack_forget()
            self.panels.pack(side=tk.TOP, before=self.controls)

    def _blink_tick(self) -> None:
        self._blink_phase ^= 1
        self._render_blink()
        self._blink_job = self.root.after(500, self._blink_tick)

    def _render_blink(self) -> None:
        showing_after = self._blink_phase == 1
        img = self.after_img if showing_after else self.before_img
        scale = self.blink_size / max(img.size)
        photo = ImageTk.PhotoImage(img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                                              Image.BILINEAR))
        self._blink_photo = photo
        ox, oy = self._nudge_offset(scale) if showing_after else (0, 0)
        c = self.blink_canvas
        c.delete("all")
        c.create_image(ox, oy, image=photo, anchor="nw")
        c.create_text(12, 12, anchor="nw", text="AFTER" if showing_after else "BEFORE",
                      fill="yellow", font=("TkDefaultFont", 16, "bold"))

    # ---------- manual alignment ----------
    def _nudge(self, delta, step: int) -> None:
        if self.mask_mode:
            return
        self.nudge[0] += delta[0] * step
        self.nudge[1] += delta[1] * step
        self._render_panels()
        if self.blinking:
            self._render_blink()
        self._update_status()
        if self._nudge_job:
            self.root.after_cancel(self._nudge_job)
        self._nudge_job = self.root.after(800, self._flush_nudge)

    def _flush_nudge(self) -> None:
        if self._nudge_job:
            self.root.after_cancel(self._nudge_job)
            self._nudge_job = None
        if self.nudge == [0, 0]:
            return
        manual = (self.meta["shift_y_px"] + self.nudge[0], self.meta["shift_x_px"] + self.nudge[1])
        self.nudge = [0, 0]
        self._recompute(manual=manual)

    def _reset_alignment(self) -> None:
        if self.mask_mode:
            return
        self.nudge = [0, 0]
        self._recompute(manual=None)

    def _recompute(self, **kwargs) -> None:
        self._update_status("Recomputing from the full-resolution originals…")
        self.root.update_idletasks()
        try:
            prepare.recompute_pair(self.state.cache, self.state.current, **kwargs)
        except Exception as exc:
            messagebox.showerror("Recompute failed", str(exc))
        self.state.reload(self.state.current)
        self.show()

    # ---------- masks ----------
    def _toggle_mask_mode(self) -> None:
        if self.mask_mode:
            self._leave_mask_mode()
            return
        if self.blinking:
            self._toggle_blink()
        self._flush_nudge()
        self.mask_mode = True
        self.points = []
        self._update_status()

    def _leave_mask_mode(self) -> None:
        self.mask_mode = False
        self.points = []
        if self.masks_dirty:
            self.masks_dirty = False
            path = self.state.cache / "masks.json"
            all_masks = load_masks(path)
            if self.masks:
                all_masks[self.state.current] = self.masks
            else:
                all_masks.pop(self.state.current, None)
            save_masks(path, all_masks)
            self._recompute(manual="keep", mask_changed=True)
        else:
            self._render_panels()
            self._update_status()

    def _close_polygon(self) -> None:
        if not self.mask_mode or not self.points:
            return
        try:
            validate_polygon(self.points)
        except ValueError as exc:
            messagebox.showwarning("Invalid polygon", str(exc))
            return
        self.masks.append(list(self.points))
        self.points = []
        self.masks_dirty = True
        self._render_panels()
        self._update_status()

    def _undo_point(self) -> None:
        if self.mask_mode and self.points:
            self.points.pop()
            self._render_panels()

    def _delete_polygon(self) -> None:
        if not self.mask_mode:
            return
        c = self.canvases["before"]
        point = self._canvas_to_um(c.winfo_pointerx() - c.winfo_rootx(), c.winfo_pointery() - c.winfo_rooty())
        for i, poly in enumerate(self.masks):
            if point_in_polygon(point, poly):
                del self.masks[i]
                self.masks_dirty = True
                self._render_panels()
                self._update_status()
                return

    # ---------- click: mask point or zoom ----------
    def _on_click(self, name: str, event) -> None:
        if self.mask_mode:
            if name == "before":
                self.points.append(self._canvas_to_um(event.x, event.y))
                self._render_panels()
            return
        self._zoom(event.x, event.y)

    def _zoom(self, x: float, y: float) -> None:
        m, s = self.meta, self.state
        row = int(y / self.scale * m["display_step"]) + m["before_origin_px"][0]
        col = int(x / self.scale * m["display_step"]) + m["before_origin_px"][1]
        size = max(200, min(600, (self.root.winfo_screenwidth() - 60) // 3))
        self._update_status("Loading full resolution…")
        self.root.update_idletasks()
        try:
            before, after, diff = prepare.zoom_region(s.cache, s.current, (row, col), size=size)
        except Exception as exc:
            messagebox.showerror("Zoom failed", str(exc))
            return
        top = tk.Toplevel(self.root)
        top.title(f"{s.current}: zoom at x {col * m['pixel_um']:.0f} µm, y {row * m['pixel_um']:.0f} µm "
                  f"(full resolution)")
        photos = []
        panels = [("before", before), ("after", after),
                  ("difference", render_difference(diff, s.threshold, s.signed, after))]
        for i, (label, arr) in enumerate(panels):
            tk.Label(top, text=label.upper()).grid(row=0, column=i)
            photo = ImageTk.PhotoImage(Image.fromarray(arr))
            photos.append(photo)
            tk.Label(top, image=photo).grid(row=1, column=i, padx=4, pady=4)
        top.photos = photos
        top.bind("<Escape>", lambda e: top.destroy())
        self._update_status()
```

- [ ] **Step 4: Manual check (uses `/tmp/pcmp/cache` from Task 13)**

```bash
.venv/bin/python -c "from profilometer_comparison.viewer import run_viewer; run_viewer('/tmp/pcmp/cache')"
```
Check each:
1. `b` switches to one large panel alternating BEFORE/AFTER; `b` again returns to three panels.
2. On M1, Shift+→ three times in blink mode: the AFTER image visibly moves right; after the pause the status says "(manual)", the shift is −4.1 µm in x, and the difference panel shows a band of change. `a` returns to automatic (shift 0.0).
3. Shift+Option+→ moves 10 px at once.
4. `m`, click four points on BEFORE, Enter: a yellow outline appears on all panels. Backspace removes the last pending point. `m`: status shows an "encrustation" line; `/tmp/pcmp/cache/masks.json` contains the polygon; `summary.csv` has an `encrustation` row for this sample.
5. `m`, hover inside the polygon, Delete, `m`: the polygon and the encrustation line are gone.
6. A bow-tie polygon + Enter shows "polygon edges cross each other".
7. Clicking a panel (not in mask mode) opens a full-resolution zoom window of the same spot; Escape closes it.

- [ ] **Step 5: Run all tests and commit**

```bash
.venv/bin/pytest -q
git add profilometer_comparison/viewer.py
git commit -m "Add blink view, manual alignment, mask drawing and zoom to viewer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Plain side-by-side viewer (any image type)

**Files:**
- Create: `profilometer_comparison/plain_viewer.py`

**Interfaces:**
- Consumes: `pairing.pair_folders`, `flags.FlagStore`. Thumbnails are loaded with PIL directly (JPEG draft mode) for speed; the flag key is the filename stem.
- Produces: `run_plain_viewer(before_dir, after_dir, flags_path) -> None`.

- [ ] **Step 1: Implement `profilometer_comparison/plain_viewer.py`**

```python
"""Side-by-side before/after browsing for any image type (no alignment or statistics)."""
from __future__ import annotations

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from .flags import FlagStore
from .pairing import pair_folders

Image.MAX_IMAGE_PIXELS = None
FIELDS = ["sample", "file", "flagged_at"]


class PlainViewer:
    def __init__(self, root: tk.Tk, before_dir: Path, after_dir: Path, flags_path: Path):
        self.root = root
        pairing = pair_folders(before_dir, after_dir)
        if not pairing.pairs:
            raise ValueError("no files with the same name in both folders")
        self.pairs = pairing.pairs
        self.flags = FlagStore(flags_path, FIELDS)
        self.index = 0
        self.panel = max(300, min(900, (root.winfo_screenwidth() - 60) // 2))
        self.blinking = False
        self._blink_job = None
        self._blink_phase = 0
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._futures: dict[int, object] = {}
        root.title(f"profilometer-comparison: {Path(before_dir).name} vs {Path(after_dir).name}")
        frame = tk.Frame(root)
        frame.pack()
        self.labels = []
        for col, name in enumerate(("BEFORE", "AFTER")):
            tk.Label(frame, text=name, font=("TkDefaultFont", 12, "bold")).grid(row=0, column=col)
            label = tk.Label(frame, width=self.panel, height=self.panel, bg="white")
            label.grid(row=1, column=col, padx=4, pady=4)
            self.labels.append(label)
        self.status = tk.Label(root, anchor="w", font=("TkFixedFont", 12))
        self.status.pack(fill=tk.X, padx=8)
        tk.Label(root, anchor="w", fg="grey40", text="← → pair   s flag   b blink").pack(fill=tk.X, padx=8)
        root.bind("<Left>", lambda e: self._go(-1))
        root.bind("<Right>", lambda e: self._go(1))
        root.bind("s", lambda e: self._toggle_flag())
        root.bind("b", lambda e: self._toggle_blink())
        self.show()

    def _load(self, i: int):
        def thumb(path: Path) -> Image.Image:
            img = Image.open(path)
            img.draft("RGB", (self.panel, self.panel))  # fast JPEG downscale
            img = img.convert("RGB")
            img.thumbnail((self.panel, self.panel))
            return img
        pair = self.pairs[i]
        return thumb(pair.before), thumb(pair.after)

    def _get(self, i: int):
        if i not in self._futures:
            self._futures[i] = self._pool.submit(self._load, i)
        return self._futures[i].result()

    def _prefetch(self) -> None:
        for i in (self.index + 1, self.index - 1):
            if 0 <= i < len(self.pairs) and i not in self._futures:
                self._futures[i] = self._pool.submit(self._load, i)
        for i in [k for k in self._futures if abs(k - self.index) > 2]:
            del self._futures[i]

    def show(self) -> None:
        before, after = self._get(self.index)
        self._photos = [ImageTk.PhotoImage(before), ImageTk.PhotoImage(after)]
        if self.blinking:
            photo = self._photos[self._blink_phase]
            self.labels[0].config(image=photo, width=self.panel, height=self.panel)
            self.labels[1].config(image="", width=self.panel, height=self.panel)
        else:
            for label, photo in zip(self.labels, self._photos):
                label.config(image=photo, width=self.panel, height=self.panel)
        pair = self.pairs[self.index]
        flag = "  ★ FLAGGED" if Path(pair.name).stem in self.flags else ""
        blink = f"  [blink: {'AFTER' if self._blink_phase else 'BEFORE'}]" if self.blinking else ""
        self.status.config(text=f"{pair.name}  ({self.index + 1}/{len(self.pairs)}){flag}{blink}")
        self._prefetch()

    def _go(self, step: int) -> None:
        self.index = max(0, min(self.index + step, len(self.pairs) - 1))
        self.show()

    def _toggle_flag(self) -> None:
        pair = self.pairs[self.index]
        self.flags.toggle(Path(pair.name).stem, {"file": pair.name})
        self.show()

    def _toggle_blink(self) -> None:
        self.blinking = not self.blinking
        if self.blinking:
            self._blink_tick()
        else:
            if self._blink_job:
                self.root.after_cancel(self._blink_job)
            self._blink_phase = 0
            self.show()

    def _blink_tick(self) -> None:
        self._blink_phase ^= 1
        self.show()
        self._blink_job = self.root.after(500, self._blink_tick)


def run_plain_viewer(before_dir, after_dir, flags_path) -> None:
    root = tk.Tk()
    try:
        PlainViewer(root, Path(before_dir), Path(after_dir), Path(flags_path))
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Cannot open folders", str(exc))
        root.destroy()
        return
    root.mainloop()
```

- [ ] **Step 2: Manual check with the 3D views**

```bash
MOPA="$HOME/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA"
.venv/bin/python -c "
from profilometer_comparison.plain_viewer import run_plain_viewer
run_plain_viewer('$MOPA/3D-views', '$MOPA/3D-views', '/tmp/pcmp/flags-3d.csv')
"
```
Check: two panels show M1's 3D view; → goes to M2 (after the first load, the next pair appears quickly because it was prefetched); `s` writes `/tmp/pcmp/flags-3d.csv` with `M1-3D-view`; `b` alternates BEFORE/AFTER in the left panel.

- [ ] **Step 3: Run all tests and commit**

```bash
.venv/bin/pytest -q
git add profilometer_comparison/plain_viewer.py
git commit -m "Add plain side-by-side viewer for any image type

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Command-line interface

**Files:**
- Create: `profilometer_comparison/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `prepare.Settings, run_prepare, detect_type, SettingsMismatch, TYPES`; `colourmap.build_from_studiable, save_csv`; `viewer.run_viewer`; `plain_viewer.run_plain_viewer`.
- Produces: `main(argv=None) -> int`; commands `prepare`, `view`, `make-colourmap`.

- [ ] **Step 1: Write the failing tests `tests/test_cli.py`**

```python
import csv
import shutil

import pytest

from profilometer_comparison.__main__ import main


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "0.1.0" in capsys.readouterr().out


def test_prepare_command(tmp_path, data_dir, capsys):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    shutil.copy(data_dir / "pc_M35.jpg", before / "M35-pseudo-colour-image.jpg")
    shutil.copy(data_dir / "pc_M35_highres.jpg", after / "M35-pseudo-colour-image.jpg")
    out = tmp_path / "cmp"
    assert main(["prepare", "--before", str(before), "--after", str(after), "--out", str(out),
                 "--thresholds", "2", "5"]) == 0
    header = next(csv.reader(open(out / "summary.csv", encoding="utf-8")))
    assert "pct_diff_2" in header and "pct_diff_1" not in header
    assert "view" in capsys.readouterr().out


def test_make_colourmap_command(tmp_path, data_dir):
    out = tmp_path / "cm.csv"
    # The test crop contains only the bar, so search from the left edge.
    assert main(["make-colourmap", str(data_dir / "colourbar.png"), "--out", str(out), "--search-from", "0"]) == 0
    assert len(out.read_text().splitlines()) > 350
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'profilometer_comparison.__main__'`.

- [ ] **Step 3: Implement `profilometer_comparison/__main__.py`**

```python
"""Command-line interface: python -m profilometer_comparison {prepare,view,make-colourmap}."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import __version__
from .colourmap import build_from_studiable, save_csv
from .images import read_rgb
from .prepare import TYPES, Settings, SettingsMismatch, detect_type, run_prepare


def _ask_dir(title: str) -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    if not path:
        raise SystemExit("No folder selected.")
    return Path(path)


def cmd_prepare(args) -> int:
    before = args.before or _ask_dir("Select the BEFORE folder")
    after = args.after or _ask_dir("Select the AFTER folder")
    image_type = args.type or detect_type(before)
    if image_type is None:
        raise SystemExit(f"Could not detect the image type in {before}; use --type {{{','.join(TYPES)}}}.")
    settings = Settings(type=image_type, pixel_um=args.pixel_um, scan_length_um=args.scan_length_um,
                        thresholds=tuple(args.thresholds) if args.thresholds else None,
                        plane_correction=not args.no_plane_correction, colourmap=args.colourmap)
    out = args.out or after.parent / f"comparison-{image_type}-{date.today():%Y-%m-%d}"
    print(f"Comparing {image_type}:\n  before: {before}\n  after:  {after}\n  output: {out}")
    try:
        run_prepare(before, after, out, settings, masks_file=args.masks)
    except SettingsMismatch as exc:
        raise SystemExit(str(exc))
    print(f"\nDone. Results: {out / 'summary.csv'}\nOpen the viewer with:\n"
          f"  python -m profilometer_comparison view \"{out}\"")
    return 0


def cmd_view(args) -> int:
    if args.before or args.after:
        from .plain_viewer import run_plain_viewer

        before = args.before or _ask_dir("Select the BEFORE folder")
        after = args.after or _ask_dir("Select the AFTER folder")
        flags = args.flags or Path.cwd() / f"flagged-{after.name}.csv"
        print(f"Flags are saved to {flags}")
        run_plain_viewer(before, after, flags)
        return 0
    from .viewer import run_viewer

    run_viewer(args.cache or _ask_dir("Select a comparison folder (made by 'prepare')"))
    return 0


def cmd_make_colourmap(args) -> int:
    cmap = build_from_studiable(read_rgb(args.image), args.z_min, args.z_max, args.search_from)
    save_csv(cmap, args.out)
    print(f"{len(cmap.height)} colours, {cmap.height.min():.1f} to {cmap.height.max():.1f} µm -> {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="profilometer-comparison",
                                     description="Compare before/after profilometer exports.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="align and compare all before/after pairs (stage 1)")
    p.add_argument("--before", type=Path, help="folder with before images (asked if omitted)")
    p.add_argument("--after", type=Path, help="folder with after images (asked if omitted)")
    p.add_argument("--type", choices=sorted(TYPES), help="image type (detected from filenames if omitted)")
    p.add_argument("--out", type=Path, help="output folder (default: next to the after folder)")
    p.add_argument("--masks", type=Path, help="masks.json to reuse (e.g. from another image type)")
    p.add_argument("--thresholds", type=float, nargs="+", help="thresholds for summary.csv")
    p.add_argument("--no-plane-correction", action="store_true", help="do not remove offset/tilt (height only)")
    p.add_argument("--pixel-um", type=float, default=1.35, help="pseudo-colour pixel size in µm (default 1.35)")
    p.add_argument("--scan-length-um", type=float, default=10674.0,
                   help="true-colour scan width in µm (default 10674)")
    p.add_argument("--colourmap", default="default", help="colour map CSV (default: MarSurf 0-350 µm)")
    p.set_defaults(func=cmd_prepare)

    v = sub.add_parser("view", help="browse a comparison folder (stage 2), or two folders side by side")
    v.add_argument("cache", nargs="?", type=Path, help="comparison folder made by 'prepare'")
    v.add_argument("--before", type=Path, help="plain mode: before folder (any image type)")
    v.add_argument("--after", type=Path, help="plain mode: after folder")
    v.add_argument("--flags", type=Path, help="plain mode: CSV for flagged samples")
    v.set_defaults(func=cmd_view)

    c = sub.add_parser("make-colourmap", help="build a colour map CSV from a '-studiable' export")
    c.add_argument("image", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--z-min", type=float, default=0.0)
    c.add_argument("--z-max", type=float, default=350.0)
    c.add_argument("--search-from", type=float, default=0.75,
                   help="fraction of the image width where the search for the colour bar starts")
    c.set_defaults(func=cmd_make_colourmap)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest -v`
Expected: all pass.

- [ ] **Step 5: Manual check of the folder pickers**

Run: `.venv/bin/python -m profilometer_comparison prepare --out /tmp/pcmp/cli-test`
Expected: a native dialog asks for the BEFORE folder, then the AFTER folder (choose `/tmp/pcmp/before` and `/tmp/pcmp/after`); the run completes and prints the `view` command. Then run `.venv/bin/python -m profilometer_comparison view` and choose `/tmp/pcmp/cli-test`: the viewer opens.

- [ ] **Step 6: Commit**

```bash
git add profilometer_comparison/__main__.py tests/test_cli.py
git commit -m "Add command-line interface with folder pickers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17: Validation scripts and README

**Files:**
- Create: `scripts/robustness.py`, `scripts/noise_floor.py`, `README.md`

**Interfaces:**
- Consumes: `prepare.Settings, load_image, load_colourmap, run_prepare`; `align.estimate_shift`.

- [ ] **Step 1: Write `scripts/robustness.py`**

```python
"""Alignment robustness on a full-resolution real scan.

Shifts the scan by a known amount, replaces a growing fraction of it by a changed
surface (patches 8 µm lower, smoothed), and reports whether the shift is recovered.

Usage: python scripts/robustness.py path/to/M35-pseudo-colour-image.jpg
"""
import sys

import numpy as np

from profilometer_comparison.align import estimate_shift
from profilometer_comparison.prepare import Settings, load_colourmap, load_image

TILE = 500
SHIFT = (150, -230)


def main(path: str) -> None:
    settings = Settings(type="pseudo-colour")
    height = load_image(path, settings, load_colourmap(settings)).align
    m = 400
    before = height[m:-m, m:-m]
    after = height[m - SHIFT[0]:height.shape[0] - m - SHIFT[0], m - SHIFT[1]:height.shape[1] - m - SHIFT[1]].copy()
    rng = np.random.default_rng(0)
    ny, nx = after.shape[0] // TILE, after.shape[1] // TILE
    print(f"true shift {SHIFT}; tiles of {TILE} px altered")
    print("fraction altered | recovered shift | correct | confidence")
    for fraction in (0.0, 0.1, 0.3, 0.5, 0.7, 0.85):
        altered = after.copy()
        level = np.nanmean(after)
        for k in rng.choice(ny * nx, int(round(fraction * ny * nx)), replace=False):
            i, j = divmod(int(k), nx)
            altered[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE] = level - 8 + rng.normal(0, 1, (TILE, TILE))
        al = estimate_shift(before, altered)
        ok = (al.dy, al.dx) == SHIFT
        print(f"{fraction:16.0%} | {str((al.dy, al.dx)):15s} | {'yes' if ok else 'NO':7s} | {al.confidence:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Write `scripts/noise_floor.py`**

```python
"""Noise floor: compare scans of the same untreated surface (e.g. the high-res rescans).

Usage: python scripts/noise_floor.py IMAGE_ONLY_FOLDER OUT_FOLDER M35 M45 M53 M105
Pairs '<ID>-pseudo-colour-image.jpg' (before) with '<ID>-high-res-pseudo-colour-image.jpg' (after).
"""
import csv
import sys
from pathlib import Path

from profilometer_comparison.prepare import Settings, run_prepare


def main(folder: str, out: str, samples: list[str]) -> None:
    folder, out = Path(folder), Path(out)
    before, after = out / "before", out / "after"
    before.mkdir(parents=True, exist_ok=True)
    after.mkdir(parents=True, exist_ok=True)
    for s in samples:
        name = f"{s}-pseudo-colour-image.jpg"
        for link, target in ((before / name, folder / name),
                             (after / name, folder / f"{s}-high-res-pseudo-colour-image.jpg")):
            if not link.exists():
                link.symlink_to(target)
    cache = run_prepare(before, after, out / "comparison", Settings(type="pseudo-colour"))
    with open(cache / "summary.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["region"] == "all"]
    print("\nsample | shift x/y µm | % > 1 µm | % > 2 µm | % > 5 µm | % > 10 µm")
    for r in rows:
        print(f"{r['sample']:6s} | {r['shift_x_um']:>5s}/{r['shift_y_um']:<5s} | {r['pct_diff_1']:>8s} | "
              f"{r['pct_diff_2']:>8s} | {r['pct_diff_5']:>8s} | {r['pct_diff_10']:>9s}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
```

- [ ] **Step 3: Run both on the real data and keep the output for the README**

```bash
IMG="$HOME/Desktop/Conservation and Restoration/APP/APP 2/Research-paper/Profilometry/before-measurements/MOPA/pseudo-colour-view/image-only"
.venv/bin/python scripts/robustness.py "$IMG/M35-pseudo-colour-image.jpg" | tee /tmp/pcmp/robustness.txt
.venv/bin/python scripts/noise_floor.py "$IMG" /tmp/pcmp/noise M35 M45 M53 M105 | tee /tmp/pcmp/noise.txt
```
Expected: robustness correct up to at least 0.5 (planning prototype: up to 0.85); noise floor for M35 close to 9.4 / 0.22 / 0.01 / 0.006 %.

- [ ] **Step 4: Write `README.md`**

Paste the two result tables from Step 3 where marked. Content:

````markdown
# profilometer-comparison

Compare **before and after** surface measurements of the same samples, exported from MarSurf MfM / MountainsMap. Developed to assess whether laser cleaning (MOPA fibre laser) of glazed ceramic tiles damages the glaze.

For every sample, the tool aligns the before and after measurement, computes where and how much the surface changed, and reports the percentage of changed pixels for the **glaze** and, where you mark it, for **encrustation** separately. A keyboard-driven viewer lets you browse all samples, blink between before and after, draw encrustation masks, correct the alignment by hand, and flag samples.

## Installation

Requires Python 3.10 or newer with tkinter (included in the python.org installers).

```bash
git clone https://github.com/<your-account>/profilometer-comparison.git
cd profilometer-comparison
python3 -m venv .venv
.venv/bin/pip install -e .
```

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
.venv/bin/python -m profilometer_comparison prepare
```

Folder dialogs ask for the before and after folders. Or give everything on the command line:

```bash
.venv/bin/python -m profilometer_comparison prepare --before BEFORE --after AFTER --out RESULTS
```

This writes a results folder with `summary.csv` (one row per sample and region), `prepare-log.txt`, and a cache for the viewer. An interrupted run continues where it stopped.

### 2. View

```bash
.venv/bin/python -m profilometer_comparison view RESULTS
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
| click | full-resolution zoom of that spot |

Masks are stored in µm, so a `masks.json` made for pseudo-colour can be reused for true colour: `prepare --masks RESULTS/masks.json ...`.

### Plain view (any image type)

```bash
.venv/bin/python -m profilometer_comparison view --before BEFORE --after AFTER
```

## Method

1. **Height from pseudo-colour.** The colour scale of the export (0–350 µm) is stored as a lookup table of 393 colours, about 0.9 µm per step. Each pixel gets the height of the nearest scale colour. Pixels far from any scale colour are excluded (`excluded_pct`), and pixels at the ends of the scale are counted as saturated (`saturated_pct`). Expected accuracy is about 1–2 µm, limited by 8-bit colour and JPG compression. For another scale, run `make-colourmap` on a `-studiable` export.
2. **Colour.** sRGB is converted to CIELAB (D65) and compared as ΔE\*ab (CIE76). `mean_dL` shows a global brightness change, for example from lighting.
3. **Alignment.** Translation only, since samples sit in a fixture. Phase correlation runs on 4× downsampled images and is then refined at full resolution. It is repeated using only pixels that did not change much and are not masked as encrustation, so cleaned areas do not pull the alignment. `align_confidence` is the peak-to-sidelobe ratio of the correlation. Unrelated surfaces give 7–8, and below 20 the sample is marked "low confidence".
4. **Offset and tilt.** Each scan is levelled on its own. A plane (offset + tilt), fitted on the glaze pixels, is therefore subtracted from the height difference. **Consequence:** a uniform loss of material over the whole field cannot be detected, only local changes. This can be switched off with `--no-plane-correction`.
5. **Percentages.** A pixel counts as changed when |after − before| ≥ threshold. Height changes are split into *lower* (material lost) and *higher*. Percentages are calculated from full-resolution histograms (bin 0.05) of the overlap area, per region: `all`, `glaze` (outside masks) and `encrustation` (inside masks).

## Validation

**Alignment robustness** (`scripts/robustness.py`): a real scan was shifted by (150, −230) px and a growing fraction of it was replaced by a changed surface.

<paste /tmp/pcmp/robustness.txt as a table>

**Noise floor** (`scripts/noise_floor.py`): four samples were measured twice without treatment (normal scan and high-res rescan). The percentages below are what the method reports when **nothing changed**. Thresholds should be chosen well above them.

<paste /tmp/pcmp/noise.txt as a table>

## Limitations

- Translation-only alignment; rotation is assumed negligible.
- Heights are reconstructed from exported colours, not read from the native measurement file.
- Uniform material loss over the whole field is removed by the plane correction (see Method 4).
- Encrustation masks are drawn by hand.

## How to cite

If you use this software, please cite it. GitHub shows the citation under "Cite this repository", generated from `CITATION.cff`. Please also cite the related paper (reference added on publication).

## Licence

[PolyForm Noncommercial License 1.0.0](LICENSE). Research, teaching and other non-commercial use is allowed; keep the copyright notice. Commercial use requires permission from the author. This is a source-available licence, not an OSI-approved open-source licence.
````

- [ ] **Step 5: Final full test run**

Run: `.venv/bin/pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts README.md
git commit -m "Add validation scripts and README with robustness and noise-floor results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
