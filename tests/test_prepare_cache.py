import csv
import json
import shutil

import pytest

from profilometer_comparison import __version__, prepare
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
    assert manifest["version"] == __version__ and manifest["settings"]["type"] == "pseudo-colour"


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
    b, a, d, origin = zoom_region(cache, "M35", (380, 380), size=200)
    assert b.shape == a.shape == (200, 200, 3)
    assert d.shape == (200, 200)
    assert origin == (280, 280)


def test_zoom_source_crops_and_clamps(folders, tmp_path):
    from profilometer_comparison.prepare import ZoomSource
    cache = run_prepare(*folders, tmp_path / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)
    src = ZoomSource(cache, "M35")
    b, a, d, origin, centre = src.region((380, 380), 150)
    assert b.shape == (150, 150, 3) and d.shape == (150, 150)
    assert origin == (305, 305) and centre == (380, 380)
    # near the edge the window is pushed inside the overlap and the centre moves with it
    b, a, d, origin, centre = src.region((0, 0), 300)
    assert origin == (src.overlap[0].start, src.overlap[1].start)
    assert centre == (origin[0] + 150, origin[1] + 150)
    # a window larger than the overlap is limited to the overlap
    b, *_ = src.region((380, 380), 2400)
    assert b.shape[0] <= 768 and b.shape[1] <= 768
