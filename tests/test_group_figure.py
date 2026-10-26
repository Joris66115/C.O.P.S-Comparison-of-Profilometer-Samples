import csv
import shutil

import numpy as np
import pytest
from PIL import Image

from profilometer_comparison.export import GroupColumn, render_group_export
from profilometer_comparison.groups import GroupStore
from profilometer_comparison.prepare import Settings, run_prepare, write_summary


def column(colour, w=300, h=280, heading="M1 · 1 s"):
    img = Image.new("RGB", (w, h), colour)
    diff = np.full((h, w, 3), 225, np.uint8)
    return GroupColumn(before=img, after=img.copy(), diff_rgb=diff, um_per_px=20.0, heading=heading,
                       lines=["1.00 % changed by ≥ 2 µm", "rotation +0.10° · shift +0.0/+0.0 µm"])


def test_render_group_export_places_columns_left_to_right():
    from profilometer_comparison.colourmap import default_colourmap
    cols = [column((200, 0, 0)), column((0, 200, 0), w=310, h=270), column((0, 0, 200))]
    out = render_group_export(cols, title="Group A", info="info", signed=True, threshold=2.0, cmap=default_colourmap())
    arr = np.asarray(out)
    xs = {}
    for name, colour in (("red", (200, 0, 0)), ("green", (0, 200, 0)), ("blue", (0, 0, 200))):
        _, x = np.nonzero((arr == colour).all(axis=2))
        xs[name] = x.mean()
    assert xs["red"] < xs["green"] < xs["blue"]
    assert out.width > 3 * 300


@pytest.fixture
def cache(tmp_path, data_dir):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    for sample in ("M1", "M2", "M7", "M8"):
        shutil.copy(data_dir / "pc_M35.jpg", before / f"{sample}-pseudo-colour-image.jpg")
        shutil.copy(data_dir / "pc_M35_highres.jpg", after / f"{sample}-pseudo-colour-image.jpg")
    return run_prepare(before, after, tmp_path / "cache", Settings(type="pseudo-colour"), log=lambda *_: None)


def test_summary_has_group_and_label(cache):
    GroupStore(cache / "groups.csv").assign("M7", "Settings A", "5 s")
    write_summary(cache)
    with open(cache / "summary.csv", encoding="utf-8") as fh:
        rows = {r["sample"]: r for r in csv.DictReader(fh) if r["region"] == "all"}
    assert (rows["M7"]["group"], rows["M7"]["label"]) == ("Settings A", "5 s")
    assert (rows["M1"]["group"], rows["M1"]["label"]) == ("", "")


def test_group_figure_from_cache(cache):
    from profilometer_comparison.group_figure import export_group
    from profilometer_comparison.viewer_state import ViewerState
    store = GroupStore(cache / "groups.csv")
    for sample, label in (("M8", "10 s"), ("M1", "1 s"), ("M7", "5 s"), ("M2", "2 s")):
        store.assign(sample, "Settings A/B", label)
    state = ViewerState(cache)
    path, members = export_group(state, "Settings A/B")
    assert [m for m, _ in members] == ["M1", "M2", "M7", "M8"]
    assert path.parent == cache / "exports" and path.name == "group-Settings-A-B-2um.jpg"
    img = Image.open(path)
    assert img.width > 4 * 300
