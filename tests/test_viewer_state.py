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
    with open(cache / "flagged.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
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


def test_export_texts(state):
    state.go(1)
    title, info = state.export_texts()
    assert title.startswith("M2 · pseudo-colour · threshold 2 µm · glaze ")
    assert "% different" in title and "lower" in title
    assert "shift" in info and "(auto)" in info and "profilometer-comparison 0.1.0" in info
