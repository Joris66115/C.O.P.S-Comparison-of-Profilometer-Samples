import numpy as np
import pytest
from PIL import Image

from profilometer_comparison.difference import percentages
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
    assert percentages(r.hists["all"], 2.0, signed=False)["diff"] == 0.0
    assert r.meta["mean_dL"] == pytest.approx(0.0, abs=1e-3)
