import csv
import shutil

import pytest

from profilometer_comparison import __version__
from profilometer_comparison.__main__ import main


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"cops {__version__}"


def test_prepare_command(tmp_path, data_dir, capsys):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    shutil.copy(data_dir / "pc_M35.jpg", before / "M35-pseudo-colour-image.jpg")
    shutil.copy(data_dir / "pc_M35_highres.jpg", after / "M35-pseudo-colour-image.jpg")
    out = tmp_path / "cmp"
    assert main(["prepare", "--before", str(before), "--after", str(after), "--out", str(out),
                 "--thresholds", "2", "5"]) == 0
    with open(out / "summary.csv", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert "pct_diff_2" in header and "pct_diff_1" not in header
    assert "view" in capsys.readouterr().out


def test_make_colourmap_command(tmp_path, data_dir):
    out = tmp_path / "cm.csv"
    # The test crop contains only the bar, so search from the left edge.
    assert main(["make-colourmap", str(data_dir / "colourbar.png"), "--out", str(out), "--search-from", "0"]) == 0
    assert len(out.read_text().splitlines()) > 350


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    from profilometer_comparison import __main__ as cli
    path = tmp_path / "config" / "cops" / "last-folder.json"
    monkeypatch.setattr(cli, "STATE_FILE", path)
    return path


def test_start_dir_is_parent_of_last_chosen_folder(tmp_path, state_file):
    from profilometer_comparison.__main__ import _remember_folder, _start_dir
    chosen = tmp_path / "MOPA" / "pseudo-colour-view" / "image-only"
    chosen.mkdir(parents=True)
    _remember_folder(chosen)
    assert state_file.exists()
    assert _start_dir() == chosen.parent


def test_start_dir_falls_back_to_home(tmp_path, state_file):
    from pathlib import Path
    from profilometer_comparison.__main__ import _remember_folder, _start_dir
    assert _start_dir() == Path.home()               # nothing remembered yet
    _remember_folder(tmp_path / "gone" / "folder")    # parent does not exist
    assert _start_dir() == Path.home()
    state_file.write_text("not json")                 # damaged settings file
    assert _start_dir() == Path.home()
