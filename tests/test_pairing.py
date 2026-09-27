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
