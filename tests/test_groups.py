import pytest

from profilometer_comparison.groups import GroupStore, label_number


@pytest.mark.parametrize("label, number", [
    ("1 s", 1.0), ("10 s", 10.0), ("0.5 s", 0.5), ("0,5 s", 0.5), ("2 × 5 s", 2.0), ("t=12s", 12.0), ("reference", None),
])
def test_label_number(label, number):
    assert label_number(label) == number


def test_assign_order_and_persist(tmp_path):
    path = tmp_path / "groups.csv"
    store = GroupStore(path)
    store.assign("M8", "Settings A", "10 s")
    store.assign("M1", "Settings A", "1 s")
    store.assign("M7", "Settings A", "5 s")
    store.assign("M2", "Settings A", "2 s")
    store.assign("M30", "Settings A", "reference")  # no number: goes last
    store.assign("M3", "Settings B", "0.5 s")
    assert store.members("Settings A") == [("M1", "1 s"), ("M2", "2 s"), ("M7", "5 s"), ("M8", "10 s"),
                                           ("M30", "reference")]
    assert store.groups() == ["Settings A", "Settings B"]
    assert store.group_of("M7") == ("Settings A", "5 s")
    assert store.group_of("M99") is None

    reopened = GroupStore(path)
    assert reopened.members("Settings A") == store.members("Settings A")


def test_reassign_moves_and_relabels(tmp_path):
    store = GroupStore(tmp_path / "groups.csv")
    store.assign("M1", "Settings A", "1 s")
    store.assign("M1", "Settings B", "3 s")  # moves to another group with another time
    assert store.members("Settings A") == []
    assert store.group_of("M1") == ("Settings B", "3 s")
    assert store.groups() == ["Settings B"]


def test_remove(tmp_path):
    store = GroupStore(tmp_path / "groups.csv")
    store.assign("M1", "Settings A", "1 s")
    store.remove("M1")
    assert store.group_of("M1") is None
    assert GroupStore(tmp_path / "groups.csv").groups() == []


def test_names_are_trimmed_and_empty_group_refused(tmp_path):
    store = GroupStore(tmp_path / "groups.csv")
    store.assign("M1", "  Settings A ", " 1 s ")
    assert store.group_of("M1") == ("Settings A", "1 s")
    with pytest.raises(ValueError):
        store.assign("M2", "   ", "1 s")


def test_import_from_other_results(tmp_path):
    (tmp_path / "pseudo").mkdir()
    source = GroupStore(tmp_path / "pseudo" / "groups.csv")
    source.assign("M1", "Settings A", "1 s")
    source.assign("M8", "Settings A", "10 s")
    source.assign("M50", "Settings C", "2 s")  # not compared here yet: kept for later batches
    target = GroupStore(tmp_path / "groups.csv")
    target.assign("M1", "Old name", "x")  # imported groups win
    target.assign("M3", "Settings B", "5 s")  # not in the import: left alone
    assert target.import_from(tmp_path / "pseudo") == 3  # a results folder works
    assert target.members("Settings A") == [("M1", "1 s"), ("M8", "10 s")]
    assert target.group_of("M3") == ("Settings B", "5 s")
    assert target.group_of("M50") == ("Settings C", "2 s")
    assert "Old name" not in target.groups()
    assert GroupStore(tmp_path / "groups.csv").group_of("M8") == ("Settings A", "10 s")  # saved
    assert target.import_from(tmp_path / "pseudo" / "groups.csv") == 3  # the file itself works too


def test_import_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        GroupStore(tmp_path / "groups.csv").import_from(tmp_path / "nowhere")
