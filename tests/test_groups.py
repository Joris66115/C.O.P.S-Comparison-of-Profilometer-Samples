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
    assert store.groups_of("M7") == [("Settings A", "5 s")]
    assert store.groups_of("M99") == []

    reopened = GroupStore(path)
    assert reopened.members("Settings A") == store.members("Settings A")


def test_sample_in_several_groups_with_own_labels(tmp_path):
    path = tmp_path / "groups.csv"
    store = GroupStore(path)
    store.assign("M7", "Time series A", "5 s")
    store.assign("M7", "Fluence series", "0.8 J/cm²")  # added to a second group, kept in the first
    store.assign("M8", "Time series A", "10 s")
    assert store.groups_of("M7") == [("Fluence series", "0.8 J/cm²"), ("Time series A", "5 s")]
    assert store.members("Time series A") == [("M7", "5 s"), ("M8", "10 s")]
    assert store.members("Fluence series") == [("M7", "0.8 J/cm²")]
    store.assign("M7", "Time series A", "6 s")  # same group again: only the label changes
    assert store.groups_of("M7") == [("Fluence series", "0.8 J/cm²"), ("Time series A", "6 s")]
    assert GroupStore(path).groups_of("M7") == store.groups_of("M7")  # saved


def test_old_groups_file_still_reads(tmp_path):
    path = tmp_path / "groups.csv"
    path.write_text("group,sample,label\nSettings A,M1,1 s\nSettings A,M2,2 s\n", encoding="utf-8")
    assert GroupStore(path).members("Settings A") == [("M1", "1 s"), ("M2", "2 s")]


def test_remove_from_one_group_or_all(tmp_path):
    store = GroupStore(tmp_path / "groups.csv")
    store.assign("M1", "Settings A", "1 s")
    store.assign("M1", "Settings B", "2 s")
    store.assign("M1", "Settings C", "3 s")
    store.remove("M1", "Settings B")
    assert store.groups_of("M1") == [("Settings A", "1 s"), ("Settings C", "3 s")]
    store.remove("M1")  # no group given: from all groups
    assert store.groups_of("M1") == []
    assert GroupStore(tmp_path / "groups.csv").groups() == []


def test_names_are_trimmed_and_empty_group_refused(tmp_path):
    store = GroupStore(tmp_path / "groups.csv")
    store.assign("M1", "  Settings A ", " 1 s ")
    assert store.groups_of("M1") == [("Settings A", "1 s")]
    with pytest.raises(ValueError):
        store.assign("M2", "   ", "1 s")


def test_import_from_other_results(tmp_path):
    (tmp_path / "pseudo").mkdir()
    source = GroupStore(tmp_path / "pseudo" / "groups.csv")
    source.assign("M1", "Settings A", "1 s")
    source.assign("M8", "Settings A", "10 s")
    source.assign("M50", "Settings C", "2 s")  # not compared here yet: kept for later batches
    target = GroupStore(tmp_path / "groups.csv")
    target.assign("M1", "Settings A", "old label")  # imported labels win
    target.assign("M1", "Own group", "x")  # memberships not in the import are kept
    target.assign("M3", "Settings B", "5 s")
    assert target.import_from(tmp_path / "pseudo") == 3  # a results folder works
    assert target.members("Settings A") == [("M1", "1 s"), ("M8", "10 s")]
    assert target.groups_of("M1") == [("Own group", "x"), ("Settings A", "1 s")]
    assert target.groups_of("M3") == [("Settings B", "5 s")]
    assert target.groups_of("M50") == [("Settings C", "2 s")]
    assert GroupStore(tmp_path / "groups.csv").groups_of("M8") == [("Settings A", "10 s")]  # saved
    assert target.import_from(tmp_path / "pseudo" / "groups.csv") == 3  # the file itself works too


def test_import_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        GroupStore(tmp_path / "groups.csv").import_from(tmp_path / "nowhere")
