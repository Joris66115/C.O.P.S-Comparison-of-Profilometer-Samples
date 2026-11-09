"""Sample groups (e.g. one laser setting at different irradiation times), stored in groups.csv.

One row per membership (group, sample, label); a sample may appear in several groups.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

FIELDS = ["group", "sample", "label"]
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def label_number(label: str) -> float | None:
    """First number in a label ('0.5 s' -> 0.5, '0,5 s' -> 0.5, '10 s' -> 10); None if there is none."""
    m = _NUMBER.search(label or "")
    return float(m.group(0).replace(",", ".")) if m else None


class GroupStore:
    """Group memberships: a sample can be in several groups, with its own free-text label in each.

    Rewritten on every change.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # (group, sample) -> {"group", "sample", "label"}, insertion order kept
        self.rows: dict[tuple[str, str], dict] = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("sample") and row.get("group"):
                        group, sample = row["group"].strip(), row["sample"].strip()
                        self.rows[(group, sample)] = {"group": group, "sample": sample,
                                                      "label": (row.get("label") or "").strip()}

    def import_from(self, source: str | Path) -> int:
        """Add the groups of another comparison (its results folder or groups.csv); returns the number of samples.

        Imported memberships are added (with the imported label if the sample is already in that group);
        the sample's other groups are kept. Samples that are not compared here (yet) are kept as well,
        so later batches are grouped automatically.
        """
        source = Path(source)
        if source.is_dir():
            source = source / "groups.csv"
        if not source.is_file():
            raise FileNotFoundError(f"no groups.csv found at {source}")
        imported = GroupStore(source).rows
        for key, row in imported.items():
            self.rows[key] = dict(row)
        self._save()
        return len({sample for _, sample in imported})

    def assign(self, sample: str, group: str, label: str) -> None:
        """Put the sample in the group with this label (only the label changes if it is already in it)."""
        group = group.strip()
        if not group:
            raise ValueError("a group needs a name")
        self.rows[(group, sample)] = {"group": group, "sample": sample, "label": label.strip()}
        self._save()

    def remove(self, sample: str, group: str | None = None) -> None:
        """Take the sample out of one group, or out of all its groups if no group is given."""
        keys = [k for k in self.rows if k[1] == sample and (group is None or k[0] == group)]
        for key in keys:
            del self.rows[key]
        if keys:
            self._save()

    def groups_of(self, sample: str) -> list[tuple[str, str]]:
        """(group, label) of every group the sample is in, sorted by group name."""
        return sorted(((row["group"], row["label"]) for (_, s), row in self.rows.items() if s == sample),
                      key=lambda gl: gl[0].lower())

    def groups(self) -> list[str]:
        return sorted({group for group, _ in self.rows}, key=str.lower)

    def members(self, group: str) -> list[tuple[str, str]]:
        """(sample, label) ordered by the number in the label; labels without a number last."""
        found = [(i, row) for i, row in enumerate(self.rows.values()) if row["group"] == group]

        def key(item):
            i, row = item
            n = label_number(row["label"])
            return (n is None, n if n is not None else 0.0, i)

        return [(row["sample"], row["label"]) for _, row in sorted(found, key=key)]

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            for group in self.groups():
                for sample, label in self.members(group):
                    writer.writerow({"group": group, "sample": sample, "label": label})
        tmp.replace(self.path)
