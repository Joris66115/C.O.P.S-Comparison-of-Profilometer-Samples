"""Sample groups (e.g. one laser setting at different irradiation times), stored in groups.csv."""
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
    """One group (and free-text label) per sample; rewritten on every change."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}  # sample -> {"group", "sample", "label"}, insertion order kept
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("sample") and row.get("group"):
                        self.rows[row["sample"].strip()] = {
                            "group": row["group"].strip(), "sample": row["sample"].strip(),
                            "label": (row.get("label") or "").strip()}

    def assign(self, sample: str, group: str, label: str) -> None:
        group = group.strip()
        if not group:
            raise ValueError("a group needs a name")
        self.rows.pop(sample, None)  # re-adding puts it at the end of the insertion order
        self.rows[sample] = {"group": group, "sample": sample, "label": label.strip()}
        self._save()

    def remove(self, sample: str) -> None:
        if self.rows.pop(sample, None) is not None:
            self._save()

    def group_of(self, sample: str) -> tuple[str, str] | None:
        row = self.rows.get(sample)
        return (row["group"], row["label"]) if row else None

    def groups(self) -> list[str]:
        return sorted({row["group"] for row in self.rows.values()}, key=str.lower)

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
