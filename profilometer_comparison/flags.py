"""Flagged samples, stored as a CSV that is rewritten on every change."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .pairing import natural_key


class FlagStore:
    def __init__(self, path: str | Path, fields: list[str]):
        self.path = Path(path)
        self.fields = list(fields)
        self.key_field = self.fields[0]
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.rows[row[self.key_field]] = row

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def __len__(self) -> int:
        return len(self.rows)

    def keys(self) -> list[str]:
        return sorted(self.rows, key=natural_key)

    def toggle(self, key: str, row: dict) -> bool:
        if key in self.rows:
            del self.rows[key]
            flagged = False
        else:
            self.rows[key] = {**row, self.key_field: key,
                              "flagged_at": datetime.now().isoformat(timespec="seconds")}
            flagged = True
        self._save()
        return flagged

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore")
            writer.writeheader()
            for key in self.keys():
                writer.writerow(self.rows[key])
        tmp.replace(self.path)
