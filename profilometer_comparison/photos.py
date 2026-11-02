"""Photographs of the samples before and after treatment, shown next to the measurements.

The photo folders are remembered in RESULTS/photos.json. A photo belongs to a sample when its
name is the sample ID ('M1.jpg'), or starts with it ('M1-before.jpg', not 'M10.jpg').
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageOps

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WHICH = ("before", "after")
_ID = re.compile(r"^[a-z]+\d+(?!\d)")


def fit_photo(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """The photo scaled (keeping its aspect ratio) to fit within max_w × max_h."""
    scale = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)


def _index(folder: Path) -> dict[str, Path]:
    """Lower-case name -> photo; names are also indexed by their leading sample ID unless that name exists."""
    found: dict[str, Path] = {}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXTENSIONS and not p.name.startswith("."))
    for p in files:
        found.setdefault(p.stem.lower(), p)
    for p in files:
        m = _ID.match(p.stem.lower())
        if m:
            found.setdefault(m.group(0), p)
    return found


class PhotoStore:
    def __init__(self, cache: str | Path):
        self.path = Path(cache) / "photos.json"
        self.folders: dict[str, Path | None] = {"before": None, "after": None}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.folders = {w: Path(saved[w]) if saved.get(w) else None for w in WHICH}
        self._indexes: dict[str, dict[str, Path]] = {}

    @property
    def configured(self) -> bool:
        return any(self.folders.values())

    def set_folders(self, before: str | Path | None, after: str | Path | None) -> None:
        """Set the photo folders; None keeps the current folder."""
        for which, folder in zip(WHICH, (before, after)):
            if folder:
                self.folders[which] = Path(folder)
        self._indexes = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({w: str(f) if f else None for w, f in self.folders.items()}, indent=1),
                             encoding="utf-8")

    def find(self, sample: str, which: str) -> Path | None:
        folder = self.folders[which]
        if folder is None:
            return None
        if which not in self._indexes:
            try:
                self._indexes[which] = _index(folder)
            except OSError:  # folder moved or unavailable
                self._indexes[which] = {}
        index = self._indexes[which]
        key = sample.lower()
        m = _ID.match(key)
        return index.get(key) or (index.get(m.group(0)) if m else None)

    def load(self, sample: str, which: str, max_size: int) -> Image.Image | None:
        """The photo upright (camera orientation applied), longest side at most `max_size`; None if there is none."""
        path = self.find(sample, which)
        if path is None:
            return None
        with Image.open(path) as img:
            img.draft("RGB", (max_size, max_size))  # fast reduced JPEG decoding
            img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_size, max_size), Image.BILINEAR)
        return img
