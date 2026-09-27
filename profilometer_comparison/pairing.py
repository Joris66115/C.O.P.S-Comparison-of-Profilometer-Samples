"""Match before/after files by identical filename."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Pair:
    name: str
    before: Path
    after: Path


@dataclass
class PairingResult:
    pairs: list[Pair] = field(default_factory=list)
    before_only: list[str] = field(default_factory=list)
    after_only: list[str] = field(default_factory=list)


def natural_key(name: str) -> list:
    """Sort key that orders embedded numbers numerically (M2 before M10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_images(folder: str | Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in Path(folder).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def pair_folders(before_dir: str | Path, after_dir: str | Path) -> PairingResult:
    before = list_images(before_dir)
    after = list_images(after_dir)
    common = sorted(before.keys() & after.keys(), key=natural_key)
    return PairingResult(
        pairs=[Pair(name, before[name], after[name]) for name in common],
        before_only=sorted(before.keys() - after.keys(), key=natural_key),
        after_only=sorted(after.keys() - before.keys(), key=natural_key),
    )


def sample_id(filename: str, type_suffix: str | None = None) -> str:
    """'M17-pseudo-colour-image.jpg' -> 'M17' (strip the image-type suffix if present)."""
    stem = Path(filename).stem
    if type_suffix and stem.endswith("-" + type_suffix):
        return stem[: -len(type_suffix) - 1]
    return stem
