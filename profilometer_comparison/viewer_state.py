"""Viewer logic without any GUI, so it can be tested headless."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from . import NAME, __version__
from .difference import percentages
from .flags import FlagStore
from .prepare import Settings, processed_samples, read_hists, read_manifest, read_meta

FLAG_FIELDS = ["sample", "file", "type", "pct_diff_glaze", "threshold", "shift_x_um", "shift_y_um",
               "align_method", "flagged_at"]
LOWER = (40, 90, 220)
HIGHER = (220, 50, 40)
UNCHANGED = (225, 225, 225)
NO_DATA = (255, 255, 255)
MARKED = (255, 0, 200)


REGION_LABELS = {"glaze": "Glaze (outside encrustation masks)", "encrustation": "Encrustation (inside masks)"}
WHOLE_SURFACE = "Whole surface"


def change_sentence(p: dict, threshold: float, signed: bool) -> str:
    """'x % of the area changed by ≥ t µm (y % lowered, z % raised)' for a percentages() result."""
    if signed:
        return (f"{p['diff']:.2f} % of the area changed by ≥ {threshold:g} µm "
                f"({p['lower']:.2f} % lowered, {p['higher']:.2f} % raised)")
    return f"{p['diff']:.2f} % of the area changed by ΔE ≥ {threshold:g}"


def rotation_text(meta: dict) -> str:
    """'rotation +0.42° (auto)': rotation of the after measurement relative to before (counter-clockwise +)."""
    return f"rotation {meta.get('rotation_deg', 0.0):+.2f}° ({meta.get('rotation_method', 'off')})"


def render_difference(diff: np.ndarray, threshold: float, signed: bool,
                      after_small: np.ndarray | None = None) -> np.ndarray:
    """Colour image of the difference: blue lower / red higher (height) or magenta over a grey after image (colour)."""
    d = diff.astype(np.float32)
    with np.errstate(invalid="ignore"):
        if signed:
            out = np.empty(d.shape + (3,), np.uint8)
            out[:] = UNCHANGED
            out[d <= -threshold] = LOWER
            out[d >= threshold] = HIGHER
        else:
            grey = after_small.astype(np.float32).mean(axis=2).astype(np.uint8)
            out = np.repeat(grey[..., None], 3, axis=2)
            out[d >= threshold] = MARKED
    out[np.isnan(d)] = NO_DATA
    return out


class ViewerState:
    def __init__(self, cache: str | Path):
        self.cache = Path(cache)
        self.manifest = read_manifest(self.cache)
        self.settings = Settings.from_dict(self.manifest["settings"])
        self.signed = self.settings.signed
        self.samples = processed_samples(self.cache)
        if not self.samples:
            raise ValueError(f"no processed samples in {self.cache}")
        self.flags = FlagStore(self.cache / "flagged.csv", FLAG_FIELDS)
        self.threshold = 2.0 if self.signed else 5.0
        self.flagged_only = False
        self.current = self.samples[0]
        self._meta: dict[str, dict] = {}
        self._hists: dict[str, dict] = {}
        state_file = self.cache / "state.json"
        if state_file.exists():
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            if saved.get("last_sample") in self.samples:
                self.current = saved["last_sample"]
            self.threshold = float(saved.get("threshold", self.threshold))

    def visible_samples(self) -> list[str]:
        if self.flagged_only:
            return [s for s in self.samples if s in self.flags]
        return self.samples

    def go(self, step: int) -> None:
        visible = self.visible_samples()
        if self.current in visible:
            i = visible.index(self.current) + step
        else:  # current sample was hidden by the filter: move to the nearest visible one
            order = self.samples.index(self.current)
            i = next((k for k, s in enumerate(visible) if self.samples.index(s) > order), len(visible) - 1)
        self.current = visible[max(0, min(i, len(visible) - 1))]

    def meta(self, sample: str | None = None) -> dict:
        sample = sample or self.current
        if sample not in self._meta:
            self._meta[sample] = read_meta(self.cache, sample)
        return self._meta[sample]

    def percentages(self, region: str = "glaze", sample: str | None = None) -> dict | None:
        sample = sample or self.current
        if sample not in self._hists:
            self._hists[sample] = read_hists(self.cache, sample)
        hist = self._hists[sample].get(region)
        return None if hist is None else percentages(hist, self.threshold, self.signed)

    def reload(self, sample: str) -> None:
        self._meta.pop(sample, None)
        self._hists.pop(sample, None)

    def toggle_flag(self) -> bool:
        m = self.meta()
        row = {"file": m["file"], "type": m["type"], "pct_diff_glaze": round(self.percentages()["diff"], 4),
               "threshold": self.threshold, "shift_x_um": m["shift_x_um"], "shift_y_um": m["shift_y_um"],
               "align_method": m["align_method"]}
        return self.flags.toggle(self.current, row)

    def toggle_flagged_only(self) -> bool:
        if not self.flagged_only and len(self.flags) == 0:
            return False
        self.flagged_only = not self.flagged_only
        if self.flagged_only and self.current not in self.flags:
            self.go(0)
        return self.flagged_only

    def set_threshold(self, value: float) -> None:
        self.threshold = max(0.5, round(float(value) * 2) / 2)

    def step_threshold(self, direction: int) -> None:
        self.set_threshold(self.threshold + 0.5 * direction)

    def save_position(self) -> None:
        (self.cache / "state.json").write_text(
            json.dumps({"last_sample": self.current, "threshold": self.threshold}), encoding="utf-8")

    def region_lines(self, sample: str | None = None) -> list[str]:
        """One line per region: 'Whole surface: …', or glaze and encrustation separately when masked."""
        sample = sample or self.current
        if self.percentages(region="encrustation", sample=sample) is None:
            p = self.percentages(region="glaze", sample=sample)
            return [f"{WHOLE_SURFACE}: {change_sentence(p, self.threshold, self.signed)}"]
        return [f"{label}: {change_sentence(self.percentages(region=region, sample=sample), self.threshold, self.signed)}"
                for region, label in REGION_LABELS.items()]

    def export_texts(self, sample: str | None = None) -> tuple[str, str]:
        """Title (one line per region) and info line for an exported figure."""
        sample = sample or self.current
        m = self.meta(sample)
        title = f"{sample} · {m['type']} · " + "\n".join(self.region_lines(sample))
        info = (f"{rotation_text(m)} · "
                f"shift {m['shift_x_um']:+.1f} / {m['shift_y_um']:+.1f} µm ({m['align_method']}) · "
                f"file {m['file']} · {NAME} {__version__} · {date.today():%Y-%m-%d}")
        return title, info
