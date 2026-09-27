"""Noise floor: compare scans of the same untreated surface (e.g. the high-res rescans).

Usage: python scripts/noise_floor.py IMAGE_ONLY_FOLDER OUT_FOLDER M35 M45 M53 M105
Pairs '<ID>-pseudo-colour-image.jpg' (before) with '<ID>-high-res-pseudo-colour-image.jpg' (after).
"""
import csv
import sys
from pathlib import Path

from profilometer_comparison.prepare import Settings, run_prepare


def main(folder: str, out: str, samples: list[str]) -> None:
    folder, out = Path(folder), Path(out)
    before, after = out / "before", out / "after"
    before.mkdir(parents=True, exist_ok=True)
    after.mkdir(parents=True, exist_ok=True)
    for s in samples:
        name = f"{s}-pseudo-colour-image.jpg"
        for link, target in ((before / name, folder / name),
                             (after / name, folder / f"{s}-high-res-pseudo-colour-image.jpg")):
            if not link.exists():
                link.symlink_to(target)
    cache = run_prepare(before, after, out / "comparison", Settings(type="pseudo-colour"))
    with open(cache / "summary.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["region"] == "all"]
    print("\nsample | shift x/y µm | % > 1 µm | % > 2 µm | % > 5 µm | % > 10 µm")
    for r in rows:
        print(f"{r['sample']:6s} | {r['shift_x_um']:>5s}/{r['shift_y_um']:<5s} | {r['pct_diff_1']:>8s} | "
              f"{r['pct_diff_2']:>8s} | {r['pct_diff_5']:>8s} | {r['pct_diff_10']:>9s}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
