"""Command-line interface: python -m profilometer_comparison {prepare,view,make-colourmap}."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from . import FULL_NAME, __version__
from .colourmap import build_from_studiable, save_csv
from .images import read_rgb
from .prepare import TYPES, Settings, SettingsMismatch, detect_type, run_prepare

# Last folder chosen in a folder dialog, so the next dialog opens next to it.
STATE_FILE = Path.home() / ".config" / "cops" / "last-folder.json"


def _remember_folder(folder: Path) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"last_folder": str(folder)}), encoding="utf-8")
    except OSError:
        pass  # remembering is a convenience; never fail on it


def _start_dir() -> Path:
    """Folder one level above the last chosen folder, or the home folder if unknown or gone."""
    try:
        parent = Path(json.loads(STATE_FILE.read_text(encoding="utf-8"))["last_folder"]).parent
        if parent.is_dir():
            return parent
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return Path.home()


def _ask_dir(title: str) -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title, initialdir=str(_start_dir()))
    root.destroy()
    if not path:
        raise SystemExit("No folder selected.")
    _remember_folder(Path(path))
    return Path(path)


def cmd_prepare(args) -> int:
    before = args.before or _ask_dir("Select the BEFORE folder")
    after = args.after or _ask_dir("Select the AFTER folder")
    image_type = args.type or detect_type(before)
    if image_type is None:
        raise SystemExit(f"Could not detect the image type in {before}; use --type {{{','.join(TYPES)}}}.")
    settings = Settings(type=image_type, pixel_um=args.pixel_um, scan_length_um=args.scan_length_um,
                        thresholds=tuple(args.thresholds) if args.thresholds else None,
                        plane_correction=not args.no_plane_correction, colourmap=args.colourmap)
    out = args.out or after.parent / f"comparison-{image_type}-{date.today():%Y-%m-%d}"
    print(f"Comparing {image_type}:\n  before: {before}\n  after:  {after}\n  output: {out}")
    try:
        run_prepare(before, after, out, settings, masks_file=args.masks)
    except SettingsMismatch as exc:
        raise SystemExit(str(exc))
    script = Path(sys.argv[0])
    launcher = f'python3 "{script.resolve()}"' if script.name == "run.py" else "python -m profilometer_comparison"
    print(f"\nDone. Results: {out / 'summary.csv'}\nOpen the viewer with:\n  {launcher} view \"{out}\"")
    return 0


def cmd_view(args) -> int:
    if args.before or args.after:
        from .plain_viewer import run_plain_viewer

        before = args.before or _ask_dir("Select the BEFORE folder")
        after = args.after or _ask_dir("Select the AFTER folder")
        flags = args.flags or Path.cwd() / f"flagged-{after.name}.csv"
        print(f"Flags are saved to {flags}")
        run_plain_viewer(before, after, flags)
        return 0
    from .viewer import run_viewer

    run_viewer(args.cache or _ask_dir("Select a comparison folder (made by 'prepare')"))
    return 0


def cmd_make_colourmap(args) -> int:
    cmap = build_from_studiable(read_rgb(args.image), args.z_min, args.z_max, args.search_from)
    save_csv(cmap, args.out)
    print(f"{len(cmap.height)} colours, {cmap.height.min():.1f} to {cmap.height.max():.1f} µm -> {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cops",
                                     description=f"{FULL_NAME}: compare before/after profilometer exports.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="align and compare all before/after pairs (stage 1)")
    p.add_argument("--before", type=Path, help="folder with before images (asked if omitted)")
    p.add_argument("--after", type=Path, help="folder with after images (asked if omitted)")
    p.add_argument("--type", choices=sorted(TYPES), help="image type (detected from filenames if omitted)")
    p.add_argument("--out", type=Path, help="output folder (default: next to the after folder)")
    p.add_argument("--masks", type=Path, help="masks.json to reuse (e.g. from another image type)")
    p.add_argument("--thresholds", type=float, nargs="+", help="thresholds for summary.csv")
    p.add_argument("--no-plane-correction", action="store_true", help="do not remove offset/tilt (height only)")
    p.add_argument("--pixel-um", type=float, default=1.35, help="pseudo-colour pixel size in µm (default 1.35)")
    p.add_argument("--scan-length-um", type=float, default=10674.0,
                   help="true-colour scan width in µm (default 10674)")
    p.add_argument("--colourmap", default="default", help="colour map CSV (default: MarSurf 0-350 µm)")
    p.set_defaults(func=cmd_prepare)

    v = sub.add_parser("view", help="browse a comparison folder (stage 2), or two folders side by side")
    v.add_argument("cache", nargs="?", type=Path, help="comparison folder made by 'prepare'")
    v.add_argument("--before", type=Path, help="plain mode: before folder (any image type)")
    v.add_argument("--after", type=Path, help="plain mode: after folder")
    v.add_argument("--flags", type=Path, help="plain mode: CSV for flagged samples")
    v.set_defaults(func=cmd_view)

    c = sub.add_parser("make-colourmap", help="build a colour map CSV from a '-studiable' export")
    c.add_argument("image", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--z-min", type=float, default=0.0)
    c.add_argument("--z-max", type=float, default=350.0)
    c.add_argument("--search-from", type=float, default=0.75,
                   help="fraction of the image width where the search for the colour bar starts")
    c.set_defaults(func=cmd_make_colourmap)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
