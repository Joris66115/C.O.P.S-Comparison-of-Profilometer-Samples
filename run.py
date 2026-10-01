#!/usr/bin/env python3
"""Start C.O.P.S. (Comparison of Profilometer Samples) without installing it.

    python3 run.py              menu; folders are chosen with dialogs
    python3 run.py prepare ...  same commands and options as 'python -m profilometer_comparison'

Requires numpy and Pillow (python3 -m pip install numpy Pillow).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy  # noqa: F401
    import PIL  # noqa: F401
except ImportError:
    sys.exit("numpy and Pillow are needed. Install them with:\n  python3 -m pip install numpy Pillow")

from profilometer_comparison.__main__ import _ask_dir, main

MENU = """
C.O.P.S. (Comparison of Profilometer Samples)

  1  Prepare: align and compare a BEFORE and an AFTER folder
  2  View the results of a comparison
  3  Plain side-by-side view of two folders (any image type)
  q  Quit
"""


def menu() -> int:
    print(MENU)
    choice = input("Choose 1, 2, 3 or q: ").strip().lower()
    if choice == "1":
        return main(["prepare"])
    if choice == "2":
        return main(["view"])
    if choice == "3":
        before = _ask_dir("Select the BEFORE folder")
        after = _ask_dir("Select the AFTER folder")
        return main(["view", "--before", str(before), "--after", str(after)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]) if len(sys.argv) > 1 else menu())
