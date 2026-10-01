"""Side-by-side before/after browsing for any image type (no alignment or statistics)."""
from __future__ import annotations

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from . import NAME
from .flags import FlagStore
from .pairing import pair_folders

Image.MAX_IMAGE_PIXELS = None
FIELDS = ["sample", "file", "flagged_at"]


class PlainViewer:
    def __init__(self, root: tk.Tk, before_dir: Path, after_dir: Path, flags_path: Path):
        self.root = root
        pairing = pair_folders(before_dir, after_dir)
        if not pairing.pairs:
            raise ValueError("no files with the same name in both folders")
        self.pairs = pairing.pairs
        self.flags = FlagStore(flags_path, FIELDS)
        self.index = 0
        self.panel = max(300, min(900, (root.winfo_screenwidth() - 60) // 2))
        self.blinking = False
        self._blink_job = None
        self._blink_phase = 0
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._futures: dict[int, object] = {}
        # Labels without an image measure width in characters; a blank image keeps them in pixels.
        self._blank = tk.PhotoImage(width=self.panel, height=self.panel)
        root.title(f"{NAME}: {Path(before_dir).name} vs {Path(after_dir).name}")
        frame = tk.Frame(root)
        frame.pack()
        self.labels = []
        for col, name in enumerate(("BEFORE", "AFTER")):
            tk.Label(frame, text=name, font=("TkDefaultFont", 12, "bold")).grid(row=0, column=col)
            label = tk.Label(frame, image=self._blank, bg="white")
            label.grid(row=1, column=col, padx=4, pady=4)
            self.labels.append(label)
        self.status = tk.Label(root, anchor="w", font=("TkFixedFont", 12))
        self.status.pack(fill=tk.X, padx=8)
        tk.Label(root, anchor="w", fg="grey40", text="← → pair   s flag   b blink").pack(fill=tk.X, padx=8)
        root.bind("<Left>", lambda e: self._go(-1))
        root.bind("<Right>", lambda e: self._go(1))
        root.bind("s", lambda e: self._toggle_flag())
        root.bind("b", lambda e: self._toggle_blink())
        self.show()

    def _load(self, i: int):
        def thumb(path: Path) -> Image.Image:
            img = Image.open(path)
            img.draft("RGB", (self.panel, self.panel))  # fast JPEG downscale
            img = img.convert("RGB")
            img.thumbnail((self.panel, self.panel))
            return img
        pair = self.pairs[i]
        return thumb(pair.before), thumb(pair.after)

    def _get(self, i: int):
        if i not in self._futures:
            self._futures[i] = self._pool.submit(self._load, i)
        return self._futures[i].result()

    def _prefetch(self) -> None:
        for i in (self.index + 1, self.index - 1):
            if 0 <= i < len(self.pairs) and i not in self._futures:
                self._futures[i] = self._pool.submit(self._load, i)
        for i in [k for k in self._futures if abs(k - self.index) > 2]:
            del self._futures[i]

    def show(self) -> None:
        before, after = self._get(self.index)
        self._photos = [ImageTk.PhotoImage(before), ImageTk.PhotoImage(after)]
        if self.blinking:
            self.labels[0].config(image=self._photos[self._blink_phase])
            self.labels[1].config(image=self._blank)
        else:
            for label, photo in zip(self.labels, self._photos):
                label.config(image=photo)
        pair = self.pairs[self.index]
        flag = "  ★ FLAGGED" if Path(pair.name).stem in self.flags else ""
        blink = f"  [blink: {'AFTER' if self._blink_phase else 'BEFORE'}]" if self.blinking else ""
        self.status.config(text=f"{pair.name}  ({self.index + 1}/{len(self.pairs)}){flag}{blink}")
        self._prefetch()

    def _go(self, step: int) -> None:
        self.index = max(0, min(self.index + step, len(self.pairs) - 1))
        self.show()

    def _toggle_flag(self) -> None:
        pair = self.pairs[self.index]
        self.flags.toggle(Path(pair.name).stem, {"file": pair.name})
        self.show()

    def _toggle_blink(self) -> None:
        self.blinking = not self.blinking
        if self.blinking:
            self._blink_tick()
        else:
            if self._blink_job:
                self.root.after_cancel(self._blink_job)
            self._blink_phase = 0
            self.show()

    def _blink_tick(self) -> None:
        self._blink_phase ^= 1
        self.show()
        self._blink_job = self.root.after(500, self._blink_tick)


def run_plain_viewer(before_dir, after_dir, flags_path) -> None:
    root = tk.Tk()
    try:
        PlainViewer(root, Path(before_dir), Path(after_dir), Path(flags_path))
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Cannot open folders", str(exc))
        root.destroy()
        return
    root.mainloop()
