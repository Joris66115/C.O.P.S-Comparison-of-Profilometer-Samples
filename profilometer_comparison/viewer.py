"""Stage 2: keyboard-driven before/after/difference viewer."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from .masks import load_masks
from .viewer_state import ViewerState, render_difference

HELP = ("← → sample   s flag   ↑ ↓ threshold   f flagged only   b blink   m mask   "
        "Shift+arrows nudge (Shift+Alt: ×10)   a auto-align   click: zoom")


class Viewer:
    def __init__(self, root: tk.Tk, state: ViewerState):
        self.root, self.state = root, state
        self.panel = max(300, min(700, (root.winfo_screenwidth() - 80) // 3))
        self.blink_size = max(400, min(2 * self.panel, root.winfo_screenheight() - 220))
        self.blinking = False
        self._blink_job = None
        self._blink_phase = 0
        self.mask_mode = False
        self.points: list[tuple[float, float]] = []
        self.masks_dirty = False
        self.nudge = [0, 0]
        self._nudge_job = None
        self._photos: list = []
        self._build()
        self._bind_keys()
        self.show()

    # ---------- layout ----------
    def _build(self) -> None:
        self.root.title(f"profilometer-comparison: {self.state.cache.name}")
        self.panels = tk.Frame(self.root)
        self.panels.pack(side=tk.TOP)
        self.canvases: dict[str, tk.Canvas] = {}
        for col, name in enumerate(("before", "after", "difference")):
            tk.Label(self.panels, text=name.upper(), font=("TkDefaultFont", 12, "bold")).grid(row=0, column=col)
            canvas = tk.Canvas(self.panels, width=self.panel, height=self.panel, bg="white", highlightthickness=0)
            canvas.grid(row=1, column=col, padx=4, pady=4)
            canvas.bind("<Button-1>", lambda e, n=name: self._on_click(n, e))
            self.canvases[name] = canvas
        self.blink_canvas = tk.Canvas(self.root, width=self.blink_size, height=self.blink_size, bg="black",
                                      highlightthickness=0)

        self.controls = tk.Frame(self.root)
        self.controls.pack(side=tk.TOP, fill=tk.X, padx=8)
        unit = "µm" if self.state.signed else "ΔE"
        tk.Label(self.controls, text=f"Threshold ({unit})").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.state.threshold)
        tk.Scale(self.controls, from_=0.5, to=50, resolution=0.5, orient=tk.HORIZONTAL, length=320,
                 variable=self.threshold_var, command=self._on_slider, takefocus=0).pack(side=tk.LEFT)
        self.status = tk.Label(self.root, anchor="w", justify=tk.LEFT, font=("TkFixedFont", 12))
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8)
        tk.Label(self.root, anchor="w", fg="grey40", text=HELP).pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<Left>", lambda e: self._go(-1))
        r.bind("<Right>", lambda e: self._go(1))
        r.bind("<Up>", lambda e: self._step_threshold(1))
        r.bind("<Down>", lambda e: self._step_threshold(-1))
        r.bind("s", lambda e: self._toggle_flag())
        r.bind("f", lambda e: self._toggle_flagged_only())
        r.protocol("WM_DELETE_WINDOW", self._quit)

    # ---------- drawing ----------
    def show(self) -> None:
        s = self.state
        folder = s.cache / "pairs" / s.current
        self.meta = s.meta()
        self.before_img = Image.open(folder / "before.png").convert("RGB")
        self.after_img = Image.open(folder / "after.png").convert("RGB")
        self.diff = np.load(folder / "diff.npy")
        self.scale = self.panel / max(self.before_img.size)
        self.masks = load_masks(s.cache / "masks.json").get(s.current, [])
        self._render_panels()
        if self.blinking:
            self._render_blink()
        self._update_status()

    def _nudge_offset(self, scale: float) -> tuple[float, float]:
        step = self.meta["display_step"]
        return -self.nudge[1] / step * scale, -self.nudge[0] / step * scale

    def _render_panels(self) -> None:
        s = self.state
        w, h = self.before_img.size
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        diff_rgb = render_difference(self.diff, s.threshold, s.signed, np.asarray(self.after_img))
        images = {"before": (self.before_img, Image.BILINEAR),
                  "after": (self.after_img, Image.BILINEAR),
                  "difference": (Image.fromarray(diff_rgb), Image.NEAREST)}
        self._photos = []
        for name, (img, resample) in images.items():
            canvas = self.canvases[name]
            canvas.delete("all")
            photo = ImageTk.PhotoImage(img.resize(size, resample))
            self._photos.append(photo)
            ox, oy = self._nudge_offset(self.scale) if name == "after" else (0, 0)
            canvas.create_image(ox, oy, image=photo, anchor="nw")
            self._draw_polygons(canvas)

    def _draw_polygons(self, canvas: tk.Canvas) -> None:
        for poly in self.masks:
            coords = [c for p in poly for c in self._um_to_canvas(*p)]
            canvas.create_polygon(*coords, outline="yellow", fill="", width=2)
        if self.points:
            coords = [c for p in self.points for c in self._um_to_canvas(*p)]
            if len(self.points) > 1:
                canvas.create_line(*coords, fill="yellow", width=2)
            for x, y in zip(coords[::2], coords[1::2]):
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="yellow", outline="")

    def _update_status(self, extra: str = "") -> None:
        s, m = self.state, self.meta
        visible = s.visible_samples()
        position = f"{visible.index(s.current) + 1}/{len(visible)}" if s.current in visible else "-"
        flag = "  ★ FLAGGED" if s.current in s.flags else ""
        filt = "  [flagged only]" if s.flagged_only else ""
        manual = " (manual)" if m["align_method"] == "manual" else ""
        lines = [f"{s.current}  ({position}){flag}{filt}    shift {m['shift_x_um']:+.1f} / "
                 f"{m['shift_y_um']:+.1f} µm{manual}    threshold {s.threshold:g} {'µm' if s.signed else 'ΔE'}"]
        for region in ("glaze", "encrustation"):
            p = s.percentages(region=region)
            if p is None:
                continue
            if s.signed:
                lines.append(f"{region:13s} {p['diff']:6.2f} % different   "
                             f"({p['lower']:.2f} % lower, {p['higher']:.2f} % higher)")
            else:
                lines.append(f"{region:13s} {p['diff']:6.2f} % with ΔE ≥ {s.threshold:g}")
        if m["align_warning"]:
            lines.append(f"⚠ alignment: {m['align_warning']}")
        if self.mask_mode:
            lines.append("MASK MODE: click points on BEFORE · Enter close polygon · Backspace undo point · "
                         "Delete remove polygon under cursor · m/Esc done")
        if self.nudge != [0, 0]:
            lines.append(f"nudge {self.nudge[1]:+d} / {self.nudge[0]:+d} px (applied after a short pause)")
        if extra:
            lines.append(extra)
        self.status.config(text="\n".join(lines))

    # ---------- coordinates ----------
    def _canvas_to_um(self, x: float, y: float) -> tuple[float, float]:
        m = self.meta
        row = y / self.scale * m["display_step"] + m["before_origin_px"][0]
        col = x / self.scale * m["display_step"] + m["before_origin_px"][1]
        return col * m["pixel_um"], row * m["pixel_um"]

    def _um_to_canvas(self, x_um: float, y_um: float) -> tuple[float, float]:
        m = self.meta
        col = x_um / m["pixel_um"] - m["before_origin_px"][1]
        row = y_um / m["pixel_um"] - m["before_origin_px"][0]
        return col / m["display_step"] * self.scale, row / m["display_step"] * self.scale

    # ---------- actions ----------
    def _go(self, step: int) -> None:
        if self.mask_mode:
            self._update_status("Leave mask mode (m) before changing sample.")
            return
        self._flush_nudge()
        self.state.go(step)
        self.show()

    def _step_threshold(self, direction: int) -> None:
        self.state.step_threshold(direction)
        self.threshold_var.set(self.state.threshold)
        self._render_panels()
        self._update_status()

    def _on_slider(self, value) -> None:
        self.state.set_threshold(float(value))
        self._render_panels()
        self._update_status()
        self.root.focus_set()  # keep arrow keys for navigation

    def _toggle_flag(self) -> None:
        self.state.toggle_flag()
        self._update_status()

    def _toggle_flagged_only(self) -> None:
        if not self.state.flagged_only and len(self.state.flags) == 0:
            self._update_status("No flagged samples yet.")
            return
        self.state.toggle_flagged_only()
        self.show()

    def _on_click(self, name: str, event) -> None:
        pass  # replaced in Task 14 (mask points / zoom)

    def _flush_nudge(self) -> None:
        pass  # replaced in Task 14

    def _quit(self) -> None:
        self._flush_nudge()
        self.state.save_position()
        self.root.destroy()


def run_viewer(cache: str | Path) -> None:
    root = tk.Tk()
    try:
        state = ViewerState(cache)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Cannot open comparison", str(exc))
        root.destroy()
        return
    Viewer(root, state)
    root.mainloop()
