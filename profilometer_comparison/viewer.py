"""Stage 2: keyboard-driven before/after/difference viewer."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from . import NAME, prepare
from .export import Locator, _draw_polygons, format_length, region_summary, render_export, upscale_for_export
from .masks import load_masks, point_in_polygon, save_masks, validate_polygon
from .viewer_state import ViewerState, render_difference, rotation_text

ROTATION_STEP = 0.05  # degrees per [ or ] key press

HELP = ("← → sample   s flag   ↑ ↓ threshold   f flagged only   b blink   m mask   "
        "Shift+arrows nudge (Shift+Alt: ×10)   [ ] rotate   a auto-align   e export JPG   click: zoom")


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
        self.rot_nudge = 0.0  # extra rotation (degrees) previewed before it is applied
        self._nudge_job = None
        self._photos: list = []
        self._build()
        self._bind_keys()
        self.show()

    # ---------- layout ----------
    def _build(self) -> None:
        self.root.title(f"{NAME}: {self.state.cache.name}")
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
        r.bind("b", lambda e: self._toggle_blink())
        r.bind("m", lambda e: self._toggle_mask_mode())
        r.bind("a", lambda e: self._reset_alignment())
        r.bind("<bracketleft>", lambda e: self._rotate(-1))
        r.bind("<bracketright>", lambda e: self._rotate(1))
        r.bind("e", lambda e: self._export_overview())
        r.bind("<Return>", lambda e: self._close_polygon())
        r.bind("<BackSpace>", lambda e: self._undo_point())
        r.bind("<Delete>", lambda e: self._delete_polygon())
        r.bind("<Escape>", lambda e: self._leave_mask_mode() if self.mask_mode else None)
        # Arrow = direction the AFTER image moves on screen.
        moves = {"Left": (0, 1), "Right": (0, -1), "Up": (1, 0), "Down": (-1, 0)}
        for key, delta in moves.items():
            r.bind(f"<Shift-{key}>", lambda e, d=delta: self._nudge(d, 1))
            for modifier in ("Alt", "Option"):
                try:
                    r.bind(f"<Shift-{modifier}-{key}>", lambda e, d=delta: self._nudge(d, 10))
                except tk.TclError:
                    pass  # modifier name not known on this platform

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

    def _after_preview(self, img: Image.Image) -> Image.Image:
        """The after image with a pending manual rotation applied (preview only)."""
        if not self.rot_nudge:
            return img
        return img.rotate(self.rot_nudge, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    def _render_panels(self) -> None:
        s = self.state
        w, h = self.before_img.size
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        diff_rgb = render_difference(self.diff, s.threshold, s.signed, np.asarray(self.after_img))
        images = {"before": (self.before_img, Image.BILINEAR),
                  "after": (self._after_preview(self.after_img), Image.BILINEAR),
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
                 f"{m['shift_y_um']:+.1f} µm{manual}    {rotation_text(m)}    threshold {s.threshold:g} {'µm' if s.signed else 'ΔE'}"]
        lines += s.region_lines()
        if m["align_warning"]:
            lines.append(f"⚠ alignment: {m['align_warning']}")
        if self.mask_mode:
            lines.append("MASK MODE: click points on BEFORE · Enter close polygon · Backspace undo point · "
                         "Delete remove polygon under cursor · m/Esc done")
        if self.nudge != [0, 0]:
            lines.append(f"nudge {self.nudge[1]:+d} / {self.nudge[0]:+d} px (applied after a short pause)")
        if self.rot_nudge:
            lines.append(f"rotate after image {self.rot_nudge:+.2f}° (applied after a short pause)")
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

    # ---------- blink ----------
    def _toggle_blink(self) -> None:
        if self.mask_mode:
            return
        self.blinking = not self.blinking
        if self.blinking:
            self.panels.pack_forget()
            self.blink_canvas.pack(side=tk.TOP, before=self.controls)
            self._blink_tick()
        else:
            if self._blink_job:
                self.root.after_cancel(self._blink_job)
                self._blink_job = None
            self.blink_canvas.pack_forget()
            self.panels.pack(side=tk.TOP, before=self.controls)

    def _blink_tick(self) -> None:
        self._blink_phase ^= 1
        self._render_blink()
        self._blink_job = self.root.after(500, self._blink_tick)

    def _render_blink(self) -> None:
        showing_after = self._blink_phase == 1
        img = self._after_preview(self.after_img) if showing_after else self.before_img
        scale = self.blink_size / max(img.size)
        photo = ImageTk.PhotoImage(img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                                              Image.BILINEAR))
        self._blink_photo = photo
        ox, oy = self._nudge_offset(scale) if showing_after else (0, 0)
        c = self.blink_canvas
        c.delete("all")
        c.create_image(ox, oy, image=photo, anchor="nw")
        c.create_text(12, 12, anchor="nw", text="AFTER" if showing_after else "BEFORE",
                      fill="yellow", font=("TkDefaultFont", 16, "bold"))

    # ---------- manual alignment ----------
    def _nudge(self, delta, step: int) -> None:
        if self.mask_mode:
            return
        self.nudge[0] += delta[0] * step
        self.nudge[1] += delta[1] * step
        self._render_panels()
        if self.blinking:
            self._render_blink()
        self._update_status()
        if self._nudge_job:
            self.root.after_cancel(self._nudge_job)
        self._nudge_job = self.root.after(800, self._flush_nudge)

    def _rotate(self, direction: int) -> None:
        """Turn the after image by one step (+: counter-clockwise), applied after a short pause."""
        if self.mask_mode:
            return
        self.rot_nudge = round(self.rot_nudge + direction * ROTATION_STEP, 4)
        self._render_panels()
        if self.blinking:
            self._render_blink()
        self._update_status()
        if self._nudge_job:
            self.root.after_cancel(self._nudge_job)
        self._nudge_job = self.root.after(800, self._flush_nudge)

    def _flush_nudge(self) -> None:
        if self._nudge_job:
            self.root.after_cancel(self._nudge_job)
            self._nudge_job = None
        if self.nudge == [0, 0] and not self.rot_nudge:
            return
        kwargs = {}
        if self.nudge != [0, 0]:
            kwargs["manual"] = (self.meta["shift_y_px"] + self.nudge[0], self.meta["shift_x_px"] + self.nudge[1])
        if self.rot_nudge:
            kwargs["manual_rotation"] = round(self.meta.get("rotation_correction_deg", 0.0) + self.rot_nudge, 4)
        self.nudge, self.rot_nudge = [0, 0], 0.0
        self._recompute(**kwargs)

    def _reset_alignment(self) -> None:
        if self.mask_mode:
            return
        self.nudge, self.rot_nudge = [0, 0], 0.0
        self._recompute(manual=None, manual_rotation=None)

    def _recompute(self, **kwargs) -> None:
        self._update_status("Recomputing from the full-resolution originals…")
        self.root.update_idletasks()
        try:
            prepare.recompute_pair(self.state.cache, self.state.current, **kwargs)
        except Exception as exc:
            messagebox.showerror("Recompute failed", str(exc))
        self.state.reload(self.state.current)
        self.show()

    # ---------- masks ----------
    def _toggle_mask_mode(self) -> None:
        if self.mask_mode:
            self._leave_mask_mode()
            return
        if self.blinking:
            self._toggle_blink()
        self._flush_nudge()
        self.mask_mode = True
        self.points = []
        self._update_status()

    def _leave_mask_mode(self) -> None:
        self.mask_mode = False
        self.points = []
        if self.masks_dirty:
            self.masks_dirty = False
            path = self.state.cache / "masks.json"
            all_masks = load_masks(path)
            if self.masks:
                all_masks[self.state.current] = self.masks
            else:
                all_masks.pop(self.state.current, None)
            save_masks(path, all_masks)
            self._recompute(manual="keep", mask_changed=True)
        else:
            self._render_panels()
            self._update_status()

    def _close_polygon(self) -> None:
        if not self.mask_mode or not self.points:
            return
        try:
            validate_polygon(self.points)
        except ValueError as exc:
            messagebox.showwarning("Invalid polygon", str(exc))
            return
        self.masks.append(list(self.points))
        self.points = []
        self.masks_dirty = True
        self._render_panels()
        self._update_status()

    def _undo_point(self) -> None:
        if self.mask_mode and self.points:
            self.points.pop()
            self._render_panels()

    def _delete_polygon(self) -> None:
        if not self.mask_mode:
            return
        c = self.canvases["before"]
        point = self._canvas_to_um(c.winfo_pointerx() - c.winfo_rootx(), c.winfo_pointery() - c.winfo_rooty())
        for i, poly in enumerate(self.masks):
            if point_in_polygon(point, poly):
                del self.masks[i]
                self.masks_dirty = True
                self._render_panels()
                self._update_status()
                return

    # ---------- click: mask point or zoom ----------
    def _on_click(self, name: str, event) -> None:
        if self.mask_mode:
            if name == "before":
                self.points.append(self._canvas_to_um(event.x, event.y))
                self._render_panels()
            return
        self._zoom(event.x, event.y)

    def _zoom(self, x: float, y: float) -> None:
        m = self.meta
        row = int(y / self.scale * m["display_step"]) + m["before_origin_px"][0]
        col = int(x / self.scale * m["display_step"]) + m["before_origin_px"][1]
        self._update_status("Loading full resolution…")
        self.root.update_idletasks()
        try:
            ZoomWindow(self, (row, col))
        except Exception as exc:
            messagebox.showerror("Zoom failed", str(exc))
        self._update_status()

    # ---------- export ----------
    def _mask_pixels(self, origin_px, step: int) -> list:
        """Mask polygons in pixel coordinates of an image whose top-left is `origin_px` (before image)."""
        px = self.meta["pixel_um"]
        return [[((x / px - origin_px[1]) / step, (y / px - origin_px[0]) / step) for x, y in poly]
                for poly in self.masks]

    def _save_export(self, image: Image.Image, name: str) -> None:
        folder = self.state.cache / "exports"
        folder.mkdir(exist_ok=True)
        path = folder / name
        image.save(path, quality=95)
        self._update_status(f"Exported {path}")

    def _threshold_tag(self) -> str:
        return f"{self.state.threshold:g}um" if self.state.signed else f"dE{self.state.threshold:g}"

    def _export_overview(self) -> None:
        if self.mask_mode:
            return
        s, m = self.state, self.meta
        title, info = s.export_texts()
        step = m["display_step"]
        figure = render_export(
            self.before_img, self.after_img,
            render_difference(self.diff, s.threshold, s.signed, np.asarray(self.after_img)),
            um_per_px=m["pixel_um"] * step, title=title, info=info, signed=s.signed, threshold=s.threshold,
            cmap=prepare.load_colourmap(s.settings), polygons=self._mask_pixels(m["before_origin_px"], step))
        self._save_export(figure, f"{s.current}-overview-{self._threshold_tag()}.jpg")

    def _quit(self) -> None:
        self._flush_nudge()
        self.state.save_position()
        self.root.destroy()


class ZoomWindow:
    """Full-resolution before/after/difference around a point, with adjustable field of view."""

    def __init__(self, viewer: Viewer, centre: tuple[int, int]):
        self.viewer = viewer
        state = viewer.state
        self.sample = state.current
        self.masks = list(viewer.masks)
        self.source = prepare.ZoomSource(state.cache, self.sample)
        self.centre = centre
        self.size_index = prepare.ZOOM_SIZES.index(600)
        self.panel = max(250, min(600, (viewer.root.winfo_screenwidth() - 60) // 3))

        self.top = tk.Toplevel(viewer.root)
        self.images = []
        for i, name in enumerate(("BEFORE", "AFTER (aligned)", "DIFFERENCE")):
            tk.Label(self.top, text=name).grid(row=0, column=i)
            label = tk.Label(self.top)
            label.grid(row=1, column=i, padx=4, pady=4)
            self.images.append(label)
        self.info = tk.Label(self.top, anchor="w", justify=tk.LEFT, font=("TkFixedFont", 12))
        self.info.grid(row=2, column=0, columnspan=3, sticky="w", padx=6)
        tk.Label(self.top, anchor="w", fg="grey40",
                 text="+ / − or scroll: zoom   arrows: move   e: export JPG   Esc: close").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=6)
        tk.Button(self.top, text="Export JPG (e)", command=self.export).grid(row=4, column=0, columnspan=3, pady=4)

        t = self.top
        for key in ("<plus>", "<equal>", "<KP_Add>"):
            t.bind(key, lambda e: self.zoom(-1))
        for key in ("<minus>", "<underscore>", "<KP_Subtract>"):
            t.bind(key, lambda e: self.zoom(1))
        t.bind("<MouseWheel>", lambda e: self.zoom(-1 if e.delta > 0 else 1))
        t.bind("<Button-4>", lambda e: self.zoom(-1))
        t.bind("<Button-5>", lambda e: self.zoom(1))
        for key, (dr, dc) in {"Left": (0, -1), "Right": (0, 1), "Up": (-1, 0), "Down": (1, 0)}.items():
            t.bind(f"<{key}>", lambda e, d=(dr, dc): self.pan(*d))
        t.bind("e", lambda e: self.export())
        t.bind("<Escape>", lambda e: t.destroy())
        self.render()
        t.focus_force()

    @property
    def size(self) -> int:
        return prepare.ZOOM_SIZES[self.size_index]

    def zoom(self, direction: int) -> None:
        """direction -1: zoom in (smaller field of view), +1: zoom out."""
        self.size_index = max(0, min(self.size_index + direction, len(prepare.ZOOM_SIZES) - 1))
        self.render()

    def pan(self, d_row: int, d_col: int) -> None:
        step = self.size // 2
        self.centre = (self.centre[0] + d_row * step, self.centre[1] + d_col * step)
        self.render()

    def _mask_pixels(self, origin, scale: float) -> list:
        px = self.source.pixel_um
        return [[((x / px - origin[1]) * scale, (y / px - origin[0]) * scale) for x, y in poly]
                for poly in self.masks]

    def _locator(self, origin, shape) -> Locator:
        """Whole-sample overview (cached before image) with the zoom area marked."""
        meta, px = self.source.meta, self.source.pixel_um
        step = meta["display_step"]
        r0, c0 = meta["before_origin_px"]
        overview = Image.open(self.viewer.state.cache / "pairs" / self.sample / "before.png").convert("RGB")
        h, w = shape
        rect = ((origin[1] - c0) / step, (origin[0] - r0) / step,
                (origin[1] + w - c0) / step, (origin[0] + h - r0) / step)
        polygons = [[((x / px - c0) / step, (y / px - r0) / step) for x, y in poly] for poly in self.masks]
        return Locator(image=overview, um_per_px=px * step, rect=rect, polygons=polygons)

    def render(self) -> None:
        state = self.viewer.state
        before, after, diff, origin, self.centre = self.source.region(self.centre, self.size)
        self.region = (before, after, diff, origin)
        diff_rgb = render_difference(diff, state.threshold, state.signed, after)
        h, w = diff.shape
        scale = self.panel / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        resample = Image.NEAREST if scale >= 1 else Image.BOX
        self.photos = []
        for label, arr in zip(self.images, (before, after, diff_rgb)):
            img = Image.fromarray(np.ascontiguousarray(arr)).resize(size, resample)
            if self.masks:
                _draw_polygons(img, self._mask_pixels(origin, scale), 2)
            photo = ImageTk.PhotoImage(img)
            self.photos.append(photo)
            label.config(image=photo)
        px = self.source.pixel_um
        cx, cy = self.centre[1] * px, self.centre[0] * px
        field = f"{format_length(w * px)} × {format_length(h * px)}"
        self.top.title(f"{self.sample}: zoom at x {cx:.0f} µm, y {cy:.0f} µm, field of view {field}")
        self.info.config(text=f"field of view {field} ({w} × {h} px)   centre x {cx:.0f} µm, y {cy:.0f} µm\n"
                              f"This region: {region_summary(diff, state.threshold, state.signed)}")

    def export(self) -> None:
        viewer, state = self.viewer, self.viewer.state
        before, after, diff, origin = self.region
        images = [Image.fromarray(np.ascontiguousarray(a)) for a in
                  (before, after, render_difference(diff, state.threshold, state.signed, after))]
        images, factor = upscale_for_export(images)
        px = self.source.pixel_um
        cx, cy = self.centre[1] * px, self.centre[0] * px
        _, info = state.export_texts(self.sample)
        whole = [f"Whole sample, {line[0].lower()}{line[1:]}" for line in state.region_lines(self.sample)]
        title = "\n".join([f"{self.sample} · {state.settings.type} · zoom at x {cx:.0f} µm, y {cy:.0f} µm, "
                           f"field of view {format_length(diff.shape[1] * px)}",
                           f"This region: {region_summary(diff, state.threshold, state.signed)}"] + whole)
        figure = render_export(images[0], images[1], np.asarray(images[2]), um_per_px=px / factor, title=title,
                               info=info, signed=state.signed, threshold=state.threshold,
                               cmap=prepare.load_colourmap(state.settings),
                               polygons=self._mask_pixels(origin, factor), locator=self._locator(origin, diff.shape))
        name = f"{self.sample}-zoom-x{cx:.0f}um-y{cy:.0f}um-{self.size}px-{viewer._threshold_tag()}.jpg"
        viewer._save_export(figure, name)
        self.info.config(text=self.info.cget("text") + f"\nExported {name}")


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
