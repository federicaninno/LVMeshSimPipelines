"""
Interactive slideshow + manual correction tool for LV cavity segmentations.

WHY THIS EXISTS
    Neither extract_lv_volume.py nor extract_lv_volume_conical.py let you
    touch mask voxels by hand. This script runs the sphere (v1) or conical
    (v2) segmentation exactly as those scripts do, then opens one
    interactive window to review and correct each cardiac phase's 3D mask:
        1. Phase slider (+ Play) to scrub/animate through the phases, and
           axial/coronal/sagittal sliders to move through every slice.
        2. BASE CUT: a line marking the valve plane on the panels that show
           the LV long axis (--axis). "Apply cut" erases everything base-ward
           of it in the whole 3D mask of the current phase — one click fixes
           a leak along the catheter / into the outflow tract, instead of
           erasing it slice by slice. The line position carries over when
           you move to the next phase, so you only nudge it and click again.
        3. "Keep largest" removes disconnected islands (3D).
        4. A 2D paint/erase brush for small touch-ups on the displayed slice.
        5. Undo (last 5 operations while on a phase) and "Reset phase".
        6. MYOCARDIUM as a second label (orange). "Auto myo" gives a first
           estimate around the current cavity (the wall is the dark band
           between cavity and tank fluid; see auto_myocardium()), "Auto myo:
           all empty" does every phase that has none yet. Pick "Cavity" or
           "Myocardium" to choose which label the brush, Keep largest and
           Reset act on; the two labels never overlap. "Apply cut" trims both.
           Myocardial volume should be roughly constant over the cycle — the
           CV printed at exit is a quick quality check.

    Recommended order per phase: base cut on the cavity -> Keep largest ->
    Auto myo -> brush touch-ups on either label.

    Errors this was built for (PEXA5): the mask leaking base-ward along the
    catheter and through the valve, plus small threshold islands. The cut +
    keep-largest handle those in a couple of clicks per phase.

MEMORY / SPEED
    Images are kept in memory as int16 (~140 MB per 512x512x269 phase,
    ~2.8 GB for 20 phases) so the slideshow doesn't reload from disk; masks
    as uint8. Use --phases to work on a subset. Display updates reuse the
    existing image artists (no full redraw), and while the brush is dragged
    only the panel being painted is refreshed.

SEGMENTATION CACHE
    The raw (pre-smoothing, pre-edit) masks are cached in
    <output>/segmentation_cache.nii.gz + .json the first time. On the next
    run with the same --folders, --method and --phases you're asked whether
    to reuse it instead of re-segmenting.

EDITS ARE SAVED AS YOU GO
    A phase's edited mask is written to <output>/edited_masks/ when you
    leave that phase, after every cut / keep-largest, and on exit. On the
    next launch with the same --output, saved edits are loaded back
    automatically ("Reset phase" returns a phase to the auto mask).

Usage:
    python mask_editor.py --method sphere --seed 260 280 144 \\
        --folders retro_A_frames_median3 retro_B_frames_median3 --output edit_median3

    Only some phases (percent of the R-R interval, 20 files -> 0, 5, ..., 95):
        python mask_editor.py --method sphere --seed 260 280 144 --phases 0 30 \\
            --folders retro_A_frames_median3 retro_B_frames_median3 --output edit_median3

Controls:
    - Mode "Navigate": click a panel to move the other two slices there.
    - Mode "Add"/"Erase": left-click-drag to paint/erase a disk on the
      displayed slice of the current phase only.
    - "Base cut" slider + "Apply cut", "Keep largest", "Undo",
      "Reset phase", "Save & Exit" (closing the window also saves).

Output (in --output):
    edited_masks/<folder>_frame_NN.nii.gz     - edited cavity mask per phase (only phases you changed)
    edited_masks/myo_<folder>_frame_NN.nii.gz - myocardium mask per phase (where one exists)
    volumes_edited.csv                    - auto vs. edited cavity volume + myocardial volume per phase
    volume_edited_plot.png                - auto vs. edited volume over the cycle
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import ndimage
from skimage.morphology import ball

from extract_lv_volume import (
    load_nifti, collect_frames, select_seed_interactive, estimate_threshold,
    compute_volume_ml, temporal_smooth_masks, segment_lv_cavity,
)
from extract_lv_volume_conical import segment_lv_cavity_conical


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

# Each panel fixes one voxel axis and displays the other two. Given the
# data[...].T + imshow(origin="lower") convention used throughout this
# codebase, the plotted x-axis always corresponds to the smaller-numbered
# of the two remaining voxel axes, and the plotted y-axis to the larger one.
PANELS = [
    {"axis_fixed": 2, "a_axis": 0, "b_axis": 1, "title": "Axial"},     # fixed z, vary x,y
    {"axis_fixed": 1, "a_axis": 0, "b_axis": 2, "title": "Coronal"},   # fixed y, vary x,z
    {"axis_fixed": 0, "a_axis": 1, "b_axis": 2, "title": "Sagittal"},  # fixed x, vary y,z
]
AXIS_NAMES = {0: "x", 1: "y", 2: "z"}
UNDO_DEPTH = 5


def _plane_view(vol, axis_fixed, fixed_idx):
    """A writable 2D view into `vol` at `fixed_idx` along `axis_fixed`."""
    if axis_fixed == 0:
        return vol[fixed_idx, :, :]
    if axis_fixed == 1:
        return vol[:, fixed_idx, :]
    return vol[:, :, fixed_idx]


def paint_disk(mask, axis_fixed, fixed_idx, a_idx, b_idx, va, vb, radius_mm, add):
    """Paint (add=True) or erase (add=False) a disk of `radius_mm` centered
    at (a_idx, b_idx) within the single slice `mask` takes at `fixed_idx`
    along `axis_fixed`. Edits only that one slice — a 2D brush, not a 3D ball.
    """
    plane = _plane_view(mask, axis_fixed, fixed_idx)
    na, nb = plane.shape
    ra = int(np.ceil(radius_mm / va))
    rb = int(np.ceil(radius_mm / vb))
    a_lo, a_hi = max(0, a_idx - ra), min(na - 1, a_idx + ra)
    b_lo, b_hi = max(0, b_idx - rb), min(nb - 1, b_idx + rb)
    if a_lo > a_hi or b_lo > b_hi:
        return
    aa = np.arange(a_lo, a_hi + 1)[:, None]
    bb = np.arange(b_lo, b_hi + 1)[None, :]
    dist2 = ((aa - a_idx) * va) ** 2 + ((bb - b_idx) * vb) ** 2
    sel = dist2 <= radius_mm ** 2
    sub = plane[a_lo:a_hi + 1, b_lo:b_hi + 1]
    sub[sel] = 1 if add else 0


def apply_base_cut(mask, axis, apex_dir, cut_idx):
    """Erase every voxel base-ward of `cut_idx` along `axis` (in place)."""
    sl = [slice(None)] * 3
    sl[axis] = slice(0, cut_idx) if apex_dir > 0 else slice(cut_idx + 1, None)
    mask[tuple(sl)] = 0


def keep_largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8)


CAVITY_RGB = (0.0, 1.0, 1.0)   # cyan
MYO_RGB = (1.0, 0.6, 0.0)      # orange


def _overlay_rgba(cavity_plane, myo_plane=None):
    """Translucent fill + opaque outline per label (cheaper than ax.contour)."""
    rgba = np.zeros((*cavity_plane.shape, 4), dtype=np.float32)
    for plane, rgb in ((myo_plane, MYO_RGB), (cavity_plane, CAVITY_RGB)):
        if plane is None:
            continue
        m = plane.astype(bool)
        rgba[m] = (*rgb, 0.18)
        rgba[m & ~ndimage.binary_erosion(m)] = (*rgb, 1.0)
    return rgba


def auto_myocardium(cavity, image, voxel_sizes, axis, apex_dir, cut_idx, max_wall_mm=15.0):
    """First-pass LV myocardium around a cavity mask, for manual correction.

    In both the HU and the enhanced volumes the wall is a DARK band between
    the bright contrast-filled cavity and the tank fluid (PEXA5 phase 0,
    enhanced: ~95 at 1-5 mm from the cavity vs ~150 beyond 10 mm). So:
      1. Estimate the wall level (median 1-4 mm outside the cavity) and the
         bath level (median 12-20 mm outside), threshold halfway between.
      2. Candidate wall = below threshold, within max_wall_mm of the cavity,
         apex side of the base cut; keep only pieces touching the cavity.
      3. Close small gaps, fill the shell (3D + per short-axis slice) so the
         epicardium is a closed surface, myocardium = shell minus cavity.
    Works on any intensity scale as long as the wall is darker than both
    neighbors. Returns (uint8 mask, dict of the levels used).
    """
    c_full = cavity.astype(bool)
    out = np.zeros(c_full.shape, np.uint8)
    if not c_full.any():
        return out, {}

    # work in the cavity's bounding box + margin (bath sampled up to 20 mm out)
    pad = [int(np.ceil((max(max_wall_mm, 20.0) + 2) / v)) for v in voxel_sizes]
    nz = np.argwhere(c_full)
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + pad + 1, c_full.shape)
    sub = tuple(slice(a, b) for a, b in zip(lo, hi))
    c = c_full[sub]
    im = ndimage.gaussian_filter(image[sub].astype(np.float32),
                                 sigma=[1.0 / v for v in voxel_sizes])  # ~1 mm
    dist = ndimage.distance_transform_edt(~c, sampling=voxel_sizes)

    ax_idx = np.arange(lo[axis], hi[axis])
    keep_ax = ax_idx >= cut_idx if apex_dir > 0 else ax_idx <= cut_idx
    shp = [1, 1, 1]
    shp[axis] = -1
    allowed = np.broadcast_to(keep_ax.reshape(shp), c.shape)

    near = (dist >= 1) & (dist <= 4) & allowed
    far = (dist >= 12) & (dist <= 20) & allowed
    if not near.any() or not far.any():
        return out, {}
    wall = float(np.median(im[near]))
    bath = float(np.median(im[far]))
    thresh = 0.5 * (wall + bath)

    shell = (dist > 0) & (dist <= max_wall_mm) & allowed
    cand = shell & (im < thresh)
    lab, _ = ndimage.label(cand)
    touching = np.unique(lab[cand & (dist <= 1.5)])
    myo = np.isin(lab, touching[touching > 0])

    myo = ndimage.binary_closing(myo, structure=ball(2))
    epi = ndimage.binary_fill_holes(myo | c)
    for i in range(epi.shape[axis]):
        sl = [slice(None)] * 3
        sl[axis] = i
        epi[tuple(sl)] = ndimage.binary_fill_holes(epi[tuple(sl)])
    myo = ndimage.binary_opening(epi & ~c & shell, structure=ball(1))

    out[sub] = myo
    return out, {"wall": wall, "bath": bath, "threshold": thresh}


# ──────────────────────────────────────────────────────────────────────────────
# Phase selection + segmentation
# ──────────────────────────────────────────────────────────────────────────────

def select_phases(frames, phases):
    """Attach a cardiac phase (% R-R, files assumed evenly spaced over the
    cycle — same convention as extract_lv_volume_conical.py --phases) to
    each file and keep only the requested ones. Returns [(folder, path, pct)].
    """
    step = 100.0 / len(frames)
    tagged = [(folder, path, i * step) for i, (folder, path) in enumerate(frames)]
    if phases is None:
        return tagged
    selected = []
    for p in phases:
        i = int(round(p / step))
        if not (0 <= i < len(frames)) or abs(i * step - p) > 1e-6:
            raise SystemExit(f"ERROR: phase {p:g}% not available. Phases: "
                             + ", ".join(f"{t[2]:g}" for t in tagged))
        selected.append(tagged[i])
    return selected


def segment_phase(data, voxel_sizes, seed, threshold, args):
    """Returns (mask uint8, valve_plane_idx or None)."""
    if args.method == "sphere":
        mask, _ = segment_lv_cavity(data, seed, voxel_sizes, threshold=threshold,
                                    sphere_radius_mm=args.sphere_radius)
        return mask.astype(np.uint8), None
    mask, _, valve_idx = segment_lv_cavity_conical(
        data, seed, voxel_sizes, threshold=threshold,
        axis=args.axis, apex_dir=args.apex_dir,
        base_radius_mm=args.sphere_radius, apex_radius_mm=args.apex_radius,
        cone_length_mm=args.cone_length, valve_search_mm=args.valve_search_mm,
        apex_sphere_radius_mm=args.apex_sphere_radius,
    )
    return mask.astype(np.uint8), int(valve_idx)


def _to_display(data):
    """Compact in-memory copy for display: uint8 for 0..255 data (enhanced
    TIFF-derived volumes), int16 otherwise (HU)."""
    if data.min() >= 0 and data.max() <= 255:
        return np.rint(data).astype(np.uint8)
    return np.clip(np.rint(data), -32768, 32767).astype(np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation cache
# ──────────────────────────────────────────────────────────────────────────────

CACHE_MASK_NAME = "segmentation_cache.nii.gz"
CACHE_META_NAME = "segmentation_cache.json"


def _cache_paths(output_dir):
    return (os.path.join(output_dir, CACHE_MASK_NAME),
            os.path.join(output_dir, CACHE_META_NAME))


def save_segmentation_cache(raw_masks, phases, params, output_dir):
    mask_path, json_path = _cache_paths(output_dir)
    arr = np.stack(raw_masks, axis=-1).astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, np.array(phases[0]["affine"])), mask_path)
    with open(json_path, "w") as f:
        json.dump({"params": params, "phases": phases}, f, indent=2)
    print(f"Segmentation cache saved: {mask_path}")


def maybe_load_cache(args):
    """If a cache built with the same folders/method/phases exists (and
    --no-cache wasn't passed), ask whether to use it. Returns
    (params, phases, raw_masks) or None."""
    if args.no_cache:
        return None
    mask_path, json_path = _cache_paths(args.output)
    if not (os.path.exists(mask_path) and os.path.exists(json_path)):
        return None
    with open(json_path) as f:
        cache = json.load(f)
    params = cache.get("params", {})
    key_now = (list(args.folders), args.method, args.phases)
    key_cache = (params.get("folders"), params.get("method"), params.get("phases"))
    if key_now != key_cache:
        print(f"Segmentation cache in '{args.output}' was built with different "
              f"--folders/--method/--phases — ignoring it.")
        return None

    print("\nFound an existing segmentation cache:")
    print(f"  {mask_path}")
    print("  " + ", ".join(f"{k}={params.get(k)}" for k in (
        "method", "seed", "threshold", "sphere_radius", "apex_radius",
        "cone_length", "apex_sphere_radius", "valve_search_mm", "axis", "apex_dir")))
    print(f"  {len(cache['phases'])} phases")
    if input("Load this cached segmentation instead of re-running it? [y/N] ").strip().lower() != "y":
        return None

    masks_4d = np.asarray(nib.load(mask_path).dataobj, dtype=np.uint8)
    raw_masks = [np.ascontiguousarray(masks_4d[..., i]) for i in range(masks_4d.shape[-1])]
    return params, cache["phases"], raw_masks


# ──────────────────────────────────────────────────────────────────────────────
# Interactive editor
# ──────────────────────────────────────────────────────────────────────────────

class MaskEditor:
    def __init__(self, images, masks_auto, masks, edited, myo, phases, seed, axis, apex_dir,
                 edits_dir, brush_mm, fps, vmin=-200, vmax=600, cmap="gray", max_wall_mm=15.0):
        self.cmap = cmap
        self.images = images
        self.masks_auto = masks_auto
        self.masks = masks            # cavity masks (edited)
        self.edited = edited          # cavity differs from auto
        self.myo = myo                # myocardium masks (empty until "Auto myo" / painting)
        self.phases = phases          # list of dicts (folder, frame_idx, pct, ...)
        self.n = len(images)
        self.axis, self.apex_dir = axis, apex_dir
        self.edits_dir = edits_dir
        self.vmin, self.vmax = vmin, vmax
        self.max_wall_mm = max_wall_mm

        self.dirty = [False] * self.n
        self.undo = []                # current phase only (cleared on phase change)
        self.vol_cache = [None] * self.n

        self.t = 0
        self.x, self.y, self.z = (int(v) for v in seed)
        self.cut = self._default_cut(0)

        self.brush_mm = brush_mm
        self.label = "Cavity"         # which label the brush / Keep largest / Reset act on
        self.mode = "Navigate"
        self.painting = False
        self.paint_panel = None
        self.playing = False

        self._build_figure()
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._advance_phase)

        self.update_all()
        plt.show()

    # ── helpers ─────────────────────────────────────────────────────────
    def _default_cut(self, t):
        """Detected valve plane (conical), else the auto mask's base-most
        extent along the long axis (i.e. a cut that removes nothing)."""
        valve = self.phases[t].get("valve_idx")
        if valve is not None:
            return int(valve)
        prof = self.masks_auto[t].any(axis=tuple(a for a in range(3) if a != self.axis))
        idx = np.flatnonzero(prof)
        if idx.size == 0:
            return self.images[t].shape[self.axis] // 2
        return int(idx[0] if self.apex_dir > 0 else idx[-1])

    def _volumes(self, t):
        """(cavity mL, myocardium mL) for phase t, cached until the next edit."""
        if self.vol_cache[t] is None:
            vv = self.phases[t]["voxel_vol"]
            self.vol_cache[t] = (compute_volume_ml(self.masks[t], vv),
                                 compute_volume_ml(self.myo[t], vv))
        return self.vol_cache[t]

    def _active(self):
        return self.masks if self.label == "Cavity" else self.myo

    def _push_undo(self, labels=None):
        """Snapshot the given labels ('cav', 'myo'; default: the active one)."""
        labels = labels or (("cav",) if self.label == "Cavity" else ("myo",))
        snap = {}
        if "cav" in labels:
            snap["cav"] = (self.masks[self.t].copy(), self.edited[self.t])
        if "myo" in labels:
            snap["myo"] = self.myo[self.t].copy()
        self.undo.append(snap)
        if len(self.undo) > UNDO_DEPTH:
            self.undo.pop(0)

    def _mark_changed(self, cavity_changed=True):
        if cavity_changed:
            self.edited[self.t] = True
        self.dirty[self.t] = True
        self.vol_cache[self.t] = None

    def _edit_path(self, t, prefix=""):
        p = self.phases[t]
        return os.path.join(self.edits_dir,
                            f"{prefix}{p['folder']}_frame_{p['frame_idx']:02d}.nii.gz")

    def save_phase(self, t):
        """Write phase t's edited cavity / myocardium masks (or remove stale
        files after a Reset)."""
        if not self.dirty[t]:
            return
        affine = np.array(self.phases[t]["affine"])
        for mask, keep, path in (
            (self.masks[t], self.edited[t], self._edit_path(t)),
            (self.myo[t], self.myo[t].any(), self._edit_path(t, "myo_")),
        ):
            if keep:
                os.makedirs(self.edits_dir, exist_ok=True)
                nib.save(nib.Nifti1Image(mask, affine), path)
                print(f"  saved {path}")
            elif os.path.exists(path):
                os.remove(path)
                print(f"  removed {path} (reset)")
        self.dirty[t] = False

    def save_all(self):
        for t in range(self.n):
            self.save_phase(t)

    @staticmethod
    def _set_quiet(slider, val):
        slider.eventson = False
        slider.set_val(val)
        slider.eventson = True

    # ── figure / widgets ────────────────────────────────────────────────
    def _build_figure(self):
        shape = self.images[0].shape
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 9))
        plt.subplots_adjust(bottom=0.40, top=0.90, left=0.03, right=0.97, wspace=0.05)

        # persistent artists: updated with set_data instead of redrawing
        self.im_img, self.im_ov, self.cross, self.cut_lines = [], [], [], []
        for ax, cfg in zip(self.axes, PANELS):
            plane_shape = (shape[cfg["b_axis"]], shape[cfg["a_axis"]])
            self.im_img.append(ax.imshow(np.zeros(plane_shape, np.int16), cmap=self.cmap,
                                         origin="lower", vmin=self.vmin, vmax=self.vmax,
                                         interpolation="nearest"))
            self.im_ov.append(ax.imshow(np.zeros((*plane_shape, 4), np.float32),
                                        origin="lower", interpolation="nearest"))
            self.cross.append((ax.axvline(0, color="yellow", lw=0.5, alpha=0.5),
                               ax.axhline(0, color="yellow", lw=0.5, alpha=0.5)))
            if cfg["a_axis"] == self.axis:
                line = ax.axvline(0, color="lime", lw=1.3, ls="--")
            elif cfg["b_axis"] == self.axis:
                line = ax.axhline(0, color="lime", lw=1.3, ls="--")
            else:
                line = None
            self.cut_lines.append(line)
            ax.axis("off")

        def sl(rect, label, lo, hi, val, step=1):
            return Slider(plt.axes(rect), label, lo, hi, valinit=val, valstep=step)

        self.sl_phase = sl([0.12, 0.33, 0.50, 0.025], "Cardiac phase", 0, max(self.n - 1, 1), 0)
        self.sl_phase.valtext.set_text(f"{self.phases[0]['pct']:g}%")
        self.btn_play = Button(plt.axes([0.66, 0.325, 0.07, 0.04]), "Play")

        self.sl_z = sl([0.12, 0.28, 0.50, 0.025], "Axial (z)", 0, shape[2] - 1, self.z)
        self.sl_y = sl([0.12, 0.245, 0.50, 0.025], "Coronal (y)", 0, shape[1] - 1, self.y)
        self.sl_x = sl([0.12, 0.21, 0.50, 0.025], "Sagittal (x)", 0, shape[0] - 1, self.x)

        self.sl_cut = sl([0.12, 0.15, 0.50, 0.025],
                         f"Base cut ({AXIS_NAMES[self.axis]})", 0, shape[self.axis] - 1, self.cut)
        self.btn_cut = Button(plt.axes([0.66, 0.143, 0.10, 0.04]), "Apply cut")
        self.btn_largest = Button(plt.axes([0.77, 0.143, 0.10, 0.04]), "Keep largest")

        self.btn_myo = Button(plt.axes([0.77, 0.21, 0.10, 0.04]), "Auto myo")
        self.btn_myo_all = Button(plt.axes([0.88, 0.21, 0.10, 0.04]), "Auto myo: all empty")
        self.radio_label = RadioButtons(plt.axes([0.88, 0.14, 0.10, 0.065]),
                                        ["Cavity", "Myocardium"])

        self.sl_brush = sl([0.12, 0.09, 0.50, 0.025], "Brush radius (mm)", 1, 25,
                           self.brush_mm, step=0.5)

        self.radio_mode = RadioButtons(plt.axes([0.66, 0.02, 0.10, 0.11]),
                                       ["Navigate", "Add", "Erase"])
        self.btn_undo = Button(plt.axes([0.77, 0.09, 0.10, 0.04]), "Undo")
        self.btn_reset = Button(plt.axes([0.77, 0.04, 0.10, 0.04]), "Reset phase")
        self.btn_save = Button(plt.axes([0.88, 0.04, 0.10, 0.09]), "Save & Exit")

        self.sl_phase.on_changed(self._on_phase)
        for s in (self.sl_z, self.sl_y, self.sl_x):
            s.on_changed(self._on_slice)
        self.sl_cut.on_changed(self._on_cut_slider)
        self.sl_brush.on_changed(self._on_brush)
        self.radio_mode.on_clicked(self._on_mode)
        self.btn_play.on_clicked(self._on_play)
        self.btn_cut.on_clicked(self._on_apply_cut)
        self.btn_largest.on_clicked(self._on_keep_largest)
        self.btn_myo.on_clicked(self._on_auto_myo)
        self.btn_myo_all.on_clicked(self._on_auto_myo_all)
        self.radio_label.on_clicked(self._on_label)
        self.btn_undo.on_clicked(self._on_undo)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_save.on_clicked(self._on_save)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.save_all())

    # ── drawing ─────────────────────────────────────────────────────────
    def update_panel(self, i, overlay_only=False):
        cfg = PANELS[i]
        fixed_idx = {2: self.z, 1: self.y, 0: self.x}[cfg["axis_fixed"]]
        self.im_ov[i].set_data(_overlay_rgba(
            _plane_view(self.masks[self.t], cfg["axis_fixed"], fixed_idx).T,
            _plane_view(self.myo[self.t], cfg["axis_fixed"], fixed_idx).T))
        if overlay_only:
            return
        self.im_img[i].set_data(_plane_view(self.images[self.t], cfg["axis_fixed"], fixed_idx).T)
        pos = {0: self.x, 1: self.y, 2: self.z}
        vline, hline = self.cross[i]
        vline.set_xdata([pos[cfg["a_axis"]]] * 2)
        hline.set_ydata([pos[cfg["b_axis"]]] * 2)
        self.axes[i].set_title(f"{cfg['title']} ({AXIS_NAMES[cfg['axis_fixed']]}={fixed_idx})")

    def update_cut_lines(self):
        for line, cfg in zip(self.cut_lines, PANELS):
            if line is None:
                continue
            if cfg["a_axis"] == self.axis:
                line.set_xdata([self.cut] * 2)
            else:
                line.set_ydata([self.cut] * 2)

    def update_title(self):
        p = self.phases[self.t]
        flag = "  [EDITED]" if self.edited[self.t] else ""
        v_cav, v_myo = self._volumes(self.t)
        self.fig.suptitle(
            f"phase {p['pct']:g}%  —  {p['folder']}/{p['filename']}  —  "
            f"cavity (cyan) = {v_cav:.2f} mL{flag}   myocardium (orange) = {v_myo:.2f} mL\n"
            f"editing: {self.label}   mode: {self.mode}   brush: {self.brush_mm:.1f} mm   "
            f"(green dashed = base cut; 'Apply cut' erases both labels on the base side)",
            fontsize=11)

    def update_all(self):
        for i in range(3):
            self.update_panel(i)
        self.update_cut_lines()
        self.update_title()
        self.fig.canvas.draw_idle()

    # ── widget callbacks ────────────────────────────────────────────────
    def _on_phase(self, _):
        new_t = min(int(self.sl_phase.val), self.n - 1)
        self.sl_phase.valtext.set_text(f"{self.phases[new_t]['pct']:g}%")
        if new_t == self.t:
            return
        self.save_phase(self.t)  # autosave the phase we're leaving
        self.undo = []           # undo history is per visit (keeps memory bounded)
        self.t = new_t
        if self.phases[self.t].get("cut_applied") is not None:
            self.cut = self.phases[self.t]["cut_applied"]
            self._set_quiet(self.sl_cut, self.cut)
        self.update_all()

    def _on_slice(self, _):
        self.z, self.y, self.x = int(self.sl_z.val), int(self.sl_y.val), int(self.sl_x.val)
        self.update_all()

    def _on_cut_slider(self, _):
        self.cut = int(self.sl_cut.val)
        self.update_cut_lines()
        self.fig.canvas.draw_idle()

    def _on_brush(self, _):
        self.brush_mm = float(self.sl_brush.val)
        self.update_title()
        self.fig.canvas.draw_idle()

    def _on_mode(self, label):
        self.mode = label
        self.update_title()
        self.fig.canvas.draw_idle()

    def _on_label(self, label):
        self.label = label
        self.update_title()
        self.fig.canvas.draw_idle()

    def _on_apply_cut(self, _):
        self._push_undo(("cav", "myo"))
        apply_base_cut(self.masks[self.t], self.axis, self.apex_dir, self.cut)
        apply_base_cut(self.myo[self.t], self.axis, self.apex_dir, self.cut)
        self.phases[self.t]["cut_applied"] = self.cut
        self._mark_changed()
        self.save_phase(self.t)
        self.update_all()

    def _on_keep_largest(self, _):
        self._push_undo()
        masks = self._active()
        masks[self.t] = keep_largest_component(masks[self.t])
        self._mark_changed(cavity_changed=self.label == "Cavity")
        self.save_phase(self.t)
        self.update_all()

    def _auto_myo_phase(self, t, cut):
        myo, levels = auto_myocardium(self.masks[t], self.images[t],
                                      self.phases[t]["voxel_sizes"], self.axis,
                                      self.apex_dir, cut, self.max_wall_mm)
        if levels:
            print(f"  phase {self.phases[t]['pct']:g}%: auto myocardium — wall level "
                  f"{levels['wall']:.0f}, bath level {levels['bath']:.0f}, threshold "
                  f"{levels['threshold']:.0f}; {compute_volume_ml(myo, self.phases[t]['voxel_vol']):.2f} mL")
        return myo

    def _on_auto_myo(self, _):
        """Replace this phase's myocardium with a fresh automatic estimate
        around the CURRENT cavity, apex side of the current base-cut line."""
        self._push_undo(("myo",))
        self.myo[self.t] = self._auto_myo_phase(self.t, self.cut)
        self._mark_changed(cavity_changed=False)
        self.save_phase(self.t)
        self.update_all()

    def _on_auto_myo_all(self, _):
        """Auto myocardium for every phase that has none yet (never touches a
        phase you already have a myocardium for). Uses each phase's applied
        cut, or the current line if that phase wasn't cut."""
        todo = [t for t in range(self.n) if not self.myo[t].any()]
        print(f"\nAuto myocardium for {len(todo)} phase(s) without one (~5 s each)...")
        for t in todo:
            cut = self.phases[t].get("cut_applied")
            self.myo[t] = self._auto_myo_phase(t, self.cut if cut is None else cut)
            self.dirty[t] = True
            self.vol_cache[t] = None
            self.save_phase(t)
        self.update_all()

    def _on_undo(self, _):
        if not self.undo:
            print("  nothing to undo for this phase")
            return
        snap = self.undo.pop()
        if "cav" in snap:
            self.masks[self.t], self.edited[self.t] = snap["cav"]
        if "myo" in snap:
            self.myo[self.t] = snap["myo"]
        self._mark_changed(cavity_changed=False)
        self.update_all()

    def _on_reset(self, _):
        """Cavity: back to the auto mask. Myocardium: cleared."""
        self._push_undo()
        if self.label == "Cavity":
            self.masks[self.t] = self.masks_auto[self.t].copy()
            self.phases[self.t]["cut_applied"] = None
            self.edited[self.t] = False
        else:
            self.myo[self.t] = np.zeros_like(self.myo[self.t])
        self._mark_changed(cavity_changed=False)
        self.save_phase(self.t)
        self.update_all()

    def _on_play(self, _):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text("Pause")
            self.timer.start()
        else:
            self.btn_play.label.set_text("Play")
            self.timer.stop()

    def _advance_phase(self):
        self.sl_phase.set_val((self.t + 1) % self.n)

    def _on_save(self, _):
        self.timer.stop()
        self.save_all()
        plt.close(self.fig)

    # ── mouse: navigate or paint ────────────────────────────────────────
    def _panel_index(self, event):
        for i, ax in enumerate(self.axes):
            if event.inaxes is ax:
                return i
        return None

    def _paint_at(self, panel_idx, xdata, ydata):
        cfg = PANELS[panel_idx]
        f = cfg["axis_fixed"]
        fixed_idx = {2: self.z, 1: self.y, 0: self.x}[f]
        vs = self.phases[self.t]["voxel_sizes"]
        paint_disk(self._active()[self.t], f, fixed_idx, int(round(xdata)), int(round(ydata)),
                   vs[cfg["a_axis"]], vs[cfg["b_axis"]], self.brush_mm,
                   add=(self.mode == "Add"))

    def _on_press(self, event):
        if event.button != 1:
            return
        i = self._panel_index(event)
        if i is None or event.xdata is None or event.ydata is None:
            return
        cfg = PANELS[i]
        if self.mode == "Navigate":
            pos = {0: self.x, 1: self.y, 2: self.z}
            pos[cfg["a_axis"]] = int(round(event.xdata))
            pos[cfg["b_axis"]] = int(round(event.ydata))
            self.x, self.y, self.z = pos[0], pos[1], pos[2]
            for s, v in ((self.sl_x, self.x), (self.sl_y, self.y), (self.sl_z, self.z)):
                self._set_quiet(s, v)
            self.update_all()
            return
        self._push_undo(("cav", "myo"))  # both: painting one label clears the other
        self.painting, self.paint_panel = True, i
        # Blitting: render the panel once without its overlay, keep that as a
        # background, and while dragging only redraw the overlay image on top
        # of it (instead of re-rendering the whole window on every mouse move).
        canvas = self.fig.canvas
        self.im_ov[i].set_animated(True)
        canvas.draw()
        self._blit_bg = canvas.copy_from_bbox(self.axes[i].bbox)
        self._paint_at(i, event.xdata, event.ydata)
        self._blit_overlay(i)

    def _blit_overlay(self, i):
        canvas = self.fig.canvas
        self.update_panel(i, overlay_only=True)
        canvas.restore_region(self._blit_bg)
        self.axes[i].draw_artist(self.im_ov[i])
        canvas.blit(self.axes[i].bbox)

    def _on_motion(self, event):
        if not self.painting or event.inaxes is not self.axes[self.paint_panel]:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._paint_at(self.paint_panel, event.xdata, event.ydata)
        self._blit_overlay(self.paint_panel)

    def _on_release(self, _event):
        if not self.painting:
            return
        self.im_ov[self.paint_panel].set_animated(False)
        self.painting, self.paint_panel = False, None
        # the two labels never overlap: what was just added to one leaves the other
        if self.mode == "Add":
            if self.label == "Cavity":
                self.myo[self.t][self.masks[self.t].astype(bool)] = 0
            else:
                self.masks[self.t][self.myo[self.t].astype(bool)] = 0
        self._mark_changed(cavity_changed=self.label == "Cavity" or self.mode == "Add")
        self.update_all()  # refresh the other panels + volume once per stroke


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def save_edited_outputs(editor, out_dir):
    csv_path = os.path.join(out_dir, "volumes_edited.csv")
    rows = []
    with open(csv_path, "w") as f:
        f.write("phase_pct,folder,frame_idx,filename,volume_ml_auto,volume_ml_edited,"
                "diff_ml,edited,volume_ml_myocardium\n")
        for t, p in enumerate(editor.phases):
            v_auto = compute_volume_ml(editor.masks_auto[t], p["voxel_vol"])
            v_edit = compute_volume_ml(editor.masks[t], p["voxel_vol"])
            v_myo = compute_volume_ml(editor.myo[t], p["voxel_vol"]) if editor.myo[t].any() else None
            rows.append((p["pct"], v_auto, v_edit, v_myo))
            f.write(f"{p['pct']:g},{p['folder']},{p['frame_idx']},{p['filename']},"
                    f"{v_auto:.4f},{v_edit:.4f},{v_edit - v_auto:+.4f},{editor.edited[t]},"
                    f"{'' if v_myo is None else f'{v_myo:.4f}'}\n")
    print(f"Edited volumes CSV saved: {csv_path}")

    # Myocardial volume should be ~constant over the cycle (tissue is
    # incompressible): a large spread means the wall segmentation needs work.
    myo_vals = np.array([r[3] for r in rows if r[3] is not None])
    if len(myo_vals) > 1:
        print(f"Myocardial volume over {len(myo_vals)} phases: mean {myo_vals.mean():.2f} mL, "
              f"SD {myo_vals.std(ddof=1):.2f} mL (CV {100 * myo_vals.std(ddof=1) / myo_vals.mean():.1f}%)")

    plot_path = os.path.join(out_dir, "volume_edited_plot.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    pct, auto, edit, myo = zip(*rows)
    ax.plot(pct, auto, "o--", color="tab:gray", label="cavity, auto")
    ax.plot(pct, edit, "o-", color="tab:cyan", label="cavity, manually edited")
    if any(v is not None for v in myo):
        mp = [(x, v) for x, v in zip(pct, myo) if v is not None]
        ax.plot(*zip(*mp), "s-", color="tab:orange", label="myocardium")
    ax.set_xlabel("Cardiac phase (% R-R)")
    ax.set_ylabel("Volume (mL)")
    ax.set_title("LV Cavity and Myocardial Volume Over the Cycle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Edited volume plot saved: {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Slideshow through cardiac phases + manual correction "
                     "(base cut, keep largest, brush) of the LV segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--folders", nargs="+", default=["retro_A_frames", "retro_B_frames"])
    parser.add_argument("--method", choices=["sphere", "conical"], default="sphere",
                         help="Segmentation to start from: 'sphere' (v1) or 'conical' (v2). Default: sphere")
    parser.add_argument("--phases", type=float, nargs="+", default=None, metavar="PCT",
                         help="Only load these cardiac phases (percent R-R, files assumed "
                              "evenly spaced over 0-100%%). Temporal smoothing is skipped.")
    parser.add_argument("--display-folders", nargs="+", default=None,
                         help="Show these images instead of the --folders ones, e.g. the "
                              "enhanced volumes from tiff_to_nifti.py. Must be on the same "
                              "voxel grid with the same number of phases. Segmentation "
                              "still runs on --folders.")
    parser.add_argument("--vmin", type=float, default=None,
                         help="Display window low end. Default: 0 for 0..255 images, else -200 (HU)")
    parser.add_argument("--vmax", type=float, default=None,
                         help="Display window high end. Default: 255 for 0..255 images, else 600 (HU)")
    parser.add_argument("--invert-display", action="store_true",
                         help="Show the images with an inverted grayscale (cavity dark, "
                              "like the ImageJ view of the enhanced TIFFs). Display only.")
    parser.add_argument("--seed", type=int, nargs=3, metavar=("X", "Y", "Z"),
                         help="Seed voxel coordinate inside the LV cavity (skip interactive selection)")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override intensity threshold (HU).")
    parser.add_argument("--sphere-radius", type=float, default=70,
                         help="Sphere radius (sphere) / cone base radius (conical), mm. Default: 70")
    parser.add_argument("--apex-radius", type=float, default=15,
                         help="Conical: cone radius at the apex end of the taper (mm). Default: 15")
    parser.add_argument("--cone-length", type=float, default=90,
                         help="Conical: distance (mm) over which the cone tapers. Default: 90")
    parser.add_argument("--apex-sphere-radius", type=float, default=25,
                         help="Conical: radius (mm) of the apex sphere bulge. Default: 25")
    parser.add_argument("--valve-search-mm", type=float, default=60,
                         help="Conical: how far (mm) from the seed to search for the valve plane. Default: 60")
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                         help="LV long axis in voxel space (direction of the base cut). Default: 1")
    parser.add_argument("--apex-dir", type=int, default=1, choices=[-1, 1],
                         help="+1 or -1: which direction along --axis is the apex. Default: 1")
    parser.add_argument("--temporal-window", type=int, default=1,
                         help="Temporal smoothing window applied to the auto masks before editing.")
    parser.add_argument("--max-wall-mm", type=float, default=15.0,
                         help="Auto myocardium: maximum wall thickness searched outside the "
                              "cavity (mm). Default: 15")
    parser.add_argument("--brush-radius", type=float, default=5.0,
                         help="Initial brush radius (mm). Default: 5.0")
    parser.add_argument("--fps", type=float, default=4.0,
                         help="Slideshow playback speed (phases/sec). Default: 4.0")
    parser.add_argument("--output", default="results_edit",
                         help="Output directory (default: results_edit)")
    parser.add_argument("--no-cache", action="store_true",
                         help="Ignore any existing segmentation cache and re-segment.")
    args = parser.parse_args()

    frames = collect_frames(args.folders)
    if not frames:
        print("ERROR: No NIfTI files found in specified folders.")
        return
    selected = select_phases(frames, args.phases)
    print(f"Found {len(frames)} phases across {len(args.folders)} folder(s); "
          f"loading {len(selected)}: " + ", ".join(f"{p:g}%" for _, _, p in selected))
    os.makedirs(args.output, exist_ok=True)

    # Images to display: same phase selection, from --display-folders if given
    display_paths = None
    if args.display_folders:
        display_frames = collect_frames(args.display_folders)
        if len(display_frames) != len(frames):
            raise SystemExit(f"ERROR: --display-folders have {len(display_frames)} phases, "
                             f"--folders have {len(frames)}; they must match.")
        display_paths = [path for _, path, _ in select_phases(display_frames, args.phases)]
        print("Displaying images from: " + ", ".join(args.display_folders))

    def load_display(i, seg_path, seg_data=None):
        if display_paths is None:
            data = seg_data if seg_data is not None else load_nifti(seg_path)[0]
        else:
            data = load_nifti(display_paths[i])[0]
        return _to_display(data)

    cached = maybe_load_cache(args)
    images = []
    if cached is not None:
        params, phases, raw_masks = cached
        seed = tuple(params["seed"])
        for i, p in enumerate(phases):
            print(f"[{i + 1}/{len(phases)}] loading image for phase {p['pct']:g}%")
            images.append(load_display(i, p["filepath"]))
    else:
        print(f"\nLoading first phase: {frames[0][1]}")
        first_data = load_nifti(frames[0][1])[0]
        if args.seed:
            seed = tuple(args.seed)
            print(f"Using provided seed: {seed}")
        else:
            seed = select_seed_interactive(first_data)
        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using provided threshold: {threshold:.0f} HU")
        else:
            threshold = estimate_threshold(first_data, seed)
            print(f"Auto-detected threshold: {threshold:.0f} HU")
        del first_data

        raw_masks, phases = [], []
        for i, (folder, path, pct) in enumerate(selected):
            filename = os.path.basename(path)
            print(f"\n[{i + 1}/{len(selected)}] phase {pct:g}%  ({folder}/{filename})")
            data, affine, voxel_sizes, voxel_vol = load_nifti(path)
            mask, valve_idx = segment_phase(data, voxel_sizes, seed, threshold, args)
            print(f"  {args.method} volume: {compute_volume_ml(mask, voxel_vol):.2f} mL"
                  + (f"  (valve plane @ {valve_idx})" if valve_idx is not None else ""))
            images.append(load_display(i, path, data))
            del data
            raw_masks.append(mask)
            phases.append({
                "folder": folder, "filename": filename, "filepath": path, "pct": pct,
                "frame_idx": int(filename.split("frame_")[1].split(".")[0]),
                "valve_idx": valve_idx, "voxel_sizes": [float(v) for v in voxel_sizes],
                "voxel_vol": float(voxel_vol), "affine": np.asarray(affine).tolist(),
            })

        params = {
            "folders": list(args.folders), "method": args.method, "phases": args.phases,
            "seed": [int(v) for v in seed], "threshold": float(threshold),
            "sphere_radius": args.sphere_radius, "apex_radius": args.apex_radius,
            "cone_length": args.cone_length, "apex_sphere_radius": args.apex_sphere_radius,
            "valve_search_mm": args.valve_search_mm, "axis": args.axis, "apex_dir": args.apex_dir,
        }
        save_segmentation_cache(raw_masks, phases, params, args.output)

    if args.temporal_window > 0 and args.phases is None:
        print(f"\nTemporal smoothing (window={args.temporal_window})...")
        masks_auto = temporal_smooth_masks(raw_masks, window=args.temporal_window)
    else:
        masks_auto = raw_masks

    # Resume previously saved edits
    edits_dir = os.path.join(args.output, "edited_masks")
    masks, edited, myo = [], [], []
    for t, p in enumerate(phases):
        p["cut_applied"] = None
        name = f"{p['folder']}_frame_{p['frame_idx']:02d}.nii.gz"
        path = os.path.join(edits_dir, name)
        if os.path.exists(path):
            masks.append(np.asarray(nib.load(path).dataobj, dtype=np.uint8))
            edited.append(True)
        else:
            masks.append(masks_auto[t].copy())
            edited.append(False)
        myo_path = os.path.join(edits_dir, "myo_" + name)
        myo.append(np.asarray(nib.load(myo_path).dataobj, dtype=np.uint8)
                   if os.path.exists(myo_path) else np.zeros_like(masks_auto[t]))
    n_myo = sum(m.any() for m in myo)
    if any(edited) or n_myo:
        print(f"\nResumed from {edits_dir}/: edited cavity for {sum(edited)} phase(s), "
              f"myocardium for {n_myo} phase(s) ('Reset phase' undoes a phase).")

    if images[0].shape != masks[0].shape:
        raise SystemExit(f"ERROR: display images {images[0].shape} and masks "
                         f"{masks[0].shape} are not on the same grid.")
    is_8bit = images[0].dtype == np.uint8
    vmin = args.vmin if args.vmin is not None else (0 if is_8bit else -200)
    vmax = args.vmax if args.vmax is not None else (255 if is_8bit else 600)

    print("\nOpening editor — 'Navigate' mode: click to move slices; 'Apply cut' "
          "erases base-ward of the green line; 'Save & Exit' when done.")
    editor = MaskEditor(images, masks_auto, masks, edited, myo, phases, seed,
                        args.axis, args.apex_dir, edits_dir,
                        brush_mm=args.brush_radius, fps=args.fps, vmin=vmin, vmax=vmax,
                        cmap="gray_r" if args.invert_display else "gray",
                        max_wall_mm=args.max_wall_mm)

    save_edited_outputs(editor, args.output)


if __name__ == "__main__":
    main()
