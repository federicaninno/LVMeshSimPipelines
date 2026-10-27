"""
Extract left ventricle cavity volume over time using a conical, valve-plane
aware constraint — "v2" of the segmentation in extract_lv_volume.py ("v1").

WHY THIS EXISTS
    v1 constrains region growing to a fixed-radius SPHERE around the seed,
    has no notion of the mitral valve plane, and recovers apex/lateral
    voxels eaten by morphological opening using a hardcoded direction
    tuple (extend_directions) that only makes sense for this dataset's
    specific orientation and catheter entry side. This script replaces
    those three pieces with a more anatomy-driven, less hand-tuned
    approach, inspired by a MATLAB LV-masking script (User_Mask_LVVolumesEF.m,
    Dr. Dan Midgett) that used a manually-clicked valve plane per phase and
    a manually-clicked conical taper. Here both are AUTOMATED instead of
    clicked, so the pipeline stays hands-off like v1.

CHANGE SUMMARY (vs extract_lv_volume.py):
    1. Automatic valve-plane detection (detect_valve_plane) — NEW.
       v1 has no explicit base cutoff; it relies entirely on the sphere
       radius plus the catheter narrow-neck heuristic to implicitly keep
       the mask below the base. Here we scan the cross-sectional area
       profile of a rough mask along the LV long axis, from the seed
       toward the base, and cut at the mitral-annulus "waist" — a local
       area minimum followed by re-widening into the atrium.

    2. Conical radial constraint (build_hybrid_mask) replaces v1's
       fixed-radius SPHERE (extract_lv_volume.segment_lv_cavity, step 2).
       Radius tapers linearly from base_radius_mm (at the valve plane) to
       apex_radius_mm over cone_length_mm — matching the LV's actual
       tapered anatomy instead of a uniform sphere. Note this also
       generalizes v1's lateral extension, which only ever grew along the
       x-axis: the cone's radial taper covers BOTH remaining axes (x and
       z) symmetrically.

    3. Omnidirectional cone-gated extension (extend_mask_within_cone)
       replaces v1's hardcoded extend_directions =
       ((1,+1), (0,+1), (0,-1)), which only recovers apex/lateral voxels
       along 3 fixed (axis, sign) pairs tuned for this dataset's
       orientation. Because the valve plane + cone already wall off the
       base, extension here can safely regrow in ANY direction — it just
       re-admits above-threshold voxels that fall inside the cone.

    Everything else — multi-Otsu threshold estimation, seed snapping,
    close(ball(2))->open(ball(4/3/2)) cleanup, the catheter-position
    safety net, 3D+per-slice hole filling, and temporal smoothing — is
    UNCHANGED and reused directly from extract_lv_volume.py.

KNOWN LIMITATION
    Valve-plane detection is purely intensity-profile based (a
    cross-sectional-area waist). If the left atrium/aorta above the valve
    is filled with the same contrast fluid as the LV cavity with no clear
    narrowing at the annulus in a given scan, the heuristic can fail to
    find a waist; it then falls back to NOT cutting (behaves like v1 for
    that frame). Always check qc_overlays/ for the detected valve-plane
    line before trusting the v2 numbers on a new dataset — see the two
    items below, both aimed at making that check possible/easier.

MANUAL CORRECTION (--manual-review)
    detect_valve_plane() is a heuristic and can be wrong. Pass
    --manual-review to open an interactive window per frame (Slider +
    Confirm button, same widgets extract_lv_volume.select_seed_interactive
    uses) showing the auto-detected cut so you can drag-correct it before
    it's used — the automated equivalent of the MATLAB script's per-phase
    valve-plane click. Corrected frames are flagged in
    volumes_comparison.csv (valve_plane_auto vs valve_plane_final columns).
    Only supports the default --axis 1 convention for the drawn line.

QC OVERLAY DESIGN
    save_comparison_overlay() draws v1 and v2 as outline contours: solid
    red for v1, dashed cyan for v2, plus a green dashed line for the
    valve-plane cut. Using a different LINESTYLE (not just color) for each
    mask means that where the two boundaries coincide exactly, both are
    still visible — a plain "draw cyan on top of red" would make v1 look
    like a broken/open arc wherever it agrees with v2, worst right at the
    base where the two algorithms are expected to diverge most.

WHAT THIS SCRIPT REPORTS
    Only volume-over-time, run head-to-head against v1 on the SAME seed,
    threshold, and frames, so any difference in the resulting volumes is
    attributable to the 3 changes above and nothing else. It deliberately
    does not compute EF/ED/ES or reproduce v1's 4D-NIfTI/ParaView/video
    exports — those remain available by calling the corresponding
    functions in extract_lv_volume.py directly if you need them.

Usage:
    python extract_lv_volume_conical.py --seed 260 280 144

    Tune the cone shape:
        python extract_lv_volume_conical.py --seed 260 280 144 \\
            --sphere-radius 50 --apex-radius 15 --cone-length 90

    If your dataset's LV long axis / apex direction differs from this
    dataset's convention (long axis = voxel axis 1 = "y", apex = +y,
    base/catheter = -y, matching v1's extend_directions):
        python extract_lv_volume_conical.py --seed ... --axis 0 --apex-dir -1
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import ndimage
from skimage.morphology import ball

from extract_lv_volume import (
    load_nifti, collect_frames, select_seed_interactive, estimate_threshold,
    find_nearest_above_threshold, remove_catheter_by_position,
    compute_volume_ml, temporal_smooth_masks,
    segment_lv_cavity,  # v1 — run alongside v2 for the comparison
)


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE #1: automatic valve-plane detection (NEW — no v1 equivalent)
# ──────────────────────────────────────────────────────────────────────────────

def _area_profile(mask, axis):
    """Cross-sectional voxel count of `mask` at every index along `axis`."""
    return np.array([np.take(mask, i, axis=axis).sum() for i in range(mask.shape[axis])])


def detect_valve_plane(mask, seed, voxel_sizes, axis=1, base_dir=-1,
                        search_mm=60, widen_ratio=1.15):
    """Find the mitral-valve-plane slice index by scanning cross-sectional area.

    Starting at the seed's index along `axis`, walk toward the base
    (`base_dir`: -1 or +1) and track the area profile. The mitral annulus is
    anatomically the narrowest point between the LV cavity and the left
    atrium, so the area profile should show a local minimum (the "waist")
    followed by re-widening once you cross into the atrium. The slice at
    that minimum is returned as the valve plane — voxels beyond it (further
    base-ward) are excluded from the cavity.

    If no waist+re-widen pattern appears within `search_mm`, there is no
    reliable valve signal in this frame; falls back to the last slice
    scanned (i.e. effectively no cut, same behavior as v1).
    """
    areas = _area_profile(mask, axis)
    n = len(areas)
    start = seed[axis]
    step = 1 if base_dir > 0 else -1
    max_steps = max(1, int(round(search_mm / voxel_sizes[axis])))

    min_area = areas[start] if areas[start] > 0 else areas.max()
    min_idx = start
    seen_narrowing = False
    idx = start

    for _ in range(max_steps):
        idx += step
        if idx < 0 or idx >= n:
            idx -= step
            break
        a = areas[idx]
        if a == 0:
            continue
        if a < min_area:
            min_area = a
            min_idx = idx
            seen_narrowing = True
        elif seen_narrowing and a > min_area * widen_ratio:
            return min_idx

    # No clear waist found -> don't cut anything (matches v1's behavior of
    # relying only on the sphere/catheter heuristic for this frame).
    return idx if seen_narrowing else start


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE #2: conical constraint replacing v1's fixed-radius sphere
# ──────────────────────────────────────────────────────────────────────────────

def build_conical_mask(shape, seed, voxel_sizes, axis, apex_dir, valve_plane_idx,
                        base_radius_mm, apex_radius_mm, cone_length_mm):
    """Boolean mask of a cone: radius tapers base_radius_mm -> apex_radius_mm
    over cone_length_mm, anchored at the valve plane (not the seed) and
    pointed toward apex_dir along `axis`.

    Nothing base-ward of valve_plane_idx is ever included — this is what
    replaces v1's reliance on the catheter narrow-neck heuristic to keep
    the mask below the base.
    """
    other_axes = [a for a in range(3) if a != axis]
    idx_grid = np.indices(shape)

    d0 = (idx_grid[other_axes[0]] - seed[other_axes[0]]) * voxel_sizes[other_axes[0]]
    d1 = (idx_grid[other_axes[1]] - seed[other_axes[1]]) * voxel_sizes[other_axes[1]]
    radial_mm = np.sqrt(d0 ** 2 + d1 ** 2)

    # 0 at the valve plane, growing positive toward the apex
    axis_mm = (idx_grid[axis] - valve_plane_idx) * voxel_sizes[axis] * apex_dir

    t = np.clip(axis_mm / cone_length_mm, 0.0, 1.0)
    allowed_radius_mm = base_radius_mm * (1 - t) + apex_radius_mm * t

    margin = voxel_sizes[axis]  # one-voxel tolerance right at the valve plane
    cone = (
        (radial_mm <= allowed_radius_mm)
        & (axis_mm >= -margin)
        & (axis_mm <= cone_length_mm + apex_radius_mm)  # cap so the tip closes off
    )
    return cone

def build_hybrid_mask(shape, seed, voxel_sizes, axis, apex_dir, valve_plane_idx,
                       base_radius_mm, apex_radius_mm, cone_length_mm,
                       apex_sphere_radius_mm):
    """Union of the tapered cone with a sphere bulge centered at the APEX TIP
    (axis_mm == cone_length_mm), both gated by the valve plane. The cone
    keeps the base/mid-cavity tight and does the valve exclusion; the sphere
    only adds extra reach right at the tip, instead of bulging the mid-cavity
    the way a sphere anchored at the valve plane would."""
    other_axes = [a for a in range(3) if a != axis]
    idx_grid = np.indices(shape)

    d0 = (idx_grid[other_axes[0]] - seed[other_axes[0]]) * voxel_sizes[other_axes[0]]
    d1 = (idx_grid[other_axes[1]] - seed[other_axes[1]]) * voxel_sizes[other_axes[1]]
    radial_mm = np.sqrt(d0 ** 2 + d1 ** 2)
    axis_mm = (idx_grid[axis] - valve_plane_idx) * voxel_sizes[axis] * apex_dir

    margin = voxel_sizes[axis]
    valve_gate = axis_mm >= -margin

    t = np.clip(axis_mm / cone_length_mm, 0.0, 1.0)
    cone_radius = base_radius_mm * (1 - t) + apex_radius_mm * t
    cone = (radial_mm <= cone_radius) & valve_gate & (axis_mm <= cone_length_mm + apex_radius_mm)

    # apex bulge: sphere centered at the TIP of the cone, not the valve plane —
    # only adds reach right at the apex, leaves the cone's taper untouched elsewhere
    apex_axis_mm = axis_mm - cone_length_mm
    apex_dist_mm = np.sqrt(radial_mm ** 2 + apex_axis_mm ** 2)
    apex_bulge = (apex_dist_mm <= apex_sphere_radius_mm) & valve_gate

    return cone | apex_bulge
# ──────────────────────────────────────────────────────────────────────────────
# CHANGE #3: omnidirectional cone-gated extension replacing v1's hardcoded
# extend_directions tuple
# ──────────────────────────────────────────────────────────────────────────────

def extend_mask_within_cone(mask, data, threshold, cone_mask, max_steps=50):
    """Grow `mask` outward in ALL directions, admitting only voxels that are
    above threshold AND inside the cone.

    v1's extend_mask_constrained() only grows along 3 hardcoded (axis,
    sign) pairs to avoid regrowing the catheter. That restriction isn't
    needed here: the cone + valve plane already exclude the catheter's
    base-ward path, so any above-threshold voxel inside the cone is safe
    to re-admit regardless of direction.
    """
    extended = mask.astype(bool).copy()
    allowed = (data > threshold) & cone_mask
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity

    for _ in range(max_steps):
        dilated = ndimage.binary_dilation(extended, structure=struct)
        new_voxels = dilated & allowed & (~extended)
        if not new_voxels.any():
            break
        extended |= new_voxels

    return extended.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# v2 segmentation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def auto_detect_valve_plane(data, seed, voxel_sizes, threshold, axis=1, apex_dir=1,
                             base_radius_mm=50, valve_search_mm=60):
    """Snap the seed and run the bootstrap-sphere + detect_valve_plane pass.

    Factored out of segment_lv_cavity_conical so --manual-review can show the
    auto-detected value to the user (and let them override it) BEFORE the
    full pipeline commits to it. Returns (valve_plane_idx, snapped_seed).
    """
    base_dir = -apex_dir

    orig_seed = seed
    seed = find_nearest_above_threshold(data, seed, threshold)  # UNCHANGED from v1
    if seed != orig_seed:
        print(f"  [v2] Seed snapped from {orig_seed} to {seed}")

    # Bootstrap pass: same spherical constraint v1 uses, but here it's
    # throwaway scaffolding purely to get a rough mask to measure the area
    # profile on, so we can find the valve plane below.
    zz, yy, xx = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    dist = np.sqrt(
        ((zz - seed[0]) * voxel_sizes[0]) ** 2
        + ((yy - seed[1]) * voxel_sizes[1]) ** 2
        + ((xx - seed[2]) * voxel_sizes[2]) ** 2
    )
    bootstrap_sphere = dist <= base_radius_mm
    in_range = ((data > threshold) & bootstrap_sphere).astype(np.uint8)
    labeled, _ = ndimage.label(in_range)
    seed_label = labeled[seed[0], seed[1], seed[2]]
    if seed_label == 0:
        raise RuntimeError(f"[v2] Seed {seed} not in any above-threshold component.")
    rough_mask = (labeled == seed_label).astype(np.uint8)

    valve_plane_idx = detect_valve_plane(
        rough_mask, seed, voxel_sizes, axis=axis, base_dir=base_dir,
        search_mm=valve_search_mm,
    )
    return valve_plane_idx, seed


def review_valve_plane_interactive(data, seed, voxel_sizes, axis, apex_dir,
                                    valve_plane_idx_auto, frame_name,
                                    review_range_mm=40):
    """MANUAL CORRECTION STEP (opt-in via --manual-review).

    Shows the auto-detected valve-plane cut as a line on the axial and
    sagittal views (the two views in which axis=1/"y" is a plotted image
    axis) with a slider to drag it and a Confirm button — the same
    Slider/Button pattern extract_lv_volume.select_seed_interactive() uses
    for seed picking. Mirrors the MATLAB script's per-phase valve-plane
    click, but as a correction of an automatic estimate rather than a
    from-scratch click every time.

    Returns the confirmed (possibly unchanged) valve-plane index.
    """
    if axis != 1:
        print(f"  [manual-review] axis={axis} != 1 — the quick line overlay "
              f"only supports the default y-axis convention; showing plain "
              f"views without a drawn cut line.")

    vmin, vmax = -200, 600
    cx, cy, cz = seed
    current = [valve_plane_idx_auto]
    confirmed = [False]

    span_vox = max(5, int(round(review_range_mm / voxel_sizes[axis])))
    lo = max(0, valve_plane_idx_auto - span_vox)
    hi = min(data.shape[axis] - 1, valve_plane_idx_auto + span_vox)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(f"{frame_name} — drag the slider to correct the valve plane, "
                 f"then click Confirm", fontsize=12)
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.3, 0.1, 0.4, 0.03])
    slider = Slider(ax_slider, "Valve plane idx", lo, hi,
                     valinit=valve_plane_idx_auto, valstep=1)
    ax_btn = plt.axes([0.45, 0.02, 0.1, 0.05])
    btn = Button(ax_btn, "Confirm")

    def draw():
        for ax in axes:
            ax.clear()

        axes[0].imshow(data[:, :, cz].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Axial z={cz}")
        axes[1].imshow(data[:, cy, :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Coronal y={cy} (orthogonal to the cut — shown for context)")
        axes[2].imshow(data[cx, :, :].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[2].set_title(f"Sagittal x={cx}")

        if axis == 1:
            axes[0].axhline(current[0], color="lime", lw=1.5)
            axes[2].axvline(current[0], color="lime", lw=1.5)

        for ax in axes:
            ax.axis("off")
        fig.canvas.draw_idle()

    def on_slider(_):
        current[0] = int(slider.val)
        draw()

    def on_confirm(_):
        confirmed[0] = True
        plt.close(fig)

    slider.on_changed(on_slider)
    btn.on_clicked(on_confirm)

    draw()
    plt.show()

    if not confirmed[0]:
        print("  [manual-review] window closed without confirming — keeping "
              f"auto-detected value ({valve_plane_idx_auto}).")
        return valve_plane_idx_auto

    if current[0] != valve_plane_idx_auto:
        print(f"  [manual-review] valve plane MANUALLY CORRECTED: "
              f"{valve_plane_idx_auto} -> {current[0]}")
    else:
        print(f"  [manual-review] valve plane confirmed at auto-detected value "
              f"({current[0]})")
    return current[0]


def segment_lv_cavity_conical(data, seed, voxel_sizes, threshold,
                               axis=1, apex_dir=1,
                               base_radius_mm=50, apex_radius_mm=15,
                               cone_length_mm=90, valve_search_mm=60,
                               apex_sphere_radius_mm=25,
                               valve_plane_override=None):
    """v2 of extract_lv_volume.segment_lv_cavity: conical + valve-plane aware.

    If `valve_plane_override` is given (e.g. from --manual-review), it is
    used directly instead of running auto_detect_valve_plane — this is the
    hook that lets a human correct the automatic estimate before the cone
    and the rest of the pipeline are built from it.

    Returns (mask, centroid, valve_plane_idx_used).
    """
    if valve_plane_override is not None:
        seed = find_nearest_above_threshold(data, seed, threshold)  # UNCHANGED from v1
        valve_plane_idx = valve_plane_override
        print(f"  [v2] Valve plane manually set to index {valve_plane_idx} "
              f"on axis {axis} (seed at {seed[axis]})")
    else:
        # === CHANGE #1: automatic valve-plane detection ===
        valve_plane_idx, seed = auto_detect_valve_plane(
            data, seed, voxel_sizes, threshold, axis=axis, apex_dir=apex_dir,
            base_radius_mm=base_radius_mm, valve_search_mm=valve_search_mm,
        )
        print(f"  [v2] Valve plane auto-detected at index {valve_plane_idx} "
              f"on axis {axis} (seed at {seed[axis]})")

    # === CHANGE #2: conical constraint instead of a fixed sphere ===
    cone_mask = build_hybrid_mask(
        data.shape, seed, voxel_sizes, axis, apex_dir, valve_plane_idx,
        base_radius_mm, apex_radius_mm, cone_length_mm, apex_sphere_radius_mm
    )

    in_range = ((data > threshold) & cone_mask).astype(np.uint8)
    labeled, _ = ndimage.label(in_range)
    seed_label = labeled[seed[0], seed[1], seed[2]]
    if seed_label == 0:
        raise RuntimeError(
            f"[v2] Seed {seed} not in any above-threshold component within the "
            "cone. Try a larger --sphere-radius/--cone-length or a lower threshold."
        )
    mask = (labeled == seed_label).astype(np.uint8)

    # --- Morphological cleanup: UNCHANGED from v1 ---
    closed = ndimage.binary_closing(mask, structure=ball(2)).astype(np.uint8)
    final_mask = None
    for r in [4, 3, 2]:
        opened = ndimage.binary_opening(closed, structure=ball(r)).astype(np.uint8)
        lab, n = ndimage.label(opened)
        if n > 0 and opened.sum() > 1000:
            sizes = ndimage.sum(opened, lab, range(1, n + 1))
            core = (lab == (np.argmax(sizes) + 1)).astype(np.uint8)
            m = ndimage.binary_dilation(core, structure=ball(r)).astype(np.uint8)
            m = (m & closed).astype(np.uint8)
            final_mask = m
            break
    if final_mask is None:
        raise RuntimeError("[v2] Morphological cleanup eroded the cavity completely.")
    mask = final_mask

    # --- Catheter safety net: UNCHANGED from v1, kept as a backstop. Should
    # mostly be a no-op now since the valve plane + cone already exclude the
    # base-ward catheter path. ---
    mask = remove_catheter_by_position(mask)
    if mask.sum() == 0:
        raise RuntimeError("[v2] Segmentation produced empty mask after catheter removal.")

    # === CHANGE #3: omnidirectional cone-gated extension ===
    mask = extend_mask_within_cone(mask, data, threshold, cone_mask)

    # --- Hole filling + light closing: UNCHANGED from v1 ---
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    for z in range(mask.shape[2]):
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    for y in range(mask.shape[1]):
        mask[:, y, :] = ndimage.binary_fill_holes(mask[:, y, :])
    for x in range(mask.shape[0]):
        mask[x, :, :] = ndimage.binary_fill_holes(mask[x, :, :])
    mask = ndimage.binary_closing(mask, structure=ball(2)).astype(np.uint8)

    centroid = tuple(int(round(c)) for c in ndimage.center_of_mass(mask))
    return mask, centroid, valve_plane_idx


# ──────────────────────────────────────────────────────────────────────────────
# QC + comparison outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_comparison_overlay(data, mask_v1, mask_v2, centroid, frame_name, out_path,
                             vmin=-200, vmax=600, axis=None, valve_plane_idx=None):
    """3-plane QC overlay comparing v1 and v2.

    Draws v1 and v2 as outline contours only — no agreement fill/boundary.
    To avoid the original line-occlusion problem (cyan drawn on top of a
    coincident red line makes v1 look like a broken arc even when it isn't),
    v1 and v2 use different LINESTYLES, not just colors: solid red for v1,
    dashed cyan for v2. Where the two masks agree exactly, both a solid and
    a dashed line are visible at that boundary instead of one hiding the
    other. A green dashed line marks the (possibly manually-corrected)
    valve-plane cut, when provided, so you can see directly where v2's base
    cutoff sits.

    Either mask may be None (--method sphere / conical); only the masks
    that were computed are drawn.
    """
    cx, cy, cz = centroid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    legend = []
    if mask_v1 is not None:
        legend.append("red = v1")
    if mask_v2 is not None:
        legend.append("cyan dashed = v2")
    if valve_plane_idx is not None:
        legend.append("green = valve plane")
    fig.suptitle(f"{frame_name}   ({', '.join(legend)})", fontsize=11)

    def slices(m):
        if m is None:
            return [None, None, None]
        return [m[:, :, cz].T, m[:, cy, :].T, m[cx, :, :].T]

    s1, s2 = slices(mask_v1), slices(mask_v2)
    panels = [
        (axes[0], data[:, :, cz].T, s1[0], s2[0], f"Axial z={cz}"),
        (axes[1], data[:, cy, :].T, s1[1], s2[1], f"Coronal y={cy}"),
        (axes[2], data[cx, :, :].T, s1[2], s2[2], f"Sagittal x={cx}"),
    ]

    for panel_idx, (ax, slc, m1, m2, title) in enumerate(panels):
        ax.imshow(slc, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

        if m1 is not None and m1.any():
            ax.contour(m1.astype(bool), levels=[0.5], colors=["red"],
                       linewidths=1.2, linestyles="solid")
        if m2 is not None and m2.any():
            ax.contour(m2.astype(bool), levels=[0.5], colors=["cyan"],
                       linewidths=1.2, linestyles="dashed")

        # axis==1 ("y") is drawn as a horizontal line in the axial panel
        # (index 0) and a vertical line in the sagittal panel (index 2) —
        # same convention as review_valve_plane_interactive().
        if valve_plane_idx is not None and axis == 1:
            if panel_idx == 0:
                ax.axhline(valve_plane_idx, color="lime", lw=1.2, linestyle="--")
            elif panel_idx == 2:
                ax.axvline(valve_plane_idx, color="lime", lw=1.2, linestyle="--")

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _overlay_centroid(data, mask_v1, mask_v2):
    """Slice position for the QC overlay: centroid of v2, else v1, else volume center."""
    for m in (mask_v2, mask_v1):
        if m is not None and m.any():
            return tuple(int(round(c)) for c in ndimage.center_of_mass(m))
    return tuple(s // 2 for s in data.shape)


def save_comparison_plot(results, out_path, methods=("v1", "v2")):
    """Volume-over-time for the methods that were run (v1 and/or v2) on the
    same axes, folders concatenated sequentially (same convention as
    extract_lv_volume.save_volume_plot)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    folders = []
    for r in results:
        if r["folder"] not in folders:
            folders.append(r["folder"])

    frames, v1_vals, v2_vals, boundaries = [], [], [], []
    for folder in folders:
        subset = [r for r in results if r["folder"] == folder]
        f = [r["phase_pct"] for r in subset]
        frames += f
        v1_vals += [r.get("volume_ml_v1") for r in subset]
        v2_vals += [r.get("volume_ml_v2") for r in subset]
        boundaries.append((min(f), folder))

    plotted = []
    if "v1" in methods:
        ax.plot(frames, v1_vals, "o--", color="tab:red",
                label="v1: sphere + threshold + directional extend")
        plotted += v1_vals
    if "v2" in methods:
        ax.plot(frames, v2_vals, "o-", color="tab:cyan",
                label="v2: conical + valve-plane + cone-gated extend")
        plotted += v2_vals

    for x, _ in boundaries[1:]:
        ax.axvline(x - 2.5, color="gray", linestyle=":", linewidth=1)
    ymax = max(plotted)
    for x, folder in boundaries:
        ax.annotate(folder, (x, ymax), textcoords="offset points",
                    xytext=(2, 4), fontsize=8, color="gray")

    ax.set_xlabel("Cardiac phase (% R-R)")
    ax.set_ylabel("LV cavity volume (mL)")
    if len(methods) == 2:
        ax.set_title("LV Volume Over Time: v1 vs v2 Segmentation")
    else:
        ax.set_title(f"LV Volume Over Time: {methods[0]} Segmentation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison volume plot saved: {out_path}")


def save_comparison_csv(results, out_path, methods=("v1", "v2")):
    """One row per frame; columns only for the methods that were run."""
    both = len(methods) == 2
    cols = ["phase_pct", "folder", "frame_idx", "filename"]
    cols += [f"volume_ml_{m}" for m in methods]
    if both:
        cols.append("diff_ml")
    cols += [f"voxel_count_{m}" for m in methods]
    if "v2" in methods:
        cols += ["valve_plane_auto", "valve_plane_final", "manually_corrected"]

    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            row = [f"{r['phase_pct']:g}", r["folder"], r["frame_idx"], r["filename"]]
            row += [f"{r[f'volume_ml_{m}']:.4f}" for m in methods]
            if both:
                row.append(f"{r['volume_ml_v2'] - r['volume_ml_v1']:+.4f}")
            row += [r[f"voxel_count_{m}"] for m in methods]
            if "v2" in methods:
                corrected = r["valve_plane_final"] != r["valve_plane_auto"]
                row += [r["valve_plane_auto"], r["valve_plane_final"], corrected]
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"Volume CSV saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare v1 (sphere) vs v2 (conical + valve-plane) LV segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--folders", nargs="+", default=["retro_A_frames", "retro_B_frames"])
    parser.add_argument("--method", choices=["sphere", "conical", "both"], default="both",
                         help="Which segmentation to run: 'sphere' (v1), 'conical' (v2), "
                              "or 'both' for the head-to-head comparison (~2x slower). "
                              "Default: both")
    parser.add_argument("--seed", type=int, nargs=3, metavar=("X", "Y", "Z"),
                         help="Seed voxel coordinate inside the LV cavity (skip interactive selection)")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override intensity threshold (HU). Shared by v1 and v2 for a fair comparison.")
    parser.add_argument("--sphere-radius", type=float, default=70,
                         help="v1 sphere radius (mm) AND v2 cone base radius at the valve plane (mm). Default: 50")
    parser.add_argument("--apex-radius", type=float, default=15,
                         help="v2 cone radius (mm) at the apex end of the taper. Default: 15")
    parser.add_argument("--cone-length", type=float, default=90,
                         help="v2 distance (mm) over which the cone tapers from base_radius to apex_radius. Default: 90")
    parser.add_argument("--apex-sphere-radius", type=float, default=25,
                         help="v2 hybrid: radius (mm) of the sphere bulge centered at "
                              "the apex tip, unioned with the cone to recover apex "
                              "voxels the taper alone would cut off. Default: 25")
    parser.add_argument("--valve-search-mm", type=float, default=60,
                         help="v2 how far (mm) from the seed to search for the valve-plane waist. Default: 60")
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                         help="LV long axis in voxel space. Default: 1 (matches v1's extend_directions convention)")
    parser.add_argument("--apex-dir", type=int, default=1, choices=[-1, 1],
                         help="+1 or -1: which direction along --axis is the apex. Default: 1 (+y = apex, matches v1)")
    parser.add_argument("--temporal-window", type=int, default=1,
                         help="Temporal smoothing window (frames on each side), applied identically to v1 and v2 masks.")
    parser.add_argument("--manual-review", action="store_true",
                         help="MANUAL CORRECTION STEP: before finalizing v2 for each "
                              "frame, opens an interactive window (slider + Confirm "
                              "button) showing the auto-detected valve plane so you "
                              "can drag-correct it. Requires a display; opens once per "
                              "frame (~20x for a full retro_A+retro_B run). Only the "
                              "--axis 1 (default) line overlay is drawn.")
    parser.add_argument("--phases", type=float, nargs="+", default=None, metavar="PCT",
                         help="Only segment these cardiac phases, in percent of the "
                              "R-R interval, e.g. --phases 0 40 for a quick QC check. "
                              "Assumes the files across all --folders (in order) are "
                              "evenly spaced over 0-100%% (20 files -> 0, 5, ..., 95%%). "
                              "Seed and threshold still come from the first phase. "
                              "Temporal smoothing is skipped (phases aren't contiguous).")
    parser.add_argument("--output", default="results_conical",
                         help="Output directory (default: results_conical)")
    args = parser.parse_args()

    matplotlib.use("Agg")

    frames = collect_frames(args.folders)
    if not frames:
        print("ERROR: No NIfTI files found in specified folders.")
        return
    print(f"Found {len(frames)} frames across {len(args.folders)} folder(s)")

    os.makedirs(args.output, exist_ok=True)
    qc_dir = os.path.join(args.output, "qc_overlays")
    os.makedirs(qc_dir, exist_ok=True)

    first_folder, first_path = frames[0]
    print(f"\nLoading first frame: {first_path}")
    first_data, first_affine, first_voxel_sizes, first_voxel_vol = load_nifti(first_path)

    if args.seed:
        seed = tuple(args.seed)
        print(f"Using provided seed: {seed}")
    else:
        matplotlib.use("TkAgg")
        seed = select_seed_interactive(first_data)
        matplotlib.use("Agg")

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold:.0f} HU")
    else:
        threshold = estimate_threshold(first_data, seed)
        print(f"Auto-detected threshold: {threshold:.0f} HU")

    del first_data  # free memory; each phase is loaded again in the loop

    run_v1 = args.method in ("sphere", "both")
    run_v2 = args.method in ("conical", "both")
    methods = tuple(m for m, on in (("v1", run_v1), ("v2", run_v2)) if on)
    print(f"Segmentation method(s): {', '.join(methods)}")

    # Cardiac phase (% R-R) of each file, assuming even spacing over the cycle
    phase_step = 100.0 / len(frames)
    phase_pct = [i * phase_step for i in range(len(frames))]

    if args.phases is not None:
        selected = []
        for p in args.phases:
            i = int(round(p / phase_step))
            if not (0 <= i < len(frames)) or abs(i * phase_step - p) > 1e-6:
                print(f"ERROR: phase {p:g}% not available. Phases: "
                      + ", ".join(f"{q:g}" for q in phase_pct))
                return
            selected.append(i)
        frames = [frames[i] for i in selected]
        phase_pct = [phase_pct[i] for i in selected]
        print("Only segmenting selected phase(s): "
              + ", ".join(f"{p:g}% = {fo}/{os.path.basename(fp)}"
                          for p, (fo, fp) in zip(phase_pct, frames)))

    smoothing = args.temporal_window > 0 and args.phases is None
    if args.temporal_window > 0 and not smoothing:
        print("Temporal smoothing skipped (--phases selects non-contiguous phases).")

    # Per-phase outputs written as soon as each phase finishes, so you can
    # inspect the segmentation while the run is still going. These use the
    # UNSMOOTHED masks; if temporal smoothing is on, the final smoothed
    # overlays/CSV in qc_overlays/ and volumes_comparison.csv come at the end.
    live_dir = os.path.join(args.output, "qc_overlays_live") if smoothing else qc_dir
    os.makedirs(live_dir, exist_ok=True)
    live_csv = os.path.join(args.output, "volumes_live.csv")
    with open(live_csv, "w") as f:
        f.write(",".join(["phase_pct", "folder", "frame_idx", "filename"]
                         + [f"volume_ml_{m}" for m in methods]
                         + (["valve_plane"] if run_v2 else [])) + "\n")
    print(f"Per-phase overlays -> {live_dir}/   per-phase volumes -> {live_csv}")

    # ── Pass 1: segment every phase with the selected method(s) ────────
    v1_raw, v2_raw, valve_indices, meta = [], [], [], []
    shared_affine = shared_voxel_vol = None

    for i, ((folder_name, filepath), pct) in enumerate(zip(frames, phase_pct)):
        filename = os.path.basename(filepath)
        frame_idx = int(filename.split("frame_")[1].split(".")[0])
        print(f"\n[{i + 1}/{len(frames)}] phase {pct:g}%  ({folder_name}/{filename})")

        data, affine, voxel_sizes, voxel_vol = load_nifti(filepath)
        if shared_affine is None:
            shared_affine, shared_voxel_vol = affine, voxel_vol

        mask_v1 = mask_v2 = auto_idx = final_idx = None
        if run_v1:
            mask_v1, _ = segment_lv_cavity(
                data, seed, voxel_sizes, threshold=threshold,
                sphere_radius_mm=args.sphere_radius,
            )

        if not run_v2:
            pass
        elif args.manual_review:
            # MANUAL CORRECTION STEP: detect automatically first, then let
            # the user confirm/drag-correct before the cone is built.
            auto_idx, snapped_seed = auto_detect_valve_plane(
                data, seed, voxel_sizes, threshold, axis=args.axis,
                apex_dir=args.apex_dir, base_radius_mm=args.sphere_radius,
                valve_search_mm=args.valve_search_mm,
            )
            matplotlib.use("TkAgg")
            final_idx = review_valve_plane_interactive(
                data, snapped_seed, voxel_sizes, args.axis, args.apex_dir,
                auto_idx, frame_name=f"{folder_name}/{filename}",
            )
            matplotlib.use("Agg")
            mask_v2, _, _ = segment_lv_cavity_conical(
                data, seed, voxel_sizes, threshold=threshold,
                axis=args.axis, apex_dir=args.apex_dir,
                base_radius_mm=args.sphere_radius, apex_radius_mm=args.apex_radius,
                cone_length_mm=args.cone_length, valve_search_mm=args.valve_search_mm,
                apex_sphere_radius_mm=args.apex_sphere_radius,
                valve_plane_override=final_idx,
            )
        else:
            mask_v2, _, final_idx = segment_lv_cavity_conical(
                data, seed, voxel_sizes, threshold=threshold,
                axis=args.axis, apex_dir=args.apex_dir,
                base_radius_mm=args.sphere_radius, apex_radius_mm=args.apex_radius,
                cone_length_mm=args.cone_length, valve_search_mm=args.valve_search_mm,
                apex_sphere_radius_mm=args.apex_sphere_radius,
            )
            auto_idx = final_idx  # no manual review -> auto value is final

        line = []
        if run_v1:
            v1_vol = compute_volume_ml(mask_v1, voxel_vol)
            line.append(f"v1: {v1_vol:.2f} mL")
        if run_v2:
            v2_vol = compute_volume_ml(mask_v2, voxel_vol)
            line.append(f"v2: {v2_vol:.2f} mL")
        if run_v1 and run_v2:
            line.append(f"(diff {v2_vol - v1_vol:+.2f} mL)")
        print("  " + "  |  ".join(line))

        # Live outputs for this phase (unsmoothed)
        overlay_path = os.path.join(
            live_dir, f"phase_{pct:04.1f}pct_{folder_name}_frame_{frame_idx:02d}_overlay.png")
        save_comparison_overlay(data, mask_v1, mask_v2,
                                 _overlay_centroid(data, mask_v1, mask_v2),
                                 f"phase {pct:g}% — {filename}",
                                 overlay_path, axis=args.axis, valve_plane_idx=final_idx)
        with open(live_csv, "a") as f:
            row = [f"{pct:g}", folder_name, frame_idx, filename]
            row += [f"{compute_volume_ml(m, voxel_vol):.4f}"
                    for m in (mask_v1, mask_v2) if m is not None]
            if run_v2:
                row.append(final_idx)
            f.write(",".join(str(v) for v in row) + "\n")
        print(f"  overlay saved: {overlay_path}")

        del data  # keep only the (much smaller) masks in memory
        v1_raw.append(mask_v1)
        v2_raw.append(mask_v2)
        valve_indices.append((auto_idx, final_idx))
        meta.append((folder_name, frame_idx, filename, filepath, pct))

    # ── Pass 2: temporal smoothing, applied identically to each method ──
    v1_masks, v2_masks = v1_raw, v2_raw
    if smoothing:
        print(f"\nTemporal smoothing (window={args.temporal_window})...")
        if run_v1:
            v1_masks = temporal_smooth_masks(v1_raw, window=args.temporal_window)
        if run_v2:
            v2_masks = temporal_smooth_masks(v2_raw, window=args.temporal_window)

    # ── Pass 3: final volumes, overlays, outputs ────────────────────────
    results = []
    for (folder_name, frame_idx, filename, filepath, pct), m1, m2, (auto_idx, final_idx) in zip(
        meta, v1_masks, v2_masks, valve_indices
    ):
        r = {"phase_pct": pct, "folder": folder_name, "frame_idx": frame_idx, "filename": filename,
             "valve_plane_auto": auto_idx, "valve_plane_final": final_idx}
        if run_v1:
            r["volume_ml_v1"] = compute_volume_ml(m1, shared_voxel_vol)
            r["voxel_count_v1"] = int(m1.sum())
        if run_v2:
            r["volume_ml_v2"] = compute_volume_ml(m2, shared_voxel_vol)
            r["voxel_count_v2"] = int(m2.sum())
        results.append(r)

        # Without smoothing, the live overlays already show these final masks.
        if smoothing:
            data = load_nifti(filepath)[0]
            overlay_name = f"phase_{pct:04.1f}pct_{folder_name}_frame_{frame_idx:02d}_overlay.png"
            save_comparison_overlay(data, m1, m2, _overlay_centroid(data, m1, m2),
                                     f"phase {pct:g}% — {filename} (smoothed)",
                                     os.path.join(qc_dir, overlay_name),
                                     axis=args.axis, valve_plane_idx=final_idx)
            del data

    save_comparison_csv(results, os.path.join(args.output, "volumes_comparison.csv"), methods)
    save_comparison_plot(results, os.path.join(args.output, "volume_comparison_plot.png"), methods)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for r in results:
        parts = [f"{m}={r[f'volume_ml_{m}']:.2f} mL" for m in methods]
        if run_v1 and run_v2:
            parts.append(f"diff={r['volume_ml_v2'] - r['volume_ml_v1']:+.2f} mL")
        print(f"  phase {r['phase_pct']:4g}%  ({r['folder']}/frame_{r['frame_idx']:02d}): "
              + "  ".join(parts))
    print()
    for m in methods:
        vals = np.array([r[f"volume_ml_{m}"] for r in results])
        print(f"  mean {m} volume:        {vals.mean():.2f} mL")
    if run_v1 and run_v2:
        diffs = np.array([r["volume_ml_v2"] - r["volume_ml_v1"] for r in results])
        print(f"  mean (v2-v1):          {diffs.mean():+.2f} mL")
        print(f"  mean |v2-v1|:          {np.abs(diffs).mean():.2f} mL")
        print(f"  max |v2-v1|:           {np.abs(diffs).max():.2f} mL")
    if args.manual_review and run_v2:
        n_corrected = sum(1 for r in results if r["valve_plane_final"] != r["valve_plane_auto"])
        print(f"  frames manually corrected: {n_corrected}/{len(results)}")
    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
