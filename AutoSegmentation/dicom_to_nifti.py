"""
Convert retrospectively-gated CT DICOM series (one folder per series, all
cardiac phases mixed together as IM0, IM1, ...) into one NIfTI volume per
phase, named frame_NN.nii.gz — the input format extract_lv_volume.py and
extract_lv_volume_conical.py expect.

HOW PHASES ARE SEPARATED
    The file names (IM0, IM1, IM10, IM100, ...) are not in acquisition order,
    so we sort by the InstanceNumber header instead. Every phase covers the
    same set of slice positions, so walking the slices in InstanceNumber
    order, the k-th time a given z position appears it belongs to phase k.
    Each phase must end up with the same number of slices, otherwise the
    script stops.

    For PEXA5: RetroA ("0-50-5") = 11 phases (0%..50%), RetroB = 9 phases
    (55%..95%), 269 slices each, 0.322266 x 0.322266 x 0.625 mm voxels.

OUTPUT
    Intensities are real HU (RescaleSlope/Intercept applied). Voxel array is
    indexed data[x, y, z] = data[column, row, slice], slices sorted by
    increasing patient z, and the affine carries the true voxel size (this is
    what the pipeline uses for mL).

Usage:
    python dicom_to_nifti.py                       # PEXA5 RetroA/RetroB defaults
    python dicom_to_nifti.py --median 3            # + 3x3x3 median denoise (as in
                                                   #   User_Mask_LVVolumesEF.m)
    python dicom_to_nifti.py --series DIR1 DIR2 --out-names a_frames b_frames
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import nibabel as nib
import pydicom
from scipy import ndimage

MIDGETT = "/Users/Postdoc/Midgett_data"
DEFAULT_SERIES = [
    os.path.join(MIDGETT, "PEXA9 ExVivo CTA RV+LV+T3 RetroA"),
    os.path.join(MIDGETT, "PEXA9 ExVivo CTA RV+LV+T3 RetroB"),
]
DEFAULT_OUT_NAMES = ["PEXA_9_retro_A_frames_median_denoise", "PEXA_9_retro_B_frames_median_denoise"]


def read_headers(series_dir):
    """Header-only read of every DICOM file in the folder."""
    headers = []
    for name in os.listdir(series_dir):
        path = os.path.join(series_dir, name)
        if name.startswith(".") or not os.path.isfile(path):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        headers.append((path, ds))
    if not headers:
        raise SystemExit(f"No DICOM files found in {series_dir}")
    return headers


def split_into_phases(headers):
    """Group slices into phases (see module docstring). Returns a list of
    phases, each a list of (z, path) sorted by increasing z."""
    headers.sort(key=lambda h: int(h[1].InstanceNumber))
    seen = defaultdict(int)
    phases = defaultdict(list)
    for path, ds in headers:
        z = round(float(ds.ImagePositionPatient[2]), 3)
        phases[seen[z]].append((z, path))
        seen[z] += 1

    phases = [sorted(phases[k]) for k in sorted(phases)]
    counts = {len(p) for p in phases}
    if len(counts) != 1:
        raise SystemExit(f"Phases have unequal slice counts {sorted(counts)} — "
                         "series may be incomplete or not phase-blocked.")
    return phases


def build_affine(ds, dz):
    """Voxel-index -> RAS mm affine for an axial stack indexed [col, row, slice]."""
    iop = [float(v) for v in ds.ImageOrientationPatient]
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-3):
        print(f"  WARNING: non-axial ImageOrientationPatient {iop}; affine "
              "orientation is approximate (voxel sizes are still correct).")
    dy, dx = (float(v) for v in ds.PixelSpacing)  # (row spacing, column spacing)
    x0, y0, z0 = (float(v) for v in ds.ImagePositionPatient)
    # DICOM is LPS, NIfTI is RAS -> flip the sign of x and y
    return np.array([
        [-dx, 0, 0, -x0],
        [0, -dy, 0, -y0],
        [0, 0, dz, z0],
        [0, 0, 0, 1],
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", nargs="+", default=DEFAULT_SERIES,
                   help="DICOM series folders (default: PEXA5 RetroA and RetroB)")
    p.add_argument("--out-names", nargs="+", default=DEFAULT_OUT_NAMES,
                   help="Output folder name per series (default: retro_A_frames retro_B_frames)")
    p.add_argument("--out-root", default=".", help="Where to create the output folders")
    p.add_argument("--median", type=int, default=0,
                   help="Apply an N x N x N median filter to each volume (0 = off; "
                        "the MATLAB script used 3)")
    args = p.parse_args()

    if len(args.series) != len(args.out_names):
        raise SystemExit("--series and --out-names must have the same length.")

    for series_dir, out_name in zip(args.series, args.out_names):
        print(f"\n== {series_dir}")
        headers = read_headers(series_dir)
        first = headers[0][1]
        print(f"  {len(headers)} slices, series: {getattr(first, 'SeriesDescription', '?')}")

        phases = split_into_phases(headers)
        n_slices = len(phases[0])
        zs = [z for z, _ in phases[0]]
        dz = float(np.median(np.diff(zs)))
        print(f"  -> {len(phases)} phases x {n_slices} slices, "
              f"z {zs[0]} .. {zs[-1]} mm, slice spacing {dz:.4f} mm")

        out_dir = os.path.join(args.out_root, out_name)
        os.makedirs(out_dir, exist_ok=True)

        for t, phase in enumerate(phases):
            slices = []
            for _, path in phase:
                ds = pydicom.dcmread(path)
                hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) \
                    + float(ds.RescaleIntercept)
                slices.append(hu.T)  # (rows, cols) -> (x=col, y=row)
            vol = np.stack(slices, axis=2)
            if args.median > 1:
                vol = ndimage.median_filter(vol, size=args.median)

            affine = build_affine(pydicom.dcmread(phase[0][1], stop_before_pixels=True), dz)
            out_path = os.path.join(out_dir, f"frame_{t:02d}.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), out_path)
            print(f"  {out_path}  shape {vol.shape}  "
                  f"HU [{vol.min():.0f}, {vol.max():.0f}]")

        vox = nib.affines.voxel_sizes(affine)
        print(f"  voxel size (mm): {vox[0]:.4f} x {vox[1]:.4f} x {vox[2]:.4f}")


if __name__ == "__main__":
    main()
