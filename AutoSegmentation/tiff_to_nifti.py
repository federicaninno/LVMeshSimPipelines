"""
Convert an ImageJ-enhanced TIFF stack (all cardiac phases of all series in
one file, as produced by Contrast_Enhance_PEXA.ijm on the raw DICOM slices)
into one NIfTI per phase on EXACTLY the same voxel grid as the
DICOM-derived NIfTIs from dicom_to_nifti.py.

Because the grids match, a seed, base-cut position or mask from the HU data
can be used directly on the enhanced data and vice versa — e.g. mask_editor.py
can segment on HU and display the enhanced images (--display-folders).

HOW THE TIFF MAPS ONTO THE DICOM GRID (verified for PEXA5)
    The TIFF holds the DICOM slices in InstanceNumber order: series after
    series (RetroA then RetroB), within a series phase after phase, and
    within a phase in DECREASING patient z. dicom_to_nifti.py stores slices
    in INCREASING z, so each phase's block of slices is reversed here.
    TIFF slice k  ->  phase k // n_slices,  z = n_slices - 1 - (k % n_slices).
    The script reports, per phase, the correlation between the converted
    volume and the reference volume as a check.

INTENSITIES
    Stored as uint8 (0..255), always with the same polarity as HU (the
    contrast-filled cavity BRIGHT), so the files work like the HU data in
    the pipeline. The enhancement macro ends with "Invert", so in
    PEXA_5_Smooth.tif the stored cavity is dark (~9); the polarity is
    detected from the sign of the correlation with the reference on the
    first phase and the data are inverted when it is negative. To view them
    the ImageJ way (cavity dark), use mask_editor.py --invert-display.

Input: the ENHANCED stack (PEXA_5_Smooth.tif for PEXA5).

Usage:
    python tiff_to_nifti.py /Users/Postdoc/Midgett_data/PEXA_5_Smooth.tif \\
        --reference retro_A_frames_no_median_denoise retro_B_frames_no_median_denoise \\
        --out-names retro_A_frames_enhanced retro_B_frames_enhanced
"""

import os
import argparse
from glob import glob

import numpy as np
import nibabel as nib
import tifffile


def open_stack(path):
    """Memory-map the TIFF if possible (fast), else fall back to page reads."""
    try:
        arr = tifffile.memmap(path, mode="r")
        return arr if arr.ndim == 3 else arr.reshape(-1, *arr.shape[-2:])
    except Exception:
        tif = tifffile.TiffFile(path)
        print("  (TIFF not memory-mappable; reading page by page — slower)")

        class Pages:
            shape = (len(tif.pages), *tif.pages[0].shape)

            def __getitem__(self, sl):
                return np.stack([tif.pages[i].asarray() for i in range(*sl.indices(self.shape[0]))])
        return Pages()


def central_corr(a, b):
    """Correlation over the central half of each axis (avoids the FOV border)."""
    sl = tuple(slice(s // 4, 3 * s // 4) for s in a.shape)
    x, y = a[sl].ravel().astype(np.float32), b[sl].ravel().astype(np.float32)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tiff")
    p.add_argument("--reference", nargs="+", required=True,
                   help="DICOM-derived NIfTI folders (dicom_to_nifti.py output), in the "
                        "same series order as in the TIFF, e.g. RetroA then RetroB")
    p.add_argument("--out-names", nargs="+", required=True,
                   help="Output folder per reference folder")
    p.add_argument("--out-root", default=".", help="Where to create the output folders")
    args = p.parse_args()

    if len(args.reference) != len(args.out_names):
        raise SystemExit("--reference and --out-names must have the same length.")

    # Accept both naming schemes: frame_00.nii.gz and <prefix>_frame_00.nii.gz
    refs = [sorted(set(glob(os.path.join(d, "frame_*.nii*")))
                   | set(glob(os.path.join(d, "*_frame_*.nii*"))))
            for d in args.reference]
    for d, files in zip(args.reference, refs):
        if not files:
            raise SystemExit(f"No frame_*.nii* or *_frame_*.nii* files in {d}")

    stack = open_stack(args.tiff)
    n_pages, rows, cols = stack.shape
    ref_shape = nib.load(refs[0][0]).shape
    n_slices = ref_shape[2]
    expected = sum(len(f) for f in refs) * n_slices
    print(f"TIFF: {n_pages} slices of {rows}x{cols}; reference: "
          f"{sum(len(f) for f in refs)} phases x {n_slices} slices of {ref_shape[0]}x{ref_shape[1]}")
    if n_pages != expected or (cols, rows) != tuple(ref_shape[:2]):
        raise SystemExit(f"Size mismatch: TIFF has {n_pages} slices of {rows}x{cols}, "
                         f"reference grid needs {expected} slices of {ref_shape[1]}x{ref_shape[0]}.")

    k = 0
    invert = None  # decided on the first phase
    for ref_dir, files, out_name in zip(args.reference, refs, args.out_names):
        out_dir = os.path.join(args.out_root, out_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n== {ref_dir} -> {out_dir}")
        for ref_path in files:
            ref = nib.load(ref_path)
            block = np.asarray(stack[k:k + n_slices])        # (slice, row, col), z decreasing
            k += n_slices
            vol = np.ascontiguousarray(block[::-1].transpose(2, 1, 0))  # -> (x=col, y=row, z increasing)
            ref_data = np.asarray(ref.dataobj)
            if invert is None:
                invert = central_corr(vol, ref_data) < 0
                print("  stored TIFF values are inverted relative to HU -> inverting"
                      if invert else "  stored TIFF values have HU polarity -> kept as-is")
            if invert:
                vol = 255 - vol
            r = central_corr(vol, ref_data)
            out_path = os.path.join(out_dir, os.path.basename(ref_path))
            nib.save(nib.Nifti1Image(vol.astype(np.uint8), ref.affine), out_path)
            flag = "" if r > 0.6 else "   <-- LOW: check slice order / series order!"
            print(f"  {out_path}  corr with reference r={r:.3f}{flag}")


if __name__ == "__main__":
    main()
