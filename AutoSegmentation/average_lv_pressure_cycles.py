"""
Average 5 LV-pressure cycles and map the average curve to CT phases
(0%, 5%, ..., 95%).

Required packages:
    pip install pandas numpy scipy matplotlib openpyxl

How this script interprets your selection:
- Sampling interval = 1 ms.
- The selected trace begins near the LV-pressure minimum before beat 1.
- It contains 5 complete beats and continues to the peak of beat 6.
- Therefore, six cycle-start minima should define five complete cycles.
- Any data after the sixth minimum are ignored for cycle averaging.

Important:
Without ECG, the detected LV-pressure minima are only a surrogate for the
scanner's ECG phase reference. Review the diagnostic plot and, if necessary,
enter the six cycle-boundary sample indices manually.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# =========================== USER SETTINGS ===========================

INPUT_EXCEL = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles.xlsx")
OUTPUT_EXCEL = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_processed.xlsx")
DIAGNOSTIC_PLOT = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_cycle_detection.png")
AVERAGE_CURVE_PLOT = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_average.png")
PRESSURE_VOLUME_PLOT = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_average_with_volume.png")
PV_LOOP_PLOT = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_PV_loop.png")

# Same figures after shifting the pressure curve so that peak pressure
# falls at the CT phase of peak volume, plus the PV-loop data (original
# and shifted pressure per CT phase) as CSV.
PRESSURE_VOLUME_SHIFTED_PLOT = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_average_with_volume_shifted.png")
PV_LOOP_SHIFTED_PLOT = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_PV_loop_shifted.png")
PV_LOOP_DATA_CSV = Path("/Users/federicaninno/PEXA_Heart_Simulations/PEXA12_LVpressures_5cycles_PV_loop_data.csv")

# LV volumes per CT phase, produced by the SAM segmentation pipeline.
# Expected columns: folder, frame_idx, filename, volume_ml, voxel_count.
# "retro_A_frames" frame_idx 0..10 are assumed to cover CT phases 0-50%
# and "retro_B_frames" frame_idx 0..8 are assumed to cover CT phases 55-95%,
# matching the RetroA/RetroB split already used for CT_PHASES_PERCENT below.
VOLUMES_CSV = Path("/Users/federicaninno/PEXA_Heart_Simulations/results/volumes.csv")

SHEET_NAME = 0             # First Excel sheet
PRESSURE_COLUMN = 0        # First column; may also be a column name
HEADER_ROWS_TO_SKIP = 3    # Your file has: "LV Millar", "Mean", "mmHg"

SAMPLING_INTERVAL_MS = 1.0

# Enter the beginning of the exported selection.
# Examples:
#   "04:38:54.860"
#   "2024-09-19 04:38:54.860"
# PEXA_12
START_TIME = "04:38:54.210"
# PEXA_5
#START_TIME = "03:14:15.05"

# Optional check. Set to None if not known.
# The script compares this with the end reconstructed from sample count.
# PEXA_12
END_TIME_FOR_CHECK = "04:39:02.700"
# PEXA_5
#END_TIME_FOR_CHECK = "03:14:23.685"

# Example:
# END_TIME_FOR_CHECK = "04:39:03.359"

NUMBER_OF_COMPLETE_CYCLES = 5

# Expected cycle-duration range. Adjust if the heart rate is outside
# approximately 30-240 beats/min.
MIN_CYCLE_DURATION_S = 0.25
MAX_CYCLE_DURATION_S = 2.00

# Smoothing used only for boundary detection, not for averaging.
DETECTION_SMOOTHING_MS = 25.0

# If automatic detection is imperfect, enter SIX sample indices here.
# These should be the minima that begin beats 1, 2, 3, 4, 5, and 6.
# Example: MANUAL_BOUNDARY_INDICES = [15, 1620, 3210, 4805, 6410, 7998]
MANUAL_BOUNDARY_INDICES = None

# CT reconstructions are available every 5%, from 0% through 95%.
CT_PHASES_PERCENT = np.arange(0, 100, 5)

# Average curve resolution. 0, 1, 2, ..., 100%.
NORMALIZED_PHASES_PERCENT = np.arange(0, 101, 1)

# ===================================================================


def parse_clock_or_datetime(value: str) -> datetime:
    """Parse a clock time or a full datetime."""
    formats = (
        "%H:%M:%S.%f",
        "%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            # For clock-only inputs, use an arbitrary common date.
            if fmt.startswith("%H"):
                parsed = parsed.replace(year=2000, month=1, day=1)
            return parsed
        except ValueError:
            continue
    raise ValueError(
        f"Could not parse time {value!r}. "
        "Use HH:MM:SS.sss or YYYY-MM-DD HH:MM:SS.sss."
    )


def load_pressure(path: Path) -> np.ndarray:
    """Load one pressure column and discard nonnumeric header entries."""
    raw = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        header=None,
        usecols=[PRESSURE_COLUMN] if isinstance(PRESSURE_COLUMN, int) else None,
    )

    if isinstance(PRESSURE_COLUMN, int):
        series = raw.iloc[:, 0]
    else:
        named = pd.read_excel(path, sheet_name=SHEET_NAME)
        if PRESSURE_COLUMN not in named.columns:
            raise KeyError(
                f"Column {PRESSURE_COLUMN!r} was not found. "
                f"Available columns: {list(named.columns)}"
            )
        series = named[PRESSURE_COLUMN]

    if HEADER_ROWS_TO_SKIP:
        series = series.iloc[HEADER_ROWS_TO_SKIP:]

    pressure = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)

    if pressure.size < 100:
        raise ValueError(
            f"Only {pressure.size} numeric samples were loaded. "
            "Check PRESSURE_COLUMN and HEADER_ROWS_TO_SKIP."
        )
    return pressure


def reconstruct_time_axis(
    n_samples: int,
    start_time_text: str,
    sampling_interval_ms: float,
) -> tuple[np.ndarray, list[datetime], datetime]:
    """Create elapsed seconds and absolute timestamps."""
    dt_s = sampling_interval_ms / 1000.0
    elapsed_s = np.arange(n_samples, dtype=float) * dt_s

    start_dt = parse_clock_or_datetime(start_time_text)
    timestamps = [
        start_dt + timedelta(seconds=float(seconds))
        for seconds in elapsed_s
    ]
    reconstructed_end = timestamps[-1]
    return elapsed_s, timestamps, reconstructed_end


def detect_six_cycle_start_minima(
    pressure: np.ndarray,
    sampling_interval_ms: float,
) -> np.ndarray:
    """
    Detect six minima defining five complete cycles.

    Candidate minima are detected on a smoothed copy of the signal.
    The best sequence of six candidates is selected based on regularity
    and coverage of the beginning of the exported interval.
    """
    dt_s = sampling_interval_ms / 1000.0
    sigma_samples = max(
        1.0,
        DETECTION_SMOOTHING_MS / sampling_interval_ms,
    )
    smooth = gaussian_filter1d(pressure, sigma=sigma_samples)

    min_distance_samples = max(
        1,
        int(MIN_CYCLE_DURATION_S / dt_s),
    )

    # A modest prominence threshold, scaled to the signal range.
    signal_range = float(np.nanpercentile(smooth, 99) - np.nanpercentile(smooth, 1))
    prominence = max(0.5, 0.03 * signal_range)

    candidates, properties = find_peaks(
        -smooth,
        distance=min_distance_samples,
        prominence=prominence,
    )

    # Include index 0 as a candidate if the export starts very near a minimum.
    early_window = min(
        len(smooth),
        max(3, int(0.20 / dt_s)),
    )
    early_minimum = int(np.argmin(smooth[:early_window]))
    candidates = np.unique(np.concatenate(([early_minimum], candidates)))

    if candidates.size < NUMBER_OF_COMPLETE_CYCLES + 1:
        raise RuntimeError(
            f"Automatic detection found only {candidates.size} candidate minima. "
            "Use MANUAL_BOUNDARY_INDICES to provide six cycle starts."
        )

    required = NUMBER_OF_COMPLETE_CYCLES + 1
    best_group = None
    best_score = np.inf

    # Test consecutive groups of six candidate minima.
    for start in range(candidates.size - required + 1):
        group = candidates[start : start + required]
        intervals_s = np.diff(group) * dt_s

        if np.any(intervals_s < MIN_CYCLE_DURATION_S):
            continue
        if np.any(intervals_s > MAX_CYCLE_DURATION_S):
            continue

        regularity = np.std(intervals_s) / max(np.mean(intervals_s), 1e-12)

        # Prefer a group beginning near the start of the exported selection.
        start_penalty = group[0] / max(len(pressure), 1)

        # Mild preference for a group covering most of the trace,
        # while allowing a trailing partial beat up to peak 6.
        coverage = (group[-1] - group[0]) / max(len(pressure) - 1, 1)
        coverage_penalty = abs(coverage - 0.80)

        score = regularity + 0.5 * start_penalty + 0.1 * coverage_penalty

        if score < best_score:
            best_score = score
            best_group = group

    if best_group is None:
        raise RuntimeError(
            "No plausible sequence of six minima was found within the "
            "configured cycle-duration limits. Adjust MIN/MAX_CYCLE_DURATION_S "
            "or enter MANUAL_BOUNDARY_INDICES."
        )

    return np.asarray(best_group, dtype=int)


def load_volumes_by_ct_phase(
    path: Path,
    ct_phases_percent: np.ndarray,
) -> np.ndarray:
    """
    Load LV volumes (mL) and align them to CT_PHASES_PERCENT order.

    Phases <= 50% are matched, in order, to "retro_A_frames" sorted by
    frame_idx; phases > 50% are matched to "retro_B_frames" sorted by
    frame_idx. This mirrors the RetroA/RetroB split already used for
    "Suggested_CT_dataset" in the CT-phase table.
    """
    volumes_table = pd.read_csv(path)

    volume_by_phase = []
    for group_name, mask in (
        ("retro_A_frames", ct_phases_percent <= 50),
        ("retro_B_frames", ct_phases_percent > 50),
    ):
        n_phases = int(mask.sum())
        group = (
            volumes_table[volumes_table["folder"] == group_name]
            .sort_values("frame_idx")
        )
        if len(group) != n_phases:
            raise ValueError(
                f"Expected {n_phases} rows for {group_name!r} to match the "
                f"CT phases assigned to it, found {len(group)} in {path}."
            )
        volume_by_phase.append(group["volume_ml"].to_numpy(float))

    return np.concatenate(volume_by_phase)


def normalize_cycles(
    pressure: np.ndarray,
    boundaries: np.ndarray,
    normalized_phases_percent: np.ndarray,
) -> np.ndarray:
    """Interpolate each pressure cycle to a common 0-100% phase axis."""
    cycles = []

    for cycle_number, (start_idx, end_idx) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
        start=1,
    ):
        # Include the final boundary sample in both neighboring cycles.
        segment = pressure[start_idx : end_idx + 1]

        if segment.size < 3:
            raise ValueError(f"Cycle {cycle_number} is too short.")

        original_phase = np.linspace(0.0, 100.0, segment.size)
        normalized = np.interp(
            normalized_phases_percent,
            original_phase,
            segment,
        )
        cycles.append(normalized)

    return np.vstack(cycles)


def shift_periodic_curve(
    curve: np.ndarray,
    phases_percent: np.ndarray,
    shift_percent: float,
) -> np.ndarray:
    """
    Circularly shift a 0-100% phase curve later by shift_percent.

    Values are only moved to new phase positions, never interpolated or
    rescaled. The 100% point is the same phase as 0%, so it is dropped
    before rolling and re-created from the new 0% point afterwards.
    """
    step = phases_percent[1] - phases_percent[0]
    n_steps = shift_percent / step
    if not np.isclose(n_steps, round(n_steps)):
        raise ValueError(
            f"Shift of {shift_percent}% is not a multiple of the "
            f"{step}% phase resolution."
        )

    shifted = np.roll(curve[:-1], int(round(n_steps)))
    return np.append(shifted, shifted[0])


def plot_pressure_and_volume(
    pressure_curve: np.ndarray,
    ct_pressure: np.ndarray,
    ct_volume_ml: np.ndarray,
    peak_pressure_phase: float,
    peak_volume_phase: float,
    out_path: Path,
) -> None:
    """Pressure and volume against normalized cardiac phase, peaks marked."""
    fig, pressure_axis = plt.subplots()
    volume_axis = pressure_axis.twinx()

    pressure_axis.plot(
        NORMALIZED_PHASES_PERCENT,
        pressure_curve,
        color="tab:red",
        linewidth=2.5,
        label="Mean LV pressure (5 cycles)",
    )
    pressure_axis.scatter(
        CT_PHASES_PERCENT,
        ct_pressure,
        color="tab:red",
        zorder=3,
        label="CT phases (pressure)",
    )
    volume_axis.plot(
        CT_PHASES_PERCENT,
        ct_volume_ml,
        color="tab:blue",
        linewidth=2.0,
        marker="o",
        label="LV volume (CT phases)",
    )
    pressure_axis.axvline(peak_pressure_phase, color="tab:red", linestyle="--", linewidth=1.0)
    pressure_axis.axvline(peak_volume_phase, color="tab:blue", linestyle="--", linewidth=1.0)

    pressure_axis.set_xlabel("Normalized cardiac phase (%)", fontsize=20)
    pressure_axis.set_ylabel("LV pressure (mmHg)", color="tab:red", fontsize=20)
    volume_axis.set_ylabel("LV volume (mL)", color="tab:blue", fontsize=20)
    pressure_axis.tick_params(axis="y", labelcolor="tab:red", labelsize=16)
    volume_axis.tick_params(axis="y", labelcolor="tab:blue", labelsize=16)
    pressure_axis.tick_params(axis="x", labelsize=16)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pv_loop(
    ct_volume_ml: np.ndarray,
    ct_pressure: np.ndarray,
    out_path: Path,
) -> None:
    """
    Pressure-volume loop, plotted exactly as measured (0-95% phase).
    The loop is not artificially closed back to the 0% point; if the
    95% and 0% points do not coincide, the plot will show a gap.
    """
    gap_volume = ct_volume_ml[-1] - ct_volume_ml[0]
    gap_pressure = ct_pressure[-1] - ct_pressure[0]
    print(
        f"\n{out_path.name} start/end gap (95% phase minus 0% phase): "
        f"{gap_volume:+.2f} mL, {gap_pressure:+.2f} mmHg"
    )

    plt.figure()
    plt.plot(
        ct_volume_ml,
        ct_pressure,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )
    plt.xlabel("Volume (mL)", fontsize=20)
    plt.ylabel("Pressure (mmHg)", fontsize=20)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    pressure = load_pressure(INPUT_EXCEL)

    elapsed_s, timestamps, reconstructed_end = reconstruct_time_axis(
        n_samples=len(pressure),
        start_time_text=START_TIME,
        sampling_interval_ms=SAMPLING_INTERVAL_MS,
    )

    print(f"Loaded numeric pressure samples: {len(pressure):,}")
    print(f"Sampling interval: {SAMPLING_INTERVAL_MS:.3f} ms")
    print(f"Reconstructed duration: {elapsed_s[-1]:.3f} s")
    print(
        "Reconstructed final timestamp: "
        f"{reconstructed_end.strftime('%H:%M:%S.%f')[:-3]}"
    )

    if END_TIME_FOR_CHECK is not None:
        expected_end = parse_clock_or_datetime(END_TIME_FOR_CHECK)
        difference_ms = (reconstructed_end - expected_end).total_seconds() * 1000
        print(
            "Difference between reconstructed and supplied end time: "
            f"{difference_ms:+.3f} ms"
        )
        if abs(difference_ms) > 2 * SAMPLING_INTERVAL_MS:
            print(
                "WARNING: The supplied end time is not consistent with the "
                "sample count and 1-ms interval. Check whether the end time "
                "is inclusive/exclusive or whether rows were omitted."
            )

    if MANUAL_BOUNDARY_INDICES is not None:
        boundaries = np.asarray(MANUAL_BOUNDARY_INDICES, dtype=int)
        if boundaries.size != NUMBER_OF_COMPLETE_CYCLES + 1:
            raise ValueError(
                "MANUAL_BOUNDARY_INDICES must contain exactly six indices "
                "for five complete cycles."
            )
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("Manual boundary indices must be strictly increasing.")
        if boundaries[0] < 0 or boundaries[-1] >= len(pressure):
            raise ValueError("A manual boundary index is outside the data range.")
    else:
        boundaries = detect_six_cycle_start_minima(
            pressure,
            SAMPLING_INTERVAL_MS,
        )

    cycle_lengths_samples = np.diff(boundaries)
    cycle_lengths_s = cycle_lengths_samples * SAMPLING_INTERVAL_MS / 1000.0
    heart_rates_bpm = 60.0 / cycle_lengths_s

    print("\nDetected cycle boundaries:")
    for i, idx in enumerate(boundaries, start=1):
        print(
            f"  Boundary {i}: sample {idx:,}, "
            f"elapsed {elapsed_s[idx]:.3f} s, "
            f"time {timestamps[idx].strftime('%H:%M:%S.%f')[:-3]}, "
            f"pressure {pressure[idx]:.3f} mmHg"
        )

    print("\nCycle durations and approximate heart rates:")
    for i, (duration, heart_rate) in enumerate(
        zip(cycle_lengths_s, heart_rates_bpm),
        start=1,
    ):
        print(
            f"  Cycle {i}: {duration:.3f} s "
            f"({heart_rate:.1f} beats/min)"
        )

    # Diagnostic boundary plot.
    plt.figure(figsize=(13, 5))
    plt.plot(elapsed_s, pressure, linewidth=0.8)
    plt.scatter(
        elapsed_s[boundaries],
        pressure[boundaries],
        marker="o",
        zorder=3,
        label="Detected cycle starts",
    )
    for i, idx in enumerate(boundaries, start=1):
        plt.axvline(elapsed_s[idx], linewidth=0.8, alpha=0.5)
        plt.text(
            elapsed_s[idx],
            pressure[idx],
            f" B{i}",
            verticalalignment="bottom",
        )
    plt.xlabel("Elapsed time from exported selection start (s)")
    plt.ylabel("LV pressure (mmHg)")
    plt.title("Review the six detected minima defining five complete cycles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIAGNOSTIC_PLOT, dpi=200)
    plt.close()

    normalized_cycles = normalize_cycles(
        pressure,
        boundaries,
        NORMALIZED_PHASES_PERCENT,
    )

    mean_pressure = normalized_cycles.mean(axis=0)
    sample_sd = normalized_cycles.std(axis=0, ddof=1)
    sem_pressure = sample_sd / np.sqrt(normalized_cycles.shape[0])
    min_pressure = normalized_cycles.min(axis=0)
    max_pressure = normalized_cycles.max(axis=0)

    # Extract 0, 5, ..., 95%. np.interp works even if the normalized
    # phase resolution is later changed from 1%.
    ct_mean = np.interp(
        CT_PHASES_PERCENT,
        NORMALIZED_PHASES_PERCENT,
        mean_pressure,
    )
    ct_sd = np.interp(
        CT_PHASES_PERCENT,
        NORMALIZED_PHASES_PERCENT,
        sample_sd,
    )
    ct_sem = np.interp(
        CT_PHASES_PERCENT,
        NORMALIZED_PHASES_PERCENT,
        sem_pressure,
    )
    ct_min = np.interp(
        CT_PHASES_PERCENT,
        NORMALIZED_PHASES_PERCENT,
        min_pressure,
    )
    ct_max = np.interp(
        CT_PHASES_PERCENT,
        NORMALIZED_PHASES_PERCENT,
        max_pressure,
    )

    # Average-curve plot.
    plt.figure(figsize=(10, 6))
    for i, cycle in enumerate(normalized_cycles, start=1):
        plt.plot(
            NORMALIZED_PHASES_PERCENT,
            cycle,
            linewidth=0.8,
            alpha=0.55,
            label=f"Cycle {i}",
        )
    plt.plot(
        NORMALIZED_PHASES_PERCENT,
        mean_pressure,
        linewidth=2.5,
        label="Mean of 5 cycles",
    )
    plt.scatter(
        CT_PHASES_PERCENT,
        ct_mean,
        zorder=3,
        label="CT phases",
    )
    plt.xlabel("Normalized cardiac phase (%)")
    plt.ylabel("LV pressure (mmHg)")
    plt.title("Five normalized LV-pressure cycles and their average")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(AVERAGE_CURVE_PLOT, dpi=200)
    plt.close()

    raw_table = pd.DataFrame(
        {
            "Sample_index": np.arange(len(pressure)),
            "Elapsed_time_s": elapsed_s,
            "Timestamp": [
                item.strftime("%H:%M:%S.%f")[:-3]
                for item in timestamps
            ],
            "LV_pressure_mmHg": pressure,
        }
    )

    boundary_table = pd.DataFrame(
        {
            "Boundary_number": np.arange(1, len(boundaries) + 1),
            "Sample_index": boundaries,
            "Elapsed_time_s": elapsed_s[boundaries],
            "Timestamp": [
                timestamps[idx].strftime("%H:%M:%S.%f")[:-3]
                for idx in boundaries
            ],
            "LV_pressure_mmHg": pressure[boundaries],
        }
    )

    cycle_info_table = pd.DataFrame(
        {
            "Cycle": np.arange(1, NUMBER_OF_COMPLETE_CYCLES + 1),
            "Start_sample": boundaries[:-1],
            "End_sample": boundaries[1:],
            "Duration_s": cycle_lengths_s,
            "Approx_heart_rate_bpm": heart_rates_bpm,
        }
    )

    normalized_table = pd.DataFrame(
        {
            "Cardiac_phase_percent": NORMALIZED_PHASES_PERCENT,
            **{
                f"Cycle_{i}_pressure_mmHg": normalized_cycles[i - 1]
                for i in range(1, NUMBER_OF_COMPLETE_CYCLES + 1)
            },
            "Mean_pressure_mmHg": mean_pressure,
            "SD_pressure_mmHg": sample_sd,
            "SEM_pressure_mmHg": sem_pressure,
            "Minimum_pressure_mmHg": min_pressure,
            "Maximum_pressure_mmHg": max_pressure,
        }
    )

    ct_phase_table = pd.DataFrame(
        {
            "CT_phase_percent": CT_PHASES_PERCENT,
            "Mean_LV_pressure_mmHg": ct_mean,
            "SD_across_5_cycles_mmHg": ct_sd,
            "SEM_across_5_cycles_mmHg": ct_sem,
            "Minimum_across_5_cycles_mmHg": ct_min,
            "Maximum_across_5_cycles_mmHg": ct_max,
            "Suggested_CT_dataset": [
                "RetroA" if phase <= 50 else "RetroB"
                for phase in CT_PHASES_PERCENT
            ],
            "Expected_slices_in_phase_volume": 272,
        }
    )

    ct_lv_volume_ml = load_volumes_by_ct_phase(VOLUMES_CSV, CT_PHASES_PERCENT)
    ct_phase_table["LV_volume_mL"] = ct_lv_volume_ml

    # Delay between peak pressure and peak volume. Peak pressure is taken
    # from the 1%-resolution mean curve (100% excluded, it duplicates 0%);
    # peak volume is only known at the 5% CT phases.
    peak_pressure_idx = int(np.argmax(mean_pressure[:-1]))
    peak_pressure_phase = float(NORMALIZED_PHASES_PERCENT[peak_pressure_idx])
    peak_pressure_value = float(mean_pressure[peak_pressure_idx])

    peak_volume_idx = int(np.argmax(ct_lv_volume_ml))
    peak_volume_phase = float(CT_PHASES_PERCENT[peak_volume_idx])
    peak_volume_value = float(ct_lv_volume_ml[peak_volume_idx])

    # Shift that moves peak pressure onto peak volume, wrapped into
    # [-50, 50)%. Positive = pressure curve moved later in the cycle,
    # i.e. peak volume occurs after peak pressure in the original data.
    pressure_shift_percent = (
        (peak_volume_phase - peak_pressure_phase + 50.0) % 100.0
    ) - 50.0
    mean_cycle_duration_s = float(cycle_lengths_s.mean())
    pressure_shift_ms = pressure_shift_percent / 100.0 * mean_cycle_duration_s * 1000.0

    print("\nPeak alignment:")
    print(f"  Peak pressure: {peak_pressure_value:.2f} mmHg at {peak_pressure_phase:.0f}% phase")
    print(f"  Peak volume:   {peak_volume_value:.2f} mL at {peak_volume_phase:.0f}% phase")
    print(
        f"  Delay (peak volume - peak pressure): {pressure_shift_percent:+.0f}% "
        f"of the cycle = {pressure_shift_ms:+.1f} ms "
        f"(mean cycle {mean_cycle_duration_s * 1000:.1f} ms)"
    )

    # Shifted curves: same values, reassigned to new phase positions.
    # CT phases lie on the 1% grid, so the shifted CT-phase pressures are
    # exact samples of the shifted curve.
    mean_pressure_shifted = shift_periodic_curve(
        mean_pressure, NORMALIZED_PHASES_PERCENT, pressure_shift_percent
    )
    sd_pressure_shifted = shift_periodic_curve(
        sample_sd, NORMALIZED_PHASES_PERCENT, pressure_shift_percent
    )
    ct_mean_shifted = np.interp(
        CT_PHASES_PERCENT,
        NORMALIZED_PHASES_PERCENT,
        mean_pressure_shifted,
    )
    ct_phase_table["Mean_LV_pressure_shifted_mmHg"] = ct_mean_shifted
    ct_phase_table["Original_phase_of_shifted_pressure_percent"] = (
        CT_PHASES_PERCENT - pressure_shift_percent
    ) % 100

    shifted_normalized_table = pd.DataFrame(
        {
            "Cardiac_phase_percent": NORMALIZED_PHASES_PERCENT,
            "Original_phase_percent": (
                NORMALIZED_PHASES_PERCENT - pressure_shift_percent
            ) % 100,
            "Mean_pressure_shifted_mmHg": mean_pressure_shifted,
            "SD_pressure_shifted_mmHg": sd_pressure_shifted,
        }
    )

    peak_alignment_table = pd.DataFrame(
        {
            "Quantity": [
                "Peak_pressure_phase_percent",
                "Peak_pressure_mmHg",
                "Peak_volume_phase_percent",
                "Peak_volume_mL",
                "Pressure_shift_percent",
                "Pressure_shift_ms",
                "Mean_cycle_duration_ms",
            ],
            "Value": [
                peak_pressure_phase,
                peak_pressure_value,
                peak_volume_phase,
                peak_volume_value,
                pressure_shift_percent,
                pressure_shift_ms,
                mean_cycle_duration_s * 1000.0,
            ],
        }
    )

    plot_pressure_and_volume(
        mean_pressure, ct_mean, ct_lv_volume_ml,
        peak_pressure_phase, peak_volume_phase, PRESSURE_VOLUME_PLOT,
    )
    plot_pv_loop(ct_lv_volume_ml, ct_mean, PV_LOOP_PLOT)

    plot_pressure_and_volume(
        mean_pressure_shifted, ct_mean_shifted, ct_lv_volume_ml,
        peak_volume_phase, peak_volume_phase, PRESSURE_VOLUME_SHIFTED_PLOT,
    )
    plot_pv_loop(ct_lv_volume_ml, ct_mean_shifted, PV_LOOP_SHIFTED_PLOT)

    ct_phase_table[
        [
            "CT_phase_percent",
            "LV_volume_mL",
            "Mean_LV_pressure_mmHg",
            "Mean_LV_pressure_shifted_mmHg",
            "Original_phase_of_shifted_pressure_percent",
        ]
    ].to_csv(PV_LOOP_DATA_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        raw_table.to_excel(writer, sheet_name="Reconstructed_time", index=False)
        boundary_table.to_excel(writer, sheet_name="Cycle_boundaries", index=False)
        cycle_info_table.to_excel(writer, sheet_name="Cycle_information", index=False)
        normalized_table.to_excel(writer, sheet_name="Normalized_cycles", index=False)
        ct_phase_table.to_excel(writer, sheet_name="CT_phase_pressures", index=False)
        shifted_normalized_table.to_excel(writer, sheet_name="Shifted_mean_pressure", index=False)
        peak_alignment_table.to_excel(writer, sheet_name="Peak_alignment", index=False)

    print(f"\nSaved processed workbook: {OUTPUT_EXCEL.resolve()}")
    print(f"Saved PV-loop data: {PV_LOOP_DATA_CSV.resolve()}")
    print(f"Saved cycle-detection plot: {DIAGNOSTIC_PLOT.resolve()}")
    print(f"Saved average-curve plot: {AVERAGE_CURVE_PLOT.resolve()}")
    print(f"Saved pressure-and-volume plot: {PRESSURE_VOLUME_PLOT.resolve()}")
    print(f"Saved pressure-and-volume plot (shifted): {PRESSURE_VOLUME_SHIFTED_PLOT.resolve()}")
    print(f"Saved PV-loop plot: {PV_LOOP_PLOT.resolve()}")
    print(f"Saved PV-loop plot (shifted): {PV_LOOP_SHIFTED_PLOT.resolve()}")
    print(
        "\nReview the cycle-detection plot before using the CT-phase pressures. "
        "If any boundary is misplaced, enter the six correct sample indices "
        "in MANUAL_BOUNDARY_INDICES and rerun."
    )


if __name__ == "__main__":
    main()
