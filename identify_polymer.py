"""
FTIR Polymer Identification
---------------------------
Detects absorption peaks in an FTIR spectrum and matches them against a
reference table of characteristic peaks for common plastics.

USAGE:
    python identify_polymer.py --file S10.txt
    python identify_polymer.py --all
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "sample-61.txt" 
TOLERANCE = 20
LABEL_TOLERANCE = 15

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True

REFERENCE_PEAKS = {
    "High density polyethylene (HDPE)": [
        (2915, "C-H stretching", 1),
        (2845, "C-H stretching", 1),
        (1472, "CH2 bending", 1),
        (1462, "CH2 bending", 1),
        (730, "CH2 rocking", 1),
        (717, "CH2 rocking", 1),
    ],
    "Low density polyethylene (LDPE)": [
        (2915, "C-H stretching", 1),
        (2845, "C-H stretching", 1),
        (1467, "CH2 bending", 1),
        (1462, "CH2 bending", 1),
        (1377, "CH2 bending", 1),
        (730, "CH2 rocking", 1),
        (717, "CH2 rocking", 1),
    ],
    "Polyethylene terephthalate (PET)": [
        (1713, "C=O stretching", 3),
        (1241, "C-O stretching", 1),
        (1094, "C-O stretching", 1),
        (720, "Aromatic CH out-of-plane bending", 1),
    ],
    "Polypropylene (PP)": [
        (2950, "C-H stretching", 1),
        (2915, "C-H stretching", 1),
        (2838, "C-H stretching", 1),
        (1455, "CH2 bending", 1),
        (1377, "CH3 bending", 1),
        (1166, "CH bending, CH3 rocking, C-C stretching", 1),
        (997, "CH3 rocking, CH3 bending, C-C stretching", 1),
        (972, "CH3 rocking, CH3 bending, C-C stretching", 1),
        (840, "CH2 rocking, C-CH3 stretching, C-C stretching", 1),
        (808, "CH2 rocking, C-CH3 stretching, C-C stretching", 1),
    ],
    "Polystyrene (PS)": [
        (3024, "Aromatic C-H stretching", 1),
        (2847, "C-H stretching", 1),
        (1601, "Aromatic ring stretching", 1),
        (1492, "Aromatic ring stretching", 1),
        (1451, "CH2 bending", 1),
        (1027, "Aromatic CH bending", 1),
        (694, "Aromatic CH out-of-plane bending", 1),
        (537, "Aromatic ring out-of-plane bending", 1),
    ],
    "Polyvinyl chloride (PVC)": [
        (1427, "CH2 bending", 1),
        (1331, "CH bending", 1),
        (1255, "CH bending", 1),
        (1099, "C-C stretching", 1),
        (966, "CH2 rocking", 1),
        (616, "C-Cl stretching", 1),
    ],
    "Polyurethane (PU)": [
        (2865, "C-H stretching", 1),
        (1731, "C=O stretching", 1),
        (1531, "N-H stretching", 1),
        (1451, "CH2 bending", 1),
        (1223, "C(=O)O stretching", 1),
    ],
    "Nylon (all polyamides)": [
        (3298, "N-H stretching", 1),
        (2932, "CH stretching", 1),
        (2858, "CH stretching", 1),
        (1634, "C=O stretching", 1),
        (1538, "NH bending, C-N stretching", 1),
        (1464, "CH2 bending", 1),
        (1372, "CH2 bending", 1),
        (1274, "NH bending, C-N stretching", 1),
        (1199, "CH2 bending", 1),
        (687, "NH bending, C=O bending", 1),
    ],
}


def read_jcamp_txt(filepath):
    metadata = {}
    x_vals, y_vals = [], []
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("##"):
                if "=" in line:
                    key, _, value = line[2:].partition("=")
                    metadata[key.strip()] = value.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                except ValueError:
                    continue

    y_units = metadata.get("YUNITS", "%T")
    return np.array(x_vals), np.array(y_vals), metadata, y_units


def choose_savgol_window(n, target, polyorder=3):
    """
    Pick a valid odd window length for savgol_filter such that
    polyorder < window_length <= n. Returns None if no valid window
    exists (e.g. n is too small to filter at all).
    """
    min_valid = polyorder + 2 if (polyorder + 2) % 2 else polyorder + 3  # smallest odd > polyorder
    if n < min_valid:
        return None
    window = min(n if n % 2 == 1 else n - 1, target)
    if window % 2 == 0:
        window -= 1
    if window < min_valid:
        window = min_valid
    if window > n:
        return None
    return window


def smooth_spectrum(y):
    window = choose_savgol_window(len(y), 11, polyorder=3)
    return savgol_filter(y, window, 3) if window else y


def estimate_baseline(y, window=201, polyorder=3):
    window = choose_savgol_window(len(y), window, polyorder=polyorder)
    return savgol_filter(y, window, polyorder) if window else np.full_like(y, np.min(y))


def detect_peaks(x, y, prominence=2, width=3, distance=8):
    inv_y = -y
    peak_idx, props = find_peaks(inv_y, prominence=prominence, width=width, distance=distance)
    return x[peak_idx], y[peak_idx], props["prominences"]


def find_closest_assignment(wavenumber, tolerance=LABEL_TOLERANCE):
    best = None
    for polymer, ref_peaks in REFERENCE_PEAKS.items():
        for rp, assignment, importance in ref_peaks:
            diff = abs(wavenumber - rp)
            if diff <= tolerance and (best is None or diff < best[0]):
                best = (diff, assignment, polymer, rp)
    if best is None:
        return None
    diff, assignment, polymer, rp = best
    return assignment, polymer, rp, diff


def effective_tolerance(reference_wavenumber: float, base_tolerance: float = TOLERANCE) -> float:
    """Use a slightly wider tolerance for high-frequency bands.

    CH stretching and other broad bands can shift more than low-frequency
    bending peaks, so allow a relative tolerance on top of the fixed base.
    """
    relative = max(1.0, reference_wavenumber * 0.02)
    return max(base_tolerance, relative)


def match_polymer(peak_x, peak_prom, tolerance=TOLERANCE):
    results = {}
    for polymer, ref_peaks in REFERENCE_PEAKS.items():
        matched = []
        weight_sum = 0.0
        max_possible_weight = 0.0
        for rp, assignment, importance in ref_peaks:
            max_possible_weight += importance
            if peak_x.size == 0:
                continue
            diffs = np.abs(peak_x - rp)
            best_idx = np.argmin(diffs)
            effective_tol = effective_tolerance(rp, tolerance)
            if diffs[best_idx] <= effective_tol:
                matched.append((rp, peak_x[best_idx], diffs[best_idx], assignment))
                weight_sum += peak_prom[best_idx] * importance
        fraction_matched = len(matched) / len(ref_peaks) if ref_peaks else 0.0
        confidence = min(100.0, fraction_matched * 100.0)
        results[polymer] = {
            "matched": matched,
            "n_matched": len(matched),
            "n_total": len(ref_peaks),
            "fraction_matched": fraction_matched,
            "weight_sum": weight_sum,
            "max_possible_weight": max_possible_weight,
            "confidence": confidence,
        }
    return results


def resolve_data_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix == "":
        path = path.with_suffix(".txt")
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def resolve_output_dir(data_file: Path) -> Path:
    """
    Prefer writing outputs next to the input file. If that folder isn't
    writable (e.g. a read-only uploads/network mount), fall back to the
    current working directory instead of crashing.
    """
    candidate = data_file.parent
    try:
        probe = candidate / f".write_test_{data_file.stem}"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path.cwd()
        print(f"Note: '{candidate}' is not writable, saving outputs to '{fallback}' instead.")
        return fallback


def process_file(data_file: Path):
    x, y, metadata, y_units = read_jcamp_txt(data_file)
    if x.size == 0 or y.size == 0:
        raise ValueError(
            f"No numeric (wavenumber, %T) data could be parsed from '{data_file}'. "
            "Check that the file has two whitespace-separated numeric columns "
            "after any '##HEADER=' lines."
        )
    out_dir = resolve_output_dir(data_file)
    y = smooth_spectrum(y)
    baseline = estimate_baseline(y, window=201, polyorder=3)
    y = y - baseline + baseline.max()
    y = np.clip(y, 0, None)

    peak_x, peak_y, peak_prom = detect_peaks(x, y, prominence=2, width=3, distance=8)

    print(f"\nProcessing: {data_file.name}")
    print("Detected absorption peaks (wavenumber, %T, prominence):")
    order = np.argsort(-peak_prom)
    for i in order:
        print(f"  {peak_x[i]:8.1f} cm-1   {peak_y[i]:6.2f} %T   prominence={peak_prom[i]:.2f}")

    peaks_df = pd.DataFrame({
        "Peak (cm-1)": peak_x,
        "Transmittance": peak_y,
        "Prominence": peak_prom,
    })
    peaks_csv = out_dir / f"Detected_Peaks_{data_file.stem}.csv"
    peaks_df.to_csv(peaks_csv, index=False)
    print(f"Detected peaks saved to: {peaks_csv}")

    results = match_polymer(peak_x, peak_prom)
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["confidence"], kv[1]["weight_sum"]),
        reverse=True,
    )

    report_df = pd.DataFrame({
        "Polymer": [polymer for polymer, _ in ranked],
        "Matched Peaks": [r["n_matched"] for _, r in ranked],
        "Reference Peaks": [r["n_total"] for _, r in ranked],
        "Match (%)": [round(r["fraction_matched"] * 100, 1) for _, r in ranked],
        "Confidence (%)": [round(r["confidence"], 1) for _, r in ranked],
    })
    report_csv = out_dir / f"Polymer_Ranking_{data_file.stem}.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"Polymer ranking saved to: {report_csv}")

    print("\nTop 3 polymer candidates:")
    for rank, (polymer, r) in enumerate(ranked[:3], start=1):
        print(f"{rank}. {polymer:30s} {r['confidence']:6.1f}%")

    if len(ranked) >= 2:
        top1, top2 = ranked[0][1], ranked[1][1]
        if top1["confidence"] >= 70 and top2["confidence"] >= 65 and abs(top1["confidence"] - top2["confidence"]) <= 5:
            print(f"Possible blend: {ranked[0][0]} + {ranked[1][0]}")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(x, y, color="black", linewidth=1.0, label="Spectrum", zorder=3)
    ax.plot(x, baseline, color="firebrick", linestyle="--", linewidth=0.9,
            label="Baseline", zorder=2)
    ax.fill_between(x, baseline, y, color="lightgray", alpha=0.4, zorder=1)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.grid(False)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.legend(loc="lower left", fontsize=10, frameon=False)

    label_candidates = []
    for i in order:
        if len(label_candidates) >= 6:
            break
        hit = find_closest_assignment(peak_x[i])
        if hit is None:
            continue
        assignment, polymer, rp, diff = hit
        label_candidates.append((float(peak_x[i]), float(peak_y[i]), assignment))

    # sort left-to-right (high wavenumber first, matching the inverted x-axis)
    labels_sorted = sorted(label_candidates, key=lambda t: -t[0])

    if labels_sorted:
        import textwrap

        # Wrap long assignment text onto short lines so labels take up
        # far less horizontal space and stop colliding with their neighbors.
        wrapped = [textwrap.fill(a, width=13) for _, _, a in labels_sorted]

        # Stagger labels across 4 vertical tiers (round-robin by left-to-
        # right order) so labels close together in wavenumber never land
        # at the same height. Tier spacing is measured from the ACTUAL
        # rendered text height (not guessed) so a long 5-6 line wrapped
        # label can never bleed down into the tier below it.
        n_tiers = 4
        y_guess = y.max() + 300  # generous provisional headroom for measuring
        ax.set_ylim(min(y.min() - 10, 0), y_guess)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        max_line_height = 0.0
        for (wn, py, assignment), wt in zip(labels_sorted, wrapped):
            probe = ax.text(wn, y.max() + 100, wt, ha="center", va="top",
                             fontsize=12, style="italic", linespacing=1.3)
            fig.canvas.draw()
            h = abs(probe.get_window_extent(renderer=renderer)
                     .transformed(ax.transData.inverted()).height)
            max_line_height = max(max_line_height, h)
            probe.remove()

        buffer = 20  # safety margin (cm-1 of vertical data-space)
        tier_gap = max_line_height + buffer
        base_y = y.max() + max_line_height + buffer
        y_max = base_y + (n_tiers - 1) * tier_gap + 10

        for idx, ((wn, py, assignment), wrapped_text) in enumerate(zip(labels_sorted, wrapped)):
            tier = idx % n_tiers
            label_y_number = base_y + tier * tier_gap
            label_y_text = label_y_number - 5

            # vertical leader line drawn straight from the peak apex up to
            # its OWN final label position (never moved afterward, so it
            # always stays connected to the right label)
            ax.plot([wn, wn], [py, label_y_number - 2], color="gray",
                     linewidth=0.5, zorder=1)
            ax.text(wn, label_y_number, f"{wn:.2f}", ha="center", va="bottom",
                    fontsize=12, color="black", fontweight="bold")
            ax.text(wn, label_y_text, wrapped_text, ha="center", va="top",
                    fontsize=12, color="black", style="italic", linespacing=1.3)
    else:
        y_max = y.max() + 8

    x_min, x_max = 4000, 400
    y_min = min(y.min() - 10, 0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    top3_text = [
        f"{idx+1}. {polymer} {r['confidence']:.1f}%"
        for idx, (polymer, r) in enumerate(ranked[:3])
    ]
    ax.text(
        0.03, 0.97,
        "Polymer Identification Results:\n" + "\n".join(top3_text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"),
        zorder=10,
    )

    plot_path = out_dir / f"ftir_identified_{data_file.stem}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Annotated plot saved to: {plot_path} (600 dpi)")


def main():
    parser = argparse.ArgumentParser(description="FTIR polymer identification for S1-S10 spectra.")
    parser.add_argument("-f", "--file", help="Input spectrum file, e.g. S10.txt or path/to/S10.txt")
    parser.add_argument("--all", action="store_true", help="Process all S1.txt through S10.txt files")
    args = parser.parse_args()

    if args.all:
        for i in range(1, 11):
            data_file = Path(__file__).resolve().parent / f"S{i}.txt"
            if data_file.exists():
                try:
                    process_file(data_file)
                except (ValueError, OSError) as e:
                    print(f"Error processing {data_file.name}: {e}")
            else:
                print(f"Warning: file not found, skipping {data_file}")
    else:
        data_file = resolve_data_file(args.file) if args.file else DEFAULT_DATA_FILE
        try:
            process_file(data_file)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Error: {e}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
