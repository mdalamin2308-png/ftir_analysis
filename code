"""
FTIR Polymer Identification — Streamlit Web App
------------------------------------------------

RUN LOCALLY:
    pip install streamlit numpy scipy pandas matplotlib
    streamlit run app.py
"""

import io
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks, savgol_filter

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="FTIR Polymer Identification", layout="wide")

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

# ----------------------------------------------------------------------
# Reference table (Table 3): characteristic FTIR peaks + assignment
# per polymer type. Third field = importance weight (1 = normal,
# >1 = a stronger diagnostic peak for that polymer).
# ----------------------------------------------------------------------
REFERENCE_PEAKS = {
    "High density polyethylene (HDPE)": [
        (2915, "C-H stretching", 1), (2845, "C-H stretching", 1),
        (1472, "CH2 bending", 1), (1462, "CH2 bending", 1),
        (730, "CH2 rocking", 1), (717, "CH2 rocking", 1),
    ],
    "Low density polyethylene (LDPE)": [
        (2915, "C-H stretching", 1), (2845, "C-H stretching", 1),
        (1467, "CH2 bending", 1), (1462, "CH2 bending", 1), (1377, "CH2 bending", 1),
        (730, "CH2 rocking", 1), (717, "CH2 rocking", 1),
    ],
    "Polyethylene terephthalate (PET)": [
        (1713, "C=O stretching", 3), (1241, "C-O stretching", 1),
        (1094, "C-O stretching", 1), (720, "Aromatic CH out-of-plane bending", 1),
    ],
    "Polypropylene (PP)": [
        (2950, "C-H stretching", 1), (2915, "C-H stretching", 1), (2838, "C-H stretching", 1),
        (1455, "CH2 bending", 1), (1377, "CH3 bending", 1),
        (1166, "CH bending, CH3 rocking, C-C stretching", 1),
        (997, "CH3 rocking, CH3 bending, C-C stretching", 1),
        (972, "CH3 rocking, CH3 bending, C-C stretching", 1),
        (840, "CH2 rocking, C-CH3 stretching, C-C stretching", 1),
        (808, "CH2 rocking, C-CH3 stretching, C-C stretching", 1),
    ],
    "Polystyrene (PS)": [
        (3024, "Aromatic C-H stretching", 1), (2847, "C-H stretching", 1),
        (1601, "Aromatic ring stretching", 1), (1492, "Aromatic ring stretching", 1),
        (1451, "CH2 bending", 1), (1027, "Aromatic CH bending", 1),
        (694, "Aromatic CH out-of-plane bending", 1),
        (537, "Aromatic ring out-of-plane bending", 1),
    ],
    "Polyvinyl chloride (PVC)": [
        (1427, "CH2 bending", 1), (1331, "CH bending", 1), (1255, "CH bending", 1),
        (1099, "C-C stretching", 1), (966, "CH2 rocking", 1), (616, "C-Cl stretching", 1),
    ],
    "Polyurethane (PU)": [
        (2865, "C-H stretching", 1), (1731, "C=O stretching", 1),
        (1531, "N-H stretching", 1), (1451, "CH2 bending", 1), (1223, "C(=O)O stretching", 1),
    ],
    "Nylon (all polyamides)": [
        (3298, "N-H stretching", 1), (2932, "CH stretching", 1), (2858, "CH stretching", 1),
        (1634, "C=O stretching", 1), (1538, "NH bending, C-N stretching", 1),
        (1464, "CH2 bending", 1), (1372, "CH2 bending", 1),
        (1274, "NH bending, C-N stretching", 1), (1199, "CH2 bending", 1),
        (687, "NH bending, C=O bending", 1),
    ],
}


# ----------------------------------------------------------------------
# Core processing functions
# ----------------------------------------------------------------------
def read_jcamp_txt(file_bytes):
    """Parse a JCAMP-style two-column FTIR text file from raw bytes."""
    metadata = {}
    x_vals, y_vals = [], []
    text = file_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
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
    """Pick a valid odd window length with polyorder < window <= n, or None."""
    min_valid = polyorder + 2 if (polyorder + 2) % 2 else polyorder + 3
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


def find_closest_assignment(wavenumber, tolerance):
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


def match_polymer(peak_x, peak_prom, tolerance):
    results = {}
    for polymer, ref_peaks in REFERENCE_PEAKS.items():
        matched = []
        used_peak = set()
        weight_sum = 0.0
        max_possible_weight = 0.0
        for rp, assignment, importance in ref_peaks:
            max_possible_weight += importance
            if peak_x.size == 0:
                continue
            diffs = np.abs(peak_x - rp)
            best_idx = np.argmin(diffs)
            if best_idx in used_peak:
                continue
            if diffs[best_idx] <= tolerance:
                matched.append((rp, peak_x[best_idx], diffs[best_idx], assignment))
                used_peak.add(best_idx)
                weight_sum += peak_prom[best_idx] * importance
        fraction_matched = len(matched) / len(ref_peaks) if ref_peaks else 0.0
        avg_prominence = weight_sum / max_possible_weight if max_possible_weight else 0.0
        normalized_weight = min(1.0, avg_prominence / 100.0)
        confidence = min(100.0, fraction_matched * 70.0 + normalized_weight * 100.0 * 0.30)
        results[polymer] = {
            "matched": matched,
            "n_matched": len(matched),
            "n_total": len(ref_peaks),
            "fraction_matched": fraction_matched,
            "confidence": confidence,
        }
    return results


def build_plot(x, y, baseline, peak_x, peak_y, peak_prom, label_tolerance, best_polymer, confidence):
    order = np.argsort(-peak_prom)

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
        hit = find_closest_assignment(peak_x[i], label_tolerance)
        if hit is None:
            continue
        assignment, polymer, rp, diff = hit
        label_candidates.append((float(peak_x[i]), float(peak_y[i]), assignment))

    labels_sorted = sorted(label_candidates, key=lambda t: -t[0])

    if labels_sorted:
        wrapped = [textwrap.fill(a, width=13) for _, _, a in labels_sorted]

        n_tiers = 4
        y_guess = y.max() + 300
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

        buffer = 20
        tier_gap = max_line_height + buffer
        base_y = y.max() + max_line_height + buffer
        y_max = base_y + (n_tiers - 1) * tier_gap + 10

        for idx, ((wn, py, assignment), wrapped_text) in enumerate(zip(labels_sorted, wrapped)):
            tier = idx % n_tiers
            label_y_number = base_y + tier * tier_gap
            label_y_text = label_y_number - 5

            ax.plot([wn, wn], [py, label_y_number - 2], color="gray", linewidth=0.5, zorder=1)
            ax.text(wn, label_y_number, f"{wn:.2f}", ha="center", va="bottom",
                    fontsize=12, color="black", fontweight="bold")
            ax.text(wn, label_y_text, wrapped_text, ha="center", va="top",
                    fontsize=12, color="black", style="italic", linespacing=1.3)
    else:
        y_max = y.max() + 8

    ax.set_xlim(4000, 400)
    ax.set_ylim(min(y.min() - 10, 0), y_max)

    ax.text(
        0.03, 0.97, f"Best polymer:\n{best_polymer}\nConfidence: {confidence:.1f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"),
        zorder=10,
    )
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.title("🔬 FTIR Polymer Identification")
st.caption(
    "Upload an FTIR spectrum (two-column wavenumber / %T text file) to detect "
    "peaks, match them against a literature reference table, and identify the "
    "most likely polymer."
)

with st.sidebar:
    st.header("Settings")
    tolerance = st.slider("Polymer-matching tolerance (cm⁻¹)", 5, 40, 20, 1)
    label_tolerance = st.slider("Peak-labeling tolerance (cm⁻¹)", 5, 30, 15, 1)
    prominence = st.slider("Peak detection sensitivity (prominence)", 0.5, 10.0, 2.0, 0.5)
    st.caption("Lower prominence = more peaks detected (including noise).")

uploaded = st.file_uploader("Upload FTIR data file (.txt)", type=["txt", "dat", "csv"])

if uploaded is None:
    st.info("👆 Upload a spectrum file to get started.")
    st.stop()

try:
    x, y_raw, metadata, y_units = read_jcamp_txt(uploaded.read())
    if x.size == 0 or y_raw.size == 0:
        st.error(
            "No numeric (wavenumber, %T) data could be parsed from this file. "
            "Make sure it has two whitespace-separated numeric columns after "
            "any '##HEADER=' lines."
        )
        st.stop()
except Exception as e:
    st.error(f"Could not read this file: {e}")
    st.stop()

y = smooth_spectrum(y_raw)
baseline = estimate_baseline(y, window=201, polyorder=3)
y = y - baseline + baseline.max()
y = np.clip(y, 0, None)

peak_x, peak_y, peak_prom = detect_peaks(x, y, prominence=prominence, width=3, distance=8)

if peak_x.size == 0:
    st.warning("No peaks were detected. Try lowering the peak-detection sensitivity in the sidebar.")
    st.stop()

results = match_polymer(peak_x, peak_prom, tolerance)
ranked = sorted(results.items(), key=lambda kv: kv[1]["confidence"], reverse=True)
best_polymer, best_r = ranked[0]

col1, col2 = st.columns([2, 1])

with col1:
    fig = build_plot(x, y, baseline, peak_x, peak_y, peak_prom, label_tolerance,
                      best_polymer, best_r["confidence"])
    st.pyplot(fig, use_container_width=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
    st.download_button("⬇️ Download annotated plot (PNG, 600 dpi)", buf.getvalue(),
                        file_name=f"ftir_identified_{uploaded.name.rsplit('.', 1)[0]}.png",
                        mime="image/png")

with col2:
    st.subheader("Best match")
    st.metric(best_polymer, f"{best_r['confidence']:.1f}% confidence")
    st.caption(f"{best_r['n_matched']}/{best_r['n_total']} reference peaks matched")

    st.subheader("Full ranking")
    report_df = pd.DataFrame({
        "Polymer": [p for p, _ in ranked],
        "Matched": [r["n_matched"] for _, r in ranked],
        "Total": [r["n_total"] for _, r in ranked],
        "Match %": [round(r["fraction_matched"] * 100, 1) for _, r in ranked],
        "Confidence %": [round(r["confidence"], 1) for _, r in ranked],
    })
    st.dataframe(report_df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download ranking (CSV)",
                        report_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"Polymer_Ranking_{uploaded.name.rsplit('.', 1)[0]}.csv",
                        mime="text/csv")

st.subheader("Detected peaks")
order = np.argsort(-peak_prom)
peaks_df = pd.DataFrame({
    "Peak (cm⁻¹)": peak_x[order],
    "Transmittance (%)": np.round(peak_y[order], 2),
    "Prominence": np.round(peak_prom[order], 2),
})
st.dataframe(peaks_df, hide_index=True, use_container_width=True)
st.download_button("⬇️ Download detected peaks (CSV)",
                    peaks_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"Detected_Peaks_{uploaded.name.rsplit('.', 1)[0]}.csv",
                    mime="text/csv")

with st.expander("Matched peak assignments for best match"):
    for rp, dp, diff, assignment in best_r["matched"]:
        st.write(f"**{rp:.0f} cm⁻¹** → detected at {dp:.1f} cm⁻¹ (Δ={diff:.1f}) — {assignment}")
