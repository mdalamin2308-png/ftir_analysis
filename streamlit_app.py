import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.linewidth"] = 1.1

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = None

if get_script_run_ctx is None or get_script_run_ctx() is None:
    raise SystemExit(
        "This Streamlit app must be run with `streamlit run streamlit_app.py` "
        "instead of `python streamlit_app.py`."
    )

DATA_DIR = Path(__file__).resolve().parent

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
    "Nylon (PA)": [
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


def parse_spectrum_text(contents: str):
    x_vals, y_vals = [], []
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("##"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))
        except ValueError:
            continue
    return np.array(x_vals), np.array(y_vals)


def choose_savgol_window(n, target, polyorder=3):
    min_valid = polyorder + 2 if (polyorder + 2) % 2 else polyorder + 3
    if n < min_valid:
        return None
    window = min(n if n % 2 == 1 else n - 1, target)
    if window % 2 == 0:
        window -= 1
    if window < min_valid:
        window = min_valid
    return window if window <= n else None


def estimate_baseline(y, window=151, polyorder=3):
    window = choose_savgol_window(len(y), window, polyorder)
    return savgol_filter(y, window, polyorder) if window else np.full_like(y, np.min(y))


def detect_peaks(x, y, prominence=1.5, width=3, distance=8):
    inv_y = -y
    indexes, props = find_peaks(inv_y, prominence=prominence, width=width, distance=distance)
    return x[indexes], y[indexes], props.get("prominences", np.zeros_like(indexes, dtype=float))


def effective_tolerance(reference_wavenumber, base_tolerance=20):
    relative = max(1.0, reference_wavenumber * 0.02)
    return max(base_tolerance, relative)


def find_closest_assignment(wavenumber, tolerance=15):
    best = None
    for polymer, peaks in REFERENCE_PEAKS.items():
        for rp, assignment, importance in peaks:
            diff = abs(wavenumber - rp)
            if diff <= tolerance and (best is None or diff < best[0]):
                best = (diff, assignment, polymer, rp)
    if best is None:
        return None
    diff, assignment, polymer, rp = best
    return assignment, polymer, rp, diff


def match_polymer(peak_x, peak_prom, tolerance=20):
    results = {}
    for polymer, ref_peaks in REFERENCE_PEAKS.items():
        matched = []
        weight_sum = 0.0
        for rp, assignment, importance in ref_peaks:
            if peak_x.size == 0:
                continue
            diffs = np.abs(peak_x - rp)
            best_idx = np.argmin(diffs)
            if diffs[best_idx] <= effective_tolerance(rp, tolerance):
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
            "confidence": confidence,
        }
    return results


def normalize_transmittance(y):
    y = np.asarray(y, dtype=float)
    return np.clip(y, 0.0, None)


def render_spectrum_plot(x, y, baseline, peak_x=None, peak_y=None, assignments=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#fcfdff")
    fig.patch.set_facecolor("#fcfdff")
    ax.plot(x, y, color="#172f57", linewidth=1.9, label="Spectrum")
    ax.plot(x, baseline, color="#c0392b", linestyle="--", linewidth=1.2, label="Baseline")
    ax.fill_between(x, baseline, y, color="#d8e3ff", alpha=0.45)
    ax.set_axisbelow(True)
    ax.grid(True, color="#e8ecf5", linestyle="-", linewidth=0.9)
    if peak_x is not None and peak_y is not None and peak_x.size > 0:
        ax.scatter(peak_x, peak_y, color="#d53838", edgecolor="white", linewidth=0.8, s=40, zorder=5, label="Detected peaks")
        for wx, wy in zip(peak_x, peak_y):
            ax.text(wx, wy - 3.5, f"{wx:.0f}", color="#d53838", fontsize=8, ha="center", va="top", rotation=90)
    if assignments is not None and peak_x is not None and peak_x.size > 0:
        labeled = []
        for wx, wy, text in zip(peak_x, peak_y, assignments):
            if text is not None:
                labeled.append((wx, wy, text))
        labeled = sorted(labeled, key=lambda t: -t[0])
        if labeled:
            max_labels = min(len(labeled), 10)
            row_height = 18
            for idx, (wx, wy, text) in enumerate(labeled[:max_labels]):
                row = idx // 2
                side = -1 if idx % 2 == 0 else 1
                x_offset = side * 32
                y_offset = -26 - row * row_height
                ha = "right" if side < 0 else "left"
                ax.annotate(
                    text,
                    xy=(wx, wy),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha=ha,
                    va="top",
                    fontsize=9,
                    color="#102a5f",
                    bbox={"facecolor": "white", "alpha": 0.94, "edgecolor": "#aac0ff", "boxstyle": "round,pad=0.3"},
                    arrowprops={"arrowstyle": "-|>", "color": "#7a90c8", "linewidth": 0.8, "shrinkA": 0, "shrinkB": 4},
                    annotation_clip=False,
                )
    ax.set_xlim(4000, 400)
    ax.set_ylim(min(y.min() - 74, -10), max(y.max() + 18, 100))
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#a8b8d3")
    ax.spines["left"].set_color("#a8b8d3")
    ax.tick_params(direction="in", length=6, width=1, colors="#3a4d7c", labelcolor="#3a4d7c")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#c4d3e8")
    fig.tight_layout(pad=1.0)
    return fig


def load_sample_files():
    files = sorted(DATA_DIR.glob("S[1-9].txt"), key=lambda p: int(p.stem[1:]))
    if (DATA_DIR / "S10.txt").exists():
        files.append(DATA_DIR / "S10.txt")
    return files


def main():
    st.set_page_config(page_title="FTIR Polymer ID", layout="wide")
    st.markdown(
        '''
        <style>
        body {background: linear-gradient(135deg, #f3f7ff 0%, #ffffff 100%); color: #0b1f44;}
        div.block-container {padding: 1.5rem 2rem; background: rgba(255,255,255,0.96); border-radius: 20px;}
        div[data-testid='stSidebar'] {background: linear-gradient(180deg, #1f3c88, #0f2245); color: #f8fbff;}
        div[data-testid='stSidebar'] .stTextInput>div>div>input,
        div[data-testid='stSidebar'] .stTextArea>div>div>textarea,
        div[data-testid='stSidebar'] .stSelectbox>div>div>div>div>div {background: #f4f7ff; color: #0b1f44;}
        div[data-testid='stSidebar'] button {border-radius: 12px; background-color: #0f2245; color: #f8fbff;}
        div[data-testid='stHeader'] {background: transparent;}
        h1, h2, h3, p, label, button {font-family: 'Times New Roman', serif;}
        </style>
        ''',
        unsafe_allow_html=True,
    )
    st.title("FTIR Polymer Identification")
    st.markdown(
        '''
        <div style='padding:18px 22px; border-radius:18px; background: linear-gradient(135deg, #eaf0ff, #ffffff); color:#0b1f44; font-size:14px;'>
            <strong>Smart ATR-FTIR polymer analysis with peak detection and assignment layout.</strong><br>
            <span style='color:#344e86;'>Upload, paste, or use sample data; the app highlights peaks and ranks polymer candidates.</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    input_mode = st.sidebar.radio("Input mode:", ["Upload file", "Paste direct input", "Use sample file"])

    file_buffer = None
    raw_text = ""
    selected_sample = None

    sample_files = load_sample_files()
    sample_names = [f.name for f in sample_files]

    if input_mode == "Upload file":
        file_buffer = st.sidebar.file_uploader("Upload spectrum file (.txt)", type=["txt"])
    elif input_mode == "Paste direct input":
        raw_text = st.sidebar.text_area(
            "Paste wavenumber and transmittance data", 
            value="4000 99.8\n3900 99.7\n...",
            height=220,
        )
    else:
        selected_sample = st.sidebar.selectbox("Select sample spectrum", ["Choose sample..."] + sample_names)

    if st.sidebar.button("Analyze spectrum"):
        if input_mode == "Upload file":
            if file_buffer is None:
                st.warning("Please upload a .txt file.")
                return
            try:
                raw_text = file_buffer.read().decode("utf-8", errors="replace")
                source_name = getattr(file_buffer, "name", "uploaded_spectrum.txt")
            except Exception:
                st.error("Unable to read the uploaded file. Upload a valid text file.")
                return
        elif input_mode == "Paste direct input":
            if not raw_text.strip():
                st.warning("Please paste spectrum data into the text area.")
                return
            source_name = "pasted spectrum"
        else:
            if selected_sample is None or selected_sample == "Choose sample...":
                st.warning("Please select a sample spectrum.")
                return
            sample_path = DATA_DIR / selected_sample
            if not sample_path.exists():
                st.error(f"Sample file not found: {selected_sample}")
                return
            raw_text = sample_path.read_text(encoding="utf-8", errors="replace")
            source_name = selected_sample

        x, y = parse_spectrum_text(raw_text)
        if x.size == 0 or y.size == 0:
            st.error("No valid numeric spectrum data could be parsed. Provide two columns: wavenumber and transmittance.")
            return

        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = normalize_transmittance(y[sort_idx])

        baseline = estimate_baseline(y, window=151, polyorder=3)
        baseline = np.minimum(baseline, y)

        peak_x, peak_y, peak_prom = detect_peaks(x, y, prominence=1.5, width=3, distance=8)
        results = match_polymer(peak_x, peak_prom)
        ranked = sorted(results.items(), key=lambda kv: (kv[1]["confidence"], kv[1]["weight_sum"]), reverse=True)

        assigned_labels = []
        for wn in peak_x:
            hit = find_closest_assignment(wn)
            assigned_labels.append(hit[0] if hit is not None else None)

        st.subheader(f"Spectrum: {source_name}")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = render_spectrum_plot(
                x,
                y,
                baseline,
                peak_x=peak_x,
                peak_y=peak_y,
                assignments=assigned_labels,
            )
            st.pyplot(fig)
        with col2:
            if ranked:
                best_polymer, best_result = ranked[0]
                st.markdown(f"### Best match: **{best_polymer}**")
                st.markdown(
                    f"Matched {best_result['n_matched']} of {best_result['n_total']} reference peaks "
                    f"({best_result['fraction_matched'] * 100:.1f}% match)"
                )
                st.markdown(f"Confidence: **{best_result['confidence']:.1f}%**")
            else:
                st.write("No polymer match results available.")

        if peak_x.size > 0:
            peak_table = [
                {"Wavenumber": f"{wx:.1f}", "Transmittance": f"{wy:.1f}", "Prominence": f"{wp:.2f}"}
                for wx, wy, wp in zip(peak_x, peak_y, peak_prom)
            ]
            st.markdown("### Detected peaks")
            st.table(peak_table)
        else:
            st.info("No absorption peaks were detected in this spectrum.")

        st.markdown("---")
        st.markdown("### Polymer ranking")
        rank_rows = [
            {
                "Polymer": polymer,
                "Matched": f"{result['n_matched']}/{result['n_total']}",
                "Match %": f"{result['fraction_matched'] * 100:.1f}%",
                "Confidence": f"{result['confidence']:.1f}%",
            }
            for polymer, result in ranked
        ]
        st.table(rank_rows)

        label_rows = []
        for wn, py in zip(peak_x, peak_y):
            hit = find_closest_assignment(wn)
            if hit is not None:
                assignment, _, rp, diff = hit
                label_rows.append(
                    {
                        "Peak cm-1": f"{wn:.1f}",
                        "Transmittance": f"{py:.1f}%",
                        "Assignment": assignment,
                        "Δ cm-1": f"{diff:.1f}",
                    }
                )
        if label_rows:
            st.markdown("### Identified peak assignments")
            st.table(label_rows[:10])
        else:
            st.info("No reference peak assignments were found for the detected peaks.")

    else:
        st.info("Choose an input mode and click 'Analyze spectrum' to start.")

    st.markdown(
        "<div style='margin-top:24px; padding:12px 16px; border-radius:14px; background: linear-gradient(135deg, #fff9ec, #f7f0d7); "
        "color:#3d3d3d; font-size:12px;'>"
        "<strong>Application Developed by Md. Al Amin, M.Sc.</strong>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Run this app with `streamlit run streamlit_app.py`")


if __name__ == "__main__":
    main()
