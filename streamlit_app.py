import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple

import identify_polymer as ip

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

st.set_page_config(page_title="FTIR Polymer ID", layout="wide")
st.title("FTIR Polymer Identification")
st.write(
    "Upload a JCAMP-style FTIR spectrum or select one of the sample files `S1.txt` through `S10.txt`."
)

sample_files = sorted(DATA_DIR.glob("S[1-9].txt"), key=lambda p: int(p.stem[1:]))
sample_files += [DATA_DIR / "S10.txt"] if (DATA_DIR / "S10.txt").exists() else []

selected_sample = st.sidebar.selectbox(
    "Sample spectrum file",
    ["Choose sample..."] + [f.name for f in sample_files],
)
uploaded_file = st.sidebar.file_uploader("Upload spectrum (.txt)", type=["txt"])

st.sidebar.markdown(
    "---\n"
    "**Instructions:** Select a sample or upload your own JCAMP-style text file. "
    "The plot shows the spectrum, baseline, labeled peaks, and the identified polymer."
)

file_source = None
if uploaded_file is not None and hasattr(uploaded_file, "read"):
    file_source = "upload"
elif selected_sample != "Choose sample...":
    file_source = "sample"

if file_source is None:
    st.info("Select a sample file from the sidebar or upload a spectrum file to start analysis.")
    st.stop()

if file_source == "sample":
    data_path = DATA_DIR / selected_sample
    x, y = ip.read_jcamp_txt(data_path)
    source_name = selected_sample
else:
    if uploaded_file is None or not hasattr(uploaded_file, "read"):
        st.info("Select a sample file from the sidebar or upload a spectrum file to start analysis.")
        st.stop()
    contents = uploaded_file.read().decode("utf-8", errors="replace")
    x_vals, y_vals = [], []
    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except ValueError:
                continue
    x = np.array(x_vals)
    y = np.array(y_vals)
    source_name = uploaded_file.name

peak_x, peak_y, peak_prom = ip.detect_peaks(x, y)
results = ip.match_polymer(peak_x, peak_prom)
ranked = sorted(
    results.items(),
    key=lambda kv: (kv[1]["fraction_matched"], kv[1]["weight_sum"]),
    reverse=True,
)

best_polymer, best_r = ranked[0]

st.subheader(f"Spectrum: {source_name}")
col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color="black", linewidth=1.5, label="Spectrum")
    baseline = ip.estimate_baseline(y, window=61, percentile=10)
    baseline = np.minimum(baseline, y)
    ax.plot(x, baseline, color="firebrick", linestyle="--", linewidth=1.2, label="Baseline")
    ax.fill_between(x, baseline, y, color="lightgray", alpha=0.4)
    ax.set_xlim(4000, 400)
    ax.set_ylim(min(y.min() - 10, 0), 150)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance (%)")
    ax.grid(False)
    ax.legend(loc="upper right")
    st.pyplot(fig)

with col2:
    st.markdown("### Top identification")
    st.markdown(f"**{best_polymer}**")
    st.markdown(
        f"Matched {best_r['n_matched']} of {best_r['n_total']} reference peaks "
        f"({best_r['fraction_matched'] * 100:.1f}% match)"
    )

    if best_r["matched"]:
        st.markdown("#### Matched peaks")
        matched_table = [
            {
                "Ref cm-1": f"{rp:.0f}",
                "Detected cm-1": f"{dp:.1f}",
                "Δ cm-1": f"{diff:.1f}",
                "Assignment": assignment,
            }
            for rp, dp, diff, assignment in best_r["matched"]
        ]
        st.table(matched_table)

st.markdown("---")

st.markdown("### Polymer ranking")
rank_table = [
    {
        "Polymer": polymer,
        "Matched": f"{r['n_matched']}/{r['n_total']}",
        "Match %": f"{r['fraction_matched'] * 100:.1f}%",
    }
    for polymer, r in ranked
]
st.table(rank_table)

label_rows = []
for wn, py, assignment in sorted(
    [
        (peak_x[i], peak_y[i], ip.find_closest_assignment(peak_x[i])[0])
        for i in np.argsort(-peak_prom)
        if peak_y[i] < ip.MAX_LABEL_TRANSMITTANCE and ip.find_closest_assignment(peak_x[i]) is not None
    ],
    key=lambda t: -t[0],
)[:6]:
    label_rows.append(
        {
            "Peak cm-1": f"{wn:.1f}",
            "Transmittance": f"{py:.1f}%",
            "Assignment": assignment,
        }
    )

if label_rows:
    st.markdown("### Labels shown on plot")
    st.table(label_rows)
else:
    st.info("No labeled peaks were found below the transmittance threshold.")

st.sidebar.markdown("---")
st.sidebar.write("Run this app with `streamlit run streamlit_app.py`")
