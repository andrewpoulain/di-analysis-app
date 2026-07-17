import streamlit as st
import numpy as np
import tempfile
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import your existing analysis functions directly
from di_analysis import (
    load_ir,
    apply_calibration,
    direct_field_at_bands,
    rt60_per_band_from_irs,
    spatial_average_reverberant,
    estimate_di,
    plot_results,
    save_csv,
    OCTAVE_CENTRES,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reverberant Field Analysis",
    page_icon="🔊",
    layout="wide",
)

st.title("🔊 Reverberant Field Analysis & DI Estimation")
st.caption("Processes IRs exported from Smaart Live — Schroeder integration and DI extraction")

# ---------------------------------------------------------------------------
# Sidebar — Room Parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Room Parameters")

    volume = st.number_input(
        "Room Volume (m³)",
        min_value=10.0,
        max_value=50000.0,
        value=500.0,
        step=10.0,
        help="Total room volume in cubic metres"
    )

    surface_area = st.number_input(
        "Total Surface Area (m²)",
        min_value=10.0,
        max_value=20000.0,
        value=350.0,
        step=10.0,
        help="Total room surface area in square metres"
    )

    st.divider()
    st.header("Gate Settings")

    auto_gate = st.checkbox(
        "Auto-detect gate length",
        value=True,
        help="Automatically sets gate to 90% of the direct-to-first-reflection interval"
    )

    gate_ms = None
    if not auto_gate:
        gate_ms = st.number_input(
            "Manual Gate Length (ms)",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=0.5,
        )

    st.divider()
    st.header("About")
    st.markdown(
        "Upload WAV IRs exported from **Smaart Live**. "
        "The first file uploaded per channel is treated as the "
        "**reference position** for direct field gating."
    )

# ---------------------------------------------------------------------------
# Main area — Channel setup
# ---------------------------------------------------------------------------
st.subheader("Channel Configuration")

num_channels = st.number_input(
    "Number of speaker channels to analyse",
    min_value=1,
    max_value=16,
    value=1,
    step=1,
)

# ---------------------------------------------------------------------------
# Per-channel inputs
# ---------------------------------------------------------------------------
channel_configs = []

for ch_idx in range(int(num_channels)):
    with st.expander(f"Channel {ch_idx + 1}", expanded=(ch_idx == 0)):

        col1, col2 = st.columns([1, 2])

        with col1:
            ch_name = st.text_input(
                "Channel name",
                value=f"Channel_{ch_idx + 1}",
                key=f"ch_name_{ch_idx}",
                help="Used in plot titles and output filenames"
            )

        with col2:
            uploaded_files = st.file_uploader(
                "Upload IR WAV files (one per mic position)",
                type=["wav"],
                accept_multiple_files=True,
                key=f"ch_files_{ch_idx}",
                help="First file = reference position for direct field. "
                     "All files used for RT60 and reverberant field averaging."
            )

        channel_configs.append({
            "name": ch_name,
            "files": uploaded_files,
            "gate_ms": gate_ms,
        })

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
st.divider()

run_button = st.button(
    "▶ Run Analysis",
    type="primary",
    use_container_width=True,
)

if run_button:

    # Validate inputs
    any_files = any(len(ch["files"]) > 0 for ch in channel_configs)
    if not any_files:
        st.error("Please upload at least one WAV file before running.")
        st.stop()

    all_results = []

    for ch in channel_configs:
        if not ch["files"]:
            st.warning(f"No files uploaded for **{ch['name']}** — skipping.")
            continue

        st.subheader(f"Results — {ch['name']}")

        with st.spinner(f"Processing {ch['name']}..."):

            # Write uploaded files to a temp directory so load_ir can read them
            with tempfile.TemporaryDirectory() as tmpdir:
                irs = []
                ref_ir = None
                ref_fs = None

                for i, uploaded in enumerate(ch["files"]):
                    tmp_path = os.path.join(tmpdir, uploaded.name)
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded.read())

                    ir, fs = load_ir(tmp_path)
                    # No calibration file in UI mode — can be added later
                    irs.append(ir)

                    if i == 0:
                        ref_ir = ir
                        ref_fs = fs

                # --- Direct field ---
                direct_levels, gate_ms_used = direct_field_at_bands(
                    ref_ir, ref_fs, gate_ms=ch["gate_ms"]
                )

                # --- RT60 per band ---
                rt60_bands = rt60_per_band_from_irs(irs, ref_fs)

                # --- Spatially averaged reverberant field ---
                reverb_levels = spatial_average_reverberant(irs, ref_fs)

                # --- DI estimate ---
                di = estimate_di(
                    direct_levels,
                    reverb_levels,
                    rt60_bands,
                    volume,
                    surface_area,
                )

                # --- Build results table ---
                bands_int = [int(b) for b in OCTAVE_CENTRES]
                rows = []
                for b in bands_int:
                    rt = rt60_bands.get(b)
                    rows.append({
                        "Band (Hz)": b,
                        "Direct Field (dB)": round(direct_levels.get(b, np.nan), 2),
                        "Reverberant Field (dB)": round(reverb_levels.get(b, np.nan), 2),
                        "DI Estimate (dB)": round(di.get(b, np.nan), 2),
                        "RT60 (s)": round(rt, 3) if rt is not None else None,
                    })
                df = pd.DataFrame(rows)
                all_results.append(df.assign(Channel=ch["name"]))

        # --- Display metrics row ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gate Length Applied", f"{gate_ms_used:.1f} ms")
        with col2:
            st.metric("IR Files Loaded", len(ch["files"]))
        with col3:
            mid_di = di.get(1000, np.nan)
            st.metric(
                "DI at 1 kHz",
                f"{mid_di:.1f} dB" if not np.isnan(mid_di) else "n/a"
            )

        # --- Results table ---
        st.dataframe(
            df.style.format({
                "Direct Field (dB)": "{:.2f}",
                "Reverberant Field (dB)": "{:.2f}",
                "DI Estimate (dB)": "{:.2f}",
                "RT60 (s)": "{:.3f}",
            }),
            use_container_width=True,
        )

        # --- Inline plots ---
        fig = _build_figure(
            direct_levels, reverb_levels, di,
            rt60_bands, gate_ms_used, ch["name"]
        )
        st.pyplot(fig)
        plt.close(fig)

        # --- Per-channel CSV download ---
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Download {ch['name']} CSV",
            data=csv_bytes,
            file_name=f"{ch['name']}_results.csv",
            mime="text/csv",
            key=f"dl_{ch['name']}",
        )

    # --- Session summary download ---
    if all_results:
        st.divider()
        st.subheader("Session Summary")
        summary = pd.concat(all_results, ignore_index=True)
        st.dataframe(summary, use_container_width=True)

        summary_csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Full Session Summary CSV",
            data=summary_csv,
            file_name="session_summary.csv",
            mime="text/csv",
        )
