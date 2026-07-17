import streamlit as st
import numpy as np
import tempfile
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import your existing analysis functions directly
from di_analysis import (
    load_ir,
    apply_calibration,
    direct_field_at_bands,
    rt60_per_band_from_irs,
    spatial_average_reverberant,
    estimate_di,
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
st.caption(
    "Processes IRs exported from Smaart Live — "
    "Schroeder integration and DI extraction"
)

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
        help="Total room volume in cubic metres",
    )

    surface_area = st.number_input(
        "Total Surface Area (m²)",
        min_value=10.0,
        max_value=20000.0,
        value=350.0,
        step=10.0,
        help="Total room surface area in square metres",
    )

    st.divider()
    st.header("Gate Settings")

    auto_gate = st.checkbox(
        "Auto-detect gate length",
        value=True,
        help=(
            "Automatically sets gate to 90% of the "
            "direct-to-first-reflection interval"
        ),
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
    st.header("Calibration File Format")
    st.markdown(
        "Cal files must be **CSV** with a header row and two columns:\n\n"
        "`frequency_hz, sensitivity_db`\n\n"
        "One cal file can be shared across all positions for a given channel, "
        "or left blank to skip correction."
    )

    st.divider()
    st.header("About")
    st.markdown(
        "Upload WAV IRs exported from **Smaart Live**. "
        "The **first file** uploaded per channel is treated as the "
        "**reference position** for direct field gating."
    )

# ---------------------------------------------------------------------------
# Helper — 4-panel figure
# ---------------------------------------------------------------------------
def _build_figure(
    direct_levels,
    reverberant_levels,
    di_estimates,
    rt60_per_band,
    gate_ms_used,
    channel_name,
    bands=OCTAVE_CENTRES,
):
    """Builds the 4-panel figure and returns it for st.pyplot()."""
    bands_int   = [int(b) for b in bands]
    band_labels = [str(b) for b in bands_int]
    x           = np.arange(len(bands_int))

    direct_vals = [direct_levels.get(b, np.nan) for b in bands_int]
    reverb_vals = [reverberant_levels.get(b, np.nan) for b in bands_int]
    di_vals     = [di_estimates.get(b, np.nan) for b in bands_int]
    rt60_vals   = [rt60_per_band.get(b) or np.nan for b in bands_int]

    ref_band    = 1000
    d_ref       = direct_levels.get(ref_band, 0.0) or 0.0
    r_ref       = reverberant_levels.get(ref_band, 0.0) or 0.0
    direct_norm = [v - d_ref for v in direct_vals]
    reverb_norm = [v - r_ref for v in reverb_vals]
    diff        = [d - r for d, r in zip(direct_norm, reverb_norm)]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Reverberant Field Analysis — {channel_name}", fontsize=13
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Direct vs Reverberant
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        x, direct_norm, "o-", color="steelblue",
        label=f"Direct (gated {gate_ms_used:.1f} ms)",
    )
    ax1.plot(
        x, reverb_norm, "s--", color="firebrick",
        label="Reverberant (spatially averaged)",
    )
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels, rotation=45)
    ax1.set_xlabel("Octave band (Hz)")
    ax1.set_ylabel("Level (dB, norm. 1 kHz)")
    ax1.set_title("Direct vs Reverberant Field")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: DI estimate
    ax2 = fig.add_subplot(gs[0, 1])
    valid = [(i, v) for i, v in enumerate(di_vals) if not np.isnan(v)]
    if valid:
        xi, yi = zip(*valid)
        ax2.plot(list(xi), list(yi), "D-", color="darkorange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_labels, rotation=45)
    ax2.set_xlabel("Octave band (Hz)")
    ax2.set_ylabel("DI (dB)")
    ax2.set_title("Estimated Directivity Index DI(f)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: RT60
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x, rt60_vals, color="mediumseagreen", alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(band_labels, rotation=45)
    ax3.set_xlabel("Octave band (Hz)")
    ax3.set_ylabel("RT60 (s)")
    ax3.set_title("RT60 per Octave Band")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Direct minus Reverberant (uncorrected)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, diff, "^-", color="mediumpurple")
    ax4.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax4.set_xticks(x)
    ax4.set_xticklabels(band_labels, rotation=45)
    ax4.set_xlabel("Octave band (Hz)")
    ax4.set_ylabel("Difference (dB)")
    ax4.set_title("Direct minus Reverberant (uncorrected)")
    ax4.grid(True, alpha=0.3)

    return fig


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
                help="Used in plot titles and output filenames",
            )

        with col2:
            uploaded_files = st.file_uploader(
                "Upload IR WAV files (one per mic position)",
                type=["wav"],
                accept_multiple_files=True,
                key=f"ch_files_{ch_idx}",
                help=(
                    "First file = reference position for direct field. "
                    "All files used for RT60 and reverberant field averaging."
                ),
            )

        # --- Calibration file (optional) ---
        st.markdown("**Microphone Calibration** *(optional)*")

        use_cal = st.checkbox(
            "Apply microphone calibration correction",
            value=False,
            key=f"use_cal_{ch_idx}",
            help=(
                "If enabled, upload a CSV calibration file. "
                "The same correction is applied to all positions for this channel."
            ),
        )

        cal_file_upload = None
        if use_cal:
            cal_file_upload = st.file_uploader(
                "Calibration CSV  —  columns: frequency_hz, sensitivity_db",
                type=["csv"],
                accept_multiple_files=False,
                key=f"ch_cal_{ch_idx}",
                help=(
                    "Two-column CSV with a header row. "
                    "Sensitivity values in dB are interpolated to the IR "
                    "frequency resolution and applied as a frequency-domain "
                    "correction."
                ),
            )

            if cal_file_upload is not None:
                # Preview the calibration curve
                import io
                cal_preview = pd.read_csv(io.BytesIO(cal_file_upload.read()))
                cal_file_upload.seek(0)   # reset so it can be read again later

                with st.expander("Preview calibration data", expanded=False):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.dataframe(cal_preview, use_container_width=True)
                    with col_b:
                        fig_cal, ax_cal = plt.subplots(figsize=(5, 3))
                        ax_cal.plot(
                            cal_preview.iloc[:, 0],
                            cal_preview.iloc[:, 1],
                            color="steelblue",
                        )
                        ax_cal.set_xlabel("Frequency (Hz)")
                        ax_cal.set_ylabel("Sensitivity (dB)")
                        ax_cal.set_title(
                            f"Cal curve — {cal_file_upload.name}"
                        )
                        ax_cal.set_xscale("log")
                        ax_cal.grid(True, alpha=0.3)
                        st.pyplot(fig_cal)
                        plt.close(fig_cal)

        channel_configs.append(
            {
                "name": ch_name,
                "files": uploaded_files,
                "gate_ms": gate_ms,
                "cal_file": cal_file_upload,
                "use_cal": use_cal,
            }
        )

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

    # Validate
    any_files = any(len(ch["files"]) > 0 for ch in channel_configs)
    if not any_files:
        st.error("Please upload at least one WAV file before running.")
        st.stop()

    # Warn if cal is enabled but no file provided
    for ch in channel_configs:
        if ch["use_cal"] and ch["cal_file"] is None:
            st.warning(
                f"**{ch['name']}**: calibration correction is enabled "
                f"but no CSV was uploaded — correction will be skipped."
            )

    all_results = []

    for ch in channel_configs:
        if not ch["files"]:
            st.warning(f"No files uploaded for **{ch['name']}** — skipping.")
            continue

        st.subheader(f"Results — {ch['name']}")

        with st.spinner(f"Processing {ch['name']}..."):

            with tempfile.TemporaryDirectory() as tmpdir:

                # --- Save calibration file to temp dir if provided ---
                cal_path = None
                if ch["use_cal"] and ch["cal_file"] is not None:
                    cal_path = os.path.join(tmpdir, ch["cal_file"].name)
                    with open(cal_path, "wb") as f:
                        f.write(ch["cal_file"].read())

                # --- Load and calibrate each IR ---
                irs    = []
                ref_ir = None
                ref_fs = None

                for i, uploaded in enumerate(ch["files"]):
                    tmp_wav = os.path.join(tmpdir, uploaded.name)
                    with open(tmp_wav, "wb") as f:
                        f.write(uploaded.read())

                    ir, fs = load_ir(tmp_wav)
                    ir     = apply_calibration(ir, fs, cal_path)
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

                # --- Results table ---
                bands_int = [int(b) for b in OCTAVE_CENTRES]
                rows = []
                for b in bands_int:
                    rt = rt60_bands.get(b)
                    rows.append(
                        {
                            "Band (Hz)": b,
                            "Direct Field (dB)": round(
                                direct_levels.get(b, np.nan), 2
                            ),
                            "Reverberant Field (dB)": round(
                                reverb_levels.get(b, np.nan), 2
                            ),
                            "DI Estimate (dB)": round(
                                di.get(b, np.nan), 2
                            ),
                            "RT60 (s)": round(rt, 3) if rt is not None else None,
                        }
                    )
                df = pd.DataFrame(rows)
                all_results.append(df.assign(Channel=ch["name"]))

        # --- Metrics row ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gate Length Applied", f"{gate_ms_used:.1f} ms")
        with col2:
            st.metric("IR Files Loaded", len(ch["files"]))
        with col3:
            mid_di = di.get(1000, np.nan)
            st.metric(
                "DI at 1 kHz",
                f"{mid_di:.1f} dB" if not np.isnan(mid_di) else "n/a",
            )
        with col4:
            st.metric(
                "Calibration",
                "Applied ✓" if (ch["use_cal"] and cal_path) else "None",
            )

        # --- Results table ---
        st.dataframe(
            df.style.format(
                {
                    "Direct Field (dB)": "{:.2f}",
                    "Reverberant Field (dB)": "{:.2f}",
                    "DI Estimate (dB)": "{:.2f}",
                    "RT60 (s)": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        # --- Inline plot ---
        fig = _build_figure(
            direct_levels,
            reverb_levels,
            di,
            rt60_bands,
            gate_ms_used,
            ch["name"],
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

    # --- Session summary ---
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
