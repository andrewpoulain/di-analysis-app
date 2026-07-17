#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import tempfile
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Octave band definitions
# ---------------------------------------------------------------------------
OCTAVE_CENTRES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])


def octave_band_limits(centre_hz):
    return centre_hz / np.sqrt(2), centre_hz * np.sqrt(2)


# ---------------------------------------------------------------------------
# IR loading
# ---------------------------------------------------------------------------
def load_ir(filepath):
    fs, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data[:, 0]
    return data, int(fs)


def apply_calibration(ir, fs, cal_file):
    if cal_file is None or not os.path.exists(cal_file):
        return ir
    cal = np.loadtxt(cal_file, delimiter=',', skiprows=1)
    freqs = cal[:, 0]
    sens_db = cal[:, 1]
    n = len(ir)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    interp_sens = np.interp(fft_freqs, freqs, sens_db,
                            left=sens_db[0], right=sens_db[-1])
    correction = 10.0 ** (-interp_sens / 20.0)
    spectrum = np.fft.rfft(ir)
    spectrum_corrected = spectrum * correction
    return np.fft.irfft(spectrum_corrected, n=n)


# ---------------------------------------------------------------------------
# Schroeder integration
# ---------------------------------------------------------------------------
def bandpass_ir(ir, fs, f_low, f_high):
    nyq = fs / 2.0
    f_low = max(f_low, 10.0)
    f_high = min(f_high, nyq * 0.99)
    if f_low >= f_high:
        return np.zeros_like(ir)
    sos = sig.butter(4, [f_low / nyq, f_high / nyq],
                     btype='band', output='sos')
    return sig.sosfilt(sos, ir)


def schroeder_decay(ir_band):
    power = ir_band ** 2
    decay = np.cumsum(power[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)
    decay_db = 10.0 * np.log10(decay / decay[0])
    return decay_db


def initial_decay_level(ir, fs, centre_hz):
    f_low, f_high = octave_band_limits(centre_hz)
    ir_band = bandpass_ir(ir, fs, f_low, f_high)
    decay_db = schroeder_decay(ir_band)
    n_avg = max(1, int(0.005 * fs))
    return float(np.mean(decay_db[:n_avg]))


def reverberant_spectrum(ir, fs, bands=OCTAVE_CENTRES):
    return {int(b): initial_decay_level(ir, fs, b) for b in bands}


# ---------------------------------------------------------------------------
# RT60 estimation
# ---------------------------------------------------------------------------
def rt60_from_schroeder(ir, fs, centre_hz, eval_range_db=(-5, -25)):
    f_low, f_high = octave_band_limits(centre_hz)
    ir_band = bandpass_ir(ir, fs, f_low, f_high)
    decay_db = schroeder_decay(ir_band)
    times = np.arange(len(decay_db)) / fs
    lo, hi = eval_range_db
    mask = (decay_db <= lo) & (decay_db >= hi)
    if mask.sum() < 10:
        return None
    coeffs = np.polyfit(times[mask], decay_db[mask], 1)
    slope = coeffs[0]
    if slope >= 0:
        return None
    return float(-60.0 / slope)


def room_constant(rt60_s, volume_m3, surface_area_m2):
    if rt60_s is None or rt60_s <= 0:
        return None
    alpha = 0.161 * volume_m3 / (rt60_s * surface_area_m2)
    alpha = min(alpha, 0.999)
    return surface_area_m2 * alpha / (1.0 - alpha)


def rt60_per_band_from_irs(ir_list, fs, bands=OCTAVE_CENTRES):
    rt60_all = {int(b): [] for b in bands}
    for ir in ir_list:
        for b in bands:
            rt = rt60_from_schroeder(ir, fs, b)
            if rt is not None:
                rt60_all[int(b)].append(rt)
    return {b: float(np.mean(v)) if v else None
            for b, v in rt60_all.items()}


# ---------------------------------------------------------------------------
# Gated direct field
# ---------------------------------------------------------------------------
def detect_direct_arrival(ir, fs, threshold_db=-20):
    power_db = 20.0 * np.log10(np.abs(ir) / (np.max(np.abs(ir)) + 1e-30))
    candidates = np.where(power_db >= threshold_db)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def detect_first_reflection(ir, fs, direct_idx, min_gap_ms=2.0):
    min_gap = int(min_gap_ms * fs / 1000.0)
    search_start = direct_idx + min_gap
    direct_level = np.abs(ir[direct_idx])
    threshold = direct_level * 10.0 ** (-20.0 / 20.0)
    search = np.abs(ir[search_start:])
    peaks, _ = sig.find_peaks(search, height=threshold, distance=min_gap)
    if len(peaks) == 0:
        return None
    return int(search_start + peaks[0])


def gated_direct_field(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
    direct_idx = detect_direct_arrival(ir, fs)
    reflection_idx = detect_first_reflection(ir, fs, direct_idx)
    if gate_ms is None:
        if reflection_idx is not None:
            gap_samples = reflection_idx - direct_idx
            gate_samples = int(0.9 * gap_samples)
        else:
            gate_samples = int(0.020 * fs)
    else:
        gate_samples = int(gate_ms * fs / 1000.0)
    gate_ms_used = gate_samples / fs * 1000.0
    ir_gated = ir[direct_idx: direct_idx + gate_samples].copy()
    window = np.hanning(2 * len(ir_gated))[:len(ir_gated)]
    ir_gated *= window
    n_fft = int(2 ** np.ceil(np.log2(max(len(ir_gated), 2))))
    spectrum = np.fft.rfft(ir_gated, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitude = 20.0 * np.log10(np.abs(spectrum) + 1e-30)
    magnitude -= np.max(magnitude)
    return freqs, magnitude, gate_ms_used


def direct_field_at_bands(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
    freqs, magnitude, gate_ms_used = gated_direct_field(ir, fs, gate_ms)
    levels = {}
    for b in bands:
        f_low, f_high = octave_band_limits(b)
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.sum() > 0:
            power = 10.0 ** (magnitude[mask] / 10.0)
            levels[int(b)] = float(10.0 * np.log10(np.mean(power)))
        else:
            levels[int(b)] = np.nan
    return levels, gate_ms_used


# ---------------------------------------------------------------------------
# Spatial averaging
# ---------------------------------------------------------------------------
def spatial_average_reverberant(ir_list, fs, bands=OCTAVE_CENTRES):
    all_spectra = [reverberant_spectrum(ir, fs, bands) for ir in ir_list]
    averaged = {}
    for b in bands:
        b = int(b)
        levels_db = [s[b] for s in all_spectra]
        powers = [10.0 ** (l / 10.0) for l in levels_db]
        averaged[b] = float(10.0 * np.log10(np.mean(powers)))
    return averaged


# ---------------------------------------------------------------------------
# DI estimation
# ---------------------------------------------------------------------------
def estimate_di(direct_levels, reverberant_levels,
                rt60_per_band, volume_m3, surface_area_m2,
                bands=OCTAVE_CENTRES):
    di = {}
    for b in bands:
        b = int(b)
        d = direct_levels.get(b, np.nan)
        r = reverberant_levels.get(b, np.nan)
        rt60 = rt60_per_band.get(b)
        R = room_constant(rt60, volume_m3, surface_area_m2)
        if np.isnan(d) or np.isnan(r) or R is None or R <= 0:
            di[b] = np.nan
            continue
        room_correction = 10.0 * np.log10(4.0 / R)
        di[b] = float(d - r + room_correction)
    return di


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def build_figure(direct_levels, reverberant_levels, di_estimates,
                 rt60_per_band, gate_ms_used, channel_name,
                 bands=OCTAVE_CENTRES):
    bands_int = [int(b) for b in bands]
    band_labels = [str(b) for b in bands_int]
    x = np.arange(len(bands_int))

    direct_vals = [direct_levels.get(b, np.nan) for b in bands_int]
    reverb_vals = [reverberant_levels.get(b, np.nan) for b in bands_int]
    di_vals = [di_estimates.get(b, np.nan) for b in bands_int]
    rt60_vals = [rt60_per_band.get(b) or np.nan for b in bands_int]

    ref_band = 1000
    d_ref = direct_levels.get(ref_band, 0.0) or 0.0
    r_ref = reverberant_levels.get(ref_band, 0.0) or 0.0
    direct_norm = [v - d_ref for v in direct_vals]
    reverb_norm = [v - r_ref for v in reverb_vals]
    diff = [d - r for d, r in zip(direct_norm, reverb_norm)]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Reverberant Field Analysis - {channel_name}", fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Direct vs Reverberant
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, direct_norm, 'o-', color='steelblue',
             label=f'Direct (gated {gate_ms_used:.1f} ms)')
    ax1.plot(x, reverb_norm, 's--', color='firebrick',
             label='Reverberant (spatially averaged)')
    ax1.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels, rotation=45)
    ax1.set_xlabel('Octave band (Hz)')
    ax1.set_ylabel('Level (dB, norm. 1 kHz)')
    ax1.set_title('Direct vs Reverberant Field')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: DI
    ax2 = fig.add_subplot(gs[0, 1])
    valid = [(i, v) for i, v in enumerate(di_vals) if not np.isnan(v)]
    if valid:
        xi, yi = zip(*valid)
        ax2.plot(list(xi), list(yi), 'D-', color='darkorange')
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_labels, rotation=45)
    ax2.set_xlabel('Octave band (Hz)')
    ax2.set_ylabel('DI (dB)')
    ax2.set_title('Estimated Directivity Index DI(f)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: RT60
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x, rt60_vals, color='mediumseagreen', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(band_labels, rotation=45)
    ax3.set_xlabel('Octave band (Hz)')
    ax3.set_ylabel('RT60 (s)')
    ax3.set_title('RT60 per Octave Band')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Direct minus Reverberant
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, diff, '^-', color='mediumpurple')
    ax4.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax4.set_xticks(x)
    ax4.set_xticklabels(band_labels, rotation=45)
    ax4.set_xlabel('Octave band (Hz)')
    ax4.set_ylabel('Difference (dB)')
    ax4.set_title('Direct minus Reverberant (uncorrected)')
    ax4.grid(True, alpha=0.3)

    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reverberant Field Analysis",
    page_icon="🔊",
    layout="wide",
)

st.title("🔊 Reverberant Field Analysis & DI Estimation")
st.caption(
    "Processes IRs exported from Smaart Live - "
    "Schroeder integration and DI extraction"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Room Parameters")

    volume = st.number_input(
        "Room Volume (m3)",
        min_value=10.0,
        max_value=50000.0,
        value=500.0,
        step=10.0,
        help="Total room volume in cubic metres",
    )

    surface_area = st.number_input(
        "Total Surface Area (m2)",
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
        help="Automatically sets gate to 90% of the direct-to-first-reflection interval",
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
        "One cal file applies to all positions for that channel."
    )

    st.divider()
    st.header("About")
    st.markdown(
        "Upload WAV IRs exported from **Smaart Live**. "
        "The **first file** uploaded per channel is the "
        "**reference position** for direct field gating."
    )

# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------
st.subheader("Channel Configuration")

num_channels = st.number_input(
    "Number of speaker channels to analyse",
    min_value=1,
    max_value=16,
    value=1,
    step=1,
)

channel_configs = []

for ch_idx in range(int(num_channels)):
    with st.expander(f"Channel {ch_idx + 1}", expanded=(ch_idx == 0)):

        col1, col2 = st.columns([1, 2])

        with col1:
            ch_name = st.text_input(
                "Channel name",
                value=f"Channel_{ch_idx + 1}",
                key=f"ch_name_{ch_idx}",
            )

        with col2:
            uploaded_files = st.file_uploader(
                "Upload IR WAV files (one per mic position)",
                type=["wav"],
                accept_multiple_files=True,
                key=f"ch_files_{ch_idx}",
                help="First file = reference position for direct field gating.",
            )

        st.markdown("**Microphone Calibration** *(optional)*")

        use_cal = st.checkbox(
            "Apply microphone calibration correction",
            value=False,
            key=f"use_cal_{ch_idx}",
        )

        cal_file_upload = None
        if use_cal:
            cal_file_upload = st.file_uploader(
                "Calibration CSV  -  columns: frequency_hz, sensitivity_db",
                type=["csv"],
                accept_multiple_files=False,
                key=f"ch_cal_{ch_idx}",
            )

            if cal_file_upload is not None:
                cal_preview = pd.read_csv(io.BytesIO(cal_file_upload.read()))
                cal_file_upload.seek(0)

                with st.expander("Preview calibration data", expanded=False):
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.dataframe(cal_preview, use_container_width=True)
                    with col_b:
                        fig_cal, ax_cal = plt.subplots(figsize=(5, 3))
                        ax_cal.plot(
                            cal_preview.iloc[:, 0],
                            cal_preview.iloc[:, 1],
                            color='steelblue',
                        )
                        ax_cal.set_xlabel("Frequency (Hz)")
                        ax_cal.set_ylabel("Sensitivity (dB)")
                        ax_cal.set_title(f"Cal curve - {cal_file_upload.name}")
                        ax_cal.set_xscale("log")
                        ax_cal.grid(True, alpha=0.3)
                        st.pyplot(fig_cal)
                        plt.close(fig_cal)

        channel_configs.append({
            "name": ch_name,
            "files": uploaded_files,
            "gate_ms": gate_ms,
            "cal_file": cal_file_upload,
            "use_cal": use_cal,
        })

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
st.divider()

run_button = st.button(
    "Run Analysis",
    type="primary",
    use_container_width=True,
)

if run_button:

    any_files = any(len(ch["files"]) > 0 for ch in channel_configs)
    if not any_files:
        st.error("Please upload at least one WAV file before running.")
        st.stop()

    for ch in channel_configs:
        if ch["use_cal"] and ch["cal_file"] is None:
            st.warning(
                f"**{ch['name']}**: calibration enabled but no CSV uploaded "
                f"- correction will be skipped."
            )

    all_results = []

    for ch in channel_configs:
        if not ch["files"]:
            st.warning(f"No files uploaded for **{ch['name']}** - skipping.")
            continue

        st.subheader(f"Results - {ch['name']}")

        with st.spinner(f"Processing {ch['name']}..."):

            with tempfile.TemporaryDirectory() as tmpdir:

                # Save calibration file if provided
                cal_path = None
                if ch["use_cal"] and ch["cal_file"] is not None:
                    cal_path = os.path.join(tmpdir, ch["cal_file"].name)
                    with open(cal_path, "wb") as f:
                        f.write(ch["cal_file"].read())

                # Load and calibrate each IR
                irs = []
                ref_ir = None
                ref_fs = None

                for i, uploaded in enumerate(ch["files"]):
                    tmp_wav = os.path.join(tmpdir, uploaded.name)
                    with open(tmp_wav, "wb") as f:
                        f.write(uploaded.read())
                    ir, fs = load_ir(tmp_wav)
                    ir = apply_calibration(ir, fs, cal_path)
                    irs.append(ir)
                    if i == 0:
                        ref_ir = ir
                        ref_fs = fs

                # Analysis
                direct_levels, gate_ms_used = direct_field_at_bands(
                    ref_ir, ref_fs, gate_ms=ch["gate_ms"]
                )
                rt60_bands = rt60_per_band_from_irs(irs, ref_fs)
                reverb_levels = spatial_average_reverberant(irs, ref_fs)
                di = estimate_di(
                    direct_levels, reverb_levels,
                    rt60_bands, volume, surface_area,
                )

                # Results table
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

        # Metrics
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
                "Applied" if (ch["use_cal"] and cal_path) else "None",
            )

        # Table
        st.dataframe(df, use_container_width=True)

        # Plot
        fig = build_figure(
            direct_levels, reverb_levels, di,
            rt60_bands, gate_ms_used, ch["name"],
        )
        st.pyplot(fig)
        plt.close(fig)

        # Download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {ch['name']} CSV",
            data=csv_bytes,
            file_name=f"{ch['name']}_results.csv",
            mime="text/csv",
            key=f"dl_{ch['name']}",
        )

    # Session summary
    if all_results:
        st.divider()
        st.subheader("Session Summary")
        summary = pd.concat(all_results, ignore_index=True)
        st.dataframe(summary, use_container_width=True)
        summary_csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Session Summary CSV",
            data=summary_csv,
            file_name="session_summary.csv",
            mime="text/csv",
        )
