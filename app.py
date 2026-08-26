#!/usr/bin/env python3
"""
Streamlit front end for reverberant field analysis and EQ derivation.
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from di_analysis import (
    load_ir,
    apply_calibration,
    direct_field_at_bands,
    gated_direct_field,
    rt60_per_band_from_irs,
    spatial_average_reverberant,
    estimate_di,
    derive_full_eq_target,
    design_fir_filter,
    design_lf_iir_filters,
    plot_analysis,
    plot_eq_and_filter,
    save_fir_coefficients,
    save_iir_parameters,
    save_csv,
    OCTAVE_CENTRES,
    interpolate_correction_to_freqs,
)

st.set_page_config(
    page_title="Room Analysis and EQ Tool",
    layout="wide")

st.title("Reverberant Field Analysis and EQ Target Derivation")

# ---------------------------------------------------------------------------
# Sidebar: room configuration
# ---------------------------------------------------------------------------

st.sidebar.header("Room Configuration")

room_name = st.sidebar.text_input(
    "Room name", value="Stage A")

st.sidebar.subheader("Room Dimensions")

room_length = st.sidebar.number_input(
    "Length (m)",
    min_value=1.0, max_value=200.0, value=20.0, step=0.5)
room_width = st.sidebar.number_input(
    "Width (m)",
    min_value=1.0, max_value=100.0, value=15.0, step=0.5)
room_height = st.sidebar.number_input(
    "Height (m)",
    min_value=1.0, max_value=30.0, value=8.0, step=0.5)

volume = room_length * room_width * room_height
surface = 2.0 * (
    room_length * room_width +
    room_length * room_height +
    room_width * room_height)

st.sidebar.metric("Volume (m³)", f"{volume:.1f}")
st.sidebar.metric("Surface area (m²)", f"{surface:.1f}")

st.sidebar.subheader("Measurement Settings")

n_taps = st.sidebar.selectbox(
    "FIR filter taps", [512, 1024, 2048, 4096], index=1)

st.sidebar.info(
    "Transition frequency between gated direct field "
    "and spatial average is calculated automatically "
    "from the detected gate length after measurement.")

st.sidebar.header("Channel Configuration")

channel_name = st.sidebar.text_input(
    "Channel name", value="Left")
gate_ms_input = st.sidebar.number_input(
    "Gate length ms (0 = auto)",
    min_value=0.0, max_value=100.0, value=0.0, step=0.5)
gate_ms = None if gate_ms_input == 0.0 else gate_ms_input

hf_shelf_hz = st.sidebar.number_input(
    "HF shelf frequency (Hz)",
    min_value=4000, max_value=16000, value=10000, step=1000)
hf_shelf_db = st.sidebar.number_input(
    "HF shelf level (dB)",
    min_value=-6.0, max_value=0.0, value=0.0, step=0.5)

# ---------------------------------------------------------------------------
# Helper: smooth a spectrum for display
# ---------------------------------------------------------------------------

def fractional_octave_smooth(freqs, magnitude, fraction=3):
    """
    Apply 1/N octave smoothing to a magnitude spectrum for display.
    fraction=3 gives 1/3 octave smoothing.
    Returns smoothed magnitude array.
    """
    smoothed = np.zeros_like(magnitude)
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        f_lo = f / (2 ** (1.0 / (2 * fraction)))
        f_hi = f * (2 ** (1.0 / (2 * fraction)))
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() > 0:
            power = 10.0 ** (magnitude[mask] / 10.0)
            smoothed[i] = 10.0 * np.log10(np.mean(power))
        else:
            smoothed[i] = magnitude[i]
    return smoothed


def octave_band_trace(bands_dict, name, colour, dash='solid'):
    """
    Build a Plotly scatter trace from an octave band dict.
    bands_dict: {centre_hz: level_db}
    """
    bands_sorted = sorted(bands_dict.keys())
    x = [float(b) for b in bands_sorted]
    y = [bands_dict[b] for b in bands_sorted]
    return go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        name=name,
        line=dict(color=colour, dash=dash, width=2),
        marker=dict(size=8))


# ---------------------------------------------------------------------------
# Main panel: file upload
# ---------------------------------------------------------------------------

st.header("1. Upload Impulse Response Files")

st.info(
    "Upload WAV files exported from Smaart. "
    "The first file is treated as the reference position. "
    "All remaining files are used for spatial averaging.")

uploaded_files = st.file_uploader(
    "IR WAV files (upload all positions for this channel)",
    type=["wav"],
    accept_multiple_files=True)

cal_file = st.file_uploader(
    "Microphone calibration file for reference position "
    "(two-column CSV: frequency_hz, sensitivity_db — optional)",
    type=["csv"])

# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

if uploaded_files and st.button("Run Analysis"):

    with st.spinner("Processing..."):

        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "output"
        out_dir.mkdir()

        ir_paths = []
        for uf in uploaded_files:
            p = tmp_dir / uf.name
            p.write_bytes(uf.read())
            ir_paths.append(p)

        cal_path = None
        if cal_file:
            cal_path = tmp_dir / cal_file.name
            cal_path.write_bytes(cal_file.read())

        irs = []
        ref_ir = None
        ref_fs = None

        for i, p in enumerate(sorted(ir_paths)):
            ir, fs = load_ir(str(p))
            if i == 0:
                if cal_path:
                    ir = apply_calibration(ir, fs, str(cal_path))
                ref_ir = ir
                ref_fs = fs
            irs.append(ir)

        st.success(f"Loaded {len(irs)} IR files at {ref_fs} Hz")

        # Direct field
        direct_levels, gate_ms_used = direct_field_at_bands(
            ref_ir, ref_fs, gate_ms=gate_ms)

        # Full resolution direct field for display
        freqs_full, direct_full, _ = gated_direct_field(
            ref_ir, ref_fs, gate_ms=gate_ms)

        # Transition frequency from gate
        transition_hz = max(125, int(2.0 / (gate_ms_used / 1000.0)))
        transition_hz = int(min(
            [b for b in OCTAVE_CENTRES if b >= transition_hz],
            default=250))

        st.info(
            f"Gate detected: {gate_ms_used:.1f} ms — "
            f"transition frequency set to {transition_hz} Hz")

        room_cfg = {
            'volume_m3': volume,
            'surface_area_m2': surface,
            'transition_hz': transition_hz,
            'fir_taps': n_taps,
            'microphone_calibration': [],
        }
        channel_cfg = {
            'name': channel_name,
            'gate_ms': gate_ms,
            'hf_shelf_hz': hf_shelf_hz,
            'hf_shelf_db': hf_shelf_db,
        }

        # RT60
        rt60_bands = rt60_per_band_from_irs(irs, ref_fs)

        # Reverberant field
        reverb_levels = spatial_average_reverberant(irs, ref_fs)

        # DI
        di = estimate_di(
            direct_levels, reverb_levels,
            rt60_bands, volume, surface)

        # EQ target
        hf_corr, lf_corr, all_corr, predicted = derive_full_eq_target(
            direct_levels, reverb_levels, reverb_levels,
            rt60_bands, room_cfg, channel_cfg,
            transition_hz=transition_hz)

        # FIR filter
        fir_coeffs, fir_freq_response = design_fir_filter(
            hf_corr, ref_fs, n_taps=n_taps,
            transition_hz=transition_hz)

        # IIR filters
        sos, lf_filter_params = design_lf_iir_filters(
            lf_corr, ref_fs, transition_hz=transition_hz)

        # Full resolution correction curve for display
        fir_freqs, fir_mag = fir_freq_response

        # Normalise direct field display to 0 dB at 1 kHz
        ref_level = direct_levels.get(1000, 0.0) or 0.0

        direct_levels_norm = {
            b: v - ref_level
            for b, v in direct_levels.items()}
        reverb_levels_norm = {
            b: v - ref_level
            for b, v in reverb_levels.items()}
        predicted_norm = {
            b: v - ref_level
            for b, v in predicted.items()
            if not np.isnan(v)}

        # Flat target line at 0 dB across all bands
        target_flat = {int(b): 0.0 for b in OCTAVE_CENTRES}

        # Smooth full resolution direct field for display
        mask_display = (freqs_full >= 20) & (freqs_full <= 20000)
        freqs_display = freqs_full[mask_display]
        direct_display = direct_full[mask_display]
        direct_display_norm = direct_display - np.max(direct_display)

        # Apply 1/3 octave smoothing
        direct_smoothed = fractional_octave_smooth(
            freqs_display, direct_display_norm, fraction=3)

        # Save filter files
        save_fir_coefficients(
            fir_coeffs, channel_name, ref_fs, str(out_dir))
        save_iir_parameters(
            lf_filter_params, channel_name, str(out_dir))
        df = save_csv(
            direct_levels, reverb_levels, di, rt60_bands,
            all_corr, predicted, channel_name, str(out_dir))

        # Static plots for download
        plot_analysis(
            direct_levels, reverb_levels, di, rt60_bands,
            gate_ms_used, channel_name, str(out_dir))
        plot_eq_and_filter(
            direct_levels, reverb_levels, all_corr,
            predicted, fir_freq_response, lf_filter_params,
            channel_name, str(out_dir))

    # ---------------------------------------------------------------------------
    # Main response plot
    # ---------------------------------------------------------------------------

    st.header("2. Measured Response and Target")

    fig_main = go.Figure()

    # Full resolution smoothed direct field
    fig_main.add_trace(go.Scatter(
        x=freqs_display,
        y=direct_smoothed,
        mode='lines',
        name='Direct field (1/3 oct smoothed)',
        line=dict(color='steelblue', width=1.5),
        opacity=0.6))

    # Octave band direct field
    fig_main.add_trace(octave_band_trace(
        direct_levels_norm,
        'Direct field (octave bands)',
        'steelblue'))

    # Octave band reverberant field
    fig_main.add_trace(octave_band_trace(
        reverb_levels_norm,
        'Reverberant field (spatially averaged)',
        'firebrick',
        dash='dash'))

    # Predicted steady-state after EQ
    fig_main.add_trace(octave_band_trace(
        predicted_norm,
        'Predicted steady-state after EQ',
        'darkorange'))

    # Flat target
    fig_main.add_trace(octave_band_trace(
        target_flat,
        'Target (flat direct field)',
        'green',
        dash='dot'))

    fig_main.update_layout(
        title=f"{channel_name} — Measured Response and Target",
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
            ticktext=['63', '125', '250', '500', '1k',
                      '2k', '4k', '8k', '16k'],
            range=[np.log10(50), np.log10(20000)]),
        yaxis=dict(
            title='Level (dB, normalised at 1 kHz)',
            range=[-20, 10]),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='left',
            x=0),
        height=500,
        hovermode='x unified')

    st.plotly_chart(fig_main, use_container_width=True)

    # ---------------------------------------------------------------------------
    # EQ correction and filter plot
    # ---------------------------------------------------------------------------

    st.header("3. EQ Correction and Filter Response")

    fig_eq = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'EQ Correction per Octave Band',
            'FIR Filter Frequency Response'))

    # EQ correction bars
    bands_sorted = sorted(all_corr.keys())
    corr_vals = [all_corr[b] for b in bands_sorted]
    colours = ['tomato' if v < 0 else 'steelblue'
               for v in corr_vals]

    fig_eq.add_trace(
        go.Bar(
            x=[str(b) for b in bands_sorted],
            y=corr_vals,
            marker_color=colours,
            name='EQ correction (dB)',
            showlegend=True),
        row=1, col=1)

    fig_eq.add_hline(
        y=0, line_dash='dot',
        line_color='grey', row=1, col=1)

    # FIR filter response
    fir_mask = fir_freqs > 20
    fig_eq.add_trace(
        go.Scatter(
            x=fir_freqs[fir_mask],
            y=fir_mag[fir_mask],
            mode='lines',
            name='FIR filter response',
            line=dict(color='darkorange', width=1.5)),
        row=1, col=2)

    fig_eq.add_hline(
        y=0, line_dash='dot',
        line_color='grey', row=1, col=2)

    fig_eq.update_xaxes(
        type='log',
        tickvals=[63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        ticktext=['63', '125', '250', '500', '1k',
                  '2k', '4k', '8k', '16k'],
        row=1, col=2)

    fig_eq.update_yaxes(
        title_text='Correction (dB)', row=1, col=1)
    fig_eq.update_yaxes(
        title_text='Filter magnitude (dB)', row=1, col=2)

    fig_eq.update_layout(height=400)

    st.plotly_chart(fig_eq, use_container_width=True)

    # ---------------------------------------------------------------------------
    # RT60 and DI plots
    # ---------------------------------------------------------------------------

    st.header("4. RT60 and Directivity Index")

    fig_rt_di = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'RT60 per Octave Band',
            'Estimated Directivity Index DI(f)'))

    rt60_bands_sorted = sorted(
        b for b in rt60_bands if rt60_bands[b] is not None)
    rt60_vals = [rt60_bands[b] for b in rt60_bands_sorted]

    fig_rt_di.add_trace(
        go.Bar(
            x=[str(b) for b in rt60_bands_sorted],
            y=rt60_vals,
            marker_color='mediumseagreen',
            name='RT60 (s)'),
        row=1, col=1)

    di_sorted = sorted(
        b for b in di if not np.isnan(di.get(b, np.nan)))
    di_vals = [di[b] for b in di_sorted]

    fig_rt_di.add_trace(
        go.Scatter(
            x=[str(b) for b in di_sorted],
            y=di_vals,
            mode='lines+markers',
            name='DI estimate (dB)',
            line=dict(color='mediumpurple', width=2),
            marker=dict(size=8)),
        row=1, col=2)

    fig_rt_di.update_yaxes(
        title_text='RT60 (s)', row=1, col=1)
    fig_rt_di.update_yaxes(
        title_text='DI (dB)', row=1, col=2)

    fig_rt_di.update_layout(height=400)

    st.plotly_chart(fig_rt_di, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------------------

    st.header("5. Measurement Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate length", f"{gate_ms_used:.1f} ms")
    with col2:
        st.metric("Transition frequency", f"{transition_hz} Hz")
    with col3:
        st.metric("IR files processed", len(irs))
    with col4:
        avg_rt60 = np.mean([v for v in rt60_bands.values()
                            if v is not None])
        st.metric("Mean RT60", f"{avg_rt60:.2f} s")

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------

    st.header("6. Results Table")
    st.dataframe(df)

    if lf_filter_params:
        st.header("7. LF IIR Filter Parameters")
        st.dataframe(pd.DataFrame(lf_filter_params))

    # ---------------------------------------------------------------------------
    # Downloads
    # ---------------------------------------------------------------------------

    st.header("8. Download Filter Files")

    col1, col2, col3, col4 = st.columns(4)

    fir_txt = out_dir / f"{channel_name}_fir.txt"
    fir_wav = out_dir / f"{channel_name}_fir.wav"
    csv_path = out_dir / f"{channel_name}_results.csv"
    iir_path = out_dir / f"{channel_name}_iir_params.csv"

    if fir_txt.exists():
        with col1:
            st.download_button(
                "FIR coefficients (text)",
                data=fir_txt.read_bytes(),
                file_name=fir_txt.name,
                mime="text/plain")

    if fir_wav.exists():
        with col2:
            st.download_button(
                "FIR as WAV (for FIR Designer)",
                data=fir_wav.read_bytes(),
                file_name=fir_wav.name,
                mime="audio/wav")

    if csv_path.exists():
        with col3:
            st.download_button(
                "Results CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv")

    if iir_path.exists():
        with col4:
            st.download_button(
                "IIR parameters CSV",
                data=iir_path.read_bytes(),
                file_name=iir_path.name,
                mime="text/csv")

    shutil.rmtree(tmp_dir)
