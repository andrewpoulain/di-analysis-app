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
    direct_field_at_third_octave_bands,
    gated_direct_field,
    rt60_per_band_from_irs,
    spatial_average_reverberant,
    spatial_average_reverberant_third_octave,
    estimate_di,
    derive_full_eq_target,
    predict_post_eq_steady_state_third_octave,
    design_fir_filter,
    design_lf_iir_filters,
    plot_analysis,
    plot_eq_and_filter,
    save_fir_coefficients,
    save_iir_parameters,
    save_csv,
    xcurve_at_third_octave_bands,
    interpolate_correction_to_freqs,
    validate_rt60,
    OCTAVE_CENTRES,
    THIRD_OCTAVE_CENTRES,
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

st.sidebar.header("Target Curve")

show_xcurve = st.sidebar.checkbox(
    "Show X-curve target", value=False)

xcurve_size = st.sidebar.radio(
    "X-curve variant",
    options=["large", "small"],
    index=0,
    help=(
        "Large: standard X-curve for rooms > 150 m³ "
        "(SMPTE ST 202M / ISO 2969). "
        "Small: modified X-curve for rooms < 150 m³ "
        "(SMPTE RP 200), flat region extended to 4 kHz."))

xcurve_ref_band = st.sidebar.number_input(
    "X-curve reference frequency (Hz)",
    min_value=500, max_value=2000, value=1000, step=100,
    help=(
        "The X-curve will be aligned to the measured direct "
        "field level at this frequency."))

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

        # -----------------------------------------------------------
        # Octave band analysis — used for EQ derivation and DI
        # -----------------------------------------------------------

        direct_levels, gate_ms_used = direct_field_at_bands(
            ref_ir, ref_fs, gate_ms=gate_ms)

        transition_hz = max(
            125, int(2.0 / (gate_ms_used / 1000.0)))
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

        rt60_bands = rt60_per_band_from_irs(irs, ref_fs)
        rt60_warnings = validate_rt60(rt60_bands)
        reverb_levels = spatial_average_reverberant(irs, ref_fs)

        di = estimate_di(
            direct_levels, reverb_levels,
            rt60_bands, volume, surface)

        hf_corr, lf_corr, all_corr, predicted_oct = \
            derive_full_eq_target(
                direct_levels, reverb_levels, reverb_levels,
                rt60_bands, room_cfg, channel_cfg,
                transition_hz=transition_hz)

        fir_coeffs, fir_freq_response = design_fir_filter(
            hf_corr, ref_fs, n_taps=n_taps,
            transition_hz=transition_hz)

        sos, lf_filter_params = design_lf_iir_filters(
            lf_corr, ref_fs, transition_hz=transition_hz)

        # -----------------------------------------------------------
        # 1/3 octave band analysis — used for display
        # -----------------------------------------------------------

        direct_levels_3rd, _ = direct_field_at_third_octave_bands(
            ref_ir, ref_fs, gate_ms=gate_ms)

        reverb_levels_3rd = \
            spatial_average_reverberant_third_octave(irs, ref_fs)

        # Predicted steady-state BEFORE EQ
        zero_corr = {int(b): 0.0 for b in OCTAVE_CENTRES}
        predicted_before_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd, reverb_levels_3rd, zero_corr)

        # Predicted steady-state AFTER EQ
        predicted_after_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd, reverb_levels_3rd, all_corr)

        # -----------------------------------------------------------
        # X-curve aligned to measured direct field
        # -----------------------------------------------------------

        xcurve_raw = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES,
            screen_size=xcurve_size)

        third_oct_sorted = sorted(direct_levels_3rd.keys())
        direct_vals_sorted = [direct_levels_3rd[b]
                               for b in third_oct_sorted]
        ref_direct_level = float(np.interp(
            np.log10(float(xcurve_ref_band)),
            np.log10(third_oct_sorted),
            direct_vals_sorted))

        xcurve_norm = {
            b: v + ref_direct_level
            for b, v in xcurve_raw.items()}

        # -----------------------------------------------------------
        # Normalise all traces to 0 dB at 1 kHz
        # -----------------------------------------------------------

        ref_level_3rd = direct_levels_3rd.get(1000.0, 0.0) or 0.0

        direct_3rd_norm = {
            b: v - ref_level_3rd
            for b, v in direct_levels_3rd.items()}
        reverb_3rd_norm = {
            b: v - ref_level_3rd
            for b, v in reverb_levels_3rd.items()}
        before_3rd_norm = {
            b: v - ref_level_3rd
            for b, v in predicted_before_3rd.items()
            if not np.isnan(v)}
        after_3rd_norm = {
            b: v - ref_level_3rd
            for b, v in predicted_after_3rd.items()
            if not np.isnan(v)}
        xcurve_display = {
            b: v - ref_direct_level
            for b, v in xcurve_norm.items()}

        # -----------------------------------------------------------
        # Save outputs
        # -----------------------------------------------------------

        save_fir_coefficients(
            fir_coeffs, channel_name, ref_fs, str(out_dir))
        save_iir_parameters(
            lf_filter_params, channel_name, str(out_dir))
        df = save_csv(
            direct_levels, reverb_levels, di, rt60_bands,
            all_corr, predicted_oct, channel_name, str(out_dir))
        plot_analysis(
            direct_levels, reverb_levels, di, rt60_bands,
            gate_ms_used, channel_name, str(out_dir))
        plot_eq_and_filter(
            direct_levels, reverb_levels, all_corr,
            predicted_oct, fir_freq_response, lf_filter_params,
            channel_name, str(out_dir))

    # ---------------------------------------------------------------
    # RT60 warnings
    # ---------------------------------------------------------------

    if rt60_warnings:
        st.warning("RT60 validation warnings:")
        for band, msg in rt60_warnings.items():
            st.write(f"  **{band} Hz:** {msg}")

    # ---------------------------------------------------------------
    # Helper: build Plotly trace from 1/3-octave dict
    # ---------------------------------------------------------------

    def make_trace(levels_dict, name, colour,
                   dash='solid', width=2, opacity=1.0):
        bands_sorted = sorted(levels_dict.keys())
        x = [float(b) for b in bands_sorted]
        y = [levels_dict[b] for b in bands_sorted]
        return go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name=name,
            line=dict(color=colour, dash=dash, width=width),
            marker=dict(size=5),
            opacity=opacity)

    # ---------------------------------------------------------------
    # Main response plot
    # ---------------------------------------------------------------

    st.header("2. Measured Response and Target")

    fig_main = go.Figure()

    fig_main.add_trace(make_trace(
        direct_3rd_norm,
        'Direct field (1/3 oct)',
        'steelblue', width=2))

    fig_main.add_trace(make_trace(
        reverb_3rd_norm,
        'Reverberant field (spatially averaged)',
        'firebrick', dash='dash', width=1.5, opacity=0.7))

    fig_main.add_trace(make_trace(
        before_3rd_norm,
        'Predicted steady-state before EQ',
        'grey', dash='dot', width=1.5, opacity=0.8))

    fig_main.add_trace(make_trace(
        after_3rd_norm,
        'Predicted steady-state after EQ',
        'darkorange', width=2))

    flat_target = {
        float(b): 0.0 for b in THIRD_OCTAVE_CENTRES
        if float(b) >= 63}
    fig_main.add_trace(make_trace(
        flat_target,
        'Flat target',
        'green', dash='dot', width=1.5, opacity=0.6))

    if show_xcurve:
        xcurve_label = (
            f'X-curve '
            f'({"large" if xcurve_size == "large" else "small"}'
            f' room, SMPTE ST 202M / ISO 2969)')
        fig_main.add_trace(make_trace(
            xcurve_display,
            xcurve_label,
            'purple', dash='dashdot', width=2))

    fig_main.update_layout(
        title=(f"{channel_name} — "
               f"Measured Response and Target "
               f"(1/3 octave bands)"),
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[
                20, 31.5, 63, 125, 250, 500,
                1000, 2000, 4000, 8000, 16000],
            ticktext=[
                '20', '31.5', '63', '125', '250', '500',
                '1k', '2k', '4k', '8k', '16k'],
            range=[np.log10(20), np.log10(20000)]),
        yaxis=dict(
            title='Level (dB, normalised at 1 kHz)',
            range=[-25, 15]),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.38,
            xanchor='left',
            x=0),
        height=560,
        hovermode='x unified')

    st.plotly_chart(fig_main, use_container_width=True)

    if show_xcurve:
        st.caption(
            "X-curve: flat to "
            + ("2 kHz" if xcurve_size == "large" else "4 kHz")
            + ", then −3 dB/octave. "
            "LF rolloff below 63 Hz at −3 dB/octave. "
            "Reference: SMPTE ST 202M / ISO 2969"
            + (" (modified for small rooms, SMPTE RP 200)."
               if xcurve_size == "small" else "."))

    # ---------------------------------------------------------------
    # EQ correction and filter plot
    # ---------------------------------------------------------------

    st.header("3. EQ Correction and Filter Response")

    fig_eq = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'EQ Correction per Octave Band',
            'FIR Filter Frequency Response'))

    bands_sorted = sorted(all_corr.keys())
    corr_vals = [all_corr[b] for b in bands_sorted]
    colours = ['tomato' if v < 0 else 'steelblue'
               for v in corr_vals]

    fig_eq.add_trace(
        go.Bar(
            x=[str(b) for b in bands_sorted],
            y=corr_vals,
            marker_color=colours,
            name='EQ correction (dB)'),
        row=1, col=1)

    fig_eq.add_hline(
        y=0, line_dash='dot',
        line_color='grey', row=1, col=1)

    fir_freqs, fir_mag = fir_freq_response
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
        tickvals=[
            63, 125, 250, 500, 1000,
            2000, 4000, 8000, 16000],
        ticktext=[
            '63', '125', '250', '500', '1k',
            '2k', '4k', '8k', '16k'],
        row=1, col=2)

    fig_eq.update_yaxes(
        title_text='Correction (dB)', row=1, col=1)
    fig_eq.update_yaxes(
        title_text='Filter magnitude (dB)', row=1, col=2)
    fig_eq.update_layout(height=400)

    st.plotly_chart(fig_eq, use_container_width=True)

    # ---------------------------------------------------------------
    # RT60 and DI plots
    # ---------------------------------------------------------------

    st.header("4. RT60 and Directivity Index")

    fig_rt_di = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'RT60 per Octave Band',
            'Estimated Directivity Index DI(f)'))

    rt60_bands_sorted = sorted(
        b for b in rt60_bands
        if rt60_bands[b] is not None)
    rt60_vals = [rt60_bands[b] for b in rt60_bands_sorted]

    fig_rt_di.add_trace(
        go.Bar(
            x=[str(b) for b in rt60_bands_sorted],
            y=rt60_vals,
            marker_color='mediumseagreen',
            name='RT60 (s)'),
        row=1, col=1)

    di_sorted = sorted(
        b for b in di
        if not np.isnan(di.get(b, np.nan)))
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

    st.plot
