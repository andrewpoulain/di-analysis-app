#!/usr/bin/env python3
"""
Streamlit front end for reverberant field analysis and EQ target
derivation.
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
    rt60_per_band_from_irs,
    spatial_average_reverberant,
    spatial_average_reverberant_third_octave,
    estimate_di,
    derive_full_eq_target,
    predict_post_eq_steady_state_third_octave,
    predict_steady_state_from_physics,
    compare_measured_to_predicted,
    air_absorption_at_bands,
    smooth_third_octave,
    save_csv,
    xcurve_at_third_octave_bands,
    export_target_for_smaart,
    export_xcurve_for_smaart,
    validate_rt60,
    OCTAVE_CENTRES,
    THIRD_OCTAVE_CENTRES,
)

st.set_page_config(
    page_title="Room Analysis and EQ Target Tool",
    layout="wide")

st.title("Reverberant Field Analysis and EQ Target Derivation")

# ---------------------------------------------------------------------------
# Sidebar
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

st.sidebar.subheader("Room Acoustics")

throw_m = st.sidebar.number_input(
    "Throw distance (m)",
    min_value=1.0, max_value=50.0, value=10.0, step=0.5,
    help=(
        "Distance from loudspeaker to reference microphone "
        "position. Used for air absorption correction and "
        "steady-state prediction."))

temperature_c = st.sidebar.number_input(
    "Air temperature (°C)",
    min_value=10.0, max_value=35.0, value=20.0, step=1.0,
    help="Used for ISO 9613-1 air absorption calculation.")

humidity_rh = st.sidebar.number_input(
    "Relative humidity (%)",
    min_value=10.0, max_value=90.0, value=50.0, step=5.0,
    help="Used for ISO 9613-1 air absorption calculation.")

st.sidebar.subheader("Measurement Settings")

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

st.sidebar.header("Display Options")

show_physics_prediction = st.sidebar.checkbox(
    "Show physics-based prediction",
    value=True,
    help=(
        "Predicts the expected steady-state response from "
        "RT60, room geometry, throw distance, and air "
        "absorption. Use as a verification envelope."))

show_tolerance_band = st.sidebar.checkbox(
    "Show tolerance band",
    value=True,
    help=(
        "±2 dB (100 Hz–8 kHz) and ±3 dB at extremes "
        "per Section 4.2 of the white paper."))

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
# File upload
# ---------------------------------------------------------------------------

st.header("1. Upload Impulse Response Files")

st.info(
    "Upload WAV files exported from Smaart. "
    "The first file (alphabetically) is treated as the "
    "reference position for the direct field measurement. "
    "All files are used for spatial averaging of the "
    "reverberant field.")

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

    with st.spinner("Processing — this may take a moment "
                    "for multiple IR files..."):

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
                    ir = apply_calibration(
                        ir, fs, str(cal_path))
                ref_ir = ir
                ref_fs = fs
            irs.append(ir)

        st.success(
            f"Loaded {len(irs)} IR file(s) at {ref_fs} Hz")

        # -----------------------------------------------------------
        # Octave band analysis
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

        hf_corr, lf_corr, all_corr = derive_full_eq_target(
            direct_levels, reverb_levels, reverb_levels,
            channel_cfg,
            transition_hz=transition_hz)

        # -----------------------------------------------------------
        # 1/3 octave band analysis
        # -----------------------------------------------------------

        direct_levels_3rd, _ = direct_field_at_third_octave_bands(
            ref_ir, ref_fs, gate_ms=gate_ms)

        reverb_levels_3rd = \
            spatial_average_reverberant_third_octave(irs, ref_fs)

        # Air absorption
        air_abs_3rd = air_absorption_at_bands(
            THIRD_OCTAVE_CENTRES,
            throw_m=throw_m,
            temperature_c=temperature_c,
            humidity_rh=humidity_rh)

        air_abs_1k = air_abs_3rd.get(1000.0, 0.0)
        air_abs_8k = air_abs_3rd.get(8000.0, 0.0)
        air_abs_16k = air_abs_3rd.get(16000.0, 0.0)
        st.info(
            f"Air absorption at {throw_m:.0f} m — "
            f"1 kHz: {air_abs_1k:.2f} dB, "
            f"8 kHz: {air_abs_8k:.2f} dB, "
            f"16 kHz: {air_abs_16k:.2f} dB")

        # Predicted steady-state BEFORE EQ with half-octave splice
        zero_corr = {int(b): 0.0 for b in OCTAVE_CENTRES}
        predicted_before_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                zero_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

        # Predicted steady-state AFTER EQ with half-octave splice
        predicted_after_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                all_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

        # Physics-based prediction and tolerance bands
        physics_predicted_3rd, tol_upper_3rd, tol_lower_3rd = \
            predict_steady_state_from_physics(
                direct_levels_3rd,
                reverb_levels_3rd,
                rt60_bands,
                volume_m3=volume,
                surface_area_m2=surface,
                throw_m=throw_m,
                temperature_c=temperature_c,
                humidity_rh=humidity_rh)

        # Verification comparison
        verification_results = compare_measured_to_predicted(
            predicted_before_3rd,
            physics_predicted_3rd,
            tol_upper_3rd,
            tol_lower_3rd)

        # -----------------------------------------------------------
        # Reference level and normalisation
        # -----------------------------------------------------------

        ref_level_3rd = direct_levels_3rd.get(1000.0, None)
        if ref_level_3rd is None or np.isnan(ref_level_3rd):
            available = {
                k: v for k, v in direct_levels_3rd.items()
                if not np.isnan(v)}
            if available:
                ref_level_3rd = available[
                    min(available.keys(),
                        key=lambda k: abs(k - 1000.0))]
            else:
                ref_level_3rd = 0.0

        def norm(d):
            return {b: v - ref_level_3rd
                    for b, v in d.items()
                    if not np.isnan(v)}

        direct_3rd_norm = norm(direct_levels_3rd)
        reverb_3rd_norm = norm(reverb_levels_3rd)
        before_3rd_norm = norm(predicted_before_3rd)
        after_3rd_norm = norm(predicted_after_3rd)
        physics_3rd_norm = norm(physics_predicted_3rd)
        tol_upper_norm = norm(tol_upper_3rd)
        tol_lower_norm = norm(tol_lower_3rd)

        # -----------------------------------------------------------
        # X-curve
        # -----------------------------------------------------------

        xcurve_raw = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES,
            screen_size=xcurve_size)

        third_oct_sorted = sorted(direct_levels_3rd.keys())
        direct_vals_sorted = [
            direct_levels_3rd[b] for b in third_oct_sorted]
        ref_direct_level = float(np.interp(
            np.log10(float(xcurve_ref_band)),
            np.log10(third_oct_sorted),
            direct_vals_sorted))

        xcurve_display = dict(xcurve_raw)

        # -----------------------------------------------------------
        # Save outputs
        # -----------------------------------------------------------

        df = save_csv(
            direct_levels, reverb_levels, di, rt60_bands,
            all_corr, channel_name, str(out_dir))

        eq_target_path = (
            out_dir / f"{channel_name}_eq_target.txt")
        export_target_for_smaart(
            target_levels_3rd=after_3rd_norm,
            ref_level_db=ref_level_3rd,
            output_path=eq_target_path,
            label=f'{channel_name} EQ Target — {room_name}')

        xcurve_export_path = (
            out_dir /
            f"{channel_name}_xcurve_"
            f"{'large' if xcurve_size == 'large' else 'small'}"
            f".txt")
        export_xcurve_for_smaart(
            xcurve_levels_3rd=xcurve_raw,
            ref_level_db=ref_direct_level,
            output_path=xcurve_export_path,
            screen_size=xcurve_size)

    # ---------------------------------------------------------------
    # RT60 warnings
    # ---------------------------------------------------------------

    if rt60_warnings:
        st.warning("RT60 validation warnings:")
        for band, msg in rt60_warnings.items():
            st.write(f"  **{band} Hz:** {msg}")

    # ---------------------------------------------------------------
    # Helper: build Plotly trace
    # ---------------------------------------------------------------

    def make_trace(levels_dict, name, colour,
                   dash='solid', width=2, opacity=1.0):
        if not levels_dict:
            return go.Scatter(x=[], y=[], name=name,
                              line=dict(color=colour))
        bands_sorted = sorted(levels_dict.keys())
        x = [float(b) for b in bands_sorted]
        y = [float(levels_dict[b]) for b in bands_sorted]
        return go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name=name,
            line=dict(color=colour, dash=dash, width=width),
            marker=dict(size=4),
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

    if show_physics_prediction and physics_3rd_norm:
        fig_main.add_trace(make_trace(
            physics_3rd_norm,
            'Physics-based steady-state prediction',
            'darkgreen', dash='longdash',
            width=2, opacity=0.8))

    if (show_tolerance_band
            and tol_upper_norm and tol_lower_norm):
        bands_tol = sorted(tol_upper_norm.keys())
        x_tol = [float(b) for b in bands_tol]
        y_upper = [tol_upper_norm[b] for b in bands_tol]
        y_lower = [tol_lower_norm.get(b, np.nan)
                   for b in bands_tol]
        fig_main.add_trace(go.Scatter(
            x=x_tol + x_tol[::-1],
            y=y_upper + y_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 128, 0, 0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Tolerance band (±2 dB / ±3 dB)',
            showlegend=True,
            hoverinfo='skip'))

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
        title=(
            f"{channel_name} — {room_name} — "
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
            y=-0.45,
            xanchor='left',
            x=0),
        height=580,
        hovermode='x unified')

    st.plotly_chart(fig_main, width='stretch')

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
    # EQ corrections chart
    # ---------------------------------------------------------------

    st.header("3. Derived EQ Corrections")

    fig_eq = go.Figure()

    bands_sorted = sorted(all_corr.keys())
    corr_vals = [all_corr[b] for b in bands_sorted]
    colours = ['tomato' if v < 0 else 'steelblue'
               for v in corr_vals]

    fig_eq.add_trace(go.Bar(
        x=[str(b) for b in bands_sorted],
        y=corr_vals,
        marker_color=colours,
        name='EQ correction (dB)'))

    fig_eq.add_hline(
        y=0, line_dash='dot', line_color='grey')

    fig_eq.update_layout(
        title='EQ Correction per Octave Band',
        xaxis_title='Octave band (Hz)',
        yaxis_title='Correction (dB)',
        height=350)

    st.plotly_chart(fig_eq, width='stretch')

    # ---------------------------------------------------------------
    # Verification against physics prediction
    # ---------------------------------------------------------------

    if show_physics_prediction:
        st.header("4. Verification Against Physics Prediction")

        st.caption(
            "The physics-based prediction is derived from the "
            "measured RT60, room geometry, throw distance, and "
            "air absorption. The measured steady-state should "
            "fall within ±2 dB (100 Hz–8 kHz) or ±3 dB "
            "(below 100 Hz and above 8 kHz). "
            "Bands outside tolerance indicate a system fault, "
            "not an EQ problem.")

        oct_bands_display = [
            63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        ver_rows = []
        for b in oct_bands_display:
            nearest = min(
                verification_results.keys(),
                key=lambda k: abs(k - b),
                default=None)
            if nearest is None:
                continue
            r = verification_results[nearest]
            within = r['within_tolerance']
            status = (
                '✅ Pass' if within is True
                else '❌ Fail' if within is False
                else '—')
            ver_rows.append({
                'Band (Hz)': b,
                'Predicted (dB)': r['predicted'],
                'Delta (dB)': r['delta'],
                'Tolerance': (
                    f"±{r['tolerance']:.0f} dB"
                    if not np.isnan(r['tolerance'])
                    else '—'),
                'Status': status})

        if ver_rows:
            ver_df = pd.DataFrame(ver_rows)
            st.dataframe(ver_df, hide_index=True)

        fails = [r for r in ver_rows
                 if r['Status'] == '❌ Fail']
        if fails:
            fail_bands = [r['Band (Hz)'] for r in fails]
            hf_fails = [b for b in fail_bands if b >= 4000]
            lf_fails = [b for b in fail_bands if b <= 250]
            mid_fails = [b for b in fail_bands
                         if 250 < b < 4000]
            if hf_fails:
                st.warning(
                    f"HF shortfall at {hf_fails} Hz — "
                    "check screen absorption, loudspeaker "
                    "aim, or screen insertion loss. "
                    "This is an installation issue, "
                    "not an EQ issue.")
            if lf_fails:
                st.warning(
                    f"LF deviation at {lf_fails} Hz — "
                    "check for modal problems or boundary "
                    "gain. Spatial averaging may need more "
                    "positions.")
            if mid_fails:
                st.warning(
                    f"Mid-band deviation at {mid_fails} Hz — "
                    "check level calibration and gain "
                    "alignment.")
        else:
            st.success(
                "All bands within tolerance. "
                "The system is behaving as the physics "
                "predict.")

        with st.expander("Air absorption detail"):
            air_rows = []
            for b in oct_bands_display:
                nearest = min(
                    air_abs_3rd.keys(),
                    key=lambda k: abs(k - b))
                air_rows.append({
                    'Band (Hz)': b,
                    'Absorption (dB)': round(
                        air_abs_3rd[nearest], 3)})
            st.dataframe(
                pd.DataFrame(air_rows), hide_index=True)
            st.caption(
                f"Temperature: {temperature_c:.0f} °C, "
                f"Humidity: {humidity_rh:.0f}% RH, "
                f"Throw: {throw_m:.1f} m. "
                "Calculated per ISO 9613-1.")

    # ---------------------------------------------------------------
    # RT60 and DI plots
    # ---------------------------------------------------------------

    st.header("5. RT60 and Directivity Index")

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
    fig_rt_di.update_layout(height=380)

    st.plotly_chart(fig_rt_di, width='stretch')

    # ---------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------

    st.header("6. Measurement Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate length", f"{gate_ms_used:.1f} ms")
    with col2:
        st.metric("Transition frequency",
                  f"{transition_hz} Hz")
    with col3:
        st.metric("IR files processed", len(irs))
    with col4:
        valid_rt60 = [v for v in rt60_bands.values()
                      if v is not None
