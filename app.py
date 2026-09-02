#!/usr/bin/env python3
"""
Streamlit front end for reverberant field analysis and EQ
target derivation.
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from di_analysis import (
    load_ir,
    apply_calibration,
    direct_field_at_bands,
    direct_field_at_third_octave_bands,
    direct_reverb_energy_all_bands,
    rt60_per_band_from_irs,
    transition_frequency_from_gate,
    spatial_average_reverberant,
    spatial_average_reverberant_third_octave,
    estimate_di_from_multiple_irs,
    derive_full_eq_target,
    predict_post_eq_steady_state_third_octave,
    reconstruct_steady_state,
    smooth_third_octave,
    save_csv,
    get_target_levels,
    xcurve_at_third_octave_bands,
    smpte_422_at_third_octave_bands,
    export_target_for_smaart,
    export_xcurve_for_smaart,
    export_selected_target_for_smaart,
    validate_rt60,
    room_constant,
    room_constant_formula_used,
    mixing_time_ms,
    late_start_ms,
    generate_parametric_eq,
    TARGET_TYPES,
    OCTAVE_CENTRES,
    THIRD_OCTAVE_CENTRES,
)

st.set_page_config(
    page_title="Room Analysis and EQ Target Tool",
    layout="wide")

st.title(
    "Reverberant Field Analysis and EQ Target Derivation")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Room Configuration")

room_name = st.sidebar.text_input(
    "Room name", value="Stage A")

st.sidebar.subheader("Room Dimensions")

room_length = st.sidebar.number_input(
    "Length (m)",
    min_value=1.0, max_value=200.0,
    value=20.0, step=0.5)
room_width = st.sidebar.number_input(
    "Width (m)",
    min_value=1.0, max_value=100.0,
    value=15.0, step=0.5)
room_height = st.sidebar.number_input(
    "Height (m)",
    min_value=1.0, max_value=30.0,
    value=8.0, step=0.5)

volume = room_length * room_width * room_height
surface = 2.0 * (
    room_length * room_width
    + room_length * room_height
    + room_width * room_height)

st.sidebar.metric("Volume (m3)", str(round(volume, 1)))
st.sidebar.metric(
    "Surface area (m2)", str(round(surface, 1)))

listener_distance_m = room_length * (2.0 / 3.0)
st.sidebar.metric(
    "Listener distance (m)",
    str(round(listener_distance_m, 1)),
    help=(
        "Estimated as 2/3 of room length. "
        "Used for DI estimation via classical "
        "D/R inversion."))

t_mix = mixing_time_ms(volume)
late_start = late_start_ms(volume)

st.sidebar.metric(
    "Mixing time estimate (ms)",
    str(round(t_mix, 1)),
    help=(
        "Empirical: 0.0033 * V^(1/3) seconds. "
        "Mixing time constant — not the Polack "
        "RT60 constant 0.0117. "
        "Verified: 500m3=26ms, 1500m3=38ms, "
        "4000m3=52ms, 8000m3=66ms."))

st.sidebar.metric(
    "Late energy window start (ms)",
    str(round(late_start, 1)),
    help=(
        "max(50 ms floor, mixing time). "
        "50 ms floor applies for rooms < ~3500 m3."))

st.sidebar.subheader("Measurement Settings")

st.sidebar.info(
    "Transition frequency calculated automatically "
    "from gate length using the 3/T rule.")

st.sidebar.header("Channel Configuration")

channel_name = st.sidebar.text_input(
    "Channel name", value="Left")

gate_ms_input = st.sidebar.number_input(
    "Gate length ms (0 = auto)",
    min_value=0.0, max_value=100.0,
    value=0.0, step=0.5)
gate_ms = None if gate_ms_input == 0.0 else gate_ms_input

hf_shelf_hz = st.sidebar.number_input(
    "HF shelf frequency (Hz)",
    min_value=4000, max_value=16000,
    value=10000, step=1000)
hf_shelf_db = st.sidebar.number_input(
    "HF shelf level (dB)",
    min_value=-6.0, max_value=0.0,
    value=0.0, step=0.5)

st.sidebar.header("Monitor Target Response")

target_type = st.sidebar.selectbox(
    "Target",
    options=TARGET_TYPES,
    index=0,
    help=(
        "Flat Direct Field: EQ to flat response. "
        "X-Curve Large: SMPTE ST 202M / ISO 2969. "
        "X-Curve Small: SMPTE RP 200 (flat to 4 kHz). "
        "SMPTE ST 422: -1.5 dB/octave above 2 kHz."))

XCURVE_REF_HZ = 1000

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.header("1. Upload Impulse Response Files")

st.info(
    "Upload WAV files exported from Smaart. "
    "All files are used for spatial averaging of the "
    "reverberant field and RT60 estimation. "
    "Select the reference position file below — this "
    "file is used for the gated direct field measurement "
    "and should be the primary mix position IR.")

uploaded_files = st.file_uploader(
    "IR WAV files (upload all positions for this channel)",
    type=["wav"],
    accept_multiple_files=True)

cal_file = st.file_uploader(
    "Microphone calibration file for reference position "
    "(two-column CSV: frequency_hz, sensitivity_db "
    "— optional)",
    type=["csv"])

ref_filename = None
if uploaded_files:
    filenames = sorted([f.name for f in uploaded_files])
    ref_filename = st.selectbox(
        "Reference position file (primary mix position)",
        options=filenames,
        index=0,
        help=(
            "Used for gated direct field measurement. "
            "Should be the primary mix position IR."))
    st.caption(
        "Direct field from: **" + ref_filename + "**. "
        "All files contribute to reverberant field "
        "spatial average and RT60 estimation.")

# ---------------------------------------------------------------------------
# RT60 overrides
# ---------------------------------------------------------------------------

with st.expander(
        "RT60 overrides (optional)", expanded=False):
    st.caption(
        "Enter manual RT60 values in seconds to override "
        "calculated values. Leave blank to use calculated "
        "value.")

    rt60_override_cols = st.columns(len(OCTAVE_CENTRES))
    rt60_overrides = {}
    for i, b in enumerate(OCTAVE_CENTRES):
        with rt60_override_cols[i]:
            val = st.text_input(
                str(int(b)) + " Hz",
                value="",
                key="rt60_override_" + str(int(b)),
                placeholder="s")
            if val.strip():
                try:
                    parsed = float(val.strip())
                    if 0.05 <= parsed <= 20.0:
                        rt60_overrides[int(b)] = parsed
                    else:
                        st.warning(
                            str(int(b))
                            + " Hz: 0.05-20.0 s only")
                except ValueError:
                    st.warning(
                        str(int(b))
                        + " Hz: invalid number")

# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

if (uploaded_files
        and ref_filename
        and st.button("Run Analysis")):

    with st.spinner(
            "Processing — this may take a moment "
            "for multiple IR files..."):

        tmp_dir = Path(tempfile.mkdtemp())
        out_dir = tmp_dir / "output"
        out_dir.mkdir()

        ir_paths = {}
        for uf in uploaded_files:
            p = tmp_dir / uf.name
            p.write_bytes(uf.read())
            ir_paths[uf.name] = p

        cal_path = None
        if cal_file:
            cal_path = tmp_dir / cal_file.name
            cal_path.write_bytes(cal_file.read())

        irs = []
        ref_path = ir_paths[ref_filename]
        ref_ir, ref_fs = load_ir(str(ref_path))
        if cal_path:
            ref_ir = apply_calibration(
                ref_ir, ref_fs, str(cal_path))

        for name, p in sorted(ir_paths.items()):
            ir, fs = load_ir(str(p))
            irs.append(ir)

        st.success(
            "Loaded " + str(len(irs))
            + " IR file(s) at " + str(ref_fs) + " Hz. "
            "Reference: " + ref_filename)

        # ---------------------------------------------------
        # Direct field — EQ path
        # ---------------------------------------------------

        direct_levels, gate_ms_used = \
            direct_field_at_bands(
                ref_ir, ref_fs, gate_ms=gate_ms)

        transition_hz = transition_frequency_from_gate(
            gate_ms_used, bands=OCTAVE_CENTRES)

        st.info(
            "Gate: " + str(round(gate_ms_used, 1))
            + " ms — transition: "
            + str(transition_hz) + " Hz (3/T rule)")

        channel_cfg = {
            'name': channel_name,
            'gate_ms': gate_ms,
            'hf_shelf_hz': hf_shelf_hz,
            'hf_shelf_db': hf_shelf_db,
        }

        # ---------------------------------------------------
        # RT60
        # ---------------------------------------------------

        rt60_bands = rt60_per_band_from_irs(irs, ref_fs)
        measured_rt60 = dict(
            rt60_bands.get('_measured', {}))

        manual_override_bands = set()
        if rt60_overrides:
            override_applied = []
            for b, val in rt60_overrides.items():
                old = rt60_bands.get(b)
                old_str = (
                    str(round(old * 1000)) + " ms"
                    if old is not None else "None")
                rt60_bands[b] = val
                manual_override_bands.add(b)
                override_applied.append(
                    str(b) + " Hz: " + old_str
                    + " -> " + str(round(val * 1000))
                    + " ms (manual override)")
            if override_applied:
                st.info(
                    "RT60 overrides: "
                    + "; ".join(override_applied))

        rt60_warnings = validate_rt60(rt60_bands)
        reverb_levels = spatial_average_reverberant(
            irs, ref_fs)

        # ---------------------------------------------------
        # Late energy window
        # ---------------------------------------------------

        late_start_val = late_start_ms(volume)
        st.info(
            "Late energy window: "
            + str(round(late_start_val, 1))
            + " ms (mixing time: "
            + str(round(mixing_time_ms(volume), 1))
            + " ms, 50 ms floor applied where < 50 ms)")

        # ---------------------------------------------------
        # DI estimation
        # ---------------------------------------------------

        di = estimate_di_from_multiple_irs(
            irs, ref_fs,
            rt60_per_band=rt60_bands,
            volume_m3=volume,
            surface_area_m2=surface,
            listener_distance_m=listener_distance_m,
            gate_ms=gate_ms_used,
            late_start_ms_val=late_start_val)

        # ---------------------------------------------------
        # EQ corrections against selected target
        # ---------------------------------------------------

        hf_corr, lf_corr, all_corr = \
            derive_full_eq_target(
                direct_levels, reverb_levels,
                reverb_levels, channel_cfg,
                transition_hz=transition_hz,
                target_type=target_type)

        # ---------------------------------------------------
        # 1/3 octave analysis
        # ---------------------------------------------------

        direct_levels_3rd, _ = \
            direct_field_at_third_octave_bands(
                ref_ir, ref_fs, gate_ms=gate_ms)

        reverb_levels_3rd = \
            spatial_average_reverberant_third_octave(
                irs, ref_fs)

        zero_corr = {
            int(b): 0.0 for b in OCTAVE_CENTRES}

        predicted_room_curve_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                zero_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

        predicted_post_eq_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                all_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

        reconstructed_ss, tol_upper, tol_lower = \
            reconstruct_steady_state(
                direct_levels_3rd,
                reverb_levels_3rd)

        # ---------------------------------------------------
        # Reference level and normalisation
        # ---------------------------------------------------

        ref_level_3rd = direct_levels_3rd.get(
            1000.0, None)
        if (ref_level_3rd is None
                or np.isnan(ref_level_3rd)):
            available = {
                k: v
                for k, v in direct_levels_3rd.items()
                if not np.isnan(v)}
            ref_level_3rd = (
                available[min(
                    available.keys(),
                    key=lambda k: abs(k - 1000.0))]
                if available else 0.0)

        def norm(d):
            return {
                b: v - ref_level_3rd
                for b, v in d.items()
                if not np.isnan(v)}

        direct_3rd_norm = norm(direct_levels_3rd)
        reverb_3rd_norm = norm(reverb_levels_3rd)
        room_curve_norm = norm(predicted_room_curve_3rd)
        post_eq_norm = norm(predicted_post_eq_3rd)

        # ---------------------------------------------------
        # Target curve for display
        # ---------------------------------------------------

        target_display = get_target_levels(
            target_type, bands=THIRD_OCTAVE_CENTRES)

        # ---------------------------------------------------
        # X-curves for export
        # ---------------------------------------------------

        xcurve_large = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES,
            screen_size='large')
        xcurve_small = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES,
            screen_size='small')

        third_oct_sorted = sorted(
            direct_levels_3rd.keys())
        direct_vals_sorted = [
            direct_levels_3rd[b]
            for b in third_oct_sorted]
        ref_direct_level = float(np.interp(
            np.log10(float(XCURVE_REF_HZ)),
            np.log10(third_oct_sorted),
            direct_vals_sorted))

        # ---------------------------------------------------
        # Parametric EQ filters
        # ---------------------------------------------------

        peq_filters = generate_parametric_eq(
            all_corr, max_filters=10)

        # ---------------------------------------------------
        # RT60 source tracking
        # ---------------------------------------------------

        rt60_table_rows = []
        hf_fallback_bands = set()
        for w in rt60_bands.get('_hf_warnings', []):
            if '8 kHz' in w and '16 kHz' in w:
                hf_fallback_bands.add(8000)
                hf_fallback_bands.add(16000)
            elif '16 kHz' in w:
                hf_fallback_bands.add(16000)

        for b in [int(x) for x in OCTAVE_CENTRES]:
            eff = rt60_bands.get(b)
            meas = measured_rt60.get(b)
            if b in manual_override_bands:
                source = 'Manual Override'
            elif b in hf_fallback_bands:
                source = 'HF Fallback'
            elif eff is not None:
                source = 'Measured'
            else:
                source = 'None'
            rt60_table_rows.append({
                'Band (Hz)': b,
                'Measured RT60 (s)': (
                    round(meas, 3)
                    if meas is not None else None),
                'Effective RT60 (s)': (
                    round(eff, 3)
                    if eff is not None else None),
                'Source': source,
                'R formula': room_constant_formula_used(
                    eff, volume, surface)
                if eff is not None else 'N/A',
            })

        # ---------------------------------------------------
        # Save outputs
        # ---------------------------------------------------

        df = save_csv(
            direct_levels, reverb_levels, di,
            rt60_bands, all_corr,
            channel_name, str(out_dir),
            target_type=target_type,
            predicted_post_eq=predicted_post_eq_3rd)

        eq_target_path = (
            out_dir
            / (channel_name + "_eq_target.txt"))

        export_target_for_smaart(
            target_levels_3rd=post_eq_norm,
            ref_level_db=ref_level_3rd,
            output_path=eq_target_path,
            label=(channel_name
                   + " EQ Target — " + room_name))

        # Selected target export
        selected_target_path = (
            out_dir
            / (channel_name + "_selected_target.txt"))
        export_selected_target_for_smaart(
            target_type=target_type,
            target_levels_3rd=target_display,
            ref_level_db=ref_direct_level,
            output_path=selected_target_path,
            channel_name=channel_name)

        xcurve_large_path = (
            out_dir
            / (channel_name + "_xcurve_large.txt"))
        export_xcurve_for_smaart(
            xcurve_levels_3rd=xcurve_large,
            ref_level_db=ref_direct_level,
            output_path=xcurve_large_path,
            screen_size='large')

        xcurve_small_path = (
            out_dir
            / (channel_name + "_xcurve_small.txt"))
        export_xcurve_for_smaart(
            xcurve_levels_3rd=xcurve_small,
            ref_level_db=ref_direct_level,
            output_path=xcurve_small_path,
            screen_size='small')

        st.session_state.update({
            'direct_3rd_norm': direct_3rd_norm,
            'reverb_3rd_norm': reverb_3rd_norm,
            'room_curve_norm': room_curve_norm,
            'post_eq_norm': post_eq_norm,
            'target_display': target_display,
            'target_type': target_type,
            'xcurve_large': xcurve_large,
            'xcurve_small': xcurve_small,
            'rt60_bands': rt60_bands,
            'rt60_warnings': rt60_warnings,
            'rt60_table_rows': rt60_table_rows,
            'di': di,
            'gate_ms_used': gate_ms_used,
            'transition_hz': transition_hz,
            'df': df,
            'peq_filters': peq_filters,
            'eq_target_path': str(eq_target_path),
            'selected_target_path':
                str(selected_target_path),
            'xcurve_large_path': str(xcurve_large_path),
            'xcurve_small_path': str(xcurve_small_path),
            'ref_filename': ref_filename,
            'n_irs': len(irs),
            'listener_distance_m': listener_distance_m,
            'volume': volume,
            'surface': surface,
            'late_start_val': late_start_val,
            'analysis_complete': True,
        })

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

if st.session_state.get('analysis_complete'):

    direct_3rd_norm = st.session_state['direct_3rd_norm']
    reverb_3rd_norm = st.session_state['reverb_3rd_norm']
    room_curve_norm = st.session_state['room_curve_norm']
    post_eq_norm = st.session_state['post_eq_norm']
    target_display = st.session_state['target_display']
    target_type = st.session_state['target_type']
    xcurve_large = st.session_state['xcurve_large']
    xcurve_small = st.session_state['xcurve_small']
    rt60_bands = st.session_state['rt60_bands']
    rt60_warnings = st.session_state['rt60_warnings']
    rt60_table_rows = st.session_state['rt60_table_rows']
    di = st.session_state['di']
    gate_ms_used = st.session_state['gate_ms_used']
    transition_hz = st.session_state['transition_hz']
    df = st.session_state['df']
    peq_filters = st.session_state['peq_filters']
    eq_target_path = Path(
        st.session_state['eq_target_path'])
    selected_target_path = Path(
        st.session_state['selected_target_path'])
    xcurve_large_path = Path(
        st.session_state['xcurve_large_path'])
    xcurve_small_path = Path(
        st.session_state['xcurve_small_path'])
    ref_filename = st.session_state['ref_filename']
    n_irs = st.session_state['n_irs']
    listener_distance_m = st.session_state[
        'listener_distance_m']
    volume = st.session_state['volume']
    surface = st.session_state['surface']
    late_start_val = st.session_state['late_start_val']

    # ---------------------------------------------------------------
    # RT60 warnings
    # ---------------------------------------------------------------

    if rt60_warnings:
        st.warning("RT60 validation warnings:")
        for band, msg in rt60_warnings.items():
            st.write("  **" + str(band) + ":** " + msg)

    # ---------------------------------------------------------------
    # Helper: build Plotly trace
    # ---------------------------------------------------------------

    def make_trace(levels_dict, name, colour,
                   dash='solid', width=2, opacity=1.0):
        if not levels_dict:
            return go.Scatter(
                x=[], y=[], name=name,
                line=dict(color=colour))
        bands_sorted = sorted(levels_dict.keys())
        x = [float(b) for b in bands_sorted]
        y = [float(levels_dict[b]) for b in bands_sorted]
        return go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name=name,
            line=dict(
                color=colour, dash=dash, width=width),
            marker=dict(size=4),
            opacity=opacity)

    # ---------------------------------------------------------------
    # Section 2: Main response plot
    # ---------------------------------------------------------------

    st.header("2. Measured Response and Room Curves")

    with st.expander("Plot controls", expanded=True):

        st.markdown("**Trace visibility**")

        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        with vc1:
            show_direct = st.checkbox(
                "Direct field", value=True,
                key='show_direct')
        with vc2:
            show_reverb = st.checkbox(
                "Reverberant field", value=True,
                key='show_reverb')
        with vc3:
            show_room_curve = st.checkbox(
                "Current room response", value=True,
                key='show_room_curve')
        with vc4:
            show_post_eq = st.checkbox(
                "Predicted post-EQ response", value=True,
                key='show_post_eq')
        with vc5:
            show_target = st.checkbox(
                "Selected target", value=True,
                key='show_target')

        st.markdown("**Trace level offsets (dB)**")
        st.caption(
            "Display-only. Does not affect EQ corrections "
            "or exported files.")

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            direct_offset = st.slider(
                "Direct field offset (dB)",
                min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5,
                key='direct_offset')
        with oc2:
            reverb_offset = st.slider(
                "Reverberant field offset (dB)",
                min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5,
                key='reverb_offset')
        with oc3:
            room_curve_offset = st.slider(
                "Room curve offset (dB)",
                min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5,
                key='room_curve_offset')

    direct_display = {
        b: v + direct_offset
        for b, v in direct_3rd_norm.items()}
    reverb_display = {
        b: v + reverb_offset
        for b, v in reverb_3rd_norm.items()}
    room_curve_display = {
        b: v + room_curve_offset
        for b, v in room_curve_norm.items()}

    fig_main = go.Figure()

    if show_direct:
        label = "Direct field (gated, 1/3 oct)"
        if direct_offset != 0.0:
            sign = "+" if direct_offset > 0 else ""
            label += ("  " + sign
                      + str(round(direct_offset, 1))
                      + " dB")
        fig_main.add_trace(make_trace(
            direct_display, label,
            'steelblue', width=2))

    if show_reverb:
        label = "Reverberant field (spatially averaged)"
        if reverb_offset != 0.0:
            sign = "+" if reverb_offset > 0 else ""
            label += ("  " + sign
                      + str(round(reverb_offset, 1))
                      + " dB")
        fig_main.add_trace(make_trace(
            reverb_display, label,
            'firebrick', dash='dash',
            width=1.5, opacity=0.7))

    if show_room_curve:
        label = "Current room response (no EQ)"
        if room_curve_offset != 0.0:
            sign = "+" if room_curve_offset > 0 else ""
            label += ("  " + sign
                      + str(round(room_curve_offset, 1))
                      + " dB")
        fig_main.add_trace(make_trace(
            room_curve_display, label,
            'grey', dash='solid',
            width=2, opacity=0.9))

    if show_post_eq:
        fig_main.add_trace(make_trace(
            post_eq_norm,
            "Predicted room response after EQ",
            'mediumseagreen', dash='solid', width=2))

    if show_target:
        fig_main.add_trace(make_trace(
            target_display,
            "Target: " + target_type,
            'darkorange', dash='dashdot', width=2))

    fig_main.update_layout(
        title=(
            channel_name + " — " + room_name
            + " — Response and Room Curves "
            + "(1/3 octave bands)"),
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[
                20, 31.5, 63, 125, 250, 500,
                1000, 2000, 4000, 8000, 16000],
            ticktext=[
                '20', '31.5', '63', '125', '250',
                '500', '1k', '2k', '4k', '8k', '16k'],
            range=[np.log10(20), np.log10(20000)]),
        yaxis=dict(
            title='Level (dB, normalised at 1 kHz)',
            range=[-25, 15]),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.40,
            xanchor='left',
            x=0),
        height=580,
        hovermode='x unified')

    st.plotly_chart(fig_main, width='stretch')

    st.caption(
        "Direct field (blue): gated IR at reference "
        "position (" + ref_filename + "), "
        "1/6-octave smoothed above transition. "
        "Reverberant field (red): spatially averaged "
        "Schroeder decay. "
        "Current room response (grey): energy sum of "
        "direct and reverberant, no EQ. "
        "Predicted post-EQ (green): room response after "
        "applying calculated corrections. "
        "Target (orange): selected monitoring target. "
        "All traces normalised to 0 dB at 1 kHz. "
        "Offsets are display-only.")

    # ---------------------------------------------------------------
    # Section 3: RT60
    # ---------------------------------------------------------------

    st.header("3. RT60")

    fig_rt60 = go.Figure()

    rt60_bands_sorted = sorted(
        b for b in rt60_bands
        if isinstance(b, int)
        and rt60_bands[b] is not None)

    # Measured RT60 trace
    measured_rt60_dict = rt60_bands.get('_measured', {})
    meas_x = []
    meas_y = []
    for b in rt60_bands_sorted:
        meas = measured_rt60_dict.get(b)
        if meas is not None:
            meas_x.append(float(b))
            meas_y.append(meas)

    if meas_x:
        fig_rt60.add_trace(go.Scatter(
            x=meas_x, y=meas_y,
            mode='lines+markers',
            name='Measured RT60',
            line=dict(
                color='steelblue', width=2,
                dash='dot'),
            marker=dict(size=7)))

    # Effective RT60 trace
    eff_x = [float(b) for b in rt60_bands_sorted]
    eff_y = [rt60_bands[b] for b in rt60_bands_sorted]

    fig_rt60.add_trace(go.Scatter(
        x=eff_x, y=eff_y,
        mode='lines+markers',
        name='Effective RT60',
        line=dict(color='mediumseagreen', width=2),
        marker=dict(size=8)))

    fig_rt60.update_layout(
        title='RT60 per Octave Band',
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[
                63, 125, 250, 500, 1000,
                2000, 4000, 8000, 16000],
            ticktext=[
                '63', '125', '250', '500', '1k',
                '2k', '4k', '8k', '16k']),
        yaxis=dict(
            title='RT60 (s)',
            rangemode='tozero'),
        height=380,
        hovermode='x unified')

    st.plotly_chart(fig_rt60, width='stretch')

    # RT60 detail table
    if rt60_table_rows:
        with st.expander(
                "RT60 detail table", expanded=True):
            rt60_df = pd.DataFrame(rt60_table_rows)
            st.dataframe(rt60_df, hide_index=True)
            st.caption(
                "Measured RT60: raw calculated value "
                "before fallback or override. "
                "Effective RT60: value used in all "
                "calculations. "
                "Source: Measured = direct calculation, "
                "HF Fallback = substituted from lower "
                "band, Manual Override = user-entered. "
                "Sabine used when alpha <= 0.2, "
                "Eyring when alpha > 0.2.")

    # ---------------------------------------------------------------
    # Section 4: DI
    # ---------------------------------------------------------------

    st.header("4. Directivity Index")

    fig_di = go.Figure()

    di_sorted = sorted(
        b for b in di
        if not np.isnan(di.get(b, np.nan)))
    di_vals = [di[b] for b in di_sorted]

    fig_di.add_trace(go.Scatter(
        x=[float(b) for b in di_sorted],
        y=di_vals,
        mode='lines+markers',
        name='DI estimate (dB)',
        line=dict(color='mediumpurple', width=2),
        marker=dict(size=8)))

    fig_di.update_layout(
        title=(
            "Estimated Directivity Index DI(f) — "
            "listener distance "
            + str(round(listener_distance_m, 1))
            + " m (2/3 room length)"),
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[
                63, 125, 250, 500, 1000,
                2000, 4000, 8000, 16000],
            ticktext=[
                '63', '125', '250', '500', '1k',
                '2k', '4k', '8k', '16k']),
        yaxis=dict(
            title='DI (dB)',
            range=[0, 22]),
        height=350,
        hovermode='x unified')

    st.plotly_chart(fig_di, width='stretch')

    st.caption(
        "DI from classical D/R inversion: "
        "Q = (16 x pi x r^2 / R) x (D/R), "
        "DI = 10 log10(Q). "
        "Median Q across all positions. "
        "R uses Sabine (alpha <= 0.2) or Eyring "
        "(alpha > 0.2). "
        "DI clipped to [0, 20] dB. "
        "Late energy window: "
        + str(round(late_start_val, 1)) + " ms.")

    # ---------------------------------------------------------------
    # Section 5: Parametric EQ filters
    # ---------------------------------------------------------------

    st.header("5. Parametric EQ Filters")

    st.caption(
        "Filter recommendations derived from octave band "
        "EQ corrections against the selected target: "
        + target_type + ". "
        "Adjacent bands with similar corrections are "
        "collapsed into single filters. "
        "Maximum 10 filters per channel.")

    if peq_filters:
        peq_df = pd.DataFrame(peq_filters)
        peq_df = peq_df.rename(columns={
            'number': 'Filter',
            'type': 'Type',
            'frequency_hz': 'Frequency (Hz)',
            'gain_db': 'Gain (dB)',
            'q': 'Q',
        })
        st.dataframe(peq_df, hide_index=True)
        st.caption(
            "Bell Q=1.0 (single band), Q=0.7 (two bands), "
            "Q=0.5 (three or more bands). "
            "Shelf Q=0.707. "
            "Suitable for Lake, Q-SYS, XTA, "
            "Dolby CP950, Trinnov.")
    else:
        st.info(
            "No significant corrections required — "
            "all octave band corrections are within "
            "0.5 dB of the target.")

    # ---------------------------------------------------------------
    # Section 6: Measurement summary
    # ---------------------------------------------------------------

    st.header("6. Measurement Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate length",
                  str(round(gate_ms_used, 1)) + " ms")
    with col2:
        st.metric("Transition frequency",
                  str(transition_hz) + " Hz")
    with col3:
        st.metric("IR files processed", n_irs)
    with col4:
        valid_rt60 = [
            v for k, v in rt60_bands.items()
            if isinstance(k, int) and v is not None]
        avg_rt60 = (
            np.mean(valid_rt60) if valid_rt60 else 0.0)
        st.metric("Mean RT60",
                  str(round(avg_rt60, 2)) + " s")

    st.caption(
        "Reference position: " + ref_filename
        + " (direct field measurement). "
        "All " + str(n_irs)
        + " file(s) used for reverberant field "
        "spatial average and RT60. "
        "Listener distance: "
        + str(round(listener_distance_m, 1))
        + " m (2/3 room length). "
        "Late energy window: "
        + str(round(late_start_val, 1))
        + " ms. Target: " + target_type + ".")

    # ---------------------------------------------------------------
    # Section 7: Results table
    # ---------------------------------------------------------------

    st.header("7. Results Table")
    st.dataframe(df)

    # ---------------------------------------------------------------
    # Section 8: Downloads
    # ---------------------------------------------------------------

    st.header("8. Downloads")

    st.subheader("Smaart Reference Curve Files")
    st.caption(
        "Import into Smaart via "
        "Options -> Reference Curves -> Import. "
        "All curves anchored to measured direct field "
        "level at 1 kHz.")

    col1, col2, col3, col4, col5 = st.columns(5)

    if eq_target_path.exists():
        with col1:
            st.download_button(
                label="EQ target curve (Smaart)",
                data=eq_target_path.read_bytes(),
                file_name=eq_target_path.name,
                mime="text/plain",
                help=(
                    "Predicted room curve after EQ. "
                    "Use as reference in Smaart."))

    if selected_target_path.exists():
        with col2:
            st.download_button(
                label="Selected target (Smaart)",
                data=selected_target_path.read_bytes(),
                file_name=selected_target_path.name,
                mime="text/plain",
                help=(
                    "Selected monitoring target: "
                    + target_type))

    if xcurve_large_path.exists():
        with col3:
            st.download_button(
                label="X-curve large room (Smaart)",
                data=xcurve_large_path.read_bytes(),
                file_name=xcurve_large_path.name,
                mime="text/plain",
                help=(
                    "SMPTE ST 202M / ISO 2969. "
                    "Flat to 2 kHz, -3 dB/oct above."))

    if xcurve_small_path.exists():
        with col4:
            st.download_button(
                label="X-curve small room (Smaart)",
                data=xcurve_small_path.read_bytes(),
                file_name=xcurve_small_path.name,
                mime="text/plain",
                help=(
                    "SMPTE RP 200. "
                    "Flat to 4 kHz, -3 dB/oct above."))

    csv_path = eq_target_path.parent / (
        channel_name + "_results.csv")
    if csv_path.exists():
        with col5:
            st.download_button(
                label="Results CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                help=(
                    "Full per-band results including "
                    "direct field, reverberant field, "
                    "RT60, DI, EQ corrections, and "
                    "predicted post-EQ response."))

    shutil.rmtree(
        eq_target_path.parent.parent,
        ignore_errors=True)
