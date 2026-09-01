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
    xcurve_at_third_octave_bands,
    export_target_for_smaart,
    export_xcurve_for_smaart,
    validate_rt60,
    room_constant,
    room_constant_formula_used,
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

# Listener distance derived as 2/3 of room length
listener_distance_m = room_length * (2.0 / 3.0)
st.sidebar.metric(
    "Listener distance (m)",
    f"{listener_distance_m:.1f}",
    help=(
        "Estimated as 2/3 of room length. Used for "
        "DI estimation via classical D/R inversion."))

st.sidebar.subheader("Measurement Settings")

st.sidebar.info(
    "Transition frequency is calculated automatically "
    "from the detected gate length using the 3/T rule.")

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

# X-curve anchored to 1 kHz for export
XCURVE_REF_HZ = 1000

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.header("1. Upload Impulse Response Files")

st.info(
    "Upload WAV files exported from Smaart. "
    "All files are used for spatial averaging of the "
    "reverberant field and RT60 estimation. "
    "Select the reference position file below — this file "
    "is used for the gated direct field measurement and "
    "should be the primary mix position IR.")

uploaded_files = st.file_uploader(
    "IR WAV files (upload all positions for this channel)",
    type=["wav"],
    accept_multiple_files=True)

cal_file = st.file_uploader(
    "Microphone calibration file for reference position "
    "(two-column CSV: frequency_hz, sensitivity_db — optional)",
    type=["csv"])

ref_filename = None
if uploaded_files:
    filenames = sorted([f.name for f in uploaded_files])
    ref_filename = st.selectbox(
        "Reference position file (primary mix position)",
        options=filenames,
        index=0,
        help=(
            "This file is used for the gated direct field "
            "measurement. It should be the IR recorded at "
            "the primary mix position. All other files "
            "contribute to the spatially averaged reverberant "
            "field only."))
    st.caption(
        f"Direct field will be measured from: "
        f"**{ref_filename}**. "
        "All uploaded files contribute to the reverberant "
        "field spatial average and RT60 estimation.")

# ---------------------------------------------------------------------------
# RT60 overrides
# ---------------------------------------------------------------------------

with st.expander("RT60 overrides (optional)", expanded=False):
    st.caption(
        "Enter manual RT60 values in seconds to override "
        "the calculated values for specific bands. "
        "Leave blank to use the calculated value. "
        "Useful when the IR is too short or noisy to "
        "produce a reliable estimate at a specific band.")

    rt60_override_cols = st.columns(len(OCTAVE_CENTRES))
    rt60_overrides = {}
    for i, b in enumerate(OCTAVE_CENTRES):
        with rt60_override_cols[i]:
            val = st.text_input(
                f"{int(b)} Hz",
                value="",
                key=f"rt60_override_{int(b)}",
                placeholder="s")
            if val.strip():
                try:
                    parsed = float(val.strip())
                    if 0.05 <= parsed <= 20.0:
                        rt60_overrides[int(b)] = parsed
                    else:
                        st.warning(
                            f"{int(b)} Hz: value must be "
                            f"between 0.05 and 20.0 s")
                except ValueError:
                    st.warning(
                        f"{int(b)} Hz: invalid number")

# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

if uploaded_files and ref_filename and st.button("Run Analysis"):

    with st.spinner("Processing — this may take a moment "
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
        ref_ir = None
        ref_fs = None

        # Load reference file first
        ref_path = ir_paths[ref_filename]
        ref_ir, ref_fs = load_ir(str(ref_path))
        if cal_path:
            ref_ir = apply_calibration(
                ref_ir, ref_fs, str(cal_path))

        # Load all files for reverberant field averaging
        for name, p in sorted(ir_paths.items()):
            ir, fs = load_ir(str(p))
            irs.append(ir)

        st.success(
            f"Loaded {len(irs)} IR file(s) at {ref_fs} Hz. "
            f"Reference position: {ref_filename}")

        # -----------------------------------------------------------
        # Octave band analysis — EQ path (normalised)
        # -----------------------------------------------------------

        direct_levels, gate_ms_used = direct_field_at_bands(
            ref_ir, ref_fs, gate_ms=gate_ms)

        # Transition frequency using 3/T rule
        transition_hz = transition_frequency_from_gate(
            gate_ms_used, bands=OCTAVE_CENTRES)

        st.info(
            f"Gate detected: {gate_ms_used:.1f} ms — "
            f"transition frequency set to "
            f"{transition_hz} Hz (3/T rule)")

        channel_cfg = {
            'name': channel_name,
            'gate_ms': gate_ms,
            'hf_shelf_hz': hf_shelf_hz,
            'hf_shelf_db': hf_shelf_db,
        }

        # -----------------------------------------------------------
        # RT60 — calculated then apply overrides
        # -----------------------------------------------------------

        rt60_bands = rt60_per_band_from_irs(irs, ref_fs)

        if rt60_overrides:
            override_applied = []
            for b, val in rt60_overrides.items():
                old = rt60_bands.get(b)
                rt60_bands[b] = val
                old_str = (f"{old * 1000:.0f} ms"
                           if old is not None else "None")
                override_applied.append(
                    f"{b} Hz: {old_str} → "
                    f"{val * 1000:.0f} ms (manual override)")
            if override_applied:
                st.info(
                    "RT60 overrides applied: "
                    + "; ".join(override_applied))

        rt60_warnings = validate_rt60(rt60_bands)
        reverb_levels = spatial_average_reverberant(irs, ref_fs)

        # -----------------------------------------------------------
        # DI estimation — absolute energy path
        # Uses all IRs, median Q across positions
        # -----------------------------------------------------------

        di = estimate_di_from_multiple_irs(
            irs, ref_fs,
            rt60_per_band=rt60_bands,
            volume_m3=volume,
            surface_area_m2=surface,
            listener_distance_m=listener_distance_m,
            gate_ms=gate_ms_used,
            late_start_ms=50.0)

        # -----------------------------------------------------------
        # EQ corrections
        # -----------------------------------------------------------

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

        # Predicted room curve — energy sum with no EQ applied
        zero_corr = {int(b): 0.0 for b in OCTAVE_CENTRES}
        predicted_room_curve_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                zero_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

        # Steady-state reconstruction from measured fields
        reconstructed_ss, tol_upper, tol_lower = \
            reconstruct_steady_state(
                direct_levels_3rd,
                reverb_levels_3rd)

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
            return {
                b: v - ref_level_3rd
                for b, v in d.items()
                if not np.isnan(v)}

        direct_3rd_norm = norm(direct_levels_3rd)
        reverb_3rd_norm = norm(reverb_levels_3rd)
        room_curve_norm = norm(predicted_room_curve_3rd)

        # -----------------------------------------------------------
        # X-curve
        # -----------------------------------------------------------

        xcurve_large = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES, screen_size='large')
        xcurve_small = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES, screen_size='small')

        third_oct_sorted = sorted(direct_levels_3rd.keys())
        direct_vals_sorted = [
            direct_levels_3rd[b] for b in third_oct_sorted]
        ref_direct_level = float(np.interp(
            np.log10(float(XCURVE_REF_HZ)),
            np.log10(third_oct_sorted),
            direct_vals_sorted))

        # -----------------------------------------------------------
        # Save outputs
        # -----------------------------------------------------------

        df = save_csv(
            direct_levels, reverb_levels, di, rt60_bands,
            all_corr, channel_name, str(out_dir))

        eq_target_path = (
            out_dir / f"{channel_name}_eq_target.txt")

        after_3rd = predict_post_eq_steady_state_third_octave(
            direct_levels_3rd,
            reverb_levels_3rd,
            all_corr,
            transition_hz=transition_hz,
            half_octave_overlap=True)
        after_3rd_norm = norm(after_3rd)

        export_target_for_smaart(
            target_levels_3rd=after_3rd_norm,
            ref_level_db=ref_level_3rd,
            output_path=eq_target_path,
            label=f'{channel_name} EQ Target — {room_name}')

        xcurve_large_path = (
            out_dir / f"{channel_name}_xcurve_large.txt")
        export_xcurve_for_smaart(
            xcurve_levels_3rd=xcurve_large,
            ref_level_db=ref_direct_level,
            output_path=xcurve_large_path,
            screen_size='large')

        xcurve_small_path = (
            out_dir / f"{channel_name}_xcurve_small.txt")
        export_xcurve_for_smaart(
            xcurve_levels_3rd=xcurve_small,
            ref_level_db=ref_direct_level,
            output_path=xcurve_small_path,
            screen_size='small')

        # Store in session state
        st.session_state.update({
            'direct_3rd_norm': direct_3rd_norm,
            'reverb_3rd_norm': reverb_3rd_norm,
            'room_curve_norm': room_curve_norm,
            'xcurve_large': xcurve_large,
            'xcurve_small': xcurve_small,
            'rt60_bands': rt60_bands,
            'rt60_warnings': rt60_warnings,
            'di': di,
            'gate_ms_used': gate_ms_used,
            'transition_hz': transition_hz,
            'df': df,
            'eq_target_path': str(eq_target_path),
            'xcurve_large_path': str(xcurve_large_path),
            'xcurve_small_path': str(xcurve_small_path),
            'ref_filename': ref_filename,
            'n_irs': len(irs),
            'listener_distance_m': listener_distance_m,
            'volume': volume,
            'surface': surface,
            'analysis_complete': True,
        })

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

if st.session_state.get('analysis_complete'):

    direct_3rd_norm = st.session_state['direct_3rd_norm']
    reverb_3rd_norm = st.session_state['reverb_3rd_norm']
    room_curve_norm = st.session_state['room_curve_norm']
    xcurve_large = st.session_state['xcurve_large']
    xcurve_small = st.session_state['xcurve_small']
    rt60_bands = st.session_state['rt60_bands']
    rt60_warnings = st.session_state['rt60_warnings']
    di = st.session_state['di']
    gate_ms_used = st.session_state['gate_ms_used']
    transition_hz = st.session_state['transition_hz']
    df = st.session_state['df']
    eq_target_path = Path(st.session_state['eq_target_path'])
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

    # ---------------------------------------------------------------
    # RT60 warnings
    # ---------------------------------------------------------------

    if rt60_warnings:
        st.warning("RT60 validation warnings:")
        for band, msg in rt60_warnings.items():
            st.write(f"  **{band}:** {msg}")

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
            line=dict(color=colour, dash=dash, width=width),
            marker=dict(size=4),
            opacity=opacity)

    # ---------------------------------------------------------------
    # Main response plot
    # ---------------------------------------------------------------

    st.header("2. Measured Response and Predicted Room Curve")

    with st.expander("Plot controls", expanded=True):

        st.markdown("**Trace visibility**")

        vis_col1, vis_col2, vis_col3, vis_col4, vis_col5 = \
            st.columns(5)
        with vis_col1:
            show_direct = st.checkbox(
                "Direct field", value=True,
                key='show_direct')
        with vis_col2:
            show_reverb = st.checkbox(
                "Reverberant field", value=True,
                key='show_reverb')
        with vis_col3:
            show_room_curve = st.checkbox(
                "Predicted room curve", value=True,
                key='show_room_curve')
        with vis_col4:
            show_xcurve_large = st.checkbox(
                "X-curve (large room)", value=True,
                key='show_xcurve_large')
        with vis_col5:
            show_xcurve_small = st.checkbox(
                "X-curve (small room)", value=False,
                key='show_xcurve_small')

        st.markdown("**Trace level offsets (dB)**")
        st.caption(
            "Display-only. Does not affect EQ corrections "
            "or exported files.")

        off_col1, off_col2, off_col3 = st.columns(3)
        with off_col1:
            direct_offset = st.slider(
                "Direct field offset (dB)",
                min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5,
                key='direct_offset')
        with off_col2:
            reverb_offset = st.slider(
                "Reverberant field offset (dB)",
                min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5,
                key='reverb_offset')
        with off_col3:
            room_curve_offset = st.slider(
                "Predicted room curve offset (dB)",
                min_value=-20.0, max_value=20.0,
                value=0.0, step=0.5,
                key='room_curve_offset')

    # Apply offsets to display copies
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
        fig_main.add_trace(make_trace(
            direct_display,
            'Direct field (gated, 1/3 oct)'
            + (f'  {direct_offset:+.1f} dB'
               if direct_offset != 0.0 else ''),
            'steelblue',
            width=2))

    if show_reverb:
        fig_main.add_trace(make_trace(
            reverb_display,
            'Reverberant field (spatially averaged)'
            + (f'  {reverb_offset:+.1f} dB'
               if reverb_offset != 0.0 else ''),
            'firebrick',
            dash='dash',
            width=1.5,
            opacity=0.7))

    if show_room_curve:
        fig_main.add_trace(make_trace(
            room_curve_display,
            'Predicted room curve (direct + reverberant, no EQ)'
            + (f'  {room_curve_offset:+.1f} dB'
               if room_curve_offset != 0.0 else ''),
            'grey',
            dash='solid',
            width=2,
            opacity=0.9))

    if show_xcurve_large:
        fig_main.add_trace(make_trace(
            xcurve_large,
            'X-curve (large room, SMPTE ST 202M / ISO 2969)',
            'purple',
            dash='dashdot',
            width=2))

    if show_xcurve_small:
        fig_main.add_trace(make_trace(
            xcurve_small,
            'X-curve (small room, SMPTE RP 200)',
            'darkorchid',
            dash='dot',
            width=2))

    fig_main.update_layout(
        title=(
            f"{channel_name} — {room_name} — "
            f"Measured Response and Predicted Room Curve "
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
            y=-0.35,
            xanchor='left',
            x=0),
        height=560,
        hovermode='x unified')

    st.plotly_chart(fig_main, width='stretch')

    st.caption(
        "**Direct field** — gated IR at reference position "
        f"({ref_filename}), "
        "1/6-octave smoothed above transition frequency. "
        "This is what the loudspeaker produces before the "
        "room acts on it. "
        "**Reverberant field** — spatially averaged Schroeder "
        "decay across all measurement positions. "
        "**Predicted room curve** — energy sum of direct and "
        "reverberant fields with no EQ applied. "
        "This is what a steady-state pink noise measurement "
        "would show. "
        "**X-curve (large room)** — flat to 2 kHz, "
        "−3 dB/octave above, −3 dB/octave below 63 Hz "
        "(SMPTE ST 202M / ISO 2969). "
        "**X-curve (small room)** — flat to 4 kHz, "
        "−3 dB/octave above, −3 dB/octave below 63 Hz "
        "(SMPTE RP 200). "
        "All traces normalised to 0 dB at 1 kHz. "
        "Offsets are display-only.")

    # ---------------------------------------------------------------
    # RT60 and DI plots
    # ---------------------------------------------------------------

    st.header("3. RT60 and Directivity Index")

    # RT60 as a line graph on a log frequency axis
    fig_rt60 = go.Figure()

    rt60_bands_sorted = sorted(
        b for b in rt60_bands
        if isinstance(b, int)
        and rt60_bands[b] is not None)
    rt60_vals = [rt60_bands[b] for b in rt60_bands_sorted]

    fig_rt60.add_trace(go.Scatter(
        x=[float(b) for b in rt60_bands_sorted],
        y=rt60_vals,
        mode='lines+markers',
        name='RT60 (s)',
        line=dict(color='mediumseagreen', width=2),
        marker=dict(size=8)))

    fig_rt60.update_layout(
        title='RT60 per Octave Band',
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[63, 125, 250, 500, 1000,
                      2000, 4000, 8000, 16000],
            ticktext=['63', '125', '250', '500', '1k',
                      '2k', '4k', '8k', '16k']),
        yaxis=dict(
            title='RT60 (s)',
            rangemode='tozero'),
        height=350,
        hovermode='x unified')

    st.plotly_chart(fig_rt60, width='stretch')

    # Room constant formula used per band
    formula_rows = []
    for b in [63, 125, 250, 500, 1000,
              2000, 4000, 8000, 16000]:
        rt60 = rt60_bands.get(b)
        if rt60 is not None:
            formula = room_constant_formula_used(
                rt60, volume, surface)
            formula_rows.append({
                'Band (Hz)': b,
                'RT60 (s)': round(rt60, 3),
                'R formula': formula})

    if formula_rows:
        with st.expander(
                "Room constant formula used per band",
                expanded=False):
            st.dataframe(
                pd.DataFrame(formula_rows),
                hide_index=True)
            st.caption(
                "Sabine is used when mean absorption α ≤ 0.2. "
                "Eyring is used when α > 0.2. "
                "Eyring is more accurate for treated rooms "
                "with high absorption.")

    # DI plot on log frequency axis
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
            f'Estimated Directivity Index DI(f) — '
            f'listener distance {listener_distance_m:.1f} m '
            f'(2/3 room length)'),
        xaxis=dict(
            title='Frequency (Hz)',
            type='log',
            tickvals=[63, 125, 250, 500, 1000,
                      2000, 4000, 8000, 16000],
            ticktext=['63', '125', '250', '500', '1k',
                      '2k', '4k', '8k', '16k']),
        yaxis=dict(
            title='DI (dB)',
            range=[0, 22]),
        height=350,
        hovermode='x unified')

    st.plotly_chart(fig_di, width='stretch')

    st.caption(
        "DI estimated from classical D/R inversion: "
        "Q = (16πr²/R) × (D/R), DI = 10 log₁₀(Q). "
        "Median Q across all measurement positions. "
        "Room constant R derived using Sabine (α ≤ 0.2) "
        "or Eyring (α > 0.2). "
        "DI clipped to [0, 20] dB.")

    # ---------------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------------

    st.header("4. Measurement Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate length", f"{gate_ms_used:.1f} ms")
    with col2:
        st.metric("Transition frequency",
                  f"{transition_hz} Hz")
    with col3:
        st.metric("IR files processed", n_irs)
    with col4:
        valid_rt60 = [
            v for k, v in rt60_bands.items()
            if isinstance(k, int) and v is not None]
        avg_rt60 = np.mean(valid_rt60) if valid_rt60 else 0.0
        st.metric("Mean RT60", f"{avg_rt60:.2f} s")

    st.caption(
        f"Reference position: **{ref_filename}** "
        "(used for direct field measurement). "
        f"All {n_irs} file(s) used for reverberant field "
        "spatial average and RT60 estimation. "
        f"Listener distance: {listener_distance_m:.1f} m "
        "(2/3 of room length).")

    # ---------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------

    st.header("5. Results Table")
    st.dataframe(df)

    # ---------------------------------------------------------------
    # Downloads
    # ---------------------------------------------------------------

    st.header("6. Downloads")

    st.subheader("Smaart Reference Curve Files")
    st.caption(
        "Import these files into Smaart via "
        "Options → Reference Curves → Import. "
        "All curves are anchored to the measured direct field "
        "level at 1 kHz so they will align correctly when "
        "overlaid on a transfer function measurement at the "
        "same gain setting.")

    col1, col2, col3, col4 = st.columns(4)

    if eq_target_path.exists():
        with col1:
            st.download_button(
                label="EQ target curve (Smaart)",
                data=eq_target_path.read_bytes(),
                file_name=eq_target_path.name,
                mime="text/plain",
                help=(
                    "Predicted room curve after EQ applied. "
                    "Use as a reference curve in Smaart to "
                    "guide equalisation."))

    if xcurve_large_path.exists():
        with col2:
            st.download_button(
                label="X-curve large room (Smaart)",
                data=xcurve_large_path.read_bytes(),
                file_name=xcurve_large_path.name,
                mime="text/plain",
                help=(
                    "SMPTE ST 202M / ISO 2969 X-curve. "
                    "Flat to 2 kHz, −3 dB/octave above. "
                    "Anchored to direct field level at 1 kHz."))

    if xcurve_small_path.exists():
        with col3:
            st.download_button(
                label="X-curve small room (Smaart)",
                data=xcurve_small_path.read_bytes(),
                file_name=xcurve_small_path.name,
                mime="text/plain",
                help=(
                    "SMPTE RP 200 modified X-curve. "
                    "Flat to 4 kHz, −3 dB/octave above. "
                    "Anchored to direct field level at 1 kHz."))

    csv_path = eq_target_path.parent / \
        f"{channel_name}_results.csv"
    if csv_path.exists():
        with col4:
            st.download_button(
                label="Results CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                help=(
                    "Full per-band results including direct "
                    "field, reverberant field, RT60, DI, and "
                    "EQ corrections."))

    shutil.rmtree(
        eq_target_path.parent.parent,
        ignore_errors=True)
