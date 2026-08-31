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

# X-curve is always anchored to the measured direct field
# level at 1 kHz for export.
XCURVE_REF_HZ = 1000

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

        # Predicted room curve — energy sum with no EQ applied
        zero_corr = {int(b): 0.0 for b in OCTAVE_CENTRES}
        predicted_room_curve_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                zero_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

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

        xcurve_raw = xcurve_at_third_octave_bands(
            bands=THIRD_OCTAVE_CENTRES,
            screen_size=xcurve_size)

        third_oct_sorted = sorted(direct_levels_3rd.keys())
        direct_vals_sorted = [
            direct_levels_3rd[b] for b in third_oct_sorted]
        ref_direct_level = float(np.interp(
            np.log10(float(XCURVE_REF_HZ)),
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

        # Store results in session state so the plot controls
        # below can redraw without re-running the analysis
        st.session_state['direct_3rd_norm'] = direct_3rd_norm
        st.session_state['reverb_3rd_norm'] = reverb_3rd_norm
        st.session_state['room_curve_norm'] = room_curve_norm
        st.session_state['xcurve_display'] = xcurve_display
        st.session_state['rt60_bands'] = rt60_bands
        st.session_state['rt60_warnings'] = rt60_warnings
        st.session_state['di'] = di
        st.session_state['gate_ms_used'] = gate_ms_used
        st.session_state['transition_hz'] = transition_hz
        st.session_state['df'] = df
        st.session_state['eq_target_path'] = eq_target_path
        st.session_state['xcurve_export_path'] = \
            xcurve_export_path
        st.session_state['analysis_complete'] = True

# ---------------------------------------------------------------------------
# Display — runs whenever session state is populated, including
# after the sliders below are adjusted without re-running analysis
# ---------------------------------------------------------------------------

if st.session_state.get('analysis_complete'):

    direct_3rd_norm = st.session_state['direct_3rd_norm']
    reverb_3rd_norm = st.session_state['reverb_3rd_norm']
    room_curve_norm = st.session_state['room_curve_norm']
    xcurve_display = st.session_state['xcurve_display']
    rt60_bands = st.session_state['rt60_bands']
    rt60_warnings = st.session_state['rt60_warnings']
    di = st.session_state['di']
    gate_ms_used = st.session_state['gate_ms_used']
    transition_hz = st.session_state['transition_hz']
    df = st.session_state['df']
    eq_target_path = st.session_state['eq_target_path']
    xcurve_export_path = st.session_state['xcurve_export_path']

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

    # Trace offset controls — placed directly above the chart
    # so the engineer can adjust without scrolling to the sidebar
    with st.expander("Trace level offsets", expanded=False):
        st.caption(
            "Shift individual traces up or down to align "
            "with an external reference or to compare "
            "channels measured at different gain settings. "
            "Offsets are display-only and do not affect "
            "the EQ corrections or exported files.")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            direct_offset = st.slider(
                "Direct field offset (dB)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                key='direct_offset')

        with col_b:
            reverb_offset = st.slider(
                "Reverberant field offset (dB)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                key='reverb_offset')

        with col_c:
            room_curve_offset = st.slider(
                "Predicted room curve offset (dB)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                key='room_curve_offset')

    # Apply offsets to display copies — originals in session
    # state are unchanged so exports are not affected
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

    fig_main.add_trace(make_trace(
        direct_display,
        'Direct field (gated, 1/3 oct)'
        + (f'  {direct_offset:+.1f} dB'
           if direct_offset != 0.0 else ''),
        'steelblue',
        width=2))

    fig_main.add_trace(make_trace(
        reverb_display,
        'Reverberant field (spatially averaged)'
        + (f'  {reverb_offset:+.1f} dB'
           if reverb_offset != 0.0 else ''),
        'firebrick',
        dash='dash',
        width=1.5,
        opacity=0.7))

    fig_main.add_trace(make_trace(
        room_curve_display,
        'Predicted room curve (direct + reverberant, no EQ)'
        + (f'  {room_curve_offset:+.1f} dB'
           if room_curve_offset != 0.0 else ''),
        'grey',
        dash='solid',
        width=2,
        opacity=0.9))

    if show_xcurve:
        xcurve_label = (
            f'X-curve '
            f'({"large" if xcurve_size == "large" else "small"}'
            f' room, SMPTE ST 202M / ISO 2969)')
        fig_main.add_trace(make_trace(
            xcurve_display,
            xcurve_label,
            'purple',
            dash='dashdot',
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
        "**Direct field** — gated IR at reference position, "
        "1/6-octave smoothed above transition frequency. "
        "This is what the loudspeaker produces before the "
        "room acts on it. "
        "**Reverberant field** — spatially averaged Schroeder "
        "decay across all measurement positions. "
        "**Predicted room curve** — energy sum of direct and "
        "reverberant fields with no EQ applied. "
        "This is what a steady-state pink noise measurement "
        "would show. "
        "All traces normalised to 0 dB at 1 kHz. "
        "Offsets applied via the controls above are "
        "display-only.")

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
    # RT60 and DI plots
    # ---------------------------------------------------------------

    st.header("3. RT60 and Directivity Index")

    fig_rt_di = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'RT60 per Octave Band',
            'Estimated Directivity Index DI(f)'))

    rt60_bands_sorted = sorted(
        b for b in rt60_bands
        if isinstance(b, int)
        and rt60_bands[b] is not None)
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

    st.header("4. Measurement Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gate length", f"{gate_ms_used:.1f} ms")
    with col2:
        st.metric("Transition frequency",
                  f"{transition_hz} Hz")
    with col3:
        st.metric(
            "IR files processed",
            len(st.session_state.get('direct_3rd_norm', {})))
    with col4:
        valid_rt60 = [
            v for k, v in rt60_bands.items()
            if isinstance(k, int) and v is not None]
        avg_rt60 = np.mean(valid_rt60) if valid_rt60 else 0.0
        st.metric("Mean RT60", f"{avg_rt60:.2f} s")

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
        "The curves are anchored to the measured direct field "
        "level at 1 kHz so they will align correctly when "
        "overlaid on a transfer function measurement at the "
        "same gain setting.")

    col1, col2, col3 = st.columns(3)

    if Path(eq_target_path).exists():
        with col1:
            st.download_button(
                label="EQ target curve (Smaart)",
                data=Path(eq_target_path).read_bytes(),
                file_name=Path(eq_target_path).name,
                mime="text/plain",
                help=(
                    "Predicted room curve after EQ applied. "
                    "Use as a reference curve in Smaart to "
                    "guide equalisation."))

    if Path(xcurve_export_path).exists():
        with col2:
            st.download_button(
                label=(
                    f"X-curve "
                    f"({'large' if xcurve_size == 'large' else 'small'}"
                    f" room, Smaart)"),
                data=Path(xcurve_export_path).read_bytes(),
                file_name=Path(xcurve_export_path).name,
                mime="text/plain",
                help=(
                    "SMPTE ST 202M / ISO 2969 X-curve aligned "
                    "to the measured direct field level at "
                    "1 kHz."))

    csv_path = Path(eq_target_path).parent / \
        f"{channel_name}_results.csv"
    if csv_path.exists():
        with col3:
            st.download_button(
                label="Results CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                help=(
                    "Full per-band results including direct "
                    "field, reverberant field, RT60, DI, and "
                    "EQ corrections."))
