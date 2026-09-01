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
                            + " Hz: 0.05–20.0 s only")
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

        # Track overrides for source reporting
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

        # Predicted room curve — no EQ
        zero_corr = {int(b): 0.0 for b in OCTAVE_CENTRES}
        predicted_room_curve_3rd = \
            predict_post_eq_steady_state_third_octave(
                direct_levels_3rd,
                reverb_levels_3rd,
                zero_corr,
                transition_hz=transition_hz,
                half_octave_overlap=True)

        # Predicted room curve — after EQ
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
