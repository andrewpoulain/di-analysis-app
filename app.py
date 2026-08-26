# app.py
import streamlit as st
import yaml
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from di_analysis import (
    load_ir,
    apply_calibration,
    direct_field_at_bands,
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
)

st.set_page_config(
    page_title="Room Analysis and EQ Tool",
    layout="wide")

st.title("Reverberant Field Analysis and EQ Target Derivation")

# ---------------------------------------------------------------------------
# Sidebar: room configuration
# ---------------------------------------------------------------------------

st.sidebar.header("Room Configuration")

room_name = st.sidebar.text_input("Room name", value="Stage A")
volume = st.sidebar.number_input(
    "Room volume (m³)", min_value=10.0, max_value=10000.0, value=850.0)
surface = st.sidebar.number_input(
    "Surface area (m²)", min_value=10.0, max_value=5000.0, value=620.0)
transition_hz = st.sidebar.selectbox(
    "Transition frequency (Hz)", [125, 250, 500], index=1)
n_taps = st.sidebar.selectbox(
    "FIR filter taps", [512, 1024, 2048, 4096], index=1)

st.sidebar.header("Channel Configuration")

channel_name = st.sidebar.text_input("Channel name", value="Left")
gate_ms = st.sidebar.number_input(
    "Gate length (ms, 0 = auto)", min_value=0.0,
    max_value=100.0, value=0.0, step=0.5)
gate_ms = None if gate_ms == 0.0 else gate_ms

hf_shelf_hz = st.sidebar.number_input(
    "HF shelf frequency (Hz)", min_value=4000, max_value=16000,
    value=10000, step=1000)
hf_shelf_db = st.sidebar.number_input(
    "HF shelf level (dB)", min_value=-6.0, max_value=0.0,
    value=0.0, step=0.5)

max_boost = st.sidebar.number_input(
    "Max boost (dB)", min_value=0.0, max_value=12.0,
    value=6.0, step=0.5)
max_cut = st.sidebar.number_input(
    "Max cut (dB)", min_value=0.0, max_value=20.0,
    value=12.0, step=0.5)

# ---------------------------------------------------------------------------
# Main panel: file upload
# ---------------------------------------------------------------------------

st.header("1. Upload Impulse Response Files")

st.info(
    "Upload WAV files exported from Smaart. "
    "The first file is treated as the reference position (mix position). "
    "All remaining files are spatial averaging positions.")

uploaded_files = st.file_uploader(
    "IR WAV files (upload all positions for this channel)",
    type=["wav"],
    accept_multiple_files=True)

cal_file = st.file_uploader(
    "Microphone calibration file for reference position (CSV, optional)",
    type=["csv"])

# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

if uploaded_files and st.button("Run Analysis"):

    with st.spinner("Processing..."):

        # Write uploaded files to a temp directory
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

        # Load IRs
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

        # Build config dicts
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

        # Direct field
        direct_levels, gate_ms_used = direct_field_at_bands(
            ref_ir, ref_fs, gate_ms=gate_ms)

        # RT60
        rt60_bands = rt60_per_band_from_irs(irs, ref_fs)

        # Reverberant field
        reverb_levels = spatial_average_reverberant(irs, ref_fs)

        # DI
        di = estimate_di(direct_levels, reverb_levels,
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

        # Generate plots to files
        plot_analysis(direct_levels, reverb_levels, di, rt60_bands,
                      gate_ms_used, channel_name, str(out_dir))
        plot_eq_and_filter(direct_levels, reverb_levels, all_corr,
                           predicted, fir_freq_response, lf_filter_params,
                           channel_name, str(out_dir))

        # Save filter files
        save_fir_coefficients(fir_coeffs, channel_name, ref_fs,
                              str(out_dir))
        save_iir_parameters(lf_filter_params, channel_name, str(out_dir))

        # Save CSV
        df = save_csv(direct_levels, reverb_levels, di, rt60_bands,
                      all_corr, predicted, channel_name, str(out_dir))

    # ---------------------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------------------

    st.header("2. Analysis Results")

    col1, col2 = st.columns(2)

    analysis_img = out_dir / f"{channel_name}_analysis.png"
    eq_img = out_dir / f"{channel_name}_eq_filter.png"

    if analysis_img.exists():
        with col1:
            st.subheader("Reverberant Field Analysis")
            st.image(str(analysis_img))

    if eq_img.exists():
        with col2:
            st.subheader("EQ Target and Filter")
            st.image(str(eq_img))

    st.header("3. Results Table")
    st.dataframe(df)

    st.header("4. RT60 Summary")
    rt60_display = {str(b): f"{v:.3f} s" if v else "n/a"
                    for b, v in rt60_bands.items()}
    st.json(rt60_display)

    if lf_filter_params:
        st.header("5. LF IIR Filter Parameters")
        st.dataframe(pd.DataFrame(lf_filter_params))

    # ---------------------------------------------------------------------------
    # Downloads
    # ---------------------------------------------------------------------------

    st.header("6. Download Filter Files")

    col1, col2, col3, col4 = st.columns(4)

    fir_txt = out_dir / f"{channel_name}_fir.txt"
    fir_wav = out_dir / f"{channel_name}_fir.wav"
    fir_bin = out_dir / f"{channel_name}_fir.bin"
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

    # Clean up temp directory
    shutil.rmtree(tmp_dir)
