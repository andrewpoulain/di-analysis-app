#!/usr/bin/env python3
"""
Reverberant Field Analysis, DI Estimation, and EQ Target Derivation
Processes IRs exported from Smaart (WAV format) to compute:
  - Schroeder decay per octave band per position
  - Spatially averaged reverberant field spectrum
  - Gated direct field spectrum at reference position
  - DI estimate from direct/reverberant difference
  - EQ correction targets per octave band
  - Minimum-phase FIR filter coefficients
  - Output plots, CSV report, and filter files

Usage:
    python di_analysis.py --config room_config.yaml --session session_dir/
"""

import os
import argparse
import yaml
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Octave band definitions
# ---------------------------------------------------------------------------

OCTAVE_CENTRES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])


def octave_band_limits(centre_hz):
    """Return (f_low, f_high) for a 1-octave band centred at centre_hz."""
    return centre_hz / np.sqrt(2), centre_hz * np.sqrt(2)


# ---------------------------------------------------------------------------
# IR loading
# ---------------------------------------------------------------------------

def load_ir(filepath):
    """
    Load a WAV file exported from Smaart.
    Returns (ir_array, sample_rate).
    Converts integer WAV formats to float64 normalised to +/-1.
    """
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
    """
    Apply a microphone calibration curve to an IR.
    cal_file: path to a two-column CSV (frequency_hz, sensitivity_db).
    """
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
    """
    Bandpass filter an IR using a 4th-order Butterworth filter.
    """
    nyq = fs / 2.0
    f_low = max(f_low, 10.0)
    f_high = min(f_high, nyq * 0.99)
    if f_low >= f_high:
        return np.zeros_like(ir)
    sos = sig.butter(4, [f_low / nyq, f_high / nyq],
                     btype='band', output='sos')
    return sig.sosfilt(sos, ir)


def schroeder_decay(ir_band):
    """
    Compute the Schroeder backward integral of a bandpass-filtered IR.
    Returns the decay curve normalised so that the initial value is 0 dB.
    """
    power = ir_band ** 2
    decay = np.cumsum(power[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)
    decay_db = 10.0 * np.log10(decay / decay[0])
    return decay_db


def initial_decay_level(ir, fs, centre_hz):
    """
    Return the initial Schroeder decay level for one octave band.
    Uses the mean level over the first 5 ms to reduce sensitivity
    to the direct arrival peak.
    """
    f_low, f_high = octave_band_limits(centre_hz)
    ir_band = bandpass_ir(ir, fs, f_low, f_high)
    decay_db = schroeder_decay(ir_band)
    n_avg = max(1, int(0.005 * fs))
    return float(np.mean(decay_db[:n_avg]))


def reverberant_spectrum(ir, fs, bands=OCTAVE_CENTRES):
    """
    Return the Schroeder initial decay level for each octave band.
    """
    return {int(b): initial_decay_level(ir, fs, b) for b in bands}


# ---------------------------------------------------------------------------
# RT60 estimation
# ---------------------------------------------------------------------------

def rt60_from_schroeder(ir, fs, centre_hz, eval_range_db=(-5, -25)):
    """
    Estimate RT60 in one octave band from the Schroeder decay curve.
    """
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
    """
    Derive the room constant R from RT60 via Sabine inversion.
    """
    if rt60_s is None or rt60_s <= 0:
        return None
    alpha = 0.161 * volume_m3 / (rt60_s * surface_area_m2)
    alpha = min(alpha, 0.999)
    return surface_area_m2 * alpha / (1.0 - alpha)


def rt60_per_band_from_irs(ir_list, fs, bands=OCTAVE_CENTRES):
    """
    Estimate RT60 per octave band by averaging across all IR positions.
    Returns dict {centre_hz: RT60_seconds}.
    """
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
    """
    Find the sample index of the direct arrival.
    """
    power_db = 20.0 * np.log10(
        np.abs(ir) / (np.max(np.abs(ir)) + 1e-30))
    candidates = np.where(power_db >= threshold_db)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def detect_first_reflection(ir, fs, direct_idx, min_gap_ms=2.0):
    """
    Estimate the arrival time of the first significant reflection.
    """
    min_gap = int(min_gap_ms * fs / 1000.0)
    search_start = direct_idx + min_gap
    direct_level = np.abs(ir[direct_idx])
    threshold = direct_level * 10.0 ** (-20.0 / 20.0)
    search = np.abs(ir[search_start:])
    peaks, _ = sig.find_peaks(search, height=threshold,
                               distance=min_gap)
    if len(peaks) == 0:
        return None
    return int(search_start + peaks[0])


def gated_direct_field(ir, fs, gate_ms=None):
    """
    Extract the gated direct field magnitude response.
    Returns freqs, magnitude (dB, normalised), gate_ms_used.
    """
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

    n_fft = int(2 ** np.ceil(np.log2(max(len(ir_gated), 16))))
    spectrum = np.fft.rfft(ir_gated, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitude = 20.0 * np.log10(np.abs(spectrum) + 1e-30)
    magnitude -= np.max(magnitude)

    return freqs, magnitude, gate_ms_used


def direct_field_at_bands(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
    """
    Return the mean direct field level in each octave band (dB, relative).
    """
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
    """
    Compute the spatially averaged reverberant field spectrum.
    Averaging is performed in the power domain.
    """
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
    """
    Estimate DI(f) per octave band.
    DI(f) = Direct(f) - Reverberant(f) + 10*log10(4 / R(f))
    """
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
# EQ target derivation
# ---------------------------------------------------------------------------

def derive_direct_field_target(direct_levels, bands=OCTAVE_CENTRES,
                                ref_band=1000,
                                hf_shelf_hz=10000,
                                hf_shelf_db=0.0):
    """
    Derive the direct field EQ correction per octave band.
    Target is flat (0 dB) relative to the reference band.
    Positive correction values mean boost, negative mean cut.
    """
    ref = direct_levels.get(ref_band, 0.0) or 0.0
    corrections = {}
    for b in [int(b) for b in bands]:
        level = direct_levels.get(b, np.nan)
        if np.isnan(level):
            corrections[b] = np.nan
            continue
        normalised = level - ref
        correction = -normalised
        if b >= hf_shelf_hz and hf_shelf_db != 0.0:
            correction += hf_shelf_db
        corrections[b] = round(correction, 2)
    return corrections


def apply_correction_constraints(corrections, direct_levels,
                                  reverberant_levels,
                                  max_boost_db=6.0,
                                  max_cut_db=12.0,
                                  min_band_hz=250):
    """
    Apply engineering constraints to the raw correction values.
    """
    constrained = {}
    for b, corr in corrections.items():
        if np.isnan(corr):
            constrained[b] = np.nan
            continue
        if b < min_band_hz:
            constrained[b] = 0.0
            continue
        corr = min(corr, max_boost_db)
        corr = max(corr, -max_cut_db)
        r = reverberant_levels.get(b, np.nan)
        if not np.isnan(r) and corr > 0:
            bands_list = sorted(reverberant_levels.keys())
            if b in bands_list:
                idx = bands_list.index(b)
                if 0 < idx < len(bands_list) - 1:
                    r_below = reverberant_levels.get(
                        bands_list[idx - 1], r)
                    r_above = reverberant_levels.get(
                        bands_list[idx + 1], r)
                    r_neighbours = (r_below + r_above) / 2.0
                    if r > r_neighbours + 2.0:
                        corr = min(corr, 0.0)
        constrained[b] = round(corr, 2)
    return constrained


def lf_correction_from_spatial_average(spatial_avg_levels,
                                        transition_hz=250,
                                        ref_band=250,
                                        max_correction_db=6.0):
    """
    Derive broad LF corrections from the spatially averaged
    steady-state below the transition frequency.
    """
    bands = sorted(k for k in spatial_avg_levels
                   if k <= transition_hz)
    if not bands:
        return {}
    ref = spatial_avg_levels.get(
        ref_band, spatial_avg_levels.get(bands[-1], 0.0))
    corrections = {}
    for b in bands:
        level = spatial_avg_levels.get(b, np.nan)
        if np.isnan(level):
            corrections[b] = 0.0
            continue
        corr = -(level - ref)
        corr = max(min(corr, max_correction_db), -max_correction_db)
        corrections[b] = round(corr, 2)
    return corrections


def predict_post_eq_steady_state(direct_levels, reverberant_levels,
                                  corrections,
                                  bands=OCTAVE_CENTRES):
    """
    Predict the steady-state response after applying the EQ correction.
    """
    predicted = {}
    for b in [int(b) for b in bands]:
        d = direct_levels.get(b, np.nan)
        r = reverberant_levels.get(b, np.nan)
        c = corrections.get(b, 0.0)
        if np.isnan(d) or np.isnan(r) or np.isnan(c):
            predicted[b] = np.nan
            continue
        d_eq = d + c
        r_eq = r + c
        ss = 10.0 * np.log10(
            10.0 ** (d_eq / 10.0) + 10.0 ** (r_eq / 10.0))
        predicted[b] = round(ss, 2)
    return predicted


def derive_full_eq_target(direct_levels, reverberant_levels,
                           spatial_avg_levels, rt60_per_band,
                           room_cfg, channel_cfg,
                           transition_hz=250):
    """
    Full EQ target derivation for one channel.
    Returns hf_corrections, lf_corrections, all_corrections,
    and predicted_curve.
    """
    hf_shelf_db = channel_cfg.get('hf_shelf_db', 0.0)
    hf_shelf_hz = channel_cfg.get('hf_shelf_hz', 10000)

    raw_corrections = derive_direct_field_target(
        direct_levels,
        hf_shelf_hz=hf_shelf_hz,
        hf_shelf_db=hf_shelf_db)

    hf_corrections = apply_correction_constraints(
        raw_corrections, direct_levels, reverberant_levels)

    lf_corrections = lf_correction_from_spatial_average(
        spatial_avg_levels, transition_hz=transition_hz)

    all_corrections = {**lf_corrections, **hf_corrections}

    predicted = predict_post_eq_steady_state(
        direct_levels, reverberant_levels, all_corrections)

    return hf_corrections, lf_corrections, all_corrections, predicted


# ---------------------------------------------------------------------------
# FIR filter design
# ---------------------------------------------------------------------------

def interpolate_correction_to_freqs(corrections, target_freqs,
                                     bands=OCTAVE_CENTRES):
    """
    Interpolate octave band corrections to a full frequency array
    using log-frequency interpolation.
    """
    bands_int = [int(b) for b in bands]
    valid = [(b, corrections[b]) for b in bands_int
             if b in corrections
             and not np.isnan(corrections.get(b, np.nan))]
    if not valid:
        return np.zeros(len(target_freqs))
    freq_pts = np.array([v[0] for v in valid], dtype=float)
    corr_pts = np.array([v[1] for v in valid], dtype=float)
    log_freq_pts = np.log10(freq_pts)
    log_target = np.log10(np.maximum(target_freqs, 1.0))
    return np.interp(log_target, log_freq_pts, corr_pts,
                     left=corr_pts[0], right=corr_pts[-1])


def minimum_phase_from_magnitude(magnitude_db, n_fft):
    """
    Derive a minimum-phase frequency response from a magnitude spectrum
    using the cepstral method.
    """
    mag_full = np.concatenate([magnitude_db,
                                magnitude_db[-2:0:-1]])
    mag_linear = 10.0 ** (mag_full / 20.0)
    log_mag = np.log(np.maximum(mag_linear, 1e-30))
    cepstrum = np.fft.ifft(log_mag).real
    win = np.zeros(n_fft)
    win[0] = 1.0
    if n_fft % 2 == 0:
        win[1:n_fft // 2] = 2.0
        win[n_fft // 2] = 1.0
    else:
        win[1:(n_fft + 1) // 2] = 2.0
    min_phase_log = np.fft.fft(cepstrum * win)
    min_phase_response = np.exp(min_phase_log)
    return min_phase_response[:n_fft // 2 + 1]


def design_fir_filter(corrections, fs, n_taps=1024,
                       transition_hz=250,
                       regularisation_db=0.5,
                       bands=OCTAVE_CENTRES):
    """
    Design a minimum-phase FIR correction filter from octave band
    corrections.
    Returns fir_coeffs and (freqs, magnitude_db) of the filter response.
    """
    n_fft = int(2 ** np.ceil(np.log2(n_taps))) * 4
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    corr_full = interpolate_correction_to_freqs(
        corrections, freqs, bands)
    corr_full[freqs < transition_hz] = 0.0
    corr_full = np.clip(corr_full, -40.0, regularisation_db + 6.0)
    mp_response = minimum_phase_from_magnitude(corr_full, n_fft)
    full_response = np.concatenate([mp_response,
                                     np.conj(mp_response[-2:0:-1])])
    h_full = np.fft.ifft(full_response).real
    h_truncated = h_full[:n_taps]
    kaiser_win = np.kaiser(n_taps, beta=8.0)
    fir_coeffs = h_truncated * kaiser_win
    w, h = sig.freqz(fir_coeffs, worN=n_fft // 2, fs=fs)
    filter_magnitude_db = 20.0 * np.log10(np.abs(h) + 1e-30)
    return fir_coeffs, (w, filter_magnitude_db)


def design_lf_iir_filters(lf_corrections, fs,
                            transition_hz=250,
                            bands=OCTAVE_CENTRES):
    """
    Design parametric IIR biquad EQ stages for LF corrections.
    Returns sos array and list of filter parameter dicts.
    """
    sos_stages = []
    filter_params = []
    lf_bands = sorted(
        b for b in lf_corrections
        if b <= transition_hz
        and not np.isnan(lf_corrections.get(b, np.nan))
        and abs(lf_corrections.get(b, 0.0)) > 0.1)
    for b in lf_bands:
        gain_db = lf_corrections[b]
        centre_hz = float(b)
        Q = 1.0 / np.sqrt(2)
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * centre_hz / fs
        alpha = np.sin(w0) / (2.0 * Q)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A
        sos_stages.append([b0 / a0, b1 / a0, b2 / a0,
                           1.0, a1 / a0, a2 / a0])
        filter_params.append({
            'type': 'peaking',
            'centre_hz': centre_hz,
            'gain_db': round(gain_db, 2),
            'Q': round(Q, 3),
        })
    sos = np.array(sos_stages) if sos_stages else None
    return sos, filter_params


def save_fir_coefficients(fir_coeffs, channel_name, fs, output_dir):
    """
    Save FIR coefficients as text, binary float32, and WAV.
    """
    out_base = Path(output_dir) / channel_name
    txt_path = str(out_base) + '_fir.txt'
    np.savetxt(txt_path, fir_coeffs, fmt='%.10f')
    bin_path = str(out_base) + '_fir.bin'
    fir_coeffs.astype(np.float32).tofile(bin_path)
    wav_path = str(out_base) + '_fir.wav'
    peak = np.max(np.abs(fir_coeffs))
    if peak > 0:
        normalised = (fir_coeffs / peak * 0.99).astype(np.float32)
    else:
        normalised = fir_coeffs.astype(np.float32)
    wavfile.write(wav_path, fs, normalised)


def save_iir_parameters(filter_params, channel_name, output_dir):
    """
    Save IIR biquad parameters as CSV.
    """
    if not filter_params:
        return
    df = pd.DataFrame(filter_params)
    out_path = Path(output_dir) / f"{channel_name}_iir_params.csv"
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_analysis(direct_levels, reverberant_levels, di_estimates,
                  rt60_per_band, gate_ms_used, channel_name,
                  output_dir, bands=OCTAVE_CENTRES):
    """
    Four-panel analysis plot.
    """
    bands_int = [int(b) for b in bands]
    labels = [str(b) for b in bands_int]
    x = np.arange(len(bands_int))

    ref = direct_levels.get(1000, 0.0) or 0.0
    r_ref = reverberant_levels.get(1000, 0.0) or 0.0
    direct_norm = [direct_levels.get(b, np.nan) - ref
                   for b in bands_int]
    reverb_norm = [reverberant_levels.get(b, np.nan) - r_ref
                   for b in bands_int]
    di_vals = [di_estimates.get(b, np.nan) for b in bands_int]
    rt60_vals = [rt60_per_band.get(b) or np.nan for b in bands_int]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Reverberant Field Analysis — {channel_name}",
                 fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, direct_norm, 'o-', color='steelblue',
             label=f'Direct field (gate {gate_ms_used:.1f} ms)')
    ax1.plot(x, reverb_norm, 's--', color='firebrick',
             label='Reverberant field (spatially averaged)')
    ax1.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_xlabel('Octave band (Hz)')
    ax1.set_ylabel('Level (dB, norm at 1 kHz)')
    ax1.set_title('Direct vs Reverberant Field')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    valid = [(i, v) for i, v in enumerate(di_vals)
             if not np.isnan(v)]
    if valid:
        xi, yi = zip(*valid)
        ax2.plot(list(xi), list(yi), 'D-', color='darkorange')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_xlabel('Octave band (Hz)')
    ax2.set_ylabel('DI (dB)')
    ax2.set_title('Estimated Directivity Index DI(f)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x, rt60_vals, color='mediumseagreen', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45)
    ax3.set_xlabel('Octave band (Hz)')
    ax3.set_ylabel('RT60 (s)')
    ax3.set_title('RT60 per Octave Band')
    ax3.grid(True, alpha=0.3, axis='y')

    diff = [d - r for d, r in zip(direct_norm, reverb_norm)]
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, diff, '^-', color='mediumpurple')
    ax4.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45)
    ax4.set_xlabel('Octave band (Hz)')
    ax4.set_ylabel('Difference (dB)')
    ax4.set_title('Direct minus Reverberant (uncorrected)')
    ax4.grid(True, alpha=0.3)

    out_path = Path(output_dir) / f"{channel_name}_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_eq_and_filter(direct_levels, reverberant_levels,
                        all_corrections, predicted_curve,
                        fir_freq_response, lf_filter_params,
                        channel_name, output_dir,
                        bands=OCTAVE_CENTRES):
    """
    Four-panel EQ and filter plot.
    """
    bands_int = [int(b) for b in bands]
    labels = [str(b) for b in bands_int]
    x = np.arange(len(bands_int))

    ref = direct_levels.get(1000, 0.0) or 0.0
    r_ref = reverberant_levels.get(1000, 0.0) or 0.0
    pred_ref = predicted_curve.get(1000, 0.0) or 0.0

    direct_norm = [direct_levels.get(b, np.nan) - ref
                   for b in bands_int]
    reverb_norm = [reverberant_levels.get(b, np.nan) - r_ref
                   for b in bands_int]
    corr_vals = [all_corrections.get(b, 0.0) for b in bands_int]
    pred_norm = [predicted_curve.get(b, np.nan) - pred_ref
                 for b in bands_int]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"EQ Target and Filter Design — {channel_name}",
                 fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, direct_norm, 'o-', color='steelblue',
             label='Direct field')
    ax1.plot(x, reverb_norm, 's--', color='firebrick',
             label='Reverberant field')
    ax1.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_xlabel('Octave band (Hz)')
    ax1.set_ylabel('Level (dB, norm at 1 kHz)')
    ax1.set_title('Before EQ')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    colours = ['tomato' if v < 0 else 'steelblue'
               for v in corr_vals]
    ax2.bar(x, corr_vals, color=colours, alpha=0.7)
    ax2.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_xlabel('Octave band (Hz)')
    ax2.set_ylabel('Correction (dB)')
    ax2.set_title('Derived EQ Corrections')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[1, 0])
    if fir_freq_response is not None:
        fir_freqs, fir_mag = fir_freq_response
        mask = fir_freqs > 20
        ax3.semilogx(fir_freqs[mask], fir_mag[mask],
                     color='darkorange', linewidth=1.5,
                     label='FIR (HF correction)')
    if lf_filter_params:
        lf_label_done = False
        for p in lf_filter_params:
            label = 'IIR biquads (LF)' if not lf_label_done else None
            ax3.axvline(p['centre_hz'], color='mediumseagreen',
                        alpha=0.5, linestyle=':', linewidth=1.0,
                        label=label)
            lf_label_done = True
    ax3.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax3.set_xlim(20, 20000)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Filter magnitude (dB)')
    ax3.set_title('Filter Frequency Response')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, pred_norm, 'D-', color='darkorange',
             label='Predicted steady-state after EQ')
    ax4.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45)
    ax4.set_xlabel('Octave band (Hz)')
    ax4.set_ylabel('Level (dB, norm at 1 kHz)')
    ax4.set_title('Predicted Steady-State After EQ')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    out_path = Path(output_dir) / f"{channel_name}_eq_filter.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def save_csv(direct_levels, reverberant_levels, di_estimates,
             rt60_per_band, all_corrections, predicted_curve,
             channel_name, output_dir, bands=OCTAVE_CENTRES):
    bands_int = [int(b) for b in bands]
    rows = []
    for b in bands_int:
        rows.append({
            'channel': channel_name,
            'band_hz': b,
            'direct_field_db': round(
                direct_levels.get(b, np.nan), 2),
            'reverberant_field_db': round(
                reverberant_levels.get(b, np.nan), 2),
            'di_estimate_db': round(
                di_estimates.get(b, np.nan), 2),
            'rt60_s': round(
                rt60_per_band.get(b) or np.nan, 3),
            'eq_correction_db': round(
                all_corrections.get(b, 0.0), 2),
            'predicted_steady_state_db': round(
                predicted_curve.get(b, np.nan), 2),
        })
    df = pd.DataFrame(rows)
    out_path = Path(output_dir) / f"{channel_name}_results.csv"
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Main session processor
# ---------------------------------------------------------------------------

def process_channel(channel_cfg, room_cfg, session_dir, output_dir):
    """
    Full processing pipeline for one speaker channel.
    """
    channel_name = channel_cfg['name']
    volume = room_cfg['volume_m3']
    surface = room_cfg['surface_area_m2']
    transition_hz = room_cfg.get('transition_hz', 250)
    n_taps = room_cfg.get('fir_taps', 1024)

    ir_files = sorted(Path(session_dir).glob(
        f"{channel_cfg['ir_prefix']}*.wav"))
    if not ir_files:
        return None

    cal_map = {c['channel']: c.get('cal_file')
               for c in room_cfg.get('microphone_calibration', [])}

    irs = []
    ref_ir = None
    ref_fs = None

    for i, f in enumerate(ir_files):
        ir, fs = load_ir(str(f))
        ir = apply_calibration(ir, fs, cal_map.get(i + 1))
        irs.append(ir)
        if i == 0:
            ref_ir = ir
            ref_fs = fs

    gate_ms = channel_cfg.get('gate_ms', None)
    direct_levels, gate_ms_used = direct_field_at_bands(
        ref_ir, ref_fs, gate_ms=gate_ms)
    rt60_bands = rt60_per_band_from_irs(irs, ref_fs)
    reverb_levels = spatial_average_reverberant(irs, ref_fs)
    di = estimate_di(direct_levels, reverb_levels,
                     rt60_bands, volume, surface)
    hf_corr, lf_corr, all_corr, predicted = derive_full_eq_target(
        direct_levels, reverb_levels, reverb_levels,
        rt60_bands, room_cfg, channel_cfg,
        transition_hz=transition_hz)
    fir_coeffs, fir_freq_response = design_fir_filter(
        hf_corr, ref_fs, n_taps=n_taps,
        transition_hz=transition_hz)
    sos, lf_filter_params = design_lf_iir_filters(
        lf_corr, ref_fs, transition_hz=transition_hz)

    plot_analysis(direct_levels, reverb_levels, di, rt60_bands,
                  gate_ms_used, channel_name, output_dir)
    plot_eq_and_filter(direct_levels, reverb_levels, all_corr,
                       predicted, fir_freq_response, lf_filter_params,
                       channel_name, output_dir)
    save_fir_coefficients(fir_coeffs, channel_name, ref_fs, output_dir)
    save_iir_parameters(lf_filter_params, channel_name, output_dir)
    df = save_csv(direct_levels, reverb_levels, di, rt60_bands,
                  all_corr, predicted, channel_name, output_dir)
    return df


def run_session(config_path, session_dir, output_dir):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    room_cfg = config['room']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []
    for ch in config['channels']:
        df = process_channel(ch, room_cfg, session_dir, output_dir)
        if df is not None:
            all_results.append(df)
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = Path(output_dir) / 'session_summary.csv'
        summary.to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reverberant field analysis, DI estimation, '
                    'and EQ filter design')
    parser.add_argument('--config', required=True,
                        help='Path to room_config.yaml')
    parser.add_argument('--session', required=True,
                        help='Directory containing exported IR WAV files')
    parser.add_argument('--output', default='output',
                        help='Directory for outputs')
    args = parser.parse_args()
    run_session(args.config, args.session, args.output)
