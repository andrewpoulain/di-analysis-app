#!/usr/bin/env python3
"""
Reverberant Field Analysis and EQ Target Derivation
Processes IRs exported from Smaart (WAV format) to compute:
  - Schroeder decay per octave band per position
  - Spatially averaged reverberant field spectrum
  - Gated direct field spectrum at reference position
  - Spatially averaged direct field across all positions
  - DI estimate from classical D/R inversion
  - EQ correction targets per octave band
  - Steady-state reconstruction from measured fields
  - Parametric EQ filter recommendations
  - Output plots and CSV report

Usage:
    python di_analysis.py --config room_config.yaml
                          --session session_dir/
"""

import os
import argparse
import yaml
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import scipy.ndimage
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Octave band definitions
# ---------------------------------------------------------------------------

OCTAVE_CENTRES = np.array([
    63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])


def octave_band_limits(centre_hz):
    """Return (f_low, f_high) for a 1-octave band."""
    return centre_hz / np.sqrt(2), centre_hz * np.sqrt(2)


# ---------------------------------------------------------------------------
# Third octave band definitions
# ---------------------------------------------------------------------------

THIRD_OCTAVE_CENTRES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 16000
])


def third_octave_band_limits(centre_hz):
    """Return (f_low, f_high) for a 1/3-octave band."""
    return (centre_hz / (2 ** (1.0 / 6)),
            centre_hz * (2 ** (1.0 / 6)))


# ---------------------------------------------------------------------------
# Truncation margin scaled by frequency
# ---------------------------------------------------------------------------

def truncation_margin_for_band(centre_hz):
    """
    Return the noise floor truncation margin in dB.

      Below 500 Hz:    10 dB
      500 Hz to 4 kHz: 12 dB
      Above 4 kHz:     15 dB
    """
    if centre_hz >= 4000:
        return 15.0
    elif centre_hz >= 500:
        return 12.0
    else:
        return 10.0


# ---------------------------------------------------------------------------
# IR loading
# ---------------------------------------------------------------------------

def load_ir(filepath):
    """
    Load a WAV file exported from Smaart.
    Returns (ir_array, sample_rate).
    Converts integer WAV formats to float64
    normalised to +/-1.
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
    cal_file: two-column CSV
    (frequency_hz, sensitivity_db).
    """
    if cal_file is None or not os.path.exists(cal_file):
        return ir
    cal = np.loadtxt(cal_file, delimiter=',', skiprows=1)
    freqs = cal[:, 0]
    sens_db = cal[:, 1]
    n = len(ir)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    interp_sens = np.interp(
        fft_freqs, freqs, sens_db,
        left=sens_db[0], right=sens_db[-1])
    correction = 10.0 ** (-interp_sens / 20.0)
    spectrum = np.fft.rfft(ir)
    return np.fft.irfft(spectrum * correction, n=n)


# ---------------------------------------------------------------------------
# Direct arrival detection using ETC analysis
# ---------------------------------------------------------------------------

def compute_etc(ir, fs, smooth_ms=1.0):
    """
    Compute the Energy Time Curve (ETC).
    Returns ETC in dB normalised to 0 dB at peak.
    """
    analytic = sig.hilbert(ir)
    envelope_sq = np.abs(analytic) ** 2
    window = max(1, int(smooth_ms * fs / 1000.0))
    smoothed = np.convolve(
        envelope_sq,
        np.ones(window) / window,
        mode='same')
    smoothed = np.maximum(smoothed, 1e-30)
    return 10.0 * np.log10(smoothed / np.max(smoothed))


def detect_direct_arrival(ir, fs, threshold_db=-20):
    """
    Find the sample index of the direct arrival using ETC.
    """
    etc_db = compute_etc(ir, fs, smooth_ms=0.5)
    candidates = np.where(etc_db >= threshold_db)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def detect_first_reflection(ir, fs, direct_idx,
                             min_gap_ms=2.0):
    """
    Estimate the first significant reflection using ETC.
    Returns sample index or None.
    """
    etc_db = compute_etc(ir, fs, smooth_ms=0.5)
    min_gap = int(min_gap_ms * fs / 1000.0)
    search_start = direct_idx + min_gap
    threshold_db = etc_db[direct_idx] - 20.0
    search = etc_db[search_start:]
    peaks, _ = sig.find_peaks(
        search,
        height=threshold_db,
        distance=min_gap)
    if len(peaks) == 0:
        return None
    return int(search_start + peaks[0])


# ---------------------------------------------------------------------------
# Noise floor truncation
# ---------------------------------------------------------------------------

def truncate_to_noise_floor(ir, margin_db=10.0):
    """
    Truncate an IR at margin_db above the noise floor.
    """
    peak = np.max(np.abs(ir))
    if peak == 0:
        return ir
    window_samples = max(1, len(ir) // 200)
    smoothed = scipy.ndimage.maximum_filter1d(
        np.abs(ir), size=window_samples)
    smoothed_db = 20.0 * np.log10(
        smoothed / peak + 1e-30)
    tail_start = int(len(smoothed_db) * 0.9)
    noise_floor_db = float(
        np.mean(smoothed_db[tail_start:]))
    threshold_db = noise_floor_db + margin_db
    above = np.where(smoothed_db >= threshold_db)[0]
    if len(above) == 0:
        return ir[:len(ir) // 2]
    truncated = ir.copy()
    truncated[above[-1]:] = 0.0
    return truncated


# ---------------------------------------------------------------------------
# Bandpass filtering
# ---------------------------------------------------------------------------

def bandpass_ir(ir, fs, f_low, f_high, order=4):
    """
    Zero-phase Butterworth bandpass filter.
    order=2 for 63 Hz and below to prevent ringing.
    """
    nyq = fs / 2.0
    f_low = max(f_low, 10.0)
    f_high = min(f_high, nyq * 0.99)
    if f_low >= f_high:
        return np.zeros_like(ir)
    sos = sig.butter(
        order, [f_low / nyq, f_high / nyq],
        btype='band', output='sos')
    return sig.sosfiltfilt(sos, ir)


# ---------------------------------------------------------------------------
# Schroeder integration
# ---------------------------------------------------------------------------

def schroeder_decay(ir_band):
    """
    Schroeder backward integral with noise subtraction.
    Returns decay curve normalised to 0 dB at start.
    """
    power = ir_band ** 2
    tail_start = int(len(power) * 0.9)
    noise_power = float(np.mean(power[tail_start:]))
    power_comp = np.maximum(power - noise_power, 0.0)
    decay = np.cumsum(power_comp[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)
    return 10.0 * np.log10(decay / decay[0])


def initial_decay_level(ir, fs, centre_hz):
    """
    Schroeder initial decay level for one octave band.
    Used for reverberant field display only.
    Not used for DI estimation.
    """
    f_low, f_high = octave_band_limits(centre_hz)
    order = 2 if centre_hz <= 63 else 4
    ir_band = bandpass_ir(
        ir, fs, f_low, f_high, order=order)
    direct_idx = detect_direct_arrival(
        ir_band, fs, threshold_db=-20)
    ir_band = truncate_to_noise_floor(
        ir_band[direct_idx:],
        margin_db=truncation_margin_for_band(centre_hz))
    decay_db = schroeder_decay(ir_band)
    n_avg = max(1, int(0.005 * fs))
    return float(np.mean(decay_db[:n_avg]))


def reverberant_spectrum(ir, fs, bands=OCTAVE_CENTRES):
    """
    Schroeder initial decay level per octave band.
    Returns dict {centre_hz: level_db}.
    Used for spectral shape display only.
    """
    return {int(b): initial_decay_level(ir, fs, b)
            for b in bands}


# ---------------------------------------------------------------------------
# Mixing time and late energy boundary
# ---------------------------------------------------------------------------

def mixing_time_ms(volume_m3):
    """
    Estimate mixing time in ms from room volume.

    Empirical approximation:
      t_mix = 0.0033 * V^(1/3)  seconds

    The constant 0.0033 is the mixing time constant.
    Do NOT confuse with the Polack RT60 constant 0.0117
    which predicts RT60, not mixing time.

    Verified reference values:
      V =  500 m3  (small cinema)    t_mix ~  26 ms
      V = 1500 m3  (medium cinema)   t_mix ~  38 ms
      V = 4000 m3  (large stage)     t_mix ~  52 ms
      V = 8000 m3  (very large)      t_mix ~  66 ms

    Verification:
      0.0033 * 500^(1/3)  = 0.0033 * 7.937  = 26.2 ms
      0.0033 * 1500^(1/3) = 0.0033 * 11.447 = 37.8 ms
      0.0033 * 4000^(1/3) = 0.0033 * 15.874 = 52.4 ms
      0.0033 * 8000^(1/3) = 0.0033 * 20.000 = 66.0 ms

    Returns mixing time in milliseconds.
    """
    if volume_m3 <= 0:
        return 50.0
    return float(
        0.0033 * (volume_m3 ** (1.0 / 3.0)) * 1000.0)


def late_start_ms(volume_m3, floor_ms=50.0):
    """
    Return the late energy start time in milliseconds.

    Derived as max(floor_ms, mixing_time_ms(volume_m3)).

    The 50 ms floor applies for rooms smaller than
    approximately 3500 m3 where the mixing time estimate
    falls below 50 ms.

    For larger rooms the mixing time estimate exceeds
    50 ms and governs the late start, avoiding inclusion
    of strong early reflections in the reverberant
    energy estimate.
    """
    return float(max(floor_ms, mixing_time_ms(volume_m3)))


# ---------------------------------------------------------------------------
# True direct and late energy measurement
# ---------------------------------------------------------------------------

def direct_reverb_energy(ir, fs, centre_hz,
                          gate_ms,
                          late_start_ms_val=50.0):
    """
    True direct and reverberant energies in one octave
    band.

    Direct energy:
      From direct arrival to gate end.

    Reverberant (late) energy:
      From late_start_ms_val onward.

    late_start_ms_val should be derived from
    late_start_ms() using the room volume so that the
    late energy window begins after the mixing time.

    Returns direct_db, reverb_db (absolute, not
    normalised). Preserves D/R ratio for DI inversion.

    NOTE: Completely separate from the EQ path.
      EQ path:  normalised gated spectrum
      DI path:  absolute energy ratios
    """
    f_low, f_high = octave_band_limits(centre_hz)
    order = 2 if centre_hz <= 63 else 4
    ir_band = bandpass_ir(
        ir, fs, f_low, f_high, order=order)
    direct_idx = detect_direct_arrival(ir_band, fs)
    gate_samples = int(gate_ms * fs / 1000.0)
    direct_end = min(
        len(ir_band), direct_idx + gate_samples)
    late_start = min(
        len(ir_band),
        int(late_start_ms_val * fs / 1000.0))
    direct_energy = np.sum(
        ir_band[direct_idx:direct_end] ** 2)
    late_energy = np.sum(ir_band[late_start:] ** 2)
    direct_db = (10.0 * np.log10(direct_energy)
                 if direct_energy > 0 else np.nan)
    reverb_db = (10.0 * np.log10(late_energy)
                 if late_energy > 0 else np.nan)
    return direct_db, reverb_db


def direct_reverb_energy_all_bands(ir, fs, gate_ms,
                                    late_start_ms_val=50.0,
                                    bands=OCTAVE_CENTRES):
    """
    Direct and late energy for all octave bands.

    late_start_ms_val should be derived from
    late_start_ms() using the room volume.

    Returns two dicts: direct_db, reverb_db keyed by
    band.
    """
    direct_db = {}
    reverb_db = {}
    for b in bands:
        d, r = direct_reverb_energy(
            ir, fs, int(b), gate_ms,
            late_start_ms_val=late_start_ms_val)
        direct_db[int(b)] = d
        reverb_db[int(b)] = r
    return direct_db, reverb_db


# ---------------------------------------------------------------------------
# RT60 estimation
# ---------------------------------------------------------------------------

def rt60_from_schroeder(ir, fs, centre_hz):
    """
    RT60 from Schroeder decay in one octave band.

    Evaluation order is frequency-dependent:

      Below 8 kHz:
        T20 (-5 to -25 dB) primary
        T30 (-5 to -35 dB) fallback
        EDT (0 to -10 dB)  last resort

      8 kHz and above:
        EDT (0 to -10 dB)  primary
        T20                fallback
        T30                last resort

    Filter order reduced to 2 at 63 Hz and below.
    Minimum slope threshold relaxed at high frequencies.

    Returns RT60 in seconds or None if unreliable.
    """
    f_low, f_high = octave_band_limits(centre_hz)
    order = 2 if centre_hz <= 63 else 4
    nyq = fs / 2.0
    fl = max(f_low, 10.0)
    fh = min(f_high, nyq * 0.99)
    if fl >= fh:
        return None
    sos = sig.butter(
        order, [fl / nyq, fh / nyq],
        btype='band', output='sos')
    ir_band = sig.sosfiltfilt(sos, ir)
    if np.max(np.abs(ir_band)) < 1e-10:
        return None
    direct_idx = detect_direct_arrival(
        ir_band, fs, threshold_db=-20)
    ir_band = ir_band[direct_idx:]
    if len(ir_band) < int(0.05 * fs):
        return None
    ir_band = truncate_to_noise_floor(
        ir_band,
        margin_db=truncation_margin_for_band(centre_hz))
    if np.max(np.abs(ir_band)) < 1e-10:
        return None
    decay_db = schroeder_decay(ir_band)
    times = np.arange(len(decay_db)) / fs
    if centre_hz >= 8000:
        eval_ranges = [
            (0,   -10),
            (-5,  -25),
            (-5,  -35),
        ]
        min_slope = -0.1
    else:
        eval_ranges = [
            (-5,  -25),
            (-5,  -35),
            (0,   -10),
        ]
        min_slope = -0.5
    for lo, hi in eval_ranges:
        mask = (decay_db <= lo) & (decay_db >= hi)
        if mask.sum() < 10:
            continue
        t_region = times[mask]
        d_region = decay_db[mask]
        if d_region[-1] >= d_region[0]:
            continue
        slope = np.polyfit(t_region, d_region, 1)[0]
        if slope >= min_slope:
            continue
        rt60 = -60.0 / slope
        if 0.05 <= rt60 <= 20.0:
            return float(rt60)
    return None


# ---------------------------------------------------------------------------
# HF RT60 fallback
# ---------------------------------------------------------------------------

def apply_rt60_hf_fallback(rt60_dict):
    """
    Apply high-frequency RT60 fallback logic.

    Rule 1: If RT60 at 8 kHz < 50% of RT60 at 4 kHz,
      replace 8 kHz and 16 kHz with the 4 kHz value.
      Rule 2 is then skipped.

    Rule 2 (only if Rule 1 did not trigger):
      If RT60 at 16 kHz < 50% of RT60 at 8 kHz,
      replace 16 kHz with the 8 kHz value.

    Returns corrected dict and list of warning strings.
    """
    corrected = dict(rt60_dict)
    warnings = []

    rt60_4k = corrected.get(4000)
    rt60_8k = corrected.get(8000)

    if (rt60_4k is not None
            and rt60_8k is not None
            and rt60_4k > 0
            and (rt60_8k / rt60_4k) < 0.50):
        warnings.append(
            '8 kHz RT60 ('
            + str(round(rt60_8k * 1000))
            + ' ms) < 50% of 4 kHz RT60 ('
            + str(round(rt60_4k * 1000))
            + ' ms) — substituting 4 kHz value '
            'at 8 kHz and 16 kHz')
        corrected[8000] = rt60_4k
        corrected[16000] = rt60_4k
        return corrected, warnings

    rt60_8k_c = corrected.get(8000)
    rt60_16k_c = corrected.get(16000)

    if (rt60_8k_c is not None
            and rt60_16k_c is not None
            and rt60_8k_c > 0
            and (rt60_16k_c / rt60_8k_c) < 0.50):
        warnings.append(
            '16 kHz RT60 ('
            + str(round(rt60_16k_c * 1000))
            + ' ms) < 50% of 8 kHz RT60 ('
            + str(round(rt60_8k_c * 1000))
            + ' ms) — substituting 8 kHz value '
            'at 16 kHz')
        corrected[16000] = rt60_8k_c

    return corrected, warnings


# ---------------------------------------------------------------------------
# RT60 averaging across positions
# ---------------------------------------------------------------------------

def rt60_per_band_from_irs(ir_list, fs,
                            bands=OCTAVE_CENTRES):
    """
    RT60 per octave band averaged across all IR positions.
    Applies HF fallback after averaging.

    Returns dict with integer band keys plus:
      '_hf_warnings': list of substitution warning strings
      '_measured':    dict of raw pre-fallback values
    """
    rt60_all = {int(b): [] for b in bands}
    for ir in ir_list:
        for b in bands:
            rt = rt60_from_schroeder(ir, fs, b)
            if rt is not None:
                rt60_all[int(b)].append(rt)

    averaged = {b: float(np.mean(v)) if v else None
                for b, v in rt60_all.items()}

    measured = dict(averaged)

    corrected, hf_warnings = \
        apply_rt60_hf_fallback(averaged)
    corrected['_hf_warnings'] = hf_warnings
    corrected['_measured'] = measured

    return corrected


# ---------------------------------------------------------------------------
# RT60 validation
# ---------------------------------------------------------------------------

def validate_rt60(rt60_per_band, bands=OCTAVE_CENTRES):
    """
    Check RT60 values for physical plausibility.
    Returns dict of warnings keyed by band.
    Includes HF fallback substitution warnings.
    """
    warnings = {}
    bands_int = [int(b) for b in bands]

    for i, w in enumerate(
            rt60_per_band.get('_hf_warnings', [])):
        warnings['hf_fallback_' + str(i)] = w

    valid = {b: rt60_per_band.get(b)
             for b in bands_int
             if rt60_per_band.get(b) is not None}

    if not valid:
        warnings['general'] = (
            'No valid RT60 estimates produced')
        return warnings

    for b, rt in valid.items():
        if rt > 15.0:
            warnings[b] = (
                str(round(rt, 2))
                + ' s is implausibly long')
        if rt < 0.05:
            warnings[b] = (
                str(round(rt, 3))
                + ' s is implausibly short')

    mf_bands = [
        b for b in [500, 1000, 2000] if b in valid]
    hf_bands = [
        b for b in [4000, 8000] if b in valid]
    if mf_bands and hf_bands:
        mf_avg = np.mean([valid[b] for b in mf_bands])
        hf_avg = np.mean([valid[b] for b in hf_bands])
        if hf_avg > mf_avg * 1.5:
            warnings['hf_rising'] = (
                'HF RT60 ('
                + str(round(hf_avg, 2))
                + ' s) > MF RT60 ('
                + str(round(mf_avg, 2))
                + ' s) — check IR length and '
                'noise floor')

    return warnings


# ---------------------------------------------------------------------------
# Room constant — Sabine / Eyring hybrid
# ---------------------------------------------------------------------------

def room_constant(rt60_s, volume_m3, surface_area_m2):
    """
    Derive the room constant R from RT60.

    Uses Sabine inversion when alpha <= 0.2 and Eyring
    inversion when alpha > 0.2.

    Sabine:
      alpha_s = 0.161 * V / (T * S)
      R       = S * alpha_s / (1 - alpha_s)

    Eyring:
      alpha_e = 1 - exp(-0.161 * V / (T * S))
      R       = S * alpha_e / (1 - alpha_e)

    Eyring is more accurate for alpha > 0.2 which is
    common in modern mix rooms with significant acoustic
    treatment. The threshold of 0.2 is the conventional
    crossover point in room acoustics literature.

    Returns R in m2, or None if inputs are invalid.
    """
    if rt60_s is None or rt60_s <= 0:
        return None
    if surface_area_m2 <= 0 or volume_m3 <= 0:
        return None
    alpha_s = min(
        0.161 * volume_m3 / (rt60_s * surface_area_m2),
        0.999)
    if alpha_s <= 0.2:
        alpha = alpha_s
    else:
        alpha = min(
            1.0 - np.exp(
                -0.161 * volume_m3
                / (rt60_s * surface_area_m2)),
            0.999)
    return float(
        surface_area_m2 * alpha / (1.0 - alpha))


def room_constant_formula_used(rt60_s, volume_m3,
                                surface_area_m2):
    """
    Return 'Sabine', 'Eyring', or 'invalid'.
    """
    if rt60_s is None or rt60_s <= 0:
        return 'invalid'
    if surface_area_m2 <= 0 or volume_m3 <= 0:
        return 'invalid'
    alpha_s = min(
        0.161 * volume_m3 / (rt60_s * surface_area_m2),
        0.999)
    return 'Sabine' if alpha_s <= 0.2 else 'Eyring'


# ---------------------------------------------------------------------------
# DI estimation — classical D/R inversion
# ---------------------------------------------------------------------------

def estimate_di(direct_energy_db,
                reverb_energy_db,
                rt60_per_band,
                volume_m3,
                surface_area_m2,
                listener_distance_m,
                bands=OCTAVE_CENTRES):
    """
    DI per octave band via classical D/R inversion.

      D/R = Q * R / (16 * pi * r^2)
      Q   = (16 * pi * r^2 / R) * (D/R)
      DI  = 10 * log10(Q)

    DI clipped to [0, 20] dB.
    Returns dict {centre_hz: DI_dB}.
    """
    di = {}
    for b in bands:
        b = int(b)
        d_db = direct_energy_db.get(b, np.nan)
        r_db = reverb_energy_db.get(b, np.nan)
        R = room_constant(
            rt60_per_band.get(b),
            volume_m3, surface_area_m2)
        if (np.isnan(d_db) or np.isnan(r_db)
                or R is None or R <= 0
                or listener_distance_m <= 0):
            di[b] = np.nan
            continue
        D_over_R = 10.0 ** ((d_db - r_db) / 10.0)
        Q = (16.0 * np.pi * listener_distance_m ** 2
             / R) * D_over_R
        di[b] = (
            float(np.clip(
                10.0 * np.log10(Q), 0.0, 20.0))
            if Q > 0 else np.nan)
    return di


def estimate_di_from_multiple_irs(
        ir_list, fs,
        rt60_per_band,
        volume_m3,
        surface_area_m2,
        listener_distance_m,
        gate_ms,
        late_start_ms_val=None,
        bands=OCTAVE_CENTRES):
    """
    DI per octave band using median Q across all IR
    positions.

    late_start_ms_val: if None, derived automatically
    from volume_m3 using late_start_ms(). Pass an
    explicit value to override.

    For each position the D/R ratio is computed from
    true direct and late energy. Q is derived per
    position per band. The median Q across positions
    is used to compute the final DI.

    Using the median is more robust against outliers
    from boundary positions and reflection hot spots.

    DI clipped to [0, 20] dB.
    Returns dict {centre_hz: DI_dB}.
    """
    if late_start_ms_val is None:
        late_start_ms_val = late_start_ms(volume_m3)

    q_per_band = {int(b): [] for b in bands}

    for ir in ir_list:
        for b in bands:
            b_int = int(b)
            d_db, r_db = direct_reverb_energy(
                ir, fs, b_int, gate_ms,
                late_start_ms_val=late_start_ms_val)
            R = room_constant(
                rt60_per_band.get(b_int),
                volume_m3, surface_area_m2)
            if (np.isnan(d_db) or np.isnan(r_db)
                    or R is None or R <= 0
                    or listener_distance_m <= 0):
                continue
            D_over_R = 10.0 ** ((d_db - r_db) / 10.0)
            Q = (16.0 * np.pi
                 * listener_distance_m ** 2
                 / R) * D_over_R
            if Q > 0:
                q_per_band[b_int].append(Q)

    di = {}
    for b in bands:
        b_int = int(b)
        q_vals = q_per_band[b_int]
        if not q_vals:
            di[b_int] = np.nan
            continue
        q_median = float(np.median(q_vals))
        di[b_int] = (
            float(np.clip(
                10.0 * np.log10(q_median), 0.0, 20.0))
            if q_median > 0 else np.nan)

    return di


# ---------------------------------------------------------------------------
# Gated direct field — EQ path (normalised)
# ---------------------------------------------------------------------------

def gated_direct_field(ir, fs, gate_ms=None):
    """
    Gated direct field magnitude response for EQ use.

    Uses ETC-based reflection detection to set the gate
    length. Transition frequency rule is 3/T.

    Returns freqs, magnitude (normalised to 0 dB peak),
    gate_ms_used.

    NOTE: The normalisation here (magnitude -= max) is
    correct for the EQ path only. Do NOT use for DI
    estimation. Use direct_reverb_energy() for DI which
    preserves absolute energy ratios.

    The two paths are completely separate:
      EQ path:  normalised gated spectrum
      DI path:  absolute energy ratios
    """
    direct_idx = detect_direct_arrival(ir, fs)
    reflection_idx = detect_first_reflection(
        ir, fs, direct_idx)

    if gate_ms is None:
        if reflection_idx is not None:
            gap_samples = reflection_idx - direct_idx
            gate_samples = int(0.9 * gap_samples)
        else:
            gate_samples = int(0.020 * fs)
    else:
        gate_samples = int(gate_ms * fs / 1000.0)

    gate_samples = max(gate_samples, 16)
    gate_ms_used = gate_samples / fs * 1000.0

    ir_gated = ir[
        direct_idx: direct_idx + gate_samples].copy()
    window = np.hanning(
        2 * len(ir_gated))[:len(ir_gated)]
    ir_gated *= window

    n_fft = int(2 ** np.ceil(
        np.log2(max(len(ir_gated), 16))))
    spectrum = np.fft.rfft(ir_gated, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitude = 20.0 * np.log10(
        np.abs(spectrum) + 1e-30)
    magnitude -= np.max(magnitude)

    return freqs, magnitude, gate_ms_used


def transition_frequency_from_gate(gate_ms_used,
                                    bands=OCTAVE_CENTRES):
    """
    Transition frequency from gate length using 3/T rule.

    Uses 3/T rather than 2/T. More conservative —
    reduces the risk of including early reflections in
    the direct field estimate near the transition.

    Result snapped to nearest octave band at or above
    the calculated frequency.
    """
    gate_s = gate_ms_used / 1000.0
    if gate_s <= 0:
        return 250
    f_transition = max(3.0 / gate_s, 125.0)
    candidates = [b for b in bands if b >= f_transition]
    return (int(min(candidates))
            if candidates else int(bands[-1]))


def direct_field_at_bands(ir, fs, gate_ms=None,
                           bands=OCTAVE_CENTRES):
    """
    Mean direct field level per octave band.
    dB, normalised — EQ path only.
    Returns levels dict and gate_ms_used.
    """
    freqs, magnitude, gate_ms_used = \
        gated_direct_field(ir, fs, gate_ms)
    levels = {}
    for b in bands:
        f_low, f_high = octave_band_limits(b)
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.sum() > 0:
            power = 10.0 ** (magnitude[mask] / 10.0)
            levels[int(b)] = float(
                10.0 * np.log10(np.mean(power)))
        else:
            levels[int(b)] = np.nan
    return levels, gate_ms_used


def direct_field_at_third_octave_bands(
        ir, fs, gate_ms=None,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Mean direct field level per 1/3-octave band.
    dB, normalised — EQ path only.
    Returns levels dict and gate_ms_used.
    """
    freqs, magnitude, gate_ms_used = \
        gated_direct_field(ir, fs, gate_ms)
    levels = {}
    for b in bands:
        f_low, f_high = third_octave_band_limits(b)
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.sum() > 0:
            power = 10.0 ** (magnitude[mask] / 10.0)
            levels[float(b)] = float(
                10.0 * np.log10(np.mean(power)))
        else:
            levels[float(b)] = np.nan
    return levels, gate_ms_used


# ---------------------------------------------------------------------------
# Spatially averaged direct field
# ---------------------------------------------------------------------------

def spatial_average_direct_field(
        ir_list, fs,
        gate_ms=None,
        bands=OCTAVE_CENTRES):
    """
    Spatially averaged direct field in octave bands.

    Computes the gated direct field magnitude spectrum
    for each IR position and averages in the power
    domain (zero-phase power averaging).

    Power averaging method:
      avg_power = mean(10 ** (level_db / 10))
      avg_db    = 10 * log10(avg_power)

    Explicitly NOT used:
      Complex averaging
      Coherence-weighted averaging
      Vector averaging
      Magnitude averaging in dB

    The result represents the spatially averaged direct
    field response across all measurement positions.
    This is used as the basis for EQ generation rather
    than the single reference position response.

    Returns dict {centre_hz: avg_level_db},
    gate_ms_used (from first IR).
    """
    all_levels = []
    gate_ms_used_ref = None

    for idx, ir in enumerate(ir_list):
        freqs, magnitude, gate_ms_used = \
            gated_direct_field(ir, fs, gate_ms=gate_ms)
        if idx == 0:
            gate_ms_used_ref = gate_ms_used
        levels = {}
        for b in bands:
            f_low, f_high = octave_band_limits(b)
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() > 0:
                power = 10.0 ** (magnitude[mask] / 10.0)
                levels[int(b)] = float(
                    10.0 * np.log10(np.mean(power)))
            else:
                levels[int(b)] = np.nan
        all_levels.append(levels)

    averaged = {}
    for b in bands:
        b_int = int(b)
        vals = [
            l[b_int] for l in all_levels
            if not np.isnan(l.get(b_int, np.nan))]
        if vals:
            powers = [10.0 ** (v / 10.0) for v in vals]
            averaged[b_int] = float(
                10.0 * np.log10(np.mean(powers)))
        else:
            averaged[b_int] = np.nan

    return averaged, gate_ms_used_ref


def spatial_average_direct_field_third_octave(
        ir_list, fs,
        gate_ms=None,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Spatially averaged direct field in 1/3-octave bands.

    Same power averaging method as
    spatial_average_direct_field() but at 1/3-octave
    resolution for display purposes.

    Returns dict {centre_hz: avg_level_db},
    gate_ms_used (from first IR).
    """
    all_levels = []
    gate_ms_used_ref = None

    for idx, ir in enumerate(ir_list):
        freqs, magnitude, gate_ms_used = \
            gated_direct_field(ir, fs, gate_ms=gate_ms)
        if idx == 0:
            gate_ms_used_ref = gate_ms_used
        levels = {}
        for b in bands:
            f_low, f_high = third_octave_band_limits(b)
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() > 0:
                power = 10.0 ** (magnitude[mask] / 10.0)
                levels[float(b)] = float(
                    10.0 * np.log10(np.mean(power)))
            else:
                levels[float(b)] = np.nan
        all_levels.append(levels)

    averaged = {}
    for b in bands:
        b_float = float(b)
        vals = [
            l[b_float] for l in all_levels
            if not np.isnan(l.get(b_float, np.nan))]
        if vals:
            powers = [10.0 ** (v / 10.0) for v in vals]
            averaged[b_float] = float(
                10.0 * np.log10(np.mean(powers)))
        else:
            averaged[b_float] = np.nan

    return averaged, gate_ms_used_ref


# ---------------------------------------------------------------------------
# Spatial averaging of reverberant field
# ---------------------------------------------------------------------------

def spatial_average_reverberant(ir_list, fs,
                                  bands=OCTAVE_CENTRES):
    """
    Spatially averaged reverberant field in octave bands.
    Power domain averaging.
    Used for spectral shape display only — not for DI.
    """
    all_spectra = [
        reverberant_spectrum(ir, fs, bands)
        for ir in ir_list]
    averaged = {}
    for b in bands:
        b = int(b)
        powers = [
            10.0 ** (s[b] / 10.0)
            for s in all_spectra]
        averaged[b] = float(
            10.0 * np.log10(np.mean(powers)))
    return averaged


def reverberant_spectrum_third_octave(
        ir, fs, bands=THIRD_OCTAVE_CENTRES):
    """
    Schroeder initial decay level per 1/3-octave band.
    Filter order reduced to 2 below 80 Hz.
    """
    result = {}
    for b in bands:
        f_low, f_high = third_octave_band_limits(b)
        order = 2 if b <= 80 else 4
        ir_band = bandpass_ir(
            ir, fs, f_low, f_high, order=order)
        direct_idx = detect_direct_arrival(
            ir_band, fs, threshold_db=-20)
        ir_band = truncate_to_noise_floor(
            ir_band[direct_idx:],
            margin_db=truncation_margin_for_band(b))
        decay_db = schroeder_decay(ir_band)
        n_avg = max(1, int(0.005 * fs))
        result[float(b)] = float(
            np.mean(decay_db[:n_avg]))
    return result


def spatial_average_reverberant_third_octave(
        ir_list, fs,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Spatially averaged reverberant field in 1/3-octave
    bands. Power domain averaging.
    """
    all_spectra = [
        reverberant_spectrum_third_octave(ir, fs, bands)
        for ir in ir_list]
    averaged = {}
    for b in bands:
        b = float(b)
        powers = [
            10.0 ** (s[b] / 10.0)
            for s in all_spectra]
        averaged[b] = float(
            10.0 * np.log10(np.mean(powers)))
    return averaged


# ---------------------------------------------------------------------------
# Target response generators
# ---------------------------------------------------------------------------

TARGET_TYPES = [
    'Flat Direct Field',
    'X-Curve (Large Room)',
    'X-Curve (Small Room)',
    'SMPTE ST 422',
]


def flat_target(bands=THIRD_OCTAVE_CENTRES):
    """
    Flat target: 0 dB at all bands.
    Returns dict {centre_hz: 0.0}.
    """
    return {float(b): 0.0 for b in bands}


def xcurve_target(freqs_hz, screen_size='large'):
    """
    X-curve target levels.

    Large room (SMPTE ST 202M / ISO 2969):
      Flat to 2 kHz
      -3 dB/octave above 2 kHz
      -3 dB/octave below 63 Hz

    Small room (SMPTE RP 200):
      Flat to 4 kHz
      -3 dB/octave above 4 kHz
      -3 dB/octave below 63 Hz
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    target = np.zeros_like(freqs)
    hf_corner = (
        4000.0 if screen_size == 'small' else 2000.0)
    lf_corner = 63.0
    for i, f in enumerate(freqs):
        if f <= 0:
            target[i] = np.nan
            continue
        level = 0.0
        if f > hf_corner:
            level -= 3.0 * np.log2(f / hf_corner)
        if f < lf_corner:
            level -= 3.0 * np.log2(lf_corner / f)
        target[i] = level
    return target


def xcurve_at_third_octave_bands(
        bands=THIRD_OCTAVE_CENTRES,
        screen_size='large'):
    """
    X-curve target at 1/3-octave band centres.
    Returns dict {centre_hz: target_db}.
    """
    levels = xcurve_target(
        np.array(bands, dtype=float),
        screen_size=screen_size)
    return {float(b): float(l)
            for b, l in zip(bands, levels)
            if not np.isnan(l)}


def smpte_422_target(freqs_hz):
    """
    SMPTE ST 422 target levels.

    Flat to 2 kHz.
    -1.5 dB/octave above 2 kHz.
    No LF tilt.

    Returns array of target levels in dB.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    target = np.zeros_like(freqs)
    hf_corner = 2000.0
    for i, f in enumerate(freqs):
        if f <= 0:
            target[i] = np.nan
            continue
        level = 0.0
        if f > hf_corner:
            level -= 1.5 * np.log2(f / hf_corner)
        target[i] = level
    return target


def smpte_422_at_third_octave_bands(
        bands=THIRD_OCTAVE_CENTRES):
    """
    SMPTE ST 422 target at 1/3-octave band centres.
    Returns dict {centre_hz: target_db}.
    """
    levels = smpte_422_target(
        np.array(bands, dtype=float))
    return {float(b): float(l)
            for b, l in zip(bands, levels)
            if not np.isnan(l)}


def get_target_levels(target_type,
                      bands=THIRD_OCTAVE_CENTRES):
    """
    Return target levels dict for the given target_type.

    target_type must be one of TARGET_TYPES:
      'Flat Direct Field'
      'X-Curve (Large Room)'
      'X-Curve (Small Room)'
      'SMPTE ST 422'
    """
    if target_type == 'X-Curve (Large Room)':
        return xcurve_at_third_octave_bands(
            bands, screen_size='large')
    elif target_type == 'X-Curve (Small Room)':
        return xcurve_at_third_octave_bands(
            bands, screen_size='small')
    elif target_type == 'SMPTE ST 422':
        return smpte_422_at_third_octave_bands(bands)
    else:
        return flat_target(bands)


def get_target_levels_octave(target_type,
                              bands=OCTAVE_CENTRES):
    """
    Return target levels dict at octave band resolution.
    """
    return get_target_levels(target_type, bands=bands)


# ---------------------------------------------------------------------------
# EQ target derivation — target-aware
# ---------------------------------------------------------------------------

def derive_direct_field_target(
        direct_levels,
        bands=OCTAVE_CENTRES,
        ref_band=1000,
        hf_shelf_hz=10000,
        hf_shelf_db=0.0,
        target_type='Flat Direct Field'):
    """
    Derive direct field EQ correction per octave band
    relative to the selected target.

    direct_levels should be the spatially averaged direct
    field, not the single reference position response.

    Correction = target_level - normalised_measured

    For Flat Direct Field this is identical to the
    previous flat-target behaviour. For other targets
    the correction is adjusted so the post-EQ response
    matches the selected target shape.
    """
    ref = direct_levels.get(ref_band, 0.0) or 0.0
    target_oct = get_target_levels_octave(
        target_type, bands=bands)
    corrections = {}
    for b in [int(b) for b in bands]:
        level = direct_levels.get(b, np.nan)
        if np.isnan(level):
            corrections[b] = np.nan
            continue
        normalised = level - ref
        target_level = target_oct.get(float(b), 0.0)
        correction = target_level - normalised
        if b >= hf_shelf_hz and hf_shelf_db != 0.0:
            correction += hf_shelf_db
        corrections[b] = round(correction, 2)
    return corrections


def apply_correction_constraints(
        corrections,
        direct_levels,
        reverberant_levels,
        max_boost_db=6.0,
        max_cut_db=12.0,
        min_band_hz=250):
    """
    Apply engineering constraints to raw corrections.

    Constraints:
      - Bands below min_band_hz set to zero (LF handled
        separately by spatial average method)
      - Clipped to [-max_cut_db, +max_boost_db]
      - Boost suppressed where reverberant level is
        elevated relative to neighbours by > 2 dB
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
            bands_list = sorted(
                reverberant_levels.keys())
            if b in bands_list:
                idx = bands_list.index(b)
                if 0 < idx < len(bands_list) - 1:
                    r_below = reverberant_levels.get(
                        bands_list[idx - 1], r)
                    r_above = reverberant_levels.get(
                        bands_list[idx + 1], r)
                    if r > (r_below + r_above) / 2.0 + 2.0:
                        corr = min(corr, 0.0)
        constrained[b] = round(corr, 2)
    return constrained


def lf_correction_from_spatial_average(
        spatial_avg_levels,
        transition_hz=250,
        ref_band=250,
        max_correction_db=6.0):
    """
    Broad LF corrections from spatially averaged
    steady-state below the transition frequency.
    """
    bands = sorted(
        k for k in spatial_avg_levels
        if k <= transition_hz)
    if not bands:
        return {}
    ref = spatial_avg_levels.get(
        ref_band,
        spatial_avg_levels.get(bands[-1], 0.0))
    corrections = {}
    for b in bands:
        level = spatial_avg_levels.get(b, np.nan)
        if np.isnan(level):
            corrections[b] = 0.0
            continue
        corr = -(level - ref)
        corr = max(min(corr, max_correction_db),
                   -max_correction_db)
        corrections[b] = round(corr, 2)
    return corrections


def derive_full_eq_target(
        direct_levels,
        reverberant_levels,
        spatial_avg_levels,
        channel_cfg,
        transition_hz=250,
        target_type='Flat Direct Field'):
    """
    Full EQ target derivation for one channel.

    direct_levels should be the spatially averaged direct
    field across all measurement positions, not the
    single reference position response.

    target_type selects the monitoring target against
    which corrections are calculated.

    Returns hf_corrections, lf_corrections,
    all_corrections.
    """
    hf_shelf_db = channel_cfg.get('hf_shelf_db', 0.0)
    hf_shelf_hz = channel_cfg.get('hf_shelf_hz', 10000)

    raw_corrections = derive_direct_field_target(
        direct_levels,
        hf_shelf_hz=hf_shelf_hz,
        hf_shelf_db=hf_shelf_db,
        target_type=target_type)

    hf_corrections = apply_correction_constraints(
        raw_corrections, direct_levels,
        reverberant_levels)

    lf_corrections = lf_correction_from_spatial_average(
        spatial_avg_levels,
        transition_hz=transition_hz)

    all_corrections = {
        **lf_corrections, **hf_corrections}

    return hf_corrections, lf_corrections, all_corrections


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_third_octave(levels_dict, fraction=3):
    """
    1/N-octave smoothing. Power domain averaging.

    fraction=3: 1/3-octave smoothing.
    fraction=6: 1/6-octave smoothing (used above
    transition frequency).
    """
    bands = sorted(levels_dict.keys())
    if len(bands) < 3:
        return levels_dict
    freqs = np.array(bands, dtype=float)
    levels = np.array([levels_dict[b] for b in bands])
    smoothed = np.zeros_like(levels)
    for i, f in enumerate(freqs):
        f_lo = f / (2.0 ** (1.0 / (2.0 * fraction)))
        f_hi = f * (2.0 ** (1.0 / (2.0 * fraction)))
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() > 0:
            power = 10.0 ** (levels[mask] / 10.0)
            smoothed[i] = 10.0 * np.log10(
                np.mean(power))
        else:
            smoothed[i] = levels[i]
    return {float(b): float(v)
            for b, v in zip(bands, smoothed)}


# ---------------------------------------------------------------------------
# Steady-state prediction with half-octave transition
# ---------------------------------------------------------------------------

def predict_post_eq_steady_state_third_octave(
        direct_levels_3rd,
        reverberant_levels_3rd,
        all_corrections_octave,
        bands=THIRD_OCTAVE_CENTRES,
        transition_hz=250,
        half_octave_overlap=True):
    """
    Predict steady-state response after EQ in 1/3-octave
    bands.

    direct_levels_3rd should be the spatially averaged
    direct field at 1/3-octave resolution.

    Half-octave transition splice:
      Above transition: 1/6-octave smoothed direct field
      Below transition: 1/3-octave smoothed direct field
      Splice region:    power-domain crossfade over half
                        octave — no hard step at
                        transition
    """
    oct_bands = sorted(all_corrections_octave.keys())
    oct_corr = [
        all_corrections_octave[b] for b in oct_bands]
    band_floats = [float(b) for b in bands]
    interp_corr = np.interp(
        np.log10(band_floats),
        np.log10([float(b) for b in oct_bands]),
        oct_corr,
        left=oct_corr[0],
        right=oct_corr[-1])

    direct_hf = smooth_third_octave(
        direct_levels_3rd, fraction=6)
    direct_lf = smooth_third_octave(
        direct_levels_3rd, fraction=3)
    splice_lo = transition_hz / (2.0 ** 0.5)
    splice_hi = transition_hz * (2.0 ** 0.5)

    predicted = {}
    for i, b in enumerate(band_floats):
        c = float(interp_corr[i])
        if (half_octave_overlap
                and splice_lo < b < splice_hi):
            w = ((np.log10(b) - np.log10(splice_lo))
                 / (np.log10(splice_hi)
                    - np.log10(splice_lo)))
            d_hf = direct_hf.get(b, np.nan)
            d_lf = direct_lf.get(b, np.nan)
            if np.isnan(d_hf) or np.isnan(d_lf):
                d = direct_levels_3rd.get(b, np.nan)
            else:
                d = 10.0 * np.log10(
                    (1.0 - w) * 10.0 ** (d_lf / 10.0)
                    + w * 10.0 ** (d_hf / 10.0))
        elif b >= transition_hz:
            d = direct_hf.get(b, np.nan)
        else:
            d = direct_lf.get(b, np.nan)
        r = reverberant_levels_3rd.get(b, np.nan)
        if np.isnan(d) or np.isnan(r):
            predicted[b] = np.nan
            continue
        ss = 10.0 * np.log10(
            10.0 ** ((d + c) / 10.0)
            + 10.0 ** ((r + c) / 10.0))
        predicted[b] = round(ss, 2)
    return predicted


# ---------------------------------------------------------------------------
# Steady-state reconstruction from measured fields
# ---------------------------------------------------------------------------

def reconstruct_steady_state(
        direct_levels_3rd,
        reverberant_levels_3rd,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Reconstruct steady-state by energy-summing measured
    fields.

    This is a reconstruction from measurements, not an
    independent physics-based prediction. It answers:
    "Given what we measured, what would a steady-state
    pink noise measurement show?"

    Returns:
      reconstructed dict {centre_hz: level_db}
      tolerance_upper dict {centre_hz: level_db}
      tolerance_lower dict {centre_hz: level_db}

    Tolerance bands:
      +/-2 dB from 100 Hz to 8 kHz
      +/-3 dB below 100 Hz and above 8 kHz
    """
    band_floats = [float(b) for b in bands]
    reconstructed = {}
    tolerance_upper = {}
    tolerance_lower = {}
    for b in band_floats:
        d = direct_levels_3rd.get(b, np.nan)
        r = reverberant_levels_3rd.get(b, np.nan)
        if np.isnan(d) or np.isnan(r):
            reconstructed[b] = np.nan
            tolerance_upper[b] = np.nan
            tolerance_lower[b] = np.nan
            continue
        ss = 10.0 * np.log10(
            10.0 ** (d / 10.0) + 10.0 ** (r / 10.0))
        reconstructed[b] = round(ss, 2)
        tol = 2.0 if 100.0 <= b <= 8000.0 else 3.0
        tolerance_upper[b] = round(ss + tol, 2)
        tolerance_lower[b] = round(ss - tol, 2)
    return reconstructed, tolerance_upper, tolerance_lower


# ---------------------------------------------------------------------------
# Parametric EQ filter generation
# ---------------------------------------------------------------------------

def generate_parametric_eq(all_corrections_octave,
                            max_filters=10):
    """
    Generate parametric EQ filter recommendations from
    octave band corrections.

    Strategy:
      1. Ignore bands with correction near zero
         (threshold 0.5 dB).
      2. Collapse adjacent bands with similar corrections
         into a single broad filter.
      3. Map remaining features to PEQ bell, high shelf,
         or low shelf as appropriate.
      4. Limit output to max_filters filters.

    Filter fields:
      number        int
      type          'Bell', 'High Shelf', 'Low Shelf'
      frequency_hz  float
      gain_db       float
      q             float

    Returns list of filter dicts.
    """
    bands = sorted(
        b for b in all_corrections_octave.keys()
        if isinstance(b, int))
    corrections = [
        all_corrections_octave[b] for b in bands]

    threshold = 0.5
    active = [
        (b, c) for b, c in zip(bands, corrections)
        if abs(c) >= threshold and not np.isnan(c)]

    if not active:
        return []

    groups = []
    current_group = [active[0]]
    bands_list = list(OCTAVE_CENTRES)

    for i in range(1, len(active)):
        prev_b, prev_c = active[i - 1]
        curr_b, curr_c = active[i]
        prev_idx = (bands_list.index(prev_b)
                    if prev_b in bands_list else -1)
        curr_idx = (bands_list.index(curr_b)
                    if curr_b in bands_list else -1)
        adjacent = (curr_idx - prev_idx == 1)
        similar = abs(curr_c - prev_c) <= 1.5
        if adjacent and similar:
            current_group.append(active[i])
        else:
            groups.append(current_group)
            current_group = [active[i]]
    groups.append(current_group)

    filters = []
    for group in groups:
        if len(filters) >= max_filters:
            break

        freqs = [g[0] for g in group]
        gains = [g[1] for g in group]
        centre_freq = float(np.exp(
            np.mean(np.log([float(f) for f in freqs]))))
        mean_gain = float(np.mean(gains))

        min_freq = min(freqs)
        max_freq = max(freqs)

        if min_freq <= 125 and max_freq <= 250:
            filter_type = 'Low Shelf'
            freq = float(min_freq)
            q = 0.707
        elif min_freq >= 4000:
            filter_type = 'High Shelf'
            freq = float(min_freq)
            q = 0.707
        else:
            filter_type = 'Bell'
            freq = centre_freq
            if len(group) == 1:
                q = 1.0
            elif len(group) == 2:
                q = 0.7
            else:
                q = 0.5

        filters.append({
            'number': len(filters) + 1,
            'type': filter_type,
            'frequency_hz': round(freq, 1),
            'gain_db': round(mean_gain, 1),
            'q': round(q, 2),
        })

    return filters


# ---------------------------------------------------------------------------
# Smaart-compatible target export
# ---------------------------------------------------------------------------

def export_target_for_smaart(target_levels_3rd,
                              ref_level_db,
                              output_path,
                              label='EQ Target'):
    """
    Export a target curve as Smaart-compatible CSV.
    """
    bands_sorted = sorted(target_levels_3rd.keys())
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write('* ' + label + '\n')
        for b in bands_sorted:
            level_norm = target_levels_3rd.get(
                b, np.nan)
            if np.isnan(level_norm):
                continue
            level_abs = level_norm + ref_level_db
            f.write(
                str(float(b)) + ','
                + str(round(level_abs, 3)) + '\n')
    return output_path


def export_xcurve_for_smaart(xcurve_levels_3rd,
                              ref_level_db,
                              output_path,
                              screen_size='large'):
    """
    Export X-curve as Smaart-compatible CSV.
    """
    if screen_size == 'large':
        label = ('X-curve target (large room, '
                 'SMPTE ST 202M / ISO 2969)')
    else:
        label = ('X-curve target '
                 '(small room, SMPTE RP 200)')
    return export_target_for_smaart(
        xcurve_levels_3rd, ref_level_db,
        output_path, label=label)


def export_selected_target_for_smaart(
        target_type,
        target_levels_3rd,
        ref_level_db,
        output_path,
        channel_name=''):
    """
    Export the selected monitoring target as a
    Smaart-compatible reference curve CSV.
    """
    label_map = {
        'Flat Direct Field':
            'Flat target',
        'X-Curve (Large Room)':
            'X-curve large room '
            '(SMPTE ST 202M / ISO 2969)',
        'X-Curve (Small Room)':
            'X-curve small room (SMPTE RP 200)',
        'SMPTE ST 422':
            'SMPTE ST 422 target',
    }
    label = label_map.get(target_type, target_type)
    if channel_name:
        label = channel_name + ' — ' + label
    return export_target_for_smaart(
        target_levels_3rd, ref_level_db,
        output_path, label=label)


# ---------------------------------------------------------------------------
# CSV results report
# ---------------------------------------------------------------------------

def save_csv(direct_levels,
             reverberant_levels,
             di_estimates,
             rt60_per_band,
             all_corrections,
             channel_name,
             output_dir,
             bands=OCTAVE_CENTRES,
             target_type='Flat Direct Field',
             predicted_post_eq=None):
    """
    Save full per-band results as CSV.

    direct_levels should be the spatially averaged direct
    field used for EQ generation.

    Columns:
      channel, band_hz, target_type,
      direct_field_db, reverberant_field_db,
      di_estimate_db,
      measured_rt60_s, effective_rt60_s, rt60_source,
      eq_correction_db,
      predicted_post_eq_db
    """
    bands_int = [int(b) for b in bands]
    measured_rt60 = rt60_per_band.get('_measured', {})
    rows = []
    for b in bands_int:
        eff_rt60 = rt60_per_band.get(b)
        meas_rt60 = measured_rt60.get(b)

        if eff_rt60 is None:
            rt60_source = 'None'
        elif (meas_rt60 is not None
              and abs(eff_rt60 - meas_rt60) < 0.001):
            rt60_source = 'Measured'
        elif meas_rt60 is None:
            rt60_source = 'Manual Override'
        else:
            rt60_source = 'HF Fallback'

        post_eq_db = np.nan
        if predicted_post_eq is not None:
            post_eq_db = predicted_post_eq.get(
                float(b), np.nan)

        rows.append({
            'channel': channel_name,
            'band_hz': b,
            'target_type': target_type,
            'direct_field_db': round(
                direct_levels.get(b, np.nan), 2),
            'reverberant_field_db': round(
                reverberant_levels.get(b, np.nan), 2),
            'di_estimate_db': round(
                di_estimates.get(b, np.nan), 2),
            'measured_rt60_s': round(
                meas_rt60 or np.nan, 3),
            'effective_rt60_s': round(
                eff_rt60 or np.nan, 3),
            'rt60_source': rt60_source,
            'eq_correction_db': round(
                all_corrections.get(b, 0.0), 2),
            'predicted_post_eq_db': (
                round(post_eq_db, 2)
                if not np.isnan(post_eq_db)
                else np.nan),
        })
    df = pd.DataFrame(rows)
    out_path = (Path(output_dir)
                / (channel_name + '_results.csv'))
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reverberant field analysis and EQ '
                    'target derivation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', default='output')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(
        "Use the Streamlit app for interactive analysis.")
    print("Command line mode outputs CSV only.")
