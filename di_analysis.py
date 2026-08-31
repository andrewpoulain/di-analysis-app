#!/usr/bin/env python3
"""
Reverberant Field Analysis and EQ Target Derivation
Processes IRs exported from Smaart (WAV format) to compute:
  - Schroeder decay per octave band per position
  - Spatially averaged reverberant field spectrum
  - Gated direct field spectrum at reference position
  - DI estimate from direct/reverberant difference
  - EQ correction targets per octave band
  - Physics-based steady-state prediction and verification
  - Output plots and CSV report

Usage:
    python di_analysis.py --config room_config.yaml --session session_dir/
"""

import os
import argparse
import yaml
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000
])


def third_octave_band_limits(centre_hz):
    """Return (f_low, f_high) for a 1/3-octave band."""
    return centre_hz / (2 ** (1.0 / 6)), centre_hz * (2 ** (1.0 / 6))


# ---------------------------------------------------------------------------
# Truncation margin scaled by frequency
# ---------------------------------------------------------------------------

def truncation_margin_for_band(centre_hz):
    """
    Return the noise floor truncation margin in dB for a given
    centre frequency.

    Scaled with frequency because at high frequencies the IR
    decays quickly and drops into the noise floor faster.
    A larger margin preserves more of the decay tail and prevents
    premature truncation that causes RT60 underestimation.

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
# Direct arrival detection
# ---------------------------------------------------------------------------

def detect_direct_arrival(ir, fs, threshold_db=-20):
    """
    Find the sample index of the direct arrival.
    Uses the first sample that exceeds threshold_db relative to peak.
    """
    peak = np.max(np.abs(ir))
    if peak == 0:
        return 0
    power_db = 20.0 * np.log10(np.abs(ir) / peak + 1e-30)
    candidates = np.where(power_db >= threshold_db)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def detect_first_reflection(ir, fs, direct_idx, min_gap_ms=2.0):
    """
    Estimate the arrival time of the first significant reflection.
    Returns sample index or None if not found.
    """
    min_gap = int(min_gap_ms * fs / 1000.0)
    search_start = direct_idx + min_gap
    direct_level = np.abs(ir[direct_idx])
    if direct_level == 0:
        return None
    threshold = direct_level * 10.0 ** (-20.0 / 20.0)
    search = np.abs(ir[search_start:])
    peaks, _ = sig.find_peaks(search, height=threshold,
                               distance=min_gap)
    if len(peaks) == 0:
        return None
    return int(search_start + peaks[0])


# ---------------------------------------------------------------------------
# Noise floor truncation
# ---------------------------------------------------------------------------

def truncate_to_noise_floor(ir, margin_db=10.0):
    """
    Truncate an IR at the point where the envelope drops to
    margin_db above the dynamic noise floor.

    The noise floor is estimated from the last 10% of the smoothed
    envelope. margin_db controls how far above the noise floor the
    truncation point is set.

    Use truncation_margin_for_band() to get the appropriate margin
    for each frequency band.
    """
    peak = np.max(np.abs(ir))
    if peak == 0:
        return ir

    window_samples = max(1, len(ir) // 200)
    envelope = np.abs(ir)
    smoothed = scipy.ndimage.maximum_filter1d(
        envelope, size=window_samples)
    smoothed_db = 20.0 * np.log10(smoothed / peak + 1e-30)

    tail_start = int(len(smoothed_db) * 0.9)
    noise_floor_db = float(np.mean(smoothed_db[tail_start:]))
    truncation_threshold_db = noise_floor_db + margin_db

    above = np.where(smoothed_db >= truncation_threshold_db)[0]
    if len(above) == 0:
        return ir[:len(ir) // 2]

    cutoff = above[-1]
    truncated = ir.copy()
    truncated[cutoff:] = 0.0
    return truncated


# ---------------------------------------------------------------------------
# Bandpass filtering
# ---------------------------------------------------------------------------

def bandpass_ir(ir, fs, f_low, f_high, order=4):
    """
    Bandpass filter an IR using a zero-phase Butterworth filter.

    order defaults to 4. Pass order=2 for low frequency bands
    (63 Hz and below) to prevent sosfiltfilt ringing on
    narrow-bandwidth IRs which would extend the apparent decay
    and cause RT60 overestimation at low frequencies.
    """
    nyq = fs / 2.0
    f_low = max(f_low, 10.0)
    f_high = min(f_high, nyq * 0.99)
    if f_low >= f_high:
        return np.zeros_like(ir)
    sos = sig.butter(order, [f_low / nyq, f_high / nyq],
                     btype='band', output='sos')
    return sig.sosfiltfilt(sos, ir)


# ---------------------------------------------------------------------------
# Schroeder integration
# ---------------------------------------------------------------------------

def schroeder_decay(ir_band):
    """
    Compute the Schroeder backward integral with noise energy
    subtraction.

    Estimates noise power from the last 10% of the squared IR and
    subtracts it before integration. Negative values are clamped
    to zero. Returns the decay curve normalised so the initial
    value is 0 dB.
    """
    power = ir_band ** 2
    tail_start = int(len(power) * 0.9)
    noise_power = float(np.mean(power[tail_start:]))
    power_compensated = np.maximum(power - noise_power, 0.0)
    decay = np.cumsum(power_compensated[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)
    decay_db = 10.0 * np.log10(decay / decay[0])
    return decay_db


def initial_decay_level(ir, fs, centre_hz):
    """
    Return the initial Schroeder decay level for one octave band.
    Uses frequency-scaled filter order and truncation margin.
    """
    f_low, f_high = octave_band_limits(centre_hz)
    order = 2 if centre_hz <= 63 else 4
    ir_band = bandpass_ir(ir, fs, f_low, f_high, order=order)
    direct_idx = detect_direct_arrival(
        ir_band, fs, threshold_db=-20)
    ir_band = ir_band[direct_idx:]
    margin_db = truncation_margin_for_band(centre_hz)
    ir_band = truncate_to_noise_floor(
        ir_band, margin_db=margin_db)
    decay_db = schroeder_decay(ir_band)
    n_avg = max(1, int(0.005 * fs))
    return float(np.mean(decay_db[:n_avg]))


def reverberant_spectrum(ir, fs, bands=OCTAVE_CENTRES):
    """
    Return the Schroeder initial decay level for each octave band.
    Returns dict {centre_hz: level_db}.
    """
    return {int(b): initial_decay_level(ir, fs, b) for b in bands}


# ---------------------------------------------------------------------------
# RT60 estimation
# ---------------------------------------------------------------------------

def rt60_from_schroeder(ir, fs, centre_hz):
    """
    Estimate RT60 in one octave band from the Schroeder decay curve.

    Evaluation order is frequency-dependent:

      Below 8 kHz:
        T20 (-5 to -25 dB) primary
        T30 (-5 to -35 dB) first fallback
        EDT (0 to -10 dB) last resort

      8 kHz and above:
        EDT (0 to -10 dB) primary — at very high frequencies
        the IR dynamic range is insufficient for T20.
        EDT uses the strongest part of the decay where SNR
        is highest and gives more reliable results than
        forcing T20 on a truncated decay.
        T20 fallback
        T30 last resort

    Filter order is reduced to 2 at 63 Hz and below to prevent
    sosfiltfilt ringing from extending the apparent decay and
    causing RT60 overestimation.

    The minimum slope threshold is relaxed at high frequencies
    because short decays have steeper slopes and the standard
    -0.5 dB/s threshold would reject valid fits.

    Returns RT60 in seconds, or None if unreliable.
    """
    f_low, f_high = octave_band_limits(centre_hz)
    order = 2 if centre_hz <= 63 else 4

    nyq = fs / 2.0
    fl = max(f_low, 10.0)
    fh = min(f_high, nyq * 0.99)
    if fl >= fh:
        return None

    sos = sig.butter(order, [fl / nyq, fh / nyq],
                     btype='band', output='sos')
    ir_band = sig.sosfiltfilt(sos, ir)

    if np.max(np.abs(ir_band)) < 1e-10:
        return None

    direct_idx = detect_direct_arrival(
        ir_band, fs, threshold_db=-20)
    ir_band = ir_band[direct_idx:]

    if len(ir_band) < int(0.05 * fs):
        return None

    margin_db = truncation_margin_for_band(centre_hz)
    ir_band = truncate_to_noise_floor(
        ir_band, margin_db=margin_db)

    if np.max(np.abs(ir_band)) < 1e-10:
        return None

    decay_db = schroeder_decay(ir_band)
    times = np.arange(len(decay_db)) / fs

    if centre_hz >= 8000:
        eval_ranges = [
            (0,   -10),   # EDT — primary at HF
            (-5,  -25),   # T20 — fallback
            (-5,  -35),   # T30 — last resort
        ]
        min_slope = -0.1
    else:
        eval_ranges = [
            (-5,  -25),   # T20 — primary
            (-5,  -35),   # T30 — first fallback
            (0,   -10),   # EDT — last resort
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
        coeffs = np.polyfit(t_region, d_region, 1)
        slope = coeffs[0]
        if slope >= min_slope:
            continue
        rt60 = -60.0 / slope
        if 0.05 <= rt60 <= 20.0:
            return float(rt60)

    return None


# ---------------------------------------------------------------------------
# HF RT60 fallback logic
# ---------------------------------------------------------------------------

def apply_rt60_hf_fallback(rt60_dict):
    """
    Apply high-frequency RT60 fallback logic.

    At high frequencies the IR dynamic range is often insufficient
    to produce a reliable RT60 estimate. This function detects
    implausible drops in RT60 at 8 kHz and 16 kHz and substitutes
    the value from the next lower band.

    Rules applied in order:

    Rule 1 — 8 kHz check:
      If RT60 at 8 kHz < 10% of RT60 at 4 kHz,
      replace 8 kHz RT60 with the 4 kHz value.
      Also replace 16 kHz with the 4 kHz value.
      Rule 2 is then skipped since both bands are set.

    Rule 2 — 16 kHz check (only if Rule 1 did not trigger):
      If RT60 at 16 kHz < 10% of RT60 at 8 kHz,
      replace 16 kHz RT60 with the 8 kHz value.

    The 10% threshold is chosen because a genuine room cannot
    lose 90% of its reverberation time in a single octave step.
    The largest physically plausible single-octave drop in a
    well-behaved room is roughly 40-50%. A drop to less than
    10% of the previous band is a measurement failure, not a
    room property.

    Returns a new dict with corrected values and a list of
    warning strings describing any substitutions made.
    """
    corrected = dict(rt60_dict)
    warnings = []

    rt60_4k = corrected.get(4000)
    rt60_8k = corrected.get(8000)

    # Rule 1 — check 8 kHz against 4 kHz
    if (rt60_4k is not None
            and rt60_8k is not None
            and rt60_4k > 0):
        ratio_8k = rt60_8k / rt60_4k
        if ratio_8k < 0.10:
            warnings.append(
                f'8 kHz RT60 ({rt60_8k:.3f} s) is less than '
                f'10% of 4 kHz RT60 ({rt60_4k:.3f} s) — '
                f'substituting 4 kHz value ({rt60_4k:.3f} s) '
                f'at 8 kHz and 16 kHz')
            corrected[8000] = rt60_4k
            corrected[16000] = rt60_4k
            return corrected, warnings

    # Rule 2 — check 16 kHz against 8 kHz
    rt60_8k_current = corrected.get(8000)
    rt60_16k_current = corrected.get(16000)

    if (rt60_8k_current is not None
            and rt60_16k_current is not None
            and rt60_8k_current > 0):
        ratio_16k = rt60_16k_current / rt60_8k_current
        if ratio_16k < 0.10:
            warnings.append(
                f'16 kHz RT60 ({rt60_16k_current:.3f} s) is '
                f'less than 10% of 8 kHz RT60 '
                f'({rt60_8k_current:.3f} s) — '
                f'substituting 8 kHz value '
                f'({rt60_8k_current:.3f} s) at 16 kHz')
            corrected[16000] = rt60_8k_current

    return corrected, warnings


# ---------------------------------------------------------------------------
# RT60 averaging across positions
# ---------------------------------------------------------------------------

def rt60_per_band_from_irs(ir_list, fs, bands=OCTAVE_CENTRES):
    """
    Estimate RT60 per octave band by averaging across all IR
    positions.

    After averaging, applies HF fallback logic to correct
    implausible drops at 8 kHz and 16 kHz. Any substitutions
    made are stored in the returned dict under the key
    '_hf_warnings' as a list of strings.

    Returns dict {centre_hz: RT60_seconds} with an additional
    '_hf_warnings' key containing a list of warning strings.
    """
    rt60_all = {int(b): [] for b in bands}
    for ir in ir_list:
        for b in bands:
            rt = rt60_from_schroeder(ir, fs, b)
            if rt is not None:
                rt60_all[int(b)].append(rt)

    averaged = {b: float(np.mean(v)) if v else None
                for b, v in rt60_all.items()}

    corrected, hf_warnings = apply_rt60_hf_fallback(averaged)
    corrected['_hf_warnings'] = hf_warnings

    return corrected


# ---------------------------------------------------------------------------
# RT60 validation
# ---------------------------------------------------------------------------

def validate_rt60(rt60_per_band, bands=OCTAVE_CENTRES):
    """
    Check RT60 values for physical plausibility.
    Returns a dict of warnings keyed by band.

    Includes any HF fallback substitution warnings from
    apply_rt60_hf_fallback that were stored on the dict by
    rt60_per_band_from_irs.
    """
    warnings = {}
    bands_int = [int(b) for b in bands]

    hf_warnings = rt60_per_band.get('_hf_warnings', [])
    for i, w in enumerate(hf_warnings):
        warnings[f'hf_fallback_{i}'] = w

    valid = {b: rt60_per_band.get(b) for b in bands_int
             if rt60_per_band.get(b) is not None}

    if not valid:
        warnings['general'] = 'No valid RT60 estimates produced'
        return warnings

    for b, rt in valid.items():
        if rt > 15.0:
            warnings[b] = f'{rt:.2f} s is implausibly long'
        if rt < 0.05:
            warnings[b] = f'{rt:.3f} s is implausibly short'

    mf_bands = [b for b in [500, 1000, 2000] if b in valid]
    hf_bands = [b for b in [4000, 8000] if b in valid]
    if mf_bands and hf_bands:
        mf_avg = np.mean([valid[b] for b in mf_bands])
        hf_avg = np.mean([valid[b] for b in hf_bands])
        if hf_avg > mf_avg * 1.5:
            warnings['hf_rising'] = (
                f'HF RT60 ({hf_avg:.2f} s) is higher than '
                f'MF RT60 ({mf_avg:.2f} s) — '
                f'check IR length and noise floor')

    return warnings


# ---------------------------------------------------------------------------
# Room constant
# ---------------------------------------------------------------------------

def room_constant(rt60_s, volume_m3, surface_area_m2):
    """
    Derive the room constant R from RT60 via Sabine inversion.
    Returns R in m^2, or None if inputs are invalid.
    """
    if rt60_s is None or rt60_s <= 0:
        return None
    alpha = 0.161 * volume_m3 / (rt60_s * surface_area_m2)
    alpha = min(alpha, 0.999)
    return surface_area_m2 * alpha / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Gated direct field
# ---------------------------------------------------------------------------

def gated_direct_field(ir, fs, gate_ms=None):
    """
    Extract the gated direct field magnitude response.
    Returns freqs, magnitude (dB normalised to 0 dB peak),
    and gate_ms_used.
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

    gate_samples = max(gate_samples, 16)
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


def direct_field_at_bands(ir, fs, gate_ms=None,
                           bands=OCTAVE_CENTRES):
    """
    Return the mean direct field level in each octave band
    (dB, relative).
    """
    freqs, magnitude, gate_ms_used = gated_direct_field(
        ir, fs, gate_ms)
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


def direct_field_at_third_octave_bands(ir, fs, gate_ms=None,
                                        bands=THIRD_OCTAVE_CENTRES):
    """
    Return the mean direct field level in each 1/3-octave band.
    """
    freqs, magnitude, gate_ms_used = gated_direct_field(
        ir, fs, gate_ms)
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
# Spatial averaging of reverberant field
# ---------------------------------------------------------------------------

def spatial_average_reverberant(ir_list, fs,
                                  bands=OCTAVE_CENTRES):
    """
    Spatially averaged reverberant field in octave bands.
    Averaging is performed in the power domain.
    """
    all_spectra = [reverberant_spectrum(ir, fs, bands)
                   for ir in ir_list]
    averaged = {}
    for b in bands:
        b = int(b)
        levels_db = [s[b] for s in all_spectra]
        powers = [10.0 ** (l / 10.0) for l in levels_db]
        averaged[b] = float(10.0 * np.log10(np.mean(powers)))
    return averaged


def reverberant_spectrum_third_octave(ir, fs,
                                       bands=THIRD_OCTAVE_CENTRES):
    """
    Return the Schroeder initial decay level for each 1/3-octave
    band. Uses frequency-scaled filter order and truncation margin.

    Filter order is reduced to 2 below 80 Hz to prevent sosfiltfilt
    ringing on narrow-bandwidth IRs at low frequencies.
    """
    result = {}
    for b in bands:
        f_low, f_high = third_octave_band_limits(b)
        order = 2 if b <= 80 else 4
        ir_band = bandpass_ir(
            ir, fs, f_low, f_high, order=order)
        direct_idx = detect_direct_arrival(
            ir_band, fs, threshold_db=-20)
        ir_band = ir_band[direct_idx:]
        margin_db = truncation_margin_for_band(b)
        ir_band = truncate_to_noise_floor(
            ir_band, margin_db=margin_db)
        decay_db = schroeder_decay(ir_band)
        n_avg = max(1, int(0.005 * fs))
        result[float(b)] = float(np.mean(decay_db[:n_avg]))
    return result


def spatial_average_reverberant_third_octave(
        ir_list, fs, bands=THIRD_OCTAVE_CENTRES):
    """
    Spatially averaged reverberant field in 1/3-octave bands.
    Averaging is performed in the power domain.
    """
    all_spectra = [
        reverberant_spectrum_third_octave(ir, fs, bands)
        for ir in ir_list]
    averaged = {}
    for b in bands:
        b = float(b)
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

def derive_direct_field_target(direct_levels,
                                bands=OCTAVE_CENTRES,
                                ref_band=1000,
                                hf_shelf_hz=10000,
                                hf_shelf_db=0.0):
    """
    Derive the direct field EQ correction per octave band.
    Target is flat (0 dB) relative to the reference band.
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
        corr = max(min(corr, max_correction_db),
                   -max_correction_db)
        corrections[b] = round(corr, 2)
    return corrections


def derive_full_eq_target(direct_levels, reverberant_levels,
                           spatial_avg_levels,
                           channel_cfg,
                           transition_hz=250):
    """
    Full EQ target derivation for one channel.
    Returns hf_corrections, lf_corrections, all_corrections.
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

    return hf_corrections, lf_corrections, all_corrections


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_third_octave(levels_dict, fraction=3):
    """
    Apply 1/N-octave smoothing to a dict of {freq: level_db} values.

    fraction=3 gives 1/3-octave smoothing.
    fraction=6 gives 1/6-octave smoothing (used above transition
    frequency per Section 5.5 of the white paper).

    Averaging is performed in the power domain.
    Returns smoothed dict with same keys.
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
            smoothed[i] = 10.0 * np.log10(np.mean(power))
        else:
            smoothed[i] = levels[i]

    return {float(b): float(v) for b, v in zip(bands, smoothed)}


# ---------------------------------------------------------------------------
# Steady-state prediction with half-octave transition splice
# ---------------------------------------------------------------------------

def predict_post_eq_steady_state_third_octave(
        direct_levels_3rd,
        reverberant_levels_3rd,
        all_corrections_octave,
        bands=THIRD_OCTAVE_CENTRES,
        transition_hz=250,
        half_octave_overlap=True):
    """
    Predict the steady-state response in 1/3-octave bands after
    applying octave band EQ corrections.

    Implements the half-octave transition splice from Section 5.5
    of the white paper:
      Above transition:  1/6-octave smoothing on direct field
      Below transition:  1/3-octave smoothing
      Splice region:     half-octave overlap with power-domain
                         crossfade — no hard step at transition

    Octave band corrections are interpolated to 1/3-octave
    resolution in log-frequency space.
    """
    oct_bands = sorted(all_corrections_octave.keys())
    oct_corr = [all_corrections_octave[b] for b in oct_bands]
    band_floats = [float(b) for b in bands]
    log_oct = np.log10([float(b) for b in oct_bands])
    log_third = np.log10(band_floats)
    interp_corr = np.interp(log_third, log_oct, oct_corr,
                             left=oct_corr[0], right=oct_corr[-1])

    direct_smoothed_hf = smooth_third_octave(
        direct_levels_3rd, fraction=6)
    direct_smoothed_lf = smooth_third_octave(
        direct_levels_3rd, fraction=3)

    splice_lo = transition_hz / (2.0 ** (1.0 / 2.0))
    splice_hi = transition_hz * (2.0 ** (1.0 / 2.0))

    predicted = {}
    for i, b in enumerate(band_floats):
        c = float(interp_corr[i])

        if half_octave_overlap and splice_lo < b < splice_hi:
            weight = (
                (np.log10(b) - np.log10(splice_lo)) /
                (np.log10(splice_hi) - np.log10(splice_lo)))
            d_hf = direct_smoothed_hf.get(b, np.nan)
            d_lf = direct_smoothed_lf.get(b, np.nan)
            if np.isnan(d_hf) or np.isnan(d_lf):
                d = direct_levels_3rd.get(b, np.nan)
            else:
                p_hf = 10.0 ** (d_hf / 10.0)
                p_lf = 10.0 ** (d_lf / 10.0)
                d = 10.0 * np.log10(
                    (1.0 - weight) * p_lf + weight * p_hf)
        elif b >= transition_hz:
            d = direct_smoothed_hf.get(b, np.nan)
        else:
            d = direct_smoothed_lf.get(b, np.nan)

        r = reverberant_levels_3rd.get(b, np.nan)

        if np.isnan(d) or np.isnan(r):
            predicted[b] = np.nan
            continue

        d_eq = d + c
        r_eq = r + c
        ss = 10.0 * np.log10(
            10.0 ** (d_eq / 10.0) + 10.0 ** (r_eq / 10.0))
        predicted[b] = round(ss, 2)

    return predicted


# ---------------------------------------------------------------------------
# Physics-based steady-state prediction and verification
# ---------------------------------------------------------------------------

def predict_steady_state_from_physics(
        direct_levels_3rd,
        reverberant_levels_3rd,
        rt60_per_band,
        volume_m3,
        surface_area_m2,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Predict the steady-state response from physical parameters
    per Section 4.2 of the white paper.

    Energy-sums the measured direct and reverberant fields to
    produce the expected steady-state. Used as a verification
    envelope — if the measured steady-state falls within ±2 dB
    (100 Hz to 8 kHz) or ±3 dB at the extremes the system is
    behaving correctly. Disagreement localises installation faults.

    Returns:
      predicted dict {centre_hz: level_db}
      tolerance_upper dict {centre_hz: level_db}
      tolerance_lower dict {centre_hz: level_db}
    """
    oct_bands_with_rt60 = sorted(
        b for b in [int(x) for x in OCTAVE_CENTRES]
        if rt60_per_band.get(b) is not None)

    if not oct_bands_with_rt60:
        empty = {float(b): np.nan for b in bands}
        return empty, empty, empty

    band_floats = [float(b) for b in bands]

    predicted = {}
    tolerance_upper = {}
    tolerance_lower = {}

    for b in band_floats:
        d = direct_levels_3rd.get(b, np.nan)
        r = reverberant_levels_3rd.get(b, np.nan)

        if np.isnan(d) or np.isnan(r):
            predicted[b] = np.nan
            tolerance_upper[b] = np.nan
            tolerance_lower[b] = np.nan
            continue

        ss = 10.0 * np.log10(
            10.0 ** (d / 10.0) + 10.0 ** (r / 10.0))

        predicted[b] = round(ss, 2)

        tol = 2.0 if 100.0 <= b <= 8000.0 else 3.0
        tolerance_upper[b] = round(ss + tol, 2)
        tolerance_lower[b] = round(ss - tol, 2)

    return predicted, tolerance_upper, tolerance_lower


def compare_measured_to_predicted(
        measured_steady_state_3rd,
        predicted_steady_state_3rd,
        tolerance_upper_3rd,
        tolerance_lower_3rd,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Compare a measured steady-state response against the
    physics-based prediction and tolerance band.

    Returns a dict of diagnostic results per band:
      {centre_hz: {
          'measured': float,
          'predicted': float,
          'delta': float,
          'within_tolerance': bool,
          'tolerance': float
      }}
    """
    results = {}
    for b in [float(b) for b in bands]:
        m = measured_steady_state_3rd.get(b, np.nan)
        p = predicted_steady_state_3rd.get(b, np.nan)
        tu = tolerance_upper_3rd.get(b, np.nan)
        tl = tolerance_lower_3rd.get(b, np.nan)

        if np.isnan(m) or np.isnan(p):
            results[b] = {
                'measured': m,
                'predicted': p,
                'delta': np.nan,
                'within_tolerance': None,
                'tolerance': np.nan}
            continue

        delta = m - p
        within = (not np.isnan(tu) and not np.isnan(tl)
                  and tl <= m <= tu)
        tol = (tu - p) if not np.isnan(tu) else np.nan

        results[b] = {
            'measured': round(m, 2),
            'predicted': round(p, 2),
            'delta': round(delta, 2),
            'within_tolerance': within,
            'tolerance': (round(tol, 2)
                          if not np.isnan(tol) else np.nan)}

    return results


# ---------------------------------------------------------------------------
# X-curve target (SMPTE ST 202M / ISO 2969)
# ---------------------------------------------------------------------------

def xcurve_target(freqs_hz, screen_size='large'):
    """
    Generate the X-curve target level at each frequency.

    Standard X-curve (large rooms > 150 m³, SMPTE ST 202M):
      Flat to 2 kHz, -3 dB/octave above,
      -3 dB/octave below 63 Hz.

    Modified X-curve (small rooms < 150 m³, SMPTE RP 200):
      Flat to 4 kHz, -3 dB/octave above,
      -3 dB/octave below 63 Hz.

    Returns array of target levels in dB.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    target = np.zeros_like(freqs)
    hf_corner = 4000.0 if screen_size == 'small' else 2000.0
    lf_corner = 63.0

    for i, f in enumerate(freqs):
        if f <= 0:
            target[i] = np.nan
            continue
        level = 0.0
        if f > hf_corner:
            octaves_above = np.log2(f / hf_corner)
            level -= 3.0 * octaves_above
        if f < lf_corner:
            octaves_below = np.log2(lf_corner / f)
            level -= 3.0 * octaves_below
        target[i] = level

    return target


def xcurve_at_third_octave_bands(bands=THIRD_OCTAVE_CENTRES,
                                  screen_size='large'):
    """
    Return the X-curve target level at each 1/3-octave band centre.
    Returns dict {centre_hz: target_db}.
    """
    levels = xcurve_target(
        np.array(bands, dtype=float), screen_size=screen_size)
    return {float(b): float(l)
            for b, l in zip(bands, levels)
            if not np.isnan(l)}


# ---------------------------------------------------------------------------
# Smaart-compatible target export
# ---------------------------------------------------------------------------

def export_target_for_smaart(target_levels_3rd,
                              ref_level_db,
                              output_path,
                              label='EQ Target'):
    """
    Export a target curve as a two-column CSV that can be imported
    into Smaart as a reference curve.

    Smaart reference curve format:
      Comment line starting with *
      Two columns: frequency (Hz) and level (dB)
      Comma separated, no header row
      Frequencies in ascending order

    The target is exported as absolute dB values referenced to the
    measured direct field level at 1 kHz so that when imported into
    Smaart and overlaid on a transfer function measurement at the
    same gain setting the target will align correctly.
    """
    bands_sorted = sorted(target_levels_3rd.keys())
    rows = []
    for b in bands_sorted:
        level_norm = target_levels_3rd.get(b, np.nan)
        if np.isnan(level_norm):
            continue
        level_abs = level_norm + ref_level_db
        rows.append((float(b), round(level_abs, 3)))

    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write(f'* {label}\n')
        for freq, level in rows:
            f.write(f'{freq},{level}\n')

    return output_path


def export_xcurve_for_smaart(xcurve_levels_3rd,
                              ref_level_db,
                              output_path,
                              screen_size='large'):
    """
    Export the X-curve target as a Smaart-compatible reference
    curve CSV.
    """
    label = (
        f'X-curve target '
        f'({"large room" if screen_size == "large" else "small room"}'
        f', SMPTE ST 202M / ISO 2969)')

    bands_sorted = sorted(xcurve_levels_3rd.keys())
    rows = []
    for b in bands_sorted:
        level_norm = xcurve_levels_3rd.get(b, np.nan)
        if np.isnan(level_norm):
            continue
        level_abs = level_norm + ref_level_db
        rows.append((float(b), round(level_abs, 3)))

    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write(f'* {label}\n')
        for freq, level in rows:
            f.write(f'{freq},{level}\n')

    return output_path


# ---------------------------------------------------------------------------
# CSV results report
# ---------------------------------------------------------------------------

def save_csv(direct_levels, reverberant_levels, di_estimates,
             rt60_per_band, all_corrections,
             channel_name, output_dir,
             bands=OCTAVE_CENTRES):
    """
    Save full per-band results as CSV.
    """
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
        })
    df = pd.DataFrame(rows)
    out_path = Path(output_dir) / f"{channel_name}_results.csv"
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reverberant field analysis and EQ target '
                    'derivation')
    parser.add_argument('--config', required=True,
                        help='Path to room_config.yaml')
    parser.add_argument('--session', required=True,
                        help='Directory containing IR WAV files')
    parser.add_argument('--output', default='output',
                        help='Directory for outputs')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Use the Streamlit app for interactive analysis.")
    print("Command line mode outputs CSV only.")
