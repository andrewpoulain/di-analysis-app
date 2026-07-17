#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverberant Field Analysis and DI Estimation

Processes IRs exported from Smaart (WAV format) to compute:
  - Schroeder decay per octave band per position
  - Spatially averaged reverberant field spectrum
  - Gated direct field spectrum at reference position
  - DI estimate from direct/reverberant difference
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
    # Take first channel if stereo
    if data.ndim > 1:
        data = data[:, 0]
    return data, int(fs)


def apply_calibration(ir, fs, cal_file):
    """
    Apply a microphone calibration curve to an IR.
    cal_file: path to a two-column CSV (frequency_hz, sensitivity_db).
    Interpolates to IR frequency resolution and applies as a
    frequency-domain correction.
    """
    if cal_file is None or not os.path.exists(cal_file):
        return ir
    cal = np.loadtxt(cal_file, delimiter=',', skiprows=1)
    freqs = cal[:, 0]
    sens_db = cal[:, 1]
    n = len(ir)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Interpolate calibration to FFT frequency bins
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
    Clamps band limits to valid range for the sample rate.
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
    # Backward cumulative sum
    decay = np.cumsum(power[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)  # guard against log(0)
    decay_db = 10.0 * np.log10(decay / decay[0])
    return decay_db


def initial_decay_level(ir, fs, centre_hz):
    """
    Return the initial Schroeder decay level (dB, relative) for one
    octave band. This is the reverberant energy proxy for that band.
    Uses the mean level over the first 5 ms of the decay curve to
    reduce sensitivity to the direct arrival peak.
    """
    f_low, f_high = octave_band_limits(centre_hz)
    ir_band = bandpass_ir(ir, fs, f_low, f_high)
    decay_db = schroeder_decay(ir_band)
    # Average over first 5 ms to smooth the initial level estimate
    n_avg = max(1, int(0.005 * fs))
    return float(np.mean(decay_db[:n_avg]))


def reverberant_spectrum(ir, fs, bands=OCTAVE_CENTRES):
    """
    Return the Schroeder initial decay level for each octave band.
    Result is a dict {centre_hz: level_db}.
    """
    return {int(b): initial_decay_level(ir, fs, b) for b in bands}


# ---------------------------------------------------------------------------
# RT60 estimation (Sabine inversion for room constant)
# ---------------------------------------------------------------------------
def rt60_from_schroeder(ir, fs, centre_hz, eval_range_db=(-5, -25)):
    """
    Estimate RT60 in one octave band from the Schroeder decay curve.
    Fits a line between eval_range_db[0] and eval_range_db[1] and
    extrapolates to -60 dB.
    Returns RT60 in seconds, or None if the decay is too noisy.
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
    # slope is dB/s; RT60 = -60 / slope
    slope = coeffs[0]
    if slope >= 0:
        return None
    return float(-60.0 / slope)


def room_constant(rt60_s, volume_m3, surface_area_m2):
    """
    Derive the room constant R from RT60 via Sabine inversion.
    R = S * alpha / (1 - alpha)
    where alpha = 0.161 * V / (RT60 * S)   [Sabine]
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
def detect_direct_arrival(ir, fs, threshold_db=-20):
    """
    Find the sample index of the direct arrival.
    Uses the first sample that exceeds threshold_db relative to peak.
    """
    power_db = 20.0 * np.log10(np.abs(ir) / (np.max(np.abs(ir)) + 1e-30))
    candidates = np.where(power_db >= threshold_db)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def detect_first_reflection(ir, fs, direct_idx, min_gap_ms=2.0):
    """
    Estimate the arrival time of the first significant reflection.
    Looks for the next local peak after a minimum gap from the direct
    arrival that exceeds -20 dB relative to the direct peak.
    Returns sample index, or None if not found within the IR length.
    """
    min_gap = int(min_gap_ms * fs / 1000.0)
    search_start = direct_idx + min_gap
    direct_level = np.abs(ir[direct_idx])
    threshold = direct_level * 10.0 ** (-20.0 / 20.0)
    # Find local maxima in the search region
    search = np.abs(ir[search_start:])
    peaks, _ = sig.find_peaks(search, height=threshold, distance=min_gap)
    if len(peaks) == 0:
        return None
    return int(search_start + peaks[0])


def gated_direct_field(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
    """
    Extract the gated direct field magnitude response.
    If gate_ms is None, the gate is set automatically to 90% of the
    interval between the direct arrival and the first reflection.
    Returns:
        freqs        : frequency array (Hz)
        magnitude    : magnitude response (dB, relative to peak)
        gate_ms_used : gate length actually applied (ms)
    """
    direct_idx = detect_direct_arrival(ir, fs)
    reflection_idx = detect_first_reflection(ir, fs, direct_idx)
    if gate_ms is None:
        if reflection_idx is not None:
            gap_samples = reflection_idx - direct_idx
            gate_samples = int(0.9 * gap_samples)
        else:
            # Fall back to 20 ms if no reflection found
            gate_samples = int(0.020 * fs)
    else:
        gate_samples = int(gate_ms * fs / 1000.0)
    gate_ms_used = gate_samples / fs * 1000.0
    # Trim IR to gate window starting at direct arrival
    ir_gated = ir[direct_idx: direct_idx + gate_samples].copy()
    # Apply half-Hann window
    window = np.hanning(2 * len(ir_gated))[:len(ir_gated)]
    ir_gated *= window
    # Zero-pad to next power of 2 for FFT efficiency
    n_fft = int(2 ** np.ceil(np.log2(len(ir_gated))))
    spectrum = np.fft.rfft(ir_gated, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitude = 20.0 * np.log10(np.abs(spectrum) + 1e-30)
    magnitude -= np.max(magnitude)  # normalise to 0 dB peak
    return freqs, magnitude, gate_ms_used


def direct_field_at_bands(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
    """
    Return the mean direct field level in each octave band (dB, relative).
    """
    freqs, magnitude, gate_ms_used = gated_direct_field(ir, fs, gate_ms)
    levels = {}
    for b in bands:
        f_low, f_high = octave_band_limits(b)
