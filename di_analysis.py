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
