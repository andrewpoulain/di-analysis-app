#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import tempfile
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Octave band definitions
# ---------------------------------------------------------------------------
OCTAVE_CENTRES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])


def octave_band_limits(centre_hz):
    return centre_hz / np.sqrt(2), centre_hz * np.sqrt(2)


# ---------------------------------------------------------------------------
# IR loading
# ---------------------------------------------------------------------------
def load_ir(filepath):
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
    nyq = fs / 2.0
    f_low = max(f_low, 10.0)
    f_high = min(f_high, nyq * 0.99)
    if f_low >= f_high:
        return np.zeros_like(ir)
    sos = sig.butter(4, [f_low / nyq, f_high / nyq],
                     btype='band', output='sos')
    return sig.sosfilt(sos, ir)


def schroeder_decay(ir_band):
    power = ir_band ** 2
    decay = np.cumsum(power[::-1])[::-1]
    decay = np.maximum(decay, 1e-30)
    decay_db = 10.0 * np.log10(decay / decay[0])
    return decay_db


def initial_decay_level(ir, fs, centre_hz):
    f_low, f_high = octave_band_limits(centre_hz)
    ir_band = bandpass_ir(ir, fs, f_low, f_high)
    decay_db = schroeder_decay(ir_band)
    n_avg = max(1, int(0.005 * fs))
    return float(np.mean(decay_db[:n_avg]))


def reverberant_spectrum(ir, fs, bands=OCTAVE_CENTRES):
    return {int(b): initial_decay_level(ir, fs, b) for b in bands}


# ---------------------------------------------------------------------------
# RT60 estimation
# ---------------------------------------------------------------------------
def rt60_from_schroeder(ir, fs, centre_hz, eval_range_db=(-5, -25)):
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
    if rt60_s is None or rt60_s <= 0:
        return None
    alpha = 0.161 * volume_m3 / (rt60_s * surface_area_m2)
    alpha = min(alpha, 0.999)
    return surface_area_m2 * alpha / (1.0 - alpha)


def rt60_per_band_from_irs(ir_list, fs, bands=OCTAVE_CENTRES):
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
    power_db = 20.0 * np.log10(np.abs(ir) / (np.max(np.abs(ir)) + 1e-30))
    candidates = np.where(power_db >= threshold_db)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def detect_first_reflection(ir, fs, direct_idx, min_gap_ms=2.0):
    min_gap = int(min_gap_ms * fs / 1000.0)
    search_start = direct_idx + min_gap
    direct_level = np.abs(ir[direct_idx])
    threshold = direct_level * 10.0 ** (-20.0 / 20.0)
    search = np.abs(ir[search_start:])
    peaks, _ = sig.find_peaks(search, height=threshold, distance=min_gap)
    if len(peaks) == 0:
        return None
    return int(search_start + peaks[0])


def gated_direct_field(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
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
    n_fft = int(2 ** np.ceil(np.log2(max(len(ir_gated), 2))))
    spectrum = np.fft.rfft(ir_gated, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitude = 20.0 * np.log10(np.abs(spectrum) + 1e-30)
    magnitude -= np.max(magnitude)
    return freqs, magnitude, gate_ms_used


def direct_field_at_bands(ir, fs, gate_ms=None, bands=OCTAVE_CENTRES):
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
        di[b] = float(d -
