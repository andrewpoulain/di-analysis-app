# ---------------------------------------------------------------------------
# Third octave band definitions
# ---------------------------------------------------------------------------

THIRD_OCTAVE_CENTRES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000
])


def third_octave_band_limits(centre_hz):
    """Return (f_low, f_high) for a 1/3-octave band centred at centre_hz."""
    return centre_hz / (2 ** (1.0 / 6)), centre_hz * (2 ** (1.0 / 6))


def direct_field_at_third_octave_bands(ir, fs, gate_ms=None,
                                        bands=THIRD_OCTAVE_CENTRES):
    """
    Return the mean direct field level in each 1/3-octave band.
    Uses the same gated direct field spectrum as the octave band
    version but averages into 1/3-octave bins.
    Returns levels dict and gate_ms_used.
    """
    freqs, magnitude, gate_ms_used = gated_direct_field(ir, fs, gate_ms)
    levels = {}
    for b in bands:
        f_low, f_high = third_octave_band_limits(b)
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.sum() > 0:
            power = 10.0 ** (magnitude[mask] / 10.0)
            levels[float(b)] = float(10.0 * np.log10(np.mean(power)))
        else:
            levels[float(b)] = np.nan
    return levels, gate_ms_used


def reverberant_spectrum_third_octave(ir, fs,
                                       bands=THIRD_OCTAVE_CENTRES):
    """
    Return the Schroeder initial decay level for each 1/3-octave band.
    """
    result = {}
    for b in bands:
        f_low, f_high = third_octave_band_limits(b)
        ir_band = bandpass_ir(ir, fs, f_low, f_high)
        direct_idx = detect_direct_arrival(ir_band, fs,
                                            threshold_db=-20)
        ir_band = ir_band[direct_idx:]
        ir_band = truncate_to_noise_floor(ir_band)
        decay_db = schroeder_decay(ir_band)
        n_avg = max(1, int(0.005 * fs))
        result[float(b)] = float(np.mean(decay_db[:n_avg]))
    return result


def spatial_average_reverberant_third_octave(ir_list, fs,
                                              bands=THIRD_OCTAVE_CENTRES):
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


def predict_post_eq_steady_state_third_octave(
        direct_levels_3rd, reverberant_levels_3rd,
        all_corrections_octave,
        bands=THIRD_OCTAVE_CENTRES):
    """
    Predict the steady-state response in 1/3-octave bands after
    applying the octave band EQ corrections.

    The octave band corrections are interpolated to 1/3-octave
    resolution in log frequency space before being applied.
    """
    # Interpolate octave band corrections to 1/3-octave centres
    oct_bands = sorted(all_corrections_octave.keys())
    oct_corr = [all_corrections_octave[b] for b in oct_bands]

    band_floats = [float(b) for b in bands]
    log_oct = np.log10([float(b) for b in oct_bands])
    log_third = np.log10(band_floats)
    interp_corr = np.interp(log_third, log_oct, oct_corr,
                             left=oct_corr[0], right=oct_corr[-1])

    predicted = {}
    for i, b in enumerate(band_floats):
        d = direct_levels_3rd.get(b, np.nan)
        r = reverberant_levels_3rd.get(b, np.nan)
        c = float(interp_corr[i])
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
# X-curve target (SMPTE ST 202M / ISO 2969)
# ---------------------------------------------------------------------------

def xcurve_target(freqs_hz, screen_size='large'):
    """
    Generate the X-curve target level at each frequency in freqs_hz.

    SMPTE ST 202M / ISO 2969 X-curve definition:
      Flat from 2 Hz to 2 kHz (0 dB reference)
      Roll off above 2 kHz at -3 dB per octave
      Roll off below 63 Hz at -3 dB per octave

    Two variants are provided:
      large:  Standard X-curve for rooms > 150 m3
              (large dubbing stages and cinemas)
      small:  Modified X-curve for rooms < 150 m3
              Flat extended to 4 kHz before rolloff
              as per SMPTE RP 200

    Returns array of target levels in dB at each frequency in freqs_hz.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    target = np.zeros_like(freqs)

    if screen_size == 'small':
        # Modified X-curve: flat to 4 kHz, -3 dB/oct above
        hf_corner = 4000.0
    else:
        # Standard X-curve: flat to 2 kHz, -3 dB/oct above
        hf_corner = 2000.0

    lf_corner = 63.0

    for i, f in enumerate(freqs):
        if f <= 0:
            target[i] = np.nan
            continue
        level = 0.0
        # HF rolloff above corner frequency
        if f > hf_corner:
            octaves_above = np.log2(f / hf_corner)
            level -= 3.0 * octaves_above
        # LF rolloff below 63 Hz
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
    levels = xcurve_target(np.array(bands, dtype=float),
                            screen_size=screen_size)
    return {float(b): float(l)
            for b, l in zip(bands, levels)
            if not np.isnan(l)}
