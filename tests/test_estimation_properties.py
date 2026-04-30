"""
Hypothesis property-based tests for estimation functions.

Tests edge cases with randomly generated inputs to find corner cases
that regular unit tests might miss.
"""

import numpy as np
from hypothesis import Verbosity, given, settings, assume
import hypothesis.strategies as st

from config import ScenarioConfig, scenario_colocated, scenario_oncoming
from geometry import compute_geometry
from iq_gen import generate_iq
from estimation import estimate_doppler_phase_diff


# =============================================================================
# Strategies
# =============================================================================

def snr_strategy():
    """Generate valid SNR values in dB."""
    return st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)


def sample_rate_strategy():
    """Generate valid sample rates."""
    return st.floats(min_value=1e3, max_value=50_000.0, allow_nan=False, allow_infinity=False)


def smooth_window_strategy():
    """Generate valid smooth window sizes (odd positive integers)."""
    return st.integers(min_value=3, max_value=1001)


def carrier_freq_strategy():
    """Generate valid carrier frequencies."""
    return st.floats(min_value=1e9, max_value=100e9, allow_nan=False, allow_infinity=False)


def time_duration_strategy():
    """Generate valid time durations."""
    return st.floats(min_value=0.02, max_value=0.2, allow_nan=False, allow_infinity=False)


def fast_cfg(cfg: ScenarioConfig) -> ScenarioConfig:
    """Keep property tests fast; full-rate simulations are integration tests."""
    cfg.fs = min(float(cfg.fs), 50_000.0)
    cfg.duration = min(float(cfg.duration), 0.2)
    cfg.interp_oversample = min(int(cfg.interp_oversample), 2)
    return cfg


# =============================================================================
# Property: estimate always returns valid arrays
# =============================================================================

@given(
    snr_db=snr_strategy(),
    fc=carrier_freq_strategy(),
    duration=time_duration_strategy(),
)
@settings(deadline=None, max_examples=50, verbosity=Verbosity.normal)
def test_estimate_returns_valid_arrays(snr_db, fc, duration):
    """
    Property: estimate_doppler_phase_diff always returns valid (t, delta_f) arrays.
    
    Regardless of SNR or signal characteristics, the function should return
    properly shaped arrays.
    """
    cfg = ScenarioConfig(
        fc=fc,
        fs=20_000.0,
        duration=duration,
        snr_db=snr_db,
        tx_x0=100.0,
        tx_y=3.7,
        tx_vx=-30.0,
        rx_x0=0.0,
        rx_y=0.0,
        rx_vx=30.0,
    )
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs)
    
    assert isinstance(t, np.ndarray), "t should be numpy array"
    assert isinstance(delta_f, np.ndarray), "delta_f should be numpy array"
    assert len(t) == len(delta_f), "t and delta_f should have same length"
    assert len(t) > 0, "Output should not be empty"


# =============================================================================
# Property: no NaN in output for various SNR levels
# =============================================================================

@given(
    snr_db=snr_strategy(),
    scenario=st.sampled_from(["oncoming", "colocated"]),
)
@settings(deadline=None, max_examples=50, verbosity=Verbosity.normal)
def test_no_nan_at_various_snr(snr_db, scenario):
    """
    Property: No NaN values in estimation output regardless of SNR.
    
    The estimation should handle both high and low SNR gracefully without
    producing NaN values in the output.
    """
    if scenario == "oncoming":
        cfg = fast_cfg(scenario_oncoming())
    else:
        cfg = fast_cfg(scenario_colocated())
    
    cfg.snr_db = snr_db
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs)
    
    assert not np.any(np.isnan(t)), f"NaN found in t output at SNR={snr_db}dB"
    assert not np.any(np.isnan(delta_f)), f"NaN found in delta_f output at SNR={snr_db}dB"


# =============================================================================
# Property: output length is n-1 for input length n
# =============================================================================

@given(
    snr_db=st.sampled_from([10.0, 30.0]),
    duration=time_duration_strategy(),
    fs=sample_rate_strategy(),
)
@settings(deadline=None, max_examples=30, verbosity=Verbosity.normal)
def test_output_length_consistent(snr_db, duration, fs):
    """
    Property: Output length is exactly input_length - 1.
    
    The phase-difference method requires n-1 samples to compute n-1 phase differences.
    """
    assume(duration * fs >= 10)  # Need at least 10 samples
    
    cfg = ScenarioConfig(
        fc=5.8e9,
        fs=fs,
        duration=duration,
        snr_db=snr_db,
    )
    
    data = generate_iq(cfg)
    iq = data["iq"]
    
    n_input = len(iq)
    t, delta_f = estimate_doppler_phase_diff(iq, fs)
    
    expected_len = n_input - 1
    assert len(t) == expected_len, f"Expected length {expected_len}, got {len(t)}"
    assert len(delta_f) == expected_len, f"Expected length {expected_len}, got {len(delta_f)}"


# =============================================================================
# Property: handles different smooth window sizes
# =============================================================================

@given(
    snr_db=st.sampled_from([10.0, 30.0]),
    smooth_window=smooth_window_strategy(),
)
@settings(deadline=None, max_examples=30, verbosity=Verbosity.normal)
def test_handles_different_smooth_windows(snr_db, smooth_window):
    """
    Property: Function works correctly with various smooth_window sizes.
    """
    cfg = fast_cfg(scenario_oncoming())
    cfg.snr_db = snr_db
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs, smooth_window=smooth_window)
    
    # Should return valid arrays
    assert len(t) > 0
    assert len(delta_f) > 0
    assert not np.any(np.isnan(delta_f))


# =============================================================================
# Property: delta_f values are in reasonable range
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0]),
    tx_vx=st.floats(min_value=-50.0, max_value=50.0),
    rx_vx=st.floats(min_value=-50.0, max_value=50.0),
)
@settings(deadline=None, max_examples=30, verbosity=Verbosity.normal)
def test_doppler_in_reasonable_range(snr_db, tx_vx, rx_vx):
    """
    Property: Estimated delta_f should be in reasonable range based on velocities.
    
    For typical scenarios, the Doppler shift should be bounded by the
    maximum possible relative velocity times fc/c.
    """
    max_relative_velocity = 100.0  # m/s
    fc = 5.8e9
    c = 299792458.0
    max_doppler = max_relative_velocity * fc / c
    
    cfg = ScenarioConfig(
        fc=fc,
        fs=20_000.0,
        duration=0.2,
        snr_db=60.0,
        tx_x0=200.0,
        tx_y=3.7,
        tx_vx=tx_vx,
        rx_x0=0.0,
        rx_y=0.0,
        rx_vx=rx_vx,
    )
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs, smooth_window=101)
    
    # At valid (non-NaN) positions, values should be reasonable
    trim = 101 // 2
    valid_mask = np.isfinite(delta_f)
    if len(delta_f) > 2 * trim:
        valid_mask[:trim] = False
        valid_mask[-trim:] = False
    if np.any(valid_mask):
        # Allow some tolerance for noise
        assert np.all(np.abs(delta_f[valid_mask]) < max_doppler * 2), \
            f"delta_f exceeds expected range: {np.max(np.abs(delta_f[valid_mask]))}"


# =============================================================================
# Property: handles very short signals
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0]),
    n_samples=st.integers(min_value=2, max_value=10),
)
@settings(deadline=None, max_examples=20, verbosity=Verbosity.normal)
def test_handles_short_signals(snr_db, n_samples):
    """
    Property: Function handles very short input signals gracefully.
    """
    fs = 20_000.0
    iq = np.ones(n_samples, dtype=complex)
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs)
    
    # Should return valid output (may be empty or single element)
    assert isinstance(t, np.ndarray)
    assert isinstance(delta_f, np.ndarray)


# =============================================================================
# Property: output time array is properly spaced
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0]),
    fs=sample_rate_strategy(),
)
@settings(deadline=None, max_examples=20, verbosity=Verbosity.normal)
def test_time_array_spacing(snr_db, fs):
    """
    Property: Output time array should be properly spaced at 1/fs intervals.
    """
    assume(fs >= 1000)  # Ensure reasonable sample rate
    
    cfg = ScenarioConfig(
        fc=5.8e9,
        fs=fs,
        duration=0.1,
        snr_db=snr_db,
        tx_x0=100.0,
        tx_y=3.7,
        tx_vx=-30.0,
        rx_x0=0.0,
        rx_y=0.0,
        rx_vx=30.0,
    )
    
    data = generate_iq(cfg)
    iq = data["iq"]
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs)
    
    if len(t) > 1:
        expected_dt = 1.0 / fs
        actual_dt = t[1] - t[0]
        np.testing.assert_allclose(actual_dt, expected_dt, rtol=1e-10)


# =============================================================================
# Property: handles zero signal (edge case)
# =============================================================================

def test_handles_zero_signal():
    """
    Property: Function handles zero signal gracefully.
    
    While not a typical use case, the function should not crash
    on all-zero input.
    """
    n_samples = 100
    fs = 1e6
    iq = np.zeros(n_samples, dtype=complex)
    
    t, delta_f = estimate_doppler_phase_diff(iq, fs)
    
    # Should return arrays, though values may be undefined
    assert len(t) == n_samples - 1
    assert len(delta_f) == n_samples - 1


# =============================================================================
# Property: handles constant phase signal
# =============================================================================

@given(
    fs=sample_rate_strategy(),
    smooth_window=smooth_window_strategy(),
)
@settings(deadline=None, max_examples=20, verbosity=Verbosity.normal)
def test_handles_constant_phase_signal(fs, smooth_window):
    """
    Property: Function handles constant phase (zero frequency) signal.
    """
    n_samples = int(0.1 * fs)
    t = np.arange(n_samples) / fs
    
    # Constant phase = zero frequency
    iq = np.exp(1j * np.zeros(n_samples))
    
    t_out, delta_f = estimate_doppler_phase_diff(iq, fs, smooth_window=smooth_window)
    
    # Should return valid output
    assert len(t_out) > 0
    assert len(delta_f) > 0
    # Zero frequency should give near-zero Doppler estimate
    np.testing.assert_allclose(delta_f, 0.0, atol=1.0)
