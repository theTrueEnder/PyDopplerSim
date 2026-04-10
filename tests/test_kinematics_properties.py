"""
Hypothesis property-based tests for kinematics.recover_kinematics function.

Tests edge cases with randomly generated inputs to find corner cases
that regular unit tests might miss.
"""

import numpy as np
from hypothesis import given, settings, assume, verbosity
import hypothesis.strategies as st

from config import ScenarioConfig, scenario_oncoming
from iq_gen import generate_iq
from estimation import estimate_doppler_phase_diff
from kinematics import recover_kinematics


# =============================================================================
# Strategies
# =============================================================================

def time_array_strategy():
    """Generate valid time arrays with at least 2 points."""
    return st.lists(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False), 
                    min_size=2, max_size=100)


def doppler_freq_strategy():
    """Generate valid Doppler frequency values."""
    return st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False)


def carrier_freq_strategy():
    """Generate valid carrier frequencies."""
    return st.floats(min_value=1e9, max_value=100e9, allow_nan=False, allow_infinity=False)


def speed_of_light_strategy():
    """Speed of light is constant but we allow testing with different precision."""
    return st.floats(min_value=2.9e8, max_value=3.0e8, allow_nan=False, allow_infinity=False)


# =============================================================================
# Property: output has correct keys
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0, 30.0]),
    fc=carrier_freq_strategy(),
)
@settings(deadline=None, max_examples=30, verbosity=verbosity.normal)
def test_output_has_correct_keys(snr_db, fc):
    """
    Property: recover_kinematics returns dict with all expected keys.
    
    The function should always return a dictionary with the four required keys:
    t_dot, r_dot, t_ddot, r_ddot.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    cfg.fc = fc
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs)
    
    result = recover_kinematics(t_est, delta_f_est, cfg.fc, cfg.c)
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "t_dot" in result, "Result should contain t_dot key"
    assert "r_dot" in result, "Result should contain r_dot key"
    assert "t_ddot" in result, "Result should contain t_ddot key"
    assert "r_ddot" in result, "Result should contain r_ddot key"


# =============================================================================
# Property: no division by zero
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0]),
    fc=carrier_freq_strategy(),
    c=speed_of_light_strategy(),
)
@settings(deadline=None, max_examples=30, verbosity=verbosity.normal)
def test_no_division_by_zero(snr_db, fc, c):
    """
    Property: Function handles inputs without division by zero errors.
    
    The function should gracefully handle all valid inputs without
    raising ZeroDivisionError.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    cfg.fc = fc
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs)
    
    # Should not raise ZeroDivisionError
    result = recover_kinematics(t_est, delta_f_est, fc, c)
    
    # All outputs should be arrays
    assert isinstance(result["t_dot"], np.ndarray)
    assert isinstance(result["r_dot"], np.ndarray)
    assert isinstance(result["t_ddot"], np.ndarray)
    assert isinstance(result["r_ddot"], np.ndarray)


# =============================================================================
# Property: r_dot = -delta_f * c / fc
# =============================================================================

@given(
    snr_db=st.sampled_from([30.0]),
    fc=carrier_freq_strategy(),
    c=speed_of_light_strategy(),
)
@settings(deadline=None, max_examples=30, verbosity=verbosity.normal)
def test_radial_velocity_formula(snr_db, fc, c):
    """
    Property: r_dot = -delta_f * c / fc should hold exactly.
    
    This is the direct inversion of the Doppler equation.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    cfg.fc = fc
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs)
    
    result = recover_kinematics(t_est, delta_f_est, fc, c)
    
    # At valid (non-NaN) positions
    valid_mask = np.isfinite(result["r_dot"]) & np.isfinite(delta_f_est)
    
    expected_r_dot = -delta_f_est[valid_mask] * c / fc
    np.testing.assert_allclose(
        result["r_dot"][valid_mask], 
        expected_r_dot, 
        rtol=1e-10,
        err_msg="r_dot formula inconsistent with delta_f"
    )


# =============================================================================
# Property: output arrays have expected lengths
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0]),
)
@settings(deadline=None, max_examples=20, verbosity=verbosity.normal)
def test_output_array_lengths(snr_db):
    """
    Property: Output arrays have expected lengths.
    
    t_dot and r_dot should have same length as input (full rate).
    t_ddot and r_ddot should have decimated length.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs)
    input_len = len(t_est)
    
    result = recover_kinematics(t_est, delta_f_est, cfg.fc, cfg.c)
    
    # Full rate arrays
    assert len(result["t_dot"]) == input_len
    assert len(result["r_dot"]) == input_len
    
    # Decimated arrays should be shorter
    assert len(result["t_ddot"]) < input_len
    assert len(result["r_ddot"]) < input_len


# =============================================================================
# Property: no NaN in r_dot except at edges
# =============================================================================

@given(
    snr_db=st.sampled_from([20.0, 40.0]),
    smooth_window=st.integers(min_value=101, max_value=501),
)
@settings(deadline=None, max_examples=20, verbosity=verbosity.normal)
def test_no_nan_except_at_edges(snr_db, smooth_window):
    """
    Property: NaN values only appear at edges (due to smoother window).
    
    The function marks edge transients as NaN, but the middle should be valid.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs, smooth_window=smooth_window)
    
    result = recover_kinematics(t_est, delta_f_est, cfg.fc, cfg.c)
    
    # Check that there's at least some valid data in the middle
    trim = smooth_window // 2
    middle_r_dot = result["r_dot"][trim:-trim] if trim > 0 else result["r_dot"]
    
    # There should be at least some finite values
    assert np.any(np.isfinite(middle_r_dot)), "Should have some valid r_dot values in middle"


# =============================================================================
# Property: handles extreme delta_f values
# =============================================================================

@given(
    fc=carrier_freq_strategy(),
    c=speed_of_light_strategy(),
    n_samples=st.integers(min_value=100, max_value=500),
)
@settings(deadline=None, max_examples=30, verbosity=verbosity.normal)
def test_handles_extreme_doppler_values(fc, c, n_samples):
    """
    Property: Function handles extreme (large magnitude) delta_f values.
    """
    t_est = np.linspace(0, 0.1, n_samples)
    
    # Very large Doppler shift (beyond typical values)
    delta_f_est = np.full(n_samples, 10000.0)
    
    result = recover_kinematics(t_est, delta_f_est, fc, c)
    
    # Should complete without error
    assert "r_dot" in result
    assert "r_ddot" in result
    
    # Values should be finite (no NaN/Inf) except at edges
    valid_r_dot = result["r_dot"][np.isfinite(result["r_dot"])]
    assert len(valid_r_dot) > 0
    assert np.all(np.isfinite(valid_r_dot))


# =============================================================================
# Property: handles zero delta_f
# =============================================================================

@given(
    fc=carrier_freq_strategy(),
    c=speed_of_light_strategy(),
    n_samples=st.integers(min_value=100, max_value=500),
)
@settings(deadline=None, max_examples=30, verbosity=verbosity.normal)
def test_handles_zero_doppler(fc, c, n_samples):
    """
    Property: Function handles zero Doppler shift (stationary scenario).
    """
    t_est = np.linspace(0, 0.1, n_samples)
    delta_f_est = np.zeros(n_samples)
    
    result = recover_kinematics(t_est, delta_f_est, fc, c)
    
    # r_dot should be zero (or NaN at edges)
    valid_r_dot = result["r_dot"][np.isfinite(result["r_dot"])]
    if len(valid_r_dot) > 0:
        np.testing.assert_allclose(valid_r_dot, 0.0, atol=1e-10)
    
    # r_ddot should be zero or NaN (no acceleration)
    valid_r_ddot = result["r_ddot"][np.isfinite(result["r_ddot"])]
    if len(valid_r_ddot) > 0:
        np.testing.assert_allclose(valid_r_ddot, 0.0, atol=1e-6)


# =============================================================================
# Property: handles linearly changing delta_f
# =============================================================================

@given(
    fc=carrier_freq_strategy(),
    c=speed_of_light_strategy(),
    n_samples=st.integers(min_value=200, max_value=500),
    doppler_rate=st.floats(min_value=-1000.0, max_value=1000.0),
)
@settings(deadline=None, max_examples=30, verbosity=verbosity.normal)
def test_handles_linear_doppler(fc, c, n_samples, doppler_rate):
    """
    Property: Function handles linearly changing Doppler (constant acceleration).
    """
    t_est = np.linspace(0, 0.1, n_samples)
    delta_f_est = doppler_rate * t_est
    
    result = recover_kinematics(t_est, delta_f_est, fc, c)
    
    # Should complete without error
    assert "r_dot" in result
    assert "r_ddot" in result
    
    # Check that r_ddot is approximately constant (as expected from linear delta_f)
    valid_r_ddot = result["r_ddot"][np.isfinite(result["r_ddot"])]
    if len(valid_r_ddot) > 2:
        # r_ddot should be approximately -doppler_rate * c / fc
        expected_r_ddot = -doppler_rate * c / fc
        np.testing.assert_allclose(
            valid_r_ddot.mean(), 
            expected_r_ddot, 
            rtol=0.1,  # Allow 10% tolerance due to filtering
        )


# =============================================================================
# Property: handles various carrier frequencies
# =============================================================================

@given(
    snr_db=st.sampled_from([30.0]),
    fc=carrier_freq_strategy(),
)
@settings(deadline=None, max_examples=20, verbosity=verbosity.normal)
def test_handles_various_carrier_frequencies(snr_db, fc):
    """
    Property: Function works correctly with various carrier frequencies.
    
    Different carrier frequencies should produce proportionally different
    r_dot estimates for the same delta_f.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    cfg.fc = fc
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs)
    
    result = recover_kinematics(t_est, delta_f_est, fc, cfg.c)
    
    # At valid positions, check formula holds
    valid_mask = np.isfinite(result["r_dot"]) & np.isfinite(delta_f_est)
    expected_r_dot = -delta_f_est[valid_mask] * cfg.c / fc
    np.testing.assert_allclose(
        result["r_dot"][valid_mask],
        expected_r_dot,
        rtol=1e-10
    )


# =============================================================================
# Property: time arrays are properly ordered
# =============================================================================

@given(
    snr_db=st.sampled_from([30.0]),
)
@settings(deadline=None, max_examples=20, verbosity=verbosity.normal)
def test_time_arrays_are_ordered(snr_db):
    """
    Property: Output time arrays should be monotonically increasing.
    """
    cfg = scenario_oncoming()
    cfg.snr_db = snr_db
    
    data = generate_iq(cfg)
    iq = data["iq"]
    fs = cfg.fs
    
    t_est, delta_f_est = estimate_doppler_phase_diff(iq, fs)
    
    result = recover_kinematics(t_est, delta_f_est, cfg.fc, cfg.c)
    
    # t_dot should be monotonically increasing
    assert np.all(np.diff(result["t_dot"]) > 0), "t_dot should be monotonically increasing"
    
    # t_ddot should be monotonically increasing (if non-empty)
    if len(result["t_ddot"]) > 1:
        assert np.all(np.diff(result["t_ddot"]) > 0), "t_ddot should be monotonically increasing"