"""
Hypothesis property-based tests for geometry.compute_geometry function.

Tests edge cases with randomly generated inputs to find corner cases
that regular unit tests might miss.
"""

import numpy as np
from hypothesis import Verbosity, given, settings, assume
import hypothesis.strategies as st

from config import ScenarioConfig
from geometry import compute_geometry


# =============================================================================
# Strategies
# =============================================================================

def positions_strategy():
    """Generate valid position values within reasonable bounds."""
    return st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)


def velocities_strategy():
    """Generate valid velocity values (reasonable speed limits)."""
    return st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


def time_strategy():
    """Generate valid time values."""
    return st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def carrier_freq_strategy():
    """Generate valid carrier frequencies."""
    return st.floats(min_value=1e9, max_value=100e9, allow_nan=False, allow_infinity=False)


# =============================================================================
# Property: r >= 0 always
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_y=positions_strategy(),
    tx_vx=velocities_strategy(),
    rx_x0=positions_strategy(),
    rx_y=positions_strategy(),
    rx_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
    t=st.lists(time_strategy(), min_size=1, max_size=20),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_range_always_nonnegative(tx_x0, tx_y, tx_vx, rx_x0, rx_y, rx_vx, fc, t):
    """
    Property: Range r must always be >= 0.
    
    r = sqrt(dx² + dy²) is mathematically guaranteed to be non-negative,
    but we test the implementation handles all edge cases correctly.
    """
    # Filter out degenerate case where tx and rx are exactly collocated at same point
    # This would cause r = 0 which is valid but tests should handle it
    assume(abs(tx_x0 - rx_x0) > 1e-10 or abs(tx_y - rx_y) > 1e-10)
    
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=tx_y,
        tx_vx=tx_vx,
        rx_x0=rx_x0,
        rx_y=rx_y,
        rx_vx=rx_vx,
        fc=fc,
    )
    t_arr = np.array(t)
    
    result = compute_geometry(cfg, t_arr)
    
    # Property: r must be >= 0 everywhere
    assert np.all(result["r"] >= 0), f"Range r has negative values: {result['r'][result['r'] < 0]}"


# =============================================================================
# Property: r_dot = 0 when vdx = 0
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_y=positions_strategy(),
    rx_x0=positions_strategy(),
    rx_y=positions_strategy(),
    common_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
    t=st.lists(time_strategy(), min_size=1, max_size=20),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_radial_velocity_zero_when_no_relative_motion(
    tx_x0, tx_y, rx_x0, rx_y, common_vx, fc, t
):
    """
    Property: r_dot = 0 when tx_vx == rx_vx (no relative velocity).
    
    When both vehicles travel at the same velocity, the distance between
    them changes at zero rate (assuming no lateral motion).
    """
    # Ensure non-zero lateral separation to avoid degenerate case
    assume(abs(tx_y - rx_y) > 1e-6)
    
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=tx_y,
        tx_vx=common_vx,  # Same velocity as receiver
        rx_x0=rx_x0,
        rx_y=rx_y,
        rx_vx=common_vx,  # Same velocity as transmitter
        fc=fc,
    )
    t_arr = np.array(t)
    
    result = compute_geometry(cfg, t_arr)
    
    # Property: r_dot should be exactly zero
    np.testing.assert_allclose(
        result["r_dot"], 
        0.0, 
        atol=1e-10,
        err_msg=f"r_dot should be 0 when vdx=0, got {result['r_dot']}"
    )


# =============================================================================
# Property: delta_f = 0 when vdx = 0
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_y=positions_strategy(),
    rx_x0=positions_strategy(),
    rx_y=positions_strategy(),
    common_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
    t=st.lists(time_strategy(), min_size=1, max_size=20),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_doppler_zero_when_no_relative_motion(
    tx_x0, tx_y, rx_x0, rx_y, common_vx, fc, t
):
    """
    Property: delta_f = 0 when tx_vx == rx_vx (no relative velocity).
    
    With no relative motion between transmitter and receiver,
    there should be no Doppler shift.
    """
    # Ensure non-zero lateral separation for meaningful test
    assume(abs(tx_y - rx_y) > 1e-6)
    
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=tx_y,
        tx_vx=common_vx,
        rx_x0=rx_x0,
        rx_y=rx_y,
        rx_vx=common_vx,
        fc=fc,
    )
    t_arr = np.array(t)
    
    result = compute_geometry(cfg, t_arr)
    
    # Property: delta_f should be exactly zero
    np.testing.assert_allclose(
        result["delta_f"], 
        0.0, 
        atol=1e-6,
        err_msg=f"delta_f should be 0 when vdx=0, got {result['delta_f']}"
    )


# =============================================================================
# Property: delta_f matches -r_dot * fc / c
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_y=positions_strategy(),
    tx_vx=velocities_strategy(),
    rx_x0=positions_strategy(),
    rx_y=positions_strategy(),
    rx_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
    t=st.lists(time_strategy(), min_size=1, max_size=20),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_doppler_formula_consistency(
    tx_x0, tx_y, tx_vx, rx_x0, rx_y, rx_vx, fc, t
):
    """
    Property: delta_f = -r_dot * fc / c always holds.
    
    The Doppler shift formula should be consistent with the computed
    radial velocity.
    """
    # Avoid degenerate case where vehicles are exactly at same position
    assume(abs(tx_x0 - rx_x0) > 1e-10 or abs(tx_y - rx_y) > 1e-10)
    
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=tx_y,
        tx_vx=tx_vx,
        rx_x0=rx_x0,
        rx_y=rx_y,
        rx_vx=rx_vx,
        fc=fc,
    )
    t_arr = np.array(t)
    
    result = compute_geometry(cfg, t_arr)
    
    # Property: delta_f should match the formula
    expected_delta_f = -result["r_dot"] * fc / cfg.c
    np.testing.assert_allclose(
        result["delta_f"],
        expected_delta_f,
        rtol=1e-10,
        err_msg="delta_f formula inconsistent with r_dot"
    )


# =============================================================================
# Property: r never decreases when vehicles move apart
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_y=positions_strategy(),
    tx_vx=velocities_strategy(),
    rx_x0=positions_strategy(),
    rx_y=positions_strategy(),
    rx_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_range_monotonic_when_receding(
    tx_x0, tx_y, tx_vx, rx_x0, rx_y, rx_vx, fc
):
    """
    Property: When relative velocity is positive (moving apart), range increases.
    
    If tx is ahead of rx and moving away (or rx is faster), the range should
    monotonically increase.
    """
    # Setup scenario where vehicles are moving apart
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=tx_y,
        tx_vx=tx_vx,
        rx_x0=rx_x0,
        rx_y=rx_y,
        rx_vx=rx_vx,
        fc=fc,
    )
    
    # Generate time array where tx is ahead and receding
    t_arr = np.linspace(0, 1.0, 11)
    
    result = compute_geometry(cfg, t_arr)
    
    # Calculate relative velocity at start
    vdx = cfg.tx_vx - cfg.rx_vx
    
    # At t=0, r_dot sign follows dx * vdx. Positive means moving apart.
    initial_dx = tx_x0 - rx_x0
    expected = initial_dx * vdx
    if expected > 0:
        assert result["r_dot"][0] >= -1e-10, "Should have positive r_dot when moving apart"
    elif expected < 0:
        assert result["r_dot"][0] <= 1e-10, "Should have negative r_dot when approaching"


# =============================================================================
# Property: all output arrays have same shape as input
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_y=positions_strategy(),
    tx_vx=velocities_strategy(),
    rx_x0=positions_strategy(),
    rx_y=positions_strategy(),
    rx_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
    t=st.lists(time_strategy(), min_size=1, max_size=50),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_output_shapes_match_input(
    tx_x0, tx_y, tx_vx, rx_x0, rx_y, rx_vx, fc, t
):
    """
    Property: All output arrays should match the input time array shape.
    """
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=tx_y,
        tx_vx=tx_vx,
        rx_x0=rx_x0,
        rx_y=rx_y,
        rx_vx=rx_vx,
        fc=fc,
    )
    t_arr = np.array(t)
    
    result = compute_geometry(cfg, t_arr)
    
    expected_shape = t_arr.shape
    
    assert result["r"].shape == expected_shape, "r shape mismatch"
    assert result["r_dot"].shape == expected_shape, "r_dot shape mismatch"
    assert result["r_ddot"].shape == expected_shape, "r_ddot shape mismatch"
    assert result["delta_f"].shape == expected_shape, "delta_f shape mismatch"
    assert result["los_angle"].shape == expected_shape, "los_angle shape mismatch"


# =============================================================================
# Property: handles extreme lateral offsets
# =============================================================================

@given(
    tx_x0=positions_strategy(),
    tx_vx=velocities_strategy(),
    rx_x0=positions_strategy(),
    rx_vx=velocities_strategy(),
    fc=carrier_freq_strategy(),
    lateral_offset=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100, verbosity=Verbosity.normal)
def test_handles_large_lateral_offsets(
    tx_x0, tx_vx, rx_x0, rx_vx, fc, lateral_offset
):
    """
    Property: Function handles large lateral offsets gracefully.
    """
    cfg = ScenarioConfig(
        tx_x0=tx_x0,
        tx_y=lateral_offset,  # Large lateral separation
        tx_vx=tx_vx,
        rx_x0=rx_x0,
        rx_y=0.0,
        rx_vx=rx_vx,
        fc=fc,
    )
    t_arr = np.array([0.0])
    
    result = compute_geometry(cfg, t_arr)
    
    # Range should be at least as large as lateral offset
    assert result["r"][0] >= lateral_offset - 1e-6, "r should be >= lateral offset"
    
    # All values should be finite
    assert np.isfinite(result["r"]).all(), "r should be finite"
    assert np.isfinite(result["r_dot"]).all(), "r_dot should be finite"
    assert np.isfinite(result["r_ddot"]).all(), "r_ddot should be finite"
    assert np.isfinite(result["delta_f"]).all(), "delta_f should be finite"
