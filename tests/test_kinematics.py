"""
Tests for kinematics.recover_kinematics().
"""

import numpy as np
import pytest
from kinematics import recover_kinematics


class TestRecoverKinematics:
    """Test suite for recover_kinematics function."""

    @pytest.fixture
    def simple_doppler_signal(self):
        """Create a simple linear Doppler signal for testing."""
        # Parameters
        fc = 5.8e9  # Hz
        c = 299792458.0  # m/s
        fs = 1e6  # Hz (1 MHz sample rate)
        duration = 1.0  # s

        # Time array
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        # Create ṙ that varies linearly: r_dot = 10 + 5*t m/s
        # This means r̈ = 5 m/s²
        r_dot_true = 10.0 + 5.0 * t  # m/s
        delta_f_true = -r_dot_true * fc / c  # Hz

        return t, delta_f_true, fc, c, r_dot_true

    def test_roundtrip_delta_f_to_r_dot(self, simple_doppler_signal):
        """Test round-trip: delta_f → r_dot recovers original."""
        t, delta_f, fc, c, r_dot_true = simple_doppler_signal

        result = recover_kinematics(t, delta_f, fc, c)

        # Check output keys
        assert "t_dot" in result
        assert "r_dot" in result
        assert "t_ddot" in result
        assert "r_ddot" in result

        # Check t_dot is same as input time
        np.testing.assert_array_equal(result["t_dot"], t)

        # r_dot should recover the true value (ignoring NaN edges)
        # The function sets NaN at edges due to smoother window
        r_dot_est = result["r_dot"]
        valid_mask = np.isfinite(r_dot_est)

        # Compare valid (non-NaN) estimates
        np.testing.assert_allclose(
            r_dot_est[valid_mask], r_dot_true[valid_mask], rtol=1e-10
        )

    def test_roundtrip_r_dot_to_r_ddot(self, simple_doppler_signal):
        """Test that r̈ is correctly computed from ṙ."""
        t, delta_f, fc, c, r_dot_true = simple_doppler_signal

        result = recover_kinematics(t, delta_f, fc, c)

        # With linear r_dot (r_dot = 10 + 5*t), r_ddot should be ~5 m/s²
        r_ddot = result["r_ddot"]
        t_ddot = result["t_ddot"]

        valid_mask = np.isfinite(r_ddot)

        # The decimated r_dot is averaged, so it should be approximately linear
        # After differentiation, we should get approximately 5 m/s²
        # Allow some tolerance due to smoothing and decimation
        if np.sum(valid_mask) > 0:
            r_ddot_valid = r_ddot[valid_mask]
            # The mean should be close to 5 m/s² (the true acceleration)
            mean_r_ddot = np.nanmean(r_ddot_valid)
            np.testing.assert_allclose(mean_r_ddot, 5.0, rtol=0.2)

    def test_nan_at_edges(self, simple_doppler_signal):
        """Test that NaN is present at edges due to smoother window."""
        t, delta_f, fc, c, _ = simple_doppler_signal

        result = recover_kinematics(t, delta_f, fc, c)

        r_dot = result["r_dot"]
        r_ddot = result["r_ddot"]

        # Check that NaN exists at edges for r_dot
        # Default PD_SMOOTH_WINDOW = 501, so trim = 250
        assert np.any(np.isnan(r_dot[:250]))
        assert np.any(np.isnan(r_dot[-250:]))

        # Check that NaN exists at edges for r_ddot
        # RDOT_SMOOTH_WINDOW = 21, so trim = 10
        valid_ddot = np.isfinite(r_ddot)
        if len(r_ddot) > 20:
            assert np.any(~valid_ddot[:10])
            assert np.any(~valid_ddot[-10:])

    def test_decimation_produces_correct_sample_rate(self, simple_doppler_signal):
        """Test that decimation produces the expected lower sample rate."""
        t, delta_f, fc, c, _ = simple_doppler_signal

        # Default RDOT_DECIM_HZ = 100 Hz
        from config import SimConfig
        decim_hz = SimConfig.RDOT_DECIM_HZ

        result = recover_kinematics(t, delta_f, fc, c)

        t_ddot = result["t_ddot"]

        if len(t_ddot) > 1:
            actual_fs = 1.0 / (t_ddot[1] - t_ddot[0])
            np.testing.assert_allclose(actual_fs, decim_hz, rtol=0.1)

    def test_zero_doppler(self):
        """Test with zero Doppler (stationary target)."""
        fc = 5.8e9
        c = 299792458.0
        fs = 1e6
        duration = 0.5
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs
        delta_f = np.zeros_like(t)

        result = recover_kinematics(t, delta_f, fc, c)

        r_dot = result["r_dot"]
        r_ddot = result["r_ddot"]

        # Valid (non-NaN) estimates should be near zero
        valid_rdot = r_dot[np.isfinite(r_dot)]
        np.testing.assert_allclose(valid_rdot, 0.0, atol=1e-6)

        valid_rddot = r_ddot[np.isfinite(r_ddot)]
        if len(valid_rddot) > 0:
            np.testing.assert_allclose(valid_rddot, 0.0, atol=1e-3)

    def test_constant_doppler(self):
        """Test with constant Doppler (constant range rate)."""
        fc = 5.8e9
        c = 299792458.0
        fs = 1e6
        duration = 0.5

        # Constant ṙ = 20 m/s → constant delta_f
        r_dot_true = 20.0
        delta_f = np.full(int(duration * fs), -r_dot_true * fc / c)

        t = np.arange(len(delta_f)) / fs

        result = recover_kinematics(t, delta_f, fc, c)

        r_dot = result["r_dot"]
        r_ddot = result["r_ddot"]

        # r_dot should be constant (with NaN edges)
        valid_rdot = r_dot[np.isfinite(r_dot)]
        np.testing.assert_allclose(valid_rdot, r_dot_true, rtol=1e-10)

        # r_ddot should be near zero (constant velocity)
        valid_rddot = r_ddot[np.isfinite(r_ddot)]
        if len(valid_rddot) > 0:
            np.testing.assert_allclose(valid_rddot, 0.0, atol=0.5)

    @pytest.mark.parametrize("duration", [0.1, 0.5, 1.0])
    @pytest.mark.slow
    def test_various_durations(self, duration):
        """Test recover_kinematics with various durations."""
        fc = 5.8e9
        c = 299792458.0
        fs = 1e6

        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs
        r_dot_true = 10.0 + 2.0 * t  # linear r_dot
        delta_f = -r_dot_true * fc / c

        result = recover_kinematics(t, delta_f, fc, c)

        # Should return all expected keys
        assert "t_dot" in result
        assert "r_dot" in result
        assert "t_ddot" in result
        assert "r_ddot" in result

        # t_ddot should have fewer samples than t_dot (decimated)
        assert len(result["t_ddot"]) < len(result["t_dot"])