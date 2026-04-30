"""
Tests for geometry.compute_geometry function.
"""

import numpy as np
import pytest

from config import ScenarioConfig
from geometry import compute_geometry


class TestRange:
    """Test r = sqrt(dx² + dy²) for known positions."""

    def test_horizontal_range(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["r"], 100.0)

    def test_lateral_range(self):
        cfg = ScenarioConfig(
            tx_x0=0.0,
            tx_y=10.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["r"], 10.0)

    def test_diagonal_range(self):
        cfg = ScenarioConfig(
            tx_x0=3.0,
            tx_y=4.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["r"], 5.0)

    def test_range_changes_with_time(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=10.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0, 1.0, 2.0])
        result = compute_geometry(cfg, t)
        expected = np.array([100.0, 110.0, 120.0])
        np.testing.assert_allclose(result["r"], expected)


class TestRadialVelocity:
    """Test r_dot = dx*vdx/r for known velocities."""

    def test_stationary_range(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["r_dot"], 0.0)

    def test_approaching(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=-10.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        # r_dot negative when approaching (moving apart = positive)
        assert np.isclose(result["r_dot"], -10.0)

    def test_receding(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=10.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        # r_dot positive when receding (moving apart)
        assert np.isclose(result["r_dot"], 10.0)

    def test_overtake_scenario(self):
        cfg = ScenarioConfig(
            tx_x0=50.0,
            tx_y=0.0,
            tx_vx=20.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=30.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        dx = 50.0 - 0.0
        vdx = 20.0 - 30.0
        r = np.sqrt(dx**2 + 0.0**2)
        expected_r_dot = (dx * vdx) / r
        assert np.isclose(result["r_dot"], expected_r_dot)


class TestRadialAcceleration:
    """Test r_ddot formula at key points."""

    def test_constant_range_at_cpa(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=10.0,
            tx_vx=-30.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=30.0,
        )
        t = np.array([100.0 / 60.0])
        result = compute_geometry(cfg, t)
        expected = (cfg.tx_vx - cfg.rx_vx) ** 2 / abs(cfg.tx_y - cfg.rx_y)
        np.testing.assert_allclose(result["r_ddot"], expected)

    def test_positive_acceleration_receding(self):
        cfg = ScenarioConfig(
            tx_x0=0.0,
            tx_y=10.0,
            tx_vx=0.0,
            rx_x0=50.0,
            rx_y=0.0,
            rx_vx=30.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert result["r_ddot"] > 0.0


class TestDopplerShift:
    """Test delta_f = -r_dot * fc / c."""

    def test_zero_doppler_stationary(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
            fc=1e9,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["delta_f"], 0.0, atol=1e-3)

    def test_positive_doppler_approaching(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=-30.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
            fc=1e9,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        # Approaching (r_dot negative) -> blue shift -> positive delta_f
        expected_delta_f = 30.0 * 1e9 / 299792458.0
        assert np.isclose(result["delta_f"], expected_delta_f, rtol=1e-3)

    def test_doppler_formula_matches_r_dot(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=-20.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
            fc=5.8e9,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        expected_delta_f = -result["r_dot"] * cfg.fc / cfg.c
        assert np.isclose(result["delta_f"], expected_delta_f)


class TestLineOfSightAngle:
    """Test los_angle = arctan2(dy, dx)."""

    def test_tx_ahead(self):
        cfg = ScenarioConfig(
            tx_x0=100.0,
            tx_y=0.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["los_angle"], 0.0)

    def test_tx_behind(self):
        cfg = ScenarioConfig(
            tx_x0=-100.0,
            tx_y=0.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["los_angle"], np.pi)

    def test_tx_left(self):
        cfg = ScenarioConfig(
            tx_x0=0.0,
            tx_y=10.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["los_angle"], np.pi / 2)

    def test_tx_right(self):
        cfg = ScenarioConfig(
            tx_x0=0.0,
            tx_y=-10.0,
            tx_vx=0.0,
            rx_x0=0.0,
            rx_y=0.0,
            rx_vx=0.0,
        )
        t = np.array([0.0])
        result = compute_geometry(cfg, t)
        assert np.isclose(result["los_angle"], -np.pi / 2)


class TestWithFixtures:
    """Tests using the conftest fixtures."""

    def test_colocated_scenario(self, scenario_colocated_cfg):
        t = np.linspace(0, 1.0, 11)
        result = compute_geometry(scenario_colocated_cfg, t)
        assert "r" in result
        assert "r_dot" in result
        assert "r_ddot" in result
        assert "delta_f" in result
        assert "los_angle" in result
        np.testing.assert_allclose(result["r_dot"], 0.0, atol=1e-10)

    def test_oncoming_scenario(self, scenario_oncoming_cfg):
        t = np.array([0.0])
        result = compute_geometry(scenario_oncoming_cfg, t)
        assert result["r_dot"] < 0.0

    def test_all_scenarios(self, any_scenario_cfg):
        t = np.array([0.0])
        result = compute_geometry(any_scenario_cfg, t)
        assert result["r"].shape == t.shape
        assert result["r_dot"].shape == t.shape
        assert result["r_ddot"].shape == t.shape
        assert result["delta_f"].shape == t.shape
        assert result["los_angle"].shape == t.shape
