"""
Tests for paths module: WaypointPath, parallel_path.
"""

import numpy as np
import pytest
from paths import WaypointPath, parallel_path, angled_path, curved_path


class TestWaypointPath:
    """Test suite for WaypointPath class."""

    def test_interpolation_at_waypoints_returns_exact_values(self):
        """Test that interpolation at waypoint times returns exact values."""
        waypoints = [
            (0.0, 0.0, 0.0),
            (100.0, 50.0, 1.0),
            (200.0, 100.0, 2.0),
        ]

        path = WaypointPath(waypoints)

        # Interpolate at exactly the waypoint times
        t_query = np.array([0.0, 1.0, 2.0])
        x, y = path.interpolate(t_query)

        np.testing.assert_allclose(x, [0.0, 100.0, 200.0])
        np.testing.assert_allclose(y, [0.0, 50.0, 100.0])

    def test_linear_interpolation_between_waypoints(self):
        """Test linear interpolation between waypoints."""
        waypoints = [
            (0.0, 0.0, 0.0),
            (100.0, 50.0, 1.0),
        ]

        path = WaypointPath(waypoints)

        # Interpolate at midpoint in time
        t_query = np.array([0.5])
        x, y = path.interpolate(t_query)

        # Should be exactly at midpoint
        np.testing.assert_allclose(x, 50.0)
        np.testing.assert_allclose(y, 25.0)

    def test_interpolation_multiple_points(self):
        """Test interpolation at multiple points."""
        waypoints = [
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 1.0),
            (100.0, 100.0, 2.0),
        ]

        path = WaypointPath(waypoints)

        t_query = np.linspace(0, 2, 5)
        x, y = path.interpolate(t_query)

        # Check that x increases then stays constant
        assert x[0] == 0.0
        assert x[-1] == 100.0

        # Check that y is piecewise linear
        assert y[0] == 0.0
        assert y[-1] == 100.0

    def test_insufficient_waypoints_raises_error(self):
        """Test that less than 2 waypoints raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            WaypointPath([(0.0, 0.0, 0.0)])

        with pytest.raises(ValueError, match="at least 2 waypoints"):
            WaypointPath([])

    def test_single_waypoint_raises_error(self):
        """Test that single waypoint raises ValueError."""
        with pytest.raises(ValueError):
            WaypointPath([(10.0, 20.0, 1.0)])

    def test_waypoints_are_sorted_by_time(self):
        """Test that waypoints are sorted by time."""
        waypoints = [
            (100.0, 100.0, 2.0),  # Last in input
            (0.0, 0.0, 0.0),      # First in input
            (50.0, 50.0, 1.0),    # Middle in input
        ]

        path = WaypointPath(waypoints)

        # Check internal sorting
        np.testing.assert_array_equal(path.times, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(path.x_points, [0.0, 50.0, 100.0])
        np.testing.assert_array_equal(path.y_points, [0.0, 50.0, 100.0])

    def test_duration_property(self):
        """Test the duration property."""
        waypoints = [
            (0.0, 0.0, 1.0),
            (100.0, 50.0, 4.0),
        ]

        path = WaypointPath(waypoints)

        assert path.duration == 3.0

    def test_extrapolation_beyond_waypoints(self):
        """Test extrapolation beyond first/last waypoint."""
        waypoints = [
            (0.0, 0.0, 1.0),
            (100.0, 50.0, 2.0),
        ]

        path = WaypointPath(waypoints)

        # Before first waypoint
        t_before = np.array([0.5])
        x_before, y_before = path.interpolate(t_before)
        np.testing.assert_allclose(x_before, 0.0)
        np.testing.assert_allclose(y_before, 0.0)

        # After last waypoint
        t_after = np.array([2.5])
        x_after, y_after = path.interpolate(t_after)
        np.testing.assert_allclose(x_after, 100.0)
        np.testing.assert_allclose(y_after, 50.0)


class TestParallelPath:
    """Test suite for parallel_path function."""

    def test_parallel_path_matches_direct_calculation(self):
        """Test that parallel_path matches direct calculation."""
        x0 = 0.0
        y = 10.0
        vx = 30.0
        duration = 5.0

        path = parallel_path(x0, y, vx, duration)

        # Check direct calculation
        expected_x_end = x0 + vx * duration
        expected_x = np.array([x0, expected_x_end])
        expected_y = np.array([y, y])

        t_query = np.array([0.0, duration])
        x, y_out = path.interpolate(t_query)

        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y_out, expected_y)

    def test_parallel_path_creates_two_waypoints(self):
        """Test that parallel_path creates exactly 2 waypoints."""
        path = parallel_path(0.0, 10.0, 30.0, 5.0)

        assert len(path.waypoints) == 2
        assert path.waypoints[0] == (0.0, 10.0, 0.0)
        assert path.waypoints[1] == (150.0, 10.0, 5.0)

    def test_parallel_path_velocity(self):
        """Test that parallel_path maintains constant velocity."""
        x0 = 0.0
        y = 5.0
        vx = 20.0
        duration = 2.0

        path = parallel_path(x0, y, vx, duration)

        # Sample at multiple points
        t_query = np.linspace(0, duration, 10)
        x, y_out = path.interpolate(t_query)

        # x should increase linearly
        expected_x = x0 + vx * t_query
        np.testing.assert_allclose(x, expected_x)

        # y should stay constant
        np.testing.assert_allclose(y_out, y)


class TestAngledPath:
    """Test suite for angled_path function."""

    def test_angled_path_horizontal(self):
        """Test angled_path with 0 degree angle (horizontal)."""
        path = angled_path(0.0, 0.0, 0.0, 10.0, 1.0)

        t_query = np.array([0.0, 1.0])
        x, y = path.interpolate(t_query)

        np.testing.assert_allclose(x, [0.0, 10.0])
        np.testing.assert_allclose(y, [0.0, 0.0])

    def test_angled_path_vertical(self):
        """Test angled_path with 90 degree angle (vertical)."""
        path = angled_path(0.0, 0.0, 90.0, 10.0, 1.0)

        t_query = np.array([0.0, 1.0])
        x, y = path.interpolate(t_query)

        np.testing.assert_allclose(x, [0.0, 0.0])
        np.testing.assert_allclose(y, [0.0, 10.0])

    def test_angled_path_45_degrees(self):
        """Test angled_path with 45 degree angle."""
        speed = 10.0
        duration = 1.0
        path = angled_path(0.0, 0.0, 45.0, speed, duration)

        t_query = np.array([0.0, 1.0])
        x, y = path.interpolate(t_query)

        expected_displacement = speed * duration * np.cos(np.deg2rad(45))
        np.testing.assert_allclose(x, [0.0, expected_displacement], rtol=1e-10)
        np.testing.assert_allclose(y, [0.0, expected_displacement], rtol=1e-10)


class TestCurvedPath:
    """Test suite for curved_path function."""

    def test_curved_path_is_alias(self):
        """Test that curved_path is an alias for WaypointPath."""
        waypoints = [
            (0.0, 0.0, 0.0),
            (50.0, 50.0, 1.0),
            (100.0, 0.0, 2.0),
        ]

        curved = curved_path(waypoints)

        assert isinstance(curved, WaypointPath)
        assert len(curved.waypoints) == 3


class TestIntegration:
    """Integration tests for paths with other modules."""

    def test_path_with_generate_iq(self, scenario_oncoming_cfg):
        """Test using WaypointPath with generate_iq."""
        from iq_gen import generate_iq
        from paths import parallel_path

        # Create custom path
        custom_path = parallel_path(
            scenario_oncoming_cfg.tx_x0,
            scenario_oncoming_cfg.tx_y,
            scenario_oncoming_cfg.tx_vx,
            scenario_oncoming_cfg.duration,
        )

        # Use with generate_iq
        result = generate_iq(scenario_oncoming_cfg, path_provider=custom_path)

        assert "tx_x" in result
        assert len(result["tx_x"]) > 0