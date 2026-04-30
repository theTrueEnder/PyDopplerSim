"""
Path module for PyDopplerSim.

Provides WaypointPath class for custom trajectories and preset path builders.
"""

__all__ = [
    "WaypointPath",
    "parallel_path",
    "angled_path",
    "curved_path",
]

import numpy as np
from typing import List, Tuple, Optional


class WaypointPath:
    """
    Path defined by waypoints with linear interpolation.

    Parameters
    ----------
    waypoints : List[Tuple[float, float, float]]
        List of (x, y, t) tuples defining waypoints.
        x: x position [m]
        y: y position [m]
        t: time [s]
    """

    def __init__(self, waypoints: List[Tuple[float, float, float]]):
        if len(waypoints) < 2:
            raise ValueError("WaypointPath requires at least 2 waypoints")

        # Sort by time
        self.waypoints = sorted(waypoints, key=lambda w: w[2])
        self.times = np.array([w[2] for w in self.waypoints])
        self.x_points = np.array([w[0] for w in self.waypoints])
        self.y_points = np.array([w[1] for w in self.waypoints])

    def interpolate(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate path positions at given times.

        Parameters
        ----------
        t : np.ndarray
            Time array [seconds]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (x, y) position arrays at each time
        """
        x = np.interp(t, self.times, self.x_points)
        y = np.interp(t, self.times, self.y_points)
        return x, y

    @property
    def duration(self) -> float:
        """Total path duration in seconds."""
        return self.times[-1] - self.times[0]


def parallel_path(x0: float, y: float, vx: float, duration: float) -> WaypointPath:
    """
    Create a straight-line path parallel to x-axis.

    This matches the current simulation behavior where the vehicle
    travels in a straight line at constant velocity.

    Parameters
    ----------
    x0 : float
        Initial x position [m]
    y : float
        Constant y position [m]
    vx : float
        Velocity in x direction [m/s]
    duration : float
        Path duration [s]

    Returns
    -------
    WaypointPath
        Path object for interpolation
    """
    waypoints = [
        (x0, y, 0.0),
        (x0 + vx * duration, y, duration),
    ]
    return WaypointPath(waypoints)


def angled_path(
    x0: float, y0: float, angle_deg: float, speed: float, duration: float
) -> WaypointPath:
    """
    Create a path at an angle from origin.

    Parameters
    ----------
    x0 : float
        Initial x position [m]
    y0 : float
        Initial y position [m]
    angle_deg : float
        Direction angle in degrees (0 = +x direction)
    speed : float
        Speed [m/s]
    duration : float
        Path duration [s]

    Returns
    -------
    WaypointPath
        Path object for interpolation
    """
    angle_rad = np.deg2rad(angle_deg)
    dx = speed * duration * np.cos(angle_rad)
    dy = speed * duration * np.sin(angle_rad)

    waypoints = [
        (x0, y0, 0.0),
        (x0 + dx, y0 + dy, duration),
    ]
    return WaypointPath(waypoints)


def curved_path(waypoints: List[Tuple[float, float, float]]) -> WaypointPath:
    """
    Create a curved path from waypoints.

    Convenience alias for WaypointPath with custom waypoints.

    Parameters
    ----------
    waypoints : List[Tuple[float, float, float]]
        List of (x, y, t) tuples

    Returns
    -------
    WaypointPath
        Path object for interpolation
    """
    return WaypointPath(waypoints)
