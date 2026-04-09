"""
Geometry module for PyDopplerSim.

Computes r(t), ṙ(t), r̈(t), delta_f, and bearing from positions.
"""

__all__ = ["compute_geometry"]

import numpy as np


def compute_geometry(cfg, t: np.ndarray) -> dict:
    """
    Compute ground-truth kinematic quantities at each time sample.

    Both vehicles travel in straight lines along the x-axis at constant speed,
    separated by a fixed lateral offset (dy).  All quantities are derived
    analytically — no numerical differentiation needed here.

    Parameters
    ----------
    cfg : ScenarioConfig
        Configuration object with tx_x0, tx_y, tx_vx, rx_x0, rx_y, rx_vx, fc, c
    t : np.ndarray
        Time array [seconds]

    Returns
    -------
    dict
        Dictionary containing:
        - r: range = |tx_pos - rx_pos|  [m]
        - r_dot: radial velocity = d(r)/dt  [m/s]   (+ve = moving apart)
        - r_ddot: radial acceleration = d²(r)/dt²  [m/s²]
        - delta_f: Doppler shift = -ṙ · fc / c  [Hz]
        - tx_x, rx_x: x positions [m]
        - tx_y, rx_y: y positions [m]
        - los_angle: bearing from Rx to Tx = arctan2(dy, dx)  [rad]
                      0 = ahead, π/2 = left, π = behind, -π/2 = right
    """
    # World-frame positions
    tx_x = cfg.tx_x0 + cfg.tx_vx * t
    rx_x = cfg.rx_x0 + cfg.rx_vx * t

    # Relative position vector (tx − rx)
    dx = tx_x - rx_x  # changes with time
    dy = cfg.tx_y - cfg.rx_y  # constant (no lateral motion)
    r = np.sqrt(dx**2 + dy**2)
    safe_r = np.where(r < 1e-6, 1e-6, r)  # guard against exact co-location

    # Relative velocity in x only (dy is constant → vdy = 0)
    vdx = cfg.tx_vx - cfg.rx_vx

    # ṙ = d/dt √(dx²+dy²) = (dx·vdx) / r
    r_dot = (dx * vdx) / safe_r

    # r̈ = d/dt [ṙ] via quotient rule:
    #   r̈ = (vdx²·r − dx·vdx·ṙ) / r²
    # Physical interpretation: peaks at CPA when the LoS rotates fastest.
    r_ddot = (vdx**2 * safe_r - dx * vdx * r_dot) / safe_r**2

    return dict(
        r=r,
        r_dot=r_dot,
        r_ddot=r_ddot,
        delta_f=-r_dot * cfg.fc / cfg.c,
        tx_x=tx_x,
        rx_x=rx_x,
        tx_y=np.full_like(t, cfg.tx_y),
        rx_y=np.full_like(t, cfg.rx_y),
        los_angle=np.arctan2(dy, dx),
    )
