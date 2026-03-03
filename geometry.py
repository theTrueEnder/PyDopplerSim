"""
geometry.py — Analytical 2-D kinematic geometry for the Doppler simulator.

Both vehicles travel at constant speed along the x-axis, separated by a
fixed lateral offset dy = tx_y − rx_y.  All quantities are derived in closed
form from the positions and velocities — no numerical differentiation needed.

Coordinate convention
---------------------
  +x : direction of travel (both vehicles)
  +y : left of direction of travel (lane offset)
  Origin: rx initial position

Key quantities
--------------
  r(t)      = |tx_pos − rx_pos|          range [m]
  ṙ(t)      = d/dt r                     radial velocity [m/s]  (+ve = separating)
  r̈(t)      = d²/dt² r                  radial acceleration [m/s²]
  Δf(t)     = −ṙ · fc / c               Doppler shift [Hz]
  los_angle = arctan2(dy, dx)            bearing from Rx to Tx [rad]
              0 = ahead, π/2 = left, π = behind

Physical intuition for r̈
------------------------
At CPA (closest point of approach), dx = 0 and r = dy (pure lateral separation).
The formula reduces to:

    r̈_CPA = vdx² / dy

So r̈ at CPA is large when vehicles are fast relative to each other (large vdx)
or the lane is narrow (small dy).  For oncoming at 60 m/s and 3.7 m lane:
    r̈_CPA = 60² / 3.7 ≈ 973 m/s²
"""

import numpy as np
from config import ScenarioConfig


def compute_geometry(cfg: ScenarioConfig, t: np.ndarray) -> dict:
    """
    Compute ground-truth kinematic quantities at each time sample t.

    Parameters
    ----------
    cfg : ScenarioConfig
    t   : 1-D array of time samples [s]

    Returns
    -------
    dict with keys:
        r, r_dot, r_ddot, delta_f   — kinematic scalars at each t
        los_angle                    — bearing angle [rad]
        tx_x, tx_y, rx_x, rx_y      — world positions (for plotting)
    """
    # --- World-frame positions ---
    tx_x = cfg.tx_x0 + cfg.tx_vx * t
    rx_x = cfg.rx_x0 + cfg.rx_vx * t

    # Relative displacement vector (tx − rx).
    # dy is constant: neither vehicle moves laterally.
    dx = tx_x - rx_x
    dy = cfg.tx_y - cfg.rx_y

    # Range
    r      = np.sqrt(dx**2 + dy**2)
    safe_r = np.where(r < 1e-6, 1e-6, r)   # avoid division by zero at exact co-location

    # Relative velocity in x only (vdy = 0)
    vdx = cfg.tx_vx - cfg.rx_vx

    # ṙ = d/dt √(dx² + dy²) = (dx·vdx + dy·vdy) / r = (dx·vdx) / r
    r_dot = (dx * vdx) / safe_r

    # r̈ via quotient rule on ṙ = (dx·vdx) / r:
    #   d/dt[dx·vdx] = vdx²            (vdx constant, d(dx)/dt = vdx)
    #   d/dt[r]      = ṙ
    #   → r̈ = (vdx²·r − dx·vdx·ṙ) / r²
    r_ddot = (vdx**2 * safe_r - dx * vdx * r_dot) / safe_r**2

    # Doppler shift: Δf = −ṙ·fc/c
    delta_f = -r_dot * cfg.fc / cfg.c

    # Bearing from Rx to Tx: arctan2 gives correct quadrant for all cases
    los_angle = np.arctan2(dy, dx)

    return dict(
        r=r,
        r_dot=r_dot,
        r_ddot=r_ddot,
        delta_f=delta_f,
        los_angle=los_angle,
        tx_x=tx_x,
        tx_y=np.full_like(t, cfg.tx_y),
        rx_x=rx_x,
        rx_y=np.full_like(t, cfg.rx_y),
    )