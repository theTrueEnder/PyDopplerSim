"""
dsp/kinematics.py — Kinematic recovery from estimated Doppler frequency.

Converts Δf_est → ṙ_est → r̈_est, returning raw and smoothed versions
of each so the plots can show all three lines (ground truth, raw, smoothed).

Pipeline
--------
    Δf_est (from estimators.py)
        │
        ▼  invert Doppler equation
    ṙ_est = −Δf_est · c / fc         [full sample rate, noisy]
        │
        ▼  decimate to RDOT_DECIM_HZ
    ṙ_decimated                       [low rate, averaged over bins]
        │
        ▼  finite difference + smooth
    r̈_est                             [low rate, smoothed]

Why decimate before differentiating?
-------------------------------------
Finite-difference amplifies noise by 1/Δt.  At fs=1 MHz, Δt=1 µs, so even
1 m/s of ṙ noise becomes 1×10⁶ m/s² after one difference step — useless.
Decimating to 100 Hz first makes Δt=10 ms, reducing amplification by 10,000×.
The Doppler trajectory varies on ~0.1–1 s timescales, so 100 Hz is sufficient.

NaN edge handling
-----------------
The Hann convolution (mode='same') zero-pads the edges, producing transient
artifacts over the first/last (window//2) samples.  We mark those as NaN so
they don't distort axis limits.  Any decimation bin containing a NaN sample
is dropped entirely rather than using nanmean, preventing contamination of
the differentiation step.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SimConfig
from dsp.estimators import hann_smooth


def recover_kinematics(
    t_est: np.ndarray,
    delta_f_raw: np.ndarray,
    delta_f_smoothed: np.ndarray,
    fc: float,
    c: float,
) -> dict:
    """
    Recover ṙ and r̈ from both the raw and smoothed Doppler estimates.

    We process both signals through the same pipeline so the plots can
    show three lines: ground truth, raw-derived, smoothed-derived.

    Parameters
    ----------
    t_est            : time axis for Δf estimates [s]
    delta_f_raw      : raw phase-diff Δf [Hz]
    delta_f_smoothed : smoothed Δf [Hz]
    fc               : carrier frequency [Hz]
    c                : speed of light [m/s]

    Returns
    -------
    dict with:
        t_dot            : time axis for ṙ (full rate, NaN at edges) [s]
        r_dot_raw        : ṙ derived from raw Δf    [m/s]
        r_dot_smoothed   : ṙ derived from smoothed Δf [m/s]
        t_ddot           : time axis for r̈ (decimated rate) [s]
        r_ddot_raw       : r̈ derived from raw ṙ     [m/s²]
        r_ddot_smoothed  : r̈ derived from smoothed ṙ [m/s²]
    """
    sw_pd    = SimConfig.PD_SMOOTH_WINDOW
    decim_hz = SimConfig.RDOT_DECIM_HZ
    sw_ddot  = SimConfig.RDOT_SMOOTH_WINDOW

    # --- ṙ: invert Doppler equation ---
    r_dot_raw      = -delta_f_raw      * c / fc
    r_dot_smoothed = -delta_f_smoothed * c / fc

    # Mark Hann-smoother edge artifacts as NaN on both signals
    trim = sw_pd // 2
    for arr in (r_dot_raw, r_dot_smoothed):
        arr[:trim]  = np.nan
        arr[-trim:] = np.nan

    # --- r̈: decimate then differentiate ---
    fs_orig    = 1.0 / (t_est[1] - t_est[0])
    decim_step = max(1, int(round(fs_orig / decim_hz)))

    def _decimate_and_diff(r_dot_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decimate r_dot_arr to RDOT_DECIM_HZ, then differentiate.
        Returns (t_d, r_ddot_d).
        Bins containing any NaN are skipped entirely.
        """
        n_bins = len(r_dot_arr) // decim_step
        t_d, rd_d = [], []
        for i in range(n_bins):
            sl  = slice(i * decim_step, (i + 1) * decim_step)
            seg = r_dot_arr[sl]
            if np.all(np.isfinite(seg)):
                t_d.append(t_est[sl].mean())
                rd_d.append(seg.mean())

        t_d  = np.array(t_d)
        rd_d = np.array(rd_d)

        if len(t_d) < 2:
            return t_d, np.full_like(t_d, np.nan)

        dt_d       = np.diff(t_d)
        ddot_raw   = np.concatenate([[np.nan], np.diff(rd_d) / dt_d])
        ddot_smooth = hann_smooth(ddot_raw, sw_ddot)

        # Trim ddot smoother edges
        trim_d = sw_ddot // 2
        ddot_smooth[:trim_d]  = np.nan
        ddot_smooth[-trim_d:] = np.nan

        return t_d, ddot_smooth

    t_ddot_raw, r_ddot_raw         = _decimate_and_diff(r_dot_raw.copy())
    t_ddot_smooth, r_ddot_smoothed = _decimate_and_diff(r_dot_smoothed.copy())

    return dict(
        t_dot           = t_est,
        r_dot_raw       = r_dot_raw,
        r_dot_smoothed  = r_dot_smoothed,
        t_ddot_raw      = t_ddot_raw,
        r_ddot_raw      = r_ddot_raw,
        t_ddot_smooth   = t_ddot_smooth,
        r_ddot_smoothed = r_ddot_smoothed,
    )