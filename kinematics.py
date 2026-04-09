"""
Kinematics recovery module for PyDopplerSim.

Recovers ṙ and r̈ from Doppler estimates.
"""

__all__ = ["recover_kinematics"]

import numpy as np
from typing import Optional

# Import helpers
try:
    from estimation import _hann_smooth
except ImportError:
    # Inline copy if estimation not available
    def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
        if w < 2:
            return x
        h = np.hanning(w)
        h /= h.sum()
        return np.convolve(x, h, mode="same")


# Import config for default parameters
try:
    from config import SimConfig
except ImportError:
    SimConfig = None


def recover_kinematics(
    t_est: np.ndarray, delta_f_est: np.ndarray, fc: float, c: float
) -> dict:
    """
    Recover estimated ṙ and r̈ from the phase-diff Doppler estimate.

    ṙ  (full rate)
    --------------
    Direct inversion of the Doppler equation:
        ṙ_est = −Δf_est · c / fc

    r̈  (decimated rate)
    --------------------
    Naive finite-differencing at full sample rate (1 MHz) would amplify noise
    by ~fs × σ_rdot ≈ 1e6 m/s² even for 1 m/s of ṙ noise.  Instead we:
      1. Mark Hann-smoother edge artifacts as NaN (first/last PD_SMOOTH_WINDOW//2 samples).
      2. Decimate ṙ to RDOT_DECIM_HZ by averaging bins of (fs/decim_hz) samples,
         skipping any bin that contains a NaN to avoid polluting the decimated series.
      3. Differentiate the clean, low-rate series: noise amplification is now
         only decim_hz × σ_rdot (≈100× instead of 1e6×).
      4. Apply a short Hann smoother on the decimated r̈ series.

    Parameters
    ----------
    t_est : np.ndarray
        Time array from Doppler estimator [s]
    delta_f_est : np.ndarray
        Doppler frequency estimate [Hz]
    fc : float
        Carrier frequency [Hz]
    c : float
        Speed of light [m/s]

    Returns
    -------
    dict
        Dictionary with:
        - t_dot: time axis for ṙ  [full rate]
        - r_dot: estimated ṙ      [m/s, NaN at edges]
        - t_ddot: time axis for r̈  [decimated rate]
        - r_ddot: estimated r̈      [m/s², NaN at edges]
    """
    sw_pd = SimConfig.PD_SMOOTH_WINDOW if SimConfig else 501
    decim_hz = SimConfig.RDOT_DECIM_HZ if SimConfig else 100
    sw_ddot = SimConfig.RDOT_SMOOTH_WINDOW if SimConfig else 21

    # --- ṙ at full sample rate ---
    r_dot_full = -delta_f_est * c / fc

    # Mark smoother edge transients as NaN
    trim = sw_pd // 2
    r_dot_full[:trim] = np.nan
    r_dot_full[-trim:] = np.nan

    # --- Decimate ṙ to low rate, skipping any NaN-containing bin ---
    fs_orig = 1.0 / (t_est[1] - t_est[0])
    decim_step = max(1, int(round(fs_orig / decim_hz)))
    n_bins = len(r_dot_full) // decim_step

    t_d, r_dot_d = [], []
    for i in range(n_bins):
        sl = slice(i * decim_step, (i + 1) * decim_step)
        seg = r_dot_full[sl]
        if np.all(np.isfinite(seg)):  # discard any bin touching NaN edge
            t_d.append(t_est[sl].mean())
            r_dot_d.append(seg.mean())

    t_d = np.array(t_d)
    r_dot_d = np.array(r_dot_d)

    # --- Differentiate on clean decimated grid ---
    dt_d = np.diff(t_d)
    r_ddot_raw = np.concatenate([[np.nan], np.diff(r_dot_d) / dt_d])
    r_ddot_d = _hann_smooth(r_ddot_raw, sw_ddot)

    # Trim smoother edges on r̈
    trim_d = sw_ddot // 2
    r_ddot_d[:trim_d] = np.nan
    r_ddot_d[-trim_d:] = np.nan

    return dict(t_dot=t_est, r_dot=r_dot_full, t_ddot=t_d, r_ddot=r_ddot_d)
