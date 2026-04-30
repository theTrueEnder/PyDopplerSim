"""
Kinematics recovery module for PyDopplerSim.

Recovers range-rate and range-acceleration from Doppler estimates.
"""

__all__ = ["recover_kinematics"]

import numpy as np

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


def _edge_trim(length: int, requested_trim: int) -> int:
    """Keep short synthetic arrays from being trimmed to all-NaN."""
    return min(requested_trim, max(0, (length - 1) // 4))


def recover_kinematics(
    t_est: np.ndarray,
    delta_f_est: np.ndarray,
    fc: float,
    c: float,
    *,
    rx_knows_velocity: bool = False,
    rx_vx: float | None = None,
    bearing_grid_rad: np.ndarray | None = None,
    pd_smooth_window: int | None = None,
    rdot_decim_hz: float | None = None,
    rddot_smooth_window: int | None = None,
) -> dict:
    """
    Recover estimated range-rate and range-acceleration from Doppler.

    Doppler alone directly gives range-rate:

        r_dot_est = -delta_f_est * c / fc

    Range-acceleration is estimated by decimating r_dot, differentiating on
    the lower-rate series, then smoothing. If the RX is allowed to know its own
    velocity, this function may also return RX-velocity-aware constraints, but
    it still does not reconstruct range, bearing, or TX position without
    additional explicit assumptions.
    """
    sw_pd = (
        pd_smooth_window
        if pd_smooth_window is not None
        else (SimConfig.PD_SMOOTH_WINDOW if SimConfig else 501)
    )
    decim_hz = (
        rdot_decim_hz
        if rdot_decim_hz is not None
        else (SimConfig.RDOT_DECIM_HZ if SimConfig else 100)
    )
    sw_ddot = (
        rddot_smooth_window
        if rddot_smooth_window is not None
        else (SimConfig.RDOT_SMOOTH_WINDOW if SimConfig else 21)
    )

    t_est = np.asarray(t_est, dtype=float)
    delta_f_est = np.asarray(delta_f_est, dtype=float)
    if t_est.shape != delta_f_est.shape:
        raise ValueError("t_est and delta_f_est must have matching shapes")

    # Full-rate range-rate from the Doppler equation.
    r_dot_full = -delta_f_est * c / fc

    trim = _edge_trim(len(r_dot_full), sw_pd // 2)
    if trim:
        r_dot_full[:trim] = np.nan
        r_dot_full[-trim:] = np.nan

    # Decimate r_dot to a lower rate, skipping bins touched by edge NaNs.
    if len(t_est) > 1:
        fs_orig = 1.0 / (t_est[1] - t_est[0])
        decim_step = max(1, int(round(fs_orig / decim_hz)))
    else:
        decim_step = 1

    n_bins = len(r_dot_full) // decim_step
    t_d, r_dot_d = [], []
    for i in range(n_bins):
        sl = slice(i * decim_step, (i + 1) * decim_step)
        seg = r_dot_full[sl]
        if np.all(np.isfinite(seg)):
            t_d.append(t_est[sl].mean())
            r_dot_d.append(seg.mean())

    t_d = np.array(t_d)
    r_dot_d = np.array(r_dot_d)

    if len(t_d) > 1:
        dt_d = np.diff(t_d)
        r_ddot_raw = np.concatenate([[np.nan], np.diff(r_dot_d) / dt_d])
        r_ddot_for_smooth = r_ddot_raw.copy()
        r_ddot_for_smooth[0] = r_ddot_for_smooth[1]
        smooth_w = min(sw_ddot, len(r_ddot_raw))
        r_ddot_d = (
            _hann_smooth(r_ddot_for_smooth, smooth_w)
            if smooth_w >= 3
            else r_ddot_for_smooth
        )

        trim_d = _edge_trim(len(r_ddot_d), sw_ddot // 2)
        if trim_d:
            r_ddot_d[:trim_d] = np.nan
            r_ddot_d[-trim_d:] = np.nan
    else:
        r_ddot_d = np.full_like(t_d, np.nan, dtype=float)

    result = dict(t_dot=t_est, r_dot=r_dot_full, t_ddot=t_d, r_ddot=r_ddot_d)

    if rx_knows_velocity:
        result.update(
            rx_velocity_known=rx_vx is not None,
            rx_vx=rx_vx,
            observability=dict(
                range=False,
                bearing=False,
                tx_position=False,
                note=(
                    "Doppler plus RX velocity constrains radial motion, "
                    "not range, bearing, or TX position."
                ),
            ),
        )

        if rx_vx is not None and bearing_grid_rad is not None:
            bearing_grid_rad = np.asarray(bearing_grid_rad, dtype=float)
            result["bearing_grid_rad"] = bearing_grid_rad
            result["tx_los_velocity_grid"] = (
                r_dot_full[:, np.newaxis]
                + float(rx_vx) * np.cos(bearing_grid_rad)[np.newaxis, :]
            )

    return result
