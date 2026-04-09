"""
Doppler estimation module for PyDopplerSim.

Provides phase-differentiation and STFT-based Doppler estimators.
"""

__all__ = [
    "estimate_doppler_phase_diff",
    "estimate_doppler_stft",
    "_hann_smooth",
]

import numpy as np
from typing import Optional

# Import config for default parameters
try:
    from config import SimConfig
except ImportError:
    SimConfig = None


def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Convolve x with a normalised Hann window of length w (linear-phase FIR)."""
    if w < 2:
        return x
    h = np.hanning(w)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def estimate_doppler_phase_diff(
    iq: np.ndarray, fs: float, smooth_window: Optional[int] = None
) -> tuple:
    """
    FM discriminator (lag-1 autocorrelation argument).

    The instantaneous frequency is proportional to the rate of phase change:

        f_inst[n] = angle(iq[n] · conj(iq[n−1])) · fs / (2π)

    This gives sample-rate time resolution but is noisy — the Hann smoother
    trades temporal resolution for noise suppression.

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband IQ samples
    fs : float
        Sample rate [Hz]
    smooth_window : int, optional
        Hann window length for smoothing. Defaults to SimConfig.PD_SMOOTH_WINDOW.

    Returns
    -------
    tuple
        (t, delta_f) both at sample rate fs (length N-1)
    """
    if smooth_window is None:
        if SimConfig is not None:
            smooth_window = SimConfig.PD_SMOOTH_WINDOW
        else:
            smooth_window = 501

    # Angle of complex lag-1 product = phase increment per sample
    f_inst = np.angle(iq[1:] * np.conj(iq[:-1])) / (2 * np.pi) * fs
    f_inst = _hann_smooth(f_inst, smooth_window)
    t = np.arange(len(f_inst)) / fs
    return t, f_inst


def estimate_doppler_stft(
    iq: np.ndarray,
    fs: float,
    window_dur: Optional[float] = None,
    hop_dur: Optional[float] = None,
    freq_zoom: float = 5000.0,
) -> dict:
    """
    Short-Time Fourier Transform spectrogram with peak frequency tracking.

    Design choices
    --------------
    - Hann analysis window (good sidelobe rejection for close-in peaks)
    - 4× zero-padding → smoother peak interpolation without changing resolution
    - freq_zoom: only rows within ±freq_zoom Hz are returned for display;
      auto-set to 3× max ground-truth |Δf| in the plotting code

    Parameters
    ----------
    iq : np.ndarray
        Complex baseband IQ samples
    fs : float
        Sample rate [Hz]
    window_dur : float, optional
        Analysis window duration [s]. Defaults to SimConfig.STFT_WINDOW_DUR.
    hop_dur : float, optional
        STFT hop size [s]. Defaults to SimConfig.STFT_HOP_DUR.
    freq_zoom : float
        Frequency zoom range [Hz]

    Returns
    -------
    dict
        Dictionary with: t, freq_axis, Sxx (power), peak_freq, mask
    """
    if window_dur is None:
        if SimConfig is not None:
            window_dur = SimConfig.STFT_WINDOW_DUR
        else:
            window_dur = 0.05
    if hop_dur is None:
        if SimConfig is not None:
            hop_dur = SimConfig.STFT_HOP_DUR
        else:
            hop_dur = 0.005

    win_samp = int(window_dur * fs)
    hop_samp = int(hop_dur * fs)
    N_fft = win_samp * 4  # zero-pad for interpolated peak
    win = np.hanning(win_samp)

    n_frames = (len(iq) - win_samp) // hop_samp + 1
    Sxx = np.zeros((N_fft, n_frames))

    for i in range(n_frames):
        seg = iq[i * hop_samp : i * hop_samp + win_samp] * win
        Sxx[:, i] = np.abs(np.fft.fftshift(np.fft.fft(seg, N_fft))) ** 2

    freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, 1.0 / fs))
    t_stft = (np.arange(n_frames) * hop_samp + win_samp // 2) / fs
    peak_freq = freq_axis[np.argmax(Sxx, axis=0)]
    mask = np.abs(freq_axis) <= freq_zoom

    return dict(t=t_stft, freq_axis=freq_axis, Sxx=Sxx, peak_freq=peak_freq, mask=mask)
