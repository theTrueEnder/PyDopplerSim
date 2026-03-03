"""
dsp/estimators.py — Raw Doppler frequency estimators.

Each estimator returns THREE signals that map to the three plot lines:
  1. raw      : unsmoothed output directly from the algorithm
  2. smoothed : after Hann FIR filtering (the "processed" line)
  3. (ground truth is computed in geometry.py, not here)

These functions are pure numpy — no GNURadio or matplotlib dependency.
The GNURadio blocks in dsp/gnuradio/ wrap these functions for streaming use.

Estimator summary
-----------------
phase_diff : FM discriminator (lag-1 autocorrelation argument).
             Sample-rate time resolution, noisy, works on any CW-like signal.
             Best for slow/moderate Doppler rates.

stft_peak  : Short-Time Fourier Transform peak tracking.
             Explicit time-frequency tradeoff via window length.
             Better frequency resolution than phase_diff but lower time resolution.
             Returns a spectrogram matrix as well as the peak track.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SimConfig


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    """
    Linear-phase FIR smoothing via Hann window convolution.

    A Hann window of length w has a -3 dB cutoff at roughly fs/(w/2),
    so larger w → more smoothing → lower noise but more temporal blur.
    """
    if w < 2:
        return x.copy()
    h = np.hanning(w)
    h /= h.sum()
    return np.convolve(x, h, mode='same')


# ---------------------------------------------------------------------------
# Phase-difference estimator
# ---------------------------------------------------------------------------

def phase_diff(
    iq: np.ndarray,
    fs: float,
    smooth_window: int | None = None,
) -> dict:
    """
    FM discriminator via lag-1 complex autocorrelation.

    Instantaneous frequency:
        f_inst[n] = angle(iq[n] · conj(iq[n−1])) · fs / (2π)

    This is equivalent to differentiating the unwrapped phase, but avoids
    explicit phase unwrapping (which can fail with low SNR).

    The raw output is noisy — standard deviation scales as:
        σ_f ≈ fs / (2π · √SNR_linear)

    The smoothed output trades temporal resolution for SNR improvement.
    Rule of thumb: smoothing by w samples reduces noise std by ~√w.

    Parameters
    ----------
    iq            : complex IQ samples
    fs            : sample rate [Hz]
    smooth_window : Hann window length; None → SimConfig.PD_SMOOTH_WINDOW

    Returns
    -------
    dict with:
        t        : time axis (length N−1) [s]
        raw      : unsmoothed instantaneous frequency [Hz]
        smoothed : Hann-smoothed frequency [Hz]
    """
    if smooth_window is None:
        smooth_window = SimConfig.PD_SMOOTH_WINDOW

    # Lag-1 product: angle gives phase increment per sample interval
    lag1  = iq[1:] * np.conj(iq[:-1])
    raw   = np.angle(lag1) / (2 * np.pi) * fs      # [Hz]
    smoothed = hann_smooth(raw, smooth_window)
    t     = np.arange(len(raw)) / fs

    return dict(t=t, raw=raw, smoothed=smoothed)


# ---------------------------------------------------------------------------
# STFT peak estimator
# ---------------------------------------------------------------------------

def stft_peak(
    iq: np.ndarray,
    fs: float,
    window_dur: float | None = None,
    hop_dur: float | None = None,
    freq_zoom: float = 5000.0,
) -> dict:
    """
    Short-Time Fourier Transform with peak frequency tracking.

    Design choices
    --------------
    - Hann analysis window: good main-lobe / sidelobe tradeoff for sinusoids.
    - 4× zero-padding: smoother peak interpolation without changing true
      frequency resolution (which is set by window_dur, not FFT length).
    - Peak is the raw argmax of |STFT|²; no quadratic interpolation yet.

    Time-frequency tradeoff
    -----------------------
    Frequency resolution   = 1 / window_dur   [Hz]
    Temporal resolution    = hop_dur           [s]
    Choosing window_dur = 0.05 s gives 20 Hz freq resolution, which is
    sufficient to resolve Doppler shifts > ~20 Hz.

    Parameters
    ----------
    iq          : complex IQ samples
    fs          : sample rate [Hz]
    window_dur  : analysis window length [s]
    hop_dur     : frame hop [s]
    freq_zoom   : only frequencies within ±freq_zoom Hz are included in
                  the returned Sxx and freq_axis (for display efficiency)

    Returns
    -------
    dict with:
        t          : frame time axis [s]
        freq_axis  : frequency axis (zoomed) [Hz]
        Sxx        : power spectrogram (zoomed), shape (n_freq, n_frames)
        raw        : argmax peak frequency per frame [Hz]  ← "raw" estimate
        smoothed   : Hann-smoothed peak track [Hz]
    """
    if window_dur is None: window_dur = SimConfig.STFT_WINDOW_DUR
    if hop_dur    is None: hop_dur    = SimConfig.STFT_HOP_DUR

    win_samp = int(window_dur * fs)
    hop_samp = int(hop_dur * fs)
    N_fft    = win_samp * 4         # zero-pad for smoother peak
    win      = np.hanning(win_samp)

    n_frames = (len(iq) - win_samp) // hop_samp + 1
    Sxx_full = np.zeros((N_fft, n_frames))

    for i in range(n_frames):
        seg            = iq[i*hop_samp : i*hop_samp + win_samp] * win
        Sxx_full[:, i] = np.abs(np.fft.fftshift(np.fft.fft(seg, N_fft)))**2

    freq_axis_full = np.fft.fftshift(np.fft.fftfreq(N_fft, 1.0 / fs))
    t_frames       = (np.arange(n_frames) * hop_samp + win_samp // 2) / fs

    # Raw peak track on full (unzoomed) axis
    raw = freq_axis_full[np.argmax(Sxx_full, axis=0)]

    # Smooth the peak track
    # Window length in frames: use ~10 frames as a reasonable default
    smooth_frames = max(3, int(0.1 / hop_dur))
    smoothed = hann_smooth(raw, smooth_frames)

    # Zoom for display
    mask      = np.abs(freq_axis_full) <= freq_zoom
    freq_zoom_axis = freq_axis_full[mask]
    Sxx_zoom  = Sxx_full[mask, :]

    return dict(
        t=t_frames,
        freq_axis=freq_zoom_axis,
        Sxx=Sxx_zoom,
        raw=raw,
        smoothed=smoothed,
    )