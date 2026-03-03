"""
iq_gen.py — Complex baseband IQ signal generation with time-varying Doppler.

The received signal phase is the integral of instantaneous frequency:

    φ(t) = 2π ∫₀ᵗ [f_tone + Δf(τ)] dτ

This is NOT the same as multiplying each sample by exp(j·2π·Δf[n]/fs),
which would treat each sample as phase-independent and accumulate error.
We compute the integral via cumulative sum on an oversampled time grid,
then decimate to the output sample rate.

AWGN model
----------
Signal power = 1 (unit-amplitude CW).
Total complex noise power σ² = signal_power / SNR_linear.
Split equally between I and Q: each component std = √(1 / (2·SNR_linear)).
"""

import numpy as np
from config import ScenarioConfig
from geometry import compute_geometry


def generate_iq(cfg: ScenarioConfig) -> dict:
    """
    Generate received IQ samples for a scenario.

    Parameters
    ----------
    cfg : ScenarioConfig

    Returns
    -------
    dict with keys:
        t         : time axis at output sample rate [s]
        iq        : noisy received IQ  (complex128, length N)
        iq_clean  : noiseless IQ       (complex128, length N)
        cfg       : the ScenarioConfig (passed through for downstream stages)
        **geo     : all geometry keys from compute_geometry (r, r_dot, etc.)
    """
    # Build fine time grid for accurate phase integration.
    # More oversample steps → more accurate cumsum approximation of the integral,
    # especially important near CPA where Δf changes fastest.
    N_fine  = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine  = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]

    # Instantaneous frequency on fine grid = tone offset + Doppler shift
    geo_fine  = compute_geometry(cfg, t_fine)
    inst_freq = cfg.f_tone + geo_fine["delta_f"]    # [Hz]

    # Integrate frequency → phase using cumulative sum (rectangle rule).
    # cumsum[n] = Σ_{k=0}^{n} f[k] · dt  ≈  ∫₀^{t_n} f(τ) dτ
    phase   = 2 * np.pi * np.cumsum(inst_freq) * dt_fine   # [rad]
    iq_fine = np.exp(1j * phase)                            # unit-amplitude CW

    # Decimate to output sample rate by striding (no anti-alias filter needed:
    # Doppler shifts are tiny compared to fs, so no aliasing risk)
    iq = iq_fine[::cfg.interp_oversample]
    t  = t_fine[::cfg.interp_oversample]
    N  = len(iq)

    # Complex AWGN: noise power = 1/SNR, split equally between I and Q
    snr_lin   = 10.0 ** (cfg.snr_db / 10.0)
    noise_std = np.sqrt(1.0 / (2.0 * snr_lin))
    noise     = noise_std * (np.random.randn(N) + 1j * np.random.randn(N))

    # Ground-truth geometry at the output sample rate
    geo = compute_geometry(cfg, t)

    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)