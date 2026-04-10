"""
IQ generation module for PyDopplerSim.

Generates complex baseband IQ samples with Doppler shift and AWGN.
Also provides WAV file save/load for replay capability.
"""

__all__ = [
    "generate_iq",
    "save_iq_wav",
    "load_iq_wav",
]

import numpy as np
from pathlib import Path
from typing import Optional

# Import geometry - will be available after T2
try:
    from geometry import compute_geometry, compute_geometry_from_samples
except ImportError:
    compute_geometry = None
    compute_geometry_from_samples = None


def generate_iq(cfg, path_provider: Optional[object] = None):
    """
    Generate received complex baseband IQ samples with time-varying Doppler.

    Phase integration (not per-sample frequency shift)
    ---------------------------------------------------
    The received signal phase is the integral of instantaneous frequency:

        φ(t) = 2π ∫₀ᵗ [f_tone + Δf(τ)] dτ

    This is computed via cumulative sum on an oversampled time grid, then
    decimated back to fs.  Oversampling (× INTERP_OVERSAMPLE) ensures the
    numerical integral is accurate even when Δf changes rapidly near CPA.

    AWGN model
    ----------
    Signal power = 1 (unit-amplitude CW).
    Complex noise variance σ² = 1/SNR_linear, split equally between I and Q,
    so each component has std = √(1 / (2·SNR_linear)).

    Parameters
    ----------
    cfg : ScenarioConfig
        Configuration object with fc, fs, duration, snr_db, f_tone,
        tx_x0, tx_y, tx_vx, rx_x0, rx_y, rx_vx, interp_oversample
    path_provider : WaypointPath, optional
        Optional path provider for TX trajectory. If None, uses parallel motion.

    Returns
    -------
    dict
        Dictionary with t, iq, iq_clean, cfg, and geometry keys
    """
    if compute_geometry is None:
        raise ImportError("geometry module not available")

    # Build fine time grid for accurate phase integration
    N_fine = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]

    # Compute geometry at fine resolution
    if path_provider is not None:
        # Use custom path for TX
        tx_x_fine, tx_y_fine = path_provider.interpolate(t_fine)
        rx_x_fine = cfg.rx_x0 + cfg.rx_vx * t_fine
        rx_y_fine = np.full_like(t_fine, cfg.rx_y)
        geo_f = compute_geometry_from_samples(
            tx_x_fine, tx_y_fine, rx_x_fine, rx_y_fine, t_fine, cfg.fc, cfg.c
        )
    else:
        geo_f = compute_geometry(cfg, t_fine)

    # Instantaneous frequency on fine grid: tone + Doppler
    inst_freq = cfg.f_tone + geo_f["delta_f"]  # [Hz]

    # Integrate frequency → phase, then form complex baseband signal
    phase = 2 * np.pi * np.cumsum(inst_freq) * dt_fine  # [rad]
    iq_fine = np.exp(1j * phase)  # unit-amplitude CW

    # Decimate to output sample rate by striding
    iq = iq_fine[:: cfg.interp_oversample]
    t = t_fine[:: cfg.interp_oversample]

    # Add complex AWGN
    snr_lin = 10 ** (cfg.snr_db / 10)
    noise_std = np.sqrt(1.0 / (2.0 * snr_lin))
    noise = noise_std * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))

    # Ground-truth geometry at output sample rate
    if path_provider is not None:
        tx_x, tx_y = path_provider.interpolate(t)
        rx_x = cfg.rx_x0 + cfg.rx_vx * t
        rx_y = np.full_like(t, cfg.rx_y)
        geo = compute_geometry_from_samples(tx_x, tx_y, rx_x, rx_y, t, cfg.fc, cfg.c)
    else:
        geo = compute_geometry(cfg, t)

    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)


def save_iq_wav(iq: np.ndarray, t: np.ndarray, filepath: Path, cfg) -> None:
    """
    Save IQ data to 16-bit WAV file.

    Parameters
    ----------
    iq : np.ndarray
        Complex IQ samples
    t : np.ndarray
        Time array (used to determine sample rate)
    filepath : Path
        Output WAV file path
    cfg : ScenarioConfig
        Configuration (for metadata)
    """
    import wave
    import struct

    # Determine sample rate from time array
    if len(t) > 1:
        fs = 1.0 / (t[1] - t[0])
    else:
        fs = cfg.fs

    # Convert complex IQ to interleaved float32, then to 16-bit integer
    iq_float = np.column_stack((iq.real, iq.imag)).astype(np.float32)

    # Scale to 16-bit range
    iq_int16 = np.clip(iq_float * 32767, -32768, 32767).astype(np.int16)

    # Write WAV file
    with wave.open(str(filepath), "w") as wav:
        wav.setnchannels(2)  # I and Q
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(fs)

        # Interleave I/Q and write
        interleaved = iq_int16.flatten()
        wav.writeframes(interleaved.tobytes())

    # Store metadata in a companion file (WAV INFO chunk is limited)
    metadata_path = filepath.with_suffix(".iqmeta")
    with open(metadata_path, "w") as f:
        f.write(f"fc={cfg.fc}\n")
        f.write(f"fs_orig={cfg.fs}\n")
        f.write(f"duration={cfg.duration}\n")
        f.write(f"snr_db={cfg.snr_db}\n")
        f.write(f"interp_oversample={cfg.interp_oversample}\n")


def load_iq_wav(filepath: Path) -> tuple:
    """
    Load IQ data from 16-bit WAV file.

    Parameters
    ----------
    filepath : Path
        Input WAV file path

    Returns
    -------
    tuple
        (iq, t, metadata) where:
        - iq: complex IQ samples
        - t: time array
        - metadata: dict with fc, fs, duration, etc.
    """
    import wave
    import struct

    # Read WAV file
    with wave.open(str(filepath), "r") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        fs = wav.getframerate()
        n_frames = wav.getnframes()

        if n_channels != 2:
            raise ValueError(f"Expected 2 channels (I/Q), got {n_channels}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit samples, got {sampwidth}")

        # Read interleaved data
        raw = wav.readframes(n_frames)
        data = np.frombuffer(raw, dtype=np.int16)

        # De-interleave to I and Q
        i_data = data[0::2].astype(np.float32) / 32767.0
        q_data = data[1::2].astype(np.float32) / 32767.0

        # Form complex IQ
        iq = i_data + 1j * q_data

    # Load metadata from companion file
    metadata_path = filepath.with_suffix(".iqmeta")
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            for line in f:
                key, val = line.strip().split("=")
                try:
                    metadata[key] = float(val)
                except ValueError:
                    metadata[key] = val

    # Reconstruct time array
    t = np.arange(len(iq)) / fs

    metadata["fs"] = fs

    return iq, t, metadata
