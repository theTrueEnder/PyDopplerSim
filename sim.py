"""
Baseband IQ Doppler simulator for vehicle-mounted SDR Doppler discrimination.

Geometry: 2D top-down view, vehicles travel along x-axis.
          Lateral offset 'd' is separation in y-axis.

Coordinate convention:
  - rx travels in +x direction at v_rx
  - tx travels in +x direction at v_tx (negative = oncoming)
  - lateral offset d > 0: tx is in adjacent lane (y = d), rx at y = 0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — write to files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    fc:       float = 5.8e9    # carrier frequency [Hz]
    fs:       float = 1e6      # sample rate [Hz]
    duration: float = 5.0      # simulation duration [s]

    tx_x0: float = 100.0       # initial x position of tx [m]
    tx_y:  float = 3.7         # lateral offset [m]; 0 = co-located
    tx_vx: float = 30.0        # tx velocity in x [m/s]; negative = oncoming

    rx_x0: float = 0.0
    rx_y:  float = 0.0
    rx_vx: float = 30.0        # rx velocity in x [m/s]

    snr_db:  float = 20.0
    f_tone:  float = 0.0       # baseband tone offset [Hz]

    interp_oversample: int = 8

    def __post_init__(self):
        self.c = 299_792_458.0


# ---------------------------------------------------------------------------
# Scenario factories
# ---------------------------------------------------------------------------

def scenario_colocated() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.tx_x0 = cfg.rx_x0
    cfg.tx_y  = cfg.rx_y
    cfg.tx_vx = cfg.rx_vx
    cfg.duration = 5.0
    return cfg

def scenario_same_direction() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.tx_x0 = 50.0
    cfg.tx_y  = 3.7
    cfg.tx_vx = 28.0   # 2 m/s slower → slow drift
    cfg.rx_vx = 30.0
    cfg.duration = 10.0
    return cfg

def scenario_oncoming() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.tx_x0 = 300.0
    cfg.tx_y  = 3.7
    cfg.tx_vx = -30.0  # head-on
    cfg.rx_vx = 30.0
    cfg.duration = 5.0
    return cfg


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_geometry(cfg: ScenarioConfig, t: np.ndarray) -> dict:
    """
    Returns a dict with:
        r       - range [m]
        r_dot   - radial velocity [m/s]  (+ve = moving apart)
        r_ddot  - radial acceleration [m/s²]
        delta_f - Doppler shift [Hz]
        tx_x, tx_y, rx_x, rx_y - positions for trajectory plots
    """
    tx_x = cfg.tx_x0 + cfg.tx_vx * t
    rx_x = cfg.rx_x0 + cfg.rx_vx * t

    dx   = tx_x - rx_x
    dy   = cfg.tx_y - cfg.rx_y   # constant (straight road, no lateral motion)
    r    = np.sqrt(dx**2 + dy**2)

    vdx  = cfg.tx_vx - cfg.rx_vx
    # vdy = 0  (no lateral motion)

    safe_r = np.where(r < 1e-6, 1e-6, r)

    # r_dot = (dx*vdx) / r
    r_dot  = (dx * vdx) / safe_r

    # r_ddot = d/dt [ (dx*vdx) / r ]
    #        = (vdx² * r  -  dx*vdx * r_dot) / r²
    #   (using quotient rule; vdx is constant, d(dx)/dt = vdx)
    r_ddot = (vdx**2 * safe_r - dx * vdx * r_dot) / safe_r**2

    delta_f = -r_dot * cfg.fc / cfg.c

    return dict(
        r=r, r_dot=r_dot, r_ddot=r_ddot,
        delta_f=delta_f,
        tx_x=tx_x, rx_x=rx_x,
        tx_y=np.full_like(t, cfg.tx_y),
        rx_y=np.full_like(t, cfg.rx_y),
    )


# ---------------------------------------------------------------------------
# IQ signal generation
# ---------------------------------------------------------------------------

def generate_iq(cfg: ScenarioConfig) -> dict:
    """
    Phase is computed by integrating instantaneous frequency on a fine grid
    (oversampled × interp_oversample) then decimated to fs.

    φ(t) = 2π ∫₀ᵗ [f_tone + Δf(τ)] dτ
    """
    N_fine  = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine  = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]

    geo_fine = compute_geometry(cfg, t_fine)
    inst_freq = cfg.f_tone + geo_fine["delta_f"]

    phase    = 2 * np.pi * np.cumsum(inst_freq) * dt_fine
    iq_fine  = np.exp(1j * phase)

    # Decimate
    iq   = iq_fine[::cfg.interp_oversample]
    t    = t_fine[::cfg.interp_oversample]
    N    = len(iq)

    geo  = compute_geometry(cfg, t)

    snr_lin   = 10 ** (cfg.snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_lin))
    noise     = noise_std * (np.random.randn(N) + 1j * np.random.randn(N))

    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)


# ---------------------------------------------------------------------------
# Doppler estimators
# ---------------------------------------------------------------------------

def estimate_doppler_stft(
    iq: np.ndarray,
    fs: float,
    window_dur: float = 0.05,   # 50 ms → freq resolution 20 Hz
    hop_dur:    float = 0.005,
    freq_zoom:  float = 5000.0, # display ± this many Hz
) -> dict:
    win_samp = int(window_dur * fs)
    hop_samp = int(hop_dur * fs)
    # Zero-pad to 4× window for smoother peak interpolation
    N_fft    = win_samp * 4
    win      = np.hanning(win_samp)

    n_frames = (len(iq) - win_samp) // hop_samp + 1
    Sxx      = np.zeros((N_fft, n_frames))

    for i in range(n_frames):
        seg       = iq[i*hop_samp : i*hop_samp + win_samp] * win
        Sxx[:, i] = np.abs(np.fft.fftshift(np.fft.fft(seg, N_fft)))**2

    freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs))
    t_stft    = (np.arange(n_frames) * hop_samp + win_samp // 2) / fs
    peak_freq = freq_axis[np.argmax(Sxx, axis=0)]

    # Zoom mask for display
    mask      = np.abs(freq_axis) <= freq_zoom

    return dict(
        t=t_stft,
        freq_axis=freq_axis,
        Sxx=Sxx,
        peak_freq=peak_freq,
        mask=mask,
    )


def estimate_doppler_phase_diff(
    iq: np.ndarray,
    fs: float,
    smooth_window: int = 501,
) -> Tuple[np.ndarray, np.ndarray]:
    """FM discriminator: f_inst[n] = angle(iq[n] * conj(iq[n-1])) * fs / 2π"""
    lag1   = iq[1:] * np.conj(iq[:-1])
    f_inst = np.angle(lag1) / (2 * np.pi) * fs
    if smooth_window > 1:
        h      = np.hanning(smooth_window)
        h     /= h.sum()
        f_inst = np.convolve(f_inst, h, mode='same')
    return np.arange(len(f_inst)) / fs, f_inst


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = dict(gt='black', pd='steelblue', stft='tomato', tx='orange', rx='royalblue')


def plot_trajectory(result: dict, ax_2d: plt.Axes, ax_range: plt.Axes):
    """2-D top-down trajectory + range vs time."""
    cfg  = result["cfg"]
    t    = result["t"]

    tx_x, tx_y = result["tx_x"], result["tx_y"]
    rx_x, rx_y = result["rx_x"], result["rx_y"]

    # 2-D trajectory
    ax_2d.plot(tx_x, tx_y, color=COLORS["tx"], lw=2, label="Tx")
    ax_2d.plot(rx_x, rx_y, color=COLORS["rx"], lw=2, label="Rx")

    # Mark start/end with arrows
    for xs, ys, col in [(tx_x, tx_y, COLORS["tx"]), (rx_x, ry := rx_y, COLORS["rx"])]:
        ax_2d.annotate("", xy=(xs[len(xs)//2 + 5], ys[len(ys)//2 + 5]),
                       xytext=(xs[len(xs)//2], ys[len(ys)//2]),
                       arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax_2d.scatter([tx_x[0], rx_x[0]], [tx_y[0], rx_y[0]], s=60, zorder=5,
                  color=[COLORS["tx"], COLORS["rx"]], marker='o')
    ax_2d.scatter([tx_x[-1], rx_x[-1]], [tx_y[-1], rx_y[-1]], s=60, zorder=5,
                  color=[COLORS["tx"], COLORS["rx"]], marker='s')
    ax_2d.set_xlabel("x position [m]")
    ax_2d.set_ylabel("y position [m]")
    ax_2d.set_title("2-D Trajectory (top-down)")
    ax_2d.legend(fontsize=8)
    ax_2d.grid(True, alpha=0.4)
    ax_2d.set_aspect('equal', adjustable='datalim')

    # Range
    ax_range.plot(t, result["r"], color='purple', lw=1.5)
    ax_range.set_xlabel("Time [s]")
    ax_range.set_ylabel("Range r(t) [m]")
    ax_range.set_title("Range over time")
    ax_range.grid(True, alpha=0.4)


def plot_rdot_rddot(result: dict, ax_rd: plt.Axes, ax_rdd: plt.Axes):
    t      = result["t"]
    r_dot  = result["r_dot"]
    r_ddot = result["r_ddot"]

    ax_rd.plot(t, r_dot, color=COLORS["gt"], lw=1.5)
    ax_rd.axhline(0, color='gray', lw=0.8, ls='--')
    ax_rd.set_ylabel("ṙ [m/s]")
    ax_rd.set_title("Radial velocity ṙ(t)  [+ = moving apart]")
    ax_rd.grid(True, alpha=0.4)

    ax_rdd.plot(t, r_ddot, color='darkgreen', lw=1.5)
    ax_rdd.axhline(0, color='gray', lw=0.8, ls='--')
    ax_rdd.set_ylabel("r̈ [m/s²]")
    ax_rdd.set_xlabel("Time [s]")
    ax_rdd.set_title("Radial acceleration r̈(t)  [peak at CPA]")
    ax_rdd.grid(True, alpha=0.4)


def plot_doppler_estimates(result: dict, ax_pd: plt.Axes, ax_stft: plt.Axes):
    cfg = result["cfg"]
    t   = result["t"]
    gt  = result["delta_f"]

    # Phase-diff
    t_pd, f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    ax_pd.plot(t_pd, f_pd, color=COLORS["pd"], lw=0.8, alpha=0.85, label='Phase-diff')
    ax_pd.plot(t, gt, color=COLORS["gt"], lw=1.5, ls='--', label='Ground truth')
    ax_pd.set_ylabel("Δf [Hz]")
    ax_pd.set_title("Instantaneous frequency — phase differentiation")
    ax_pd.legend(fontsize=8)
    ax_pd.grid(True, alpha=0.4)

    # STFT spectrogram (zoomed)
    # Auto zoom: use 3× the max ground-truth |Δf| or minimum 500 Hz
    zoom = max(500.0, 3.0 * np.abs(gt).max())
    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)

    mask    = stft["mask"]
    f_zoom  = stft["freq_axis"][mask]
    Sxx_dB  = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    vmin    = np.percentile(Sxx_dB, 20)

    extent  = [stft["t"][0], stft["t"][-1], f_zoom[0], f_zoom[-1]]
    ax_stft.imshow(Sxx_dB, aspect='auto', origin='lower',
                   extent=extent, cmap='inferno', vmin=vmin)
    ax_stft.plot(stft["t"], stft["peak_freq"], color='cyan', lw=1.0,
                 ls='--', label='STFT peak')
    ax_stft.plot(t, gt, color='white', lw=1.5, ls='--', label='Ground truth')
    ax_stft.set_ylabel("Δf [Hz]")
    ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (zoomed ±{zoom:.0f} Hz)")
    ax_stft.legend(fontsize=8, loc='upper right')


def save_all_plots(result: dict, scenario_name: str, out_root: str = "plots"):
    """
    Saves three figures per scenario into plots/<scenario_name>/:
      1. trajectory.png   - 2D top-down path + range
      2. rdot_rddot.png   - radial velocity & acceleration
      3. doppler.png      - phase-diff estimate + STFT spectrogram
    """
    slug = scenario_name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Trajectory ---
    fig, (ax_2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{scenario_name} — Trajectory", fontweight='bold')
    plot_trajectory(result, ax_2d, ax_r)
    plt.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: ṙ and r̈ ---
    fig, (ax_rd, ax_rdd) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{scenario_name} — Radial kinematics", fontweight='bold')
    plot_rdot_rddot(result, ax_rd, ax_rdd)
    plt.tight_layout()
    fig.savefig(out_dir / "rdot_rddot.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Doppler estimates ---
    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{scenario_name} — Doppler estimation", fontweight='bold')
    plot_doppler_estimates(result, ax_pd, ax_stft)
    plt.tight_layout()
    fig.savefig(out_dir / "doppler.png", dpi=150)
    plt.close(fig)

    print(f"  Saved plots → {out_dir}\\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    scenarios = [
        ("Co-located",     scenario_colocated()),
        ("Same-direction", scenario_same_direction()),
        ("Oncoming",       scenario_oncoming()),
    ]

    for name, cfg in scenarios:
        print(f"\n=== {name} ===")
        result = generate_iq(cfg)
        print(f"  Duration:     {cfg.duration:.1f} s  ({len(result['iq']):,} samples @ {cfg.fs/1e6:.1f} MHz)")
        print(f"  Δf range:     [{result['delta_f'].min():.2f}, {result['delta_f'].max():.2f}] Hz")
        print(f"  r̈ peak:       {np.abs(result['r_ddot']).max():.4f} m/s²")
        print(f"  Range min:    {result['r'].min():.1f} m  (CPA)")
        save_all_plots(result, name)