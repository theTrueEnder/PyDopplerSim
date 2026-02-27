"""
Baseband IQ Doppler simulator — full pipeline.

Usage
-----
Edit the TOP-LEVEL CONFIG block below, then run.
All scenario parameters, noise level, and render flags live there.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path


# =============================================================================
# ★  TOP-LEVEL CONFIG — edit this block
# =============================================================================


class SimConfig:
    # ---------- rendering ------------------------------------------------
    RENDER_ANIMATION   = True    # set False to skip video rendering entirely
    ANIMATION_FPS      = 20
    ANIMATION_N_FRAMES = 100     # fewer = faster; 100 is good for GIF
    ANIMATION_DPI_MP4  = 120
    ANIMATION_DPI_GIF  = 80

    # ---------- RF / sampling --------------------------------------------
    FC      = 5.8e9    # carrier frequency [Hz]
    FS      = 1e6      # sample rate [Hz]
    SNR_DB  = 20.0     # signal-to-noise ratio [dB]
    F_TONE  = 0.0      # optional baseband tone offset [Hz]; 0 = pure CW

    # ---------- receiver (always travels in +x) --------------------------
    RX_VX   = 30.0     # receiver velocity [m/s]  (~108 km/h, 67.8 mph)
    RX_X0   = 0.0      # receiver initial x position [m]

    # ---------- scenario-specific Tx params ------------------------------
    # Co-located
    COLOC_TX_Y      = 0.5    # tiny lateral offset so both dots are visible [m]

    # Same-direction
    SAME_TX_X0      = 50.0   # tx starts ahead [m]
    SAME_TX_Y       = 3.7    # lane width [m]
    SAME_TX_VX      = 28.0   # tx velocity [m/s]  (slower → Rx overtakes)
    SAME_DURATION   = 10.0   # [s]

    # Oncoming  (tx_vx < 0 = approaching)
    ONCO_TX_X0      = 400.0  # initial separation [m]
    ONCO_TX_Y       = 3.7    # opposing lane offset [m]
    ONCO_TX_VX      = -30.0  # tx velocity [m/s]  (negative = oncoming)
    ONCO_DURATION   = 13.4   # [s]  CPA at t = TX_X0 / (RX_VX - TX_VX)

    # ---------- estimation -----------------------------------------------
    PD_SMOOTH_WINDOW   = 501   # Hann window length for phase-diff smoother [samples]
    RDOT_SMOOTH_WINDOW = 501   # second Hann window for r_ddot smoother [samples]
    STFT_WINDOW_DUR    = 0.05  # [s]   → freq resolution = 1/window_dur Hz
    STFT_HOP_DUR       = 0.005 # [s]
    INTERP_OVERSAMPLE  = 8     # phase integration oversampling factor


# =============================================================================
# Scenario config dataclass (populated from SimConfig)
# =============================================================================

@dataclass
class ScenarioConfig:
    fc:    float = 5.8e9
    fs:    float = 1e6
    duration: float = 5.0

    tx_x0: float = 100.0
    tx_y:  float = 3.7
    tx_vx: float = 30.0

    rx_x0: float = 0.0
    rx_y:  float = 0.0
    rx_vx: float = 30.0

    snr_db: float = 20.0
    f_tone: float = 0.0
    interp_oversample: int = 8

    def __post_init__(self):
        self.c = 299_792_458.0


def _base_cfg() -> ScenarioConfig:
    """Build a ScenarioConfig from SimConfig globals."""
    return ScenarioConfig(
        fc=SimConfig.FC, fs=SimConfig.FS,
        snr_db=SimConfig.SNR_DB, f_tone=SimConfig.F_TONE,
        rx_x0=SimConfig.RX_X0, rx_vx=SimConfig.RX_VX,
        interp_oversample=SimConfig.INTERP_OVERSAMPLE,
    )


def scenario_colocated() -> ScenarioConfig:
    cfg = _base_cfg()
    cfg.tx_x0    = cfg.rx_x0
    cfg.tx_y     = SimConfig.COLOC_TX_Y
    cfg.tx_vx    = cfg.rx_vx
    cfg.duration = 5.0
    return cfg

def scenario_same_direction() -> ScenarioConfig:
    cfg = _base_cfg()
    cfg.tx_x0    = SimConfig.SAME_TX_X0
    cfg.tx_y     = SimConfig.SAME_TX_Y
    cfg.tx_vx    = SimConfig.SAME_TX_VX
    cfg.duration = SimConfig.SAME_DURATION
    return cfg

def scenario_oncoming() -> ScenarioConfig:
    cfg = _base_cfg()
    cfg.tx_x0    = SimConfig.ONCO_TX_X0
    cfg.tx_y     = SimConfig.ONCO_TX_Y
    cfg.tx_vx    = SimConfig.ONCO_TX_VX
    cfg.duration = SimConfig.ONCO_DURATION
    return cfg


# =============================================================================
# Geometry
# =============================================================================

def compute_geometry(cfg: ScenarioConfig, t: np.ndarray) -> dict:
    tx_x = cfg.tx_x0 + cfg.tx_vx * t
    rx_x = cfg.rx_x0 + cfg.rx_vx * t
    dx   = tx_x - rx_x
    dy   = cfg.tx_y - cfg.rx_y
    r    = np.sqrt(dx**2 + dy**2)
    safe_r = np.where(r < 1e-6, 1e-6, r)

    vdx    = cfg.tx_vx - cfg.rx_vx
    r_dot  = (dx * vdx) / safe_r
    r_ddot = (vdx**2 * safe_r - dx * vdx * r_dot) / safe_r**2

    # Bearing angle: 0° = Tx directly ahead (+x), 90° = Tx to the left (+y)
    los_angle = np.arctan2(dy, dx)

    return dict(
        r=r, r_dot=r_dot, r_ddot=r_ddot,
        delta_f=-r_dot * cfg.fc / cfg.c,
        tx_x=tx_x, rx_x=rx_x,
        tx_y=np.full_like(t, cfg.tx_y),
        rx_y=np.full_like(t, cfg.rx_y),
        los_angle=los_angle,
    )


# =============================================================================
# IQ generation
# =============================================================================

def generate_iq(cfg: ScenarioConfig) -> dict:
    N_fine  = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine  = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]

    geo_fine  = compute_geometry(cfg, t_fine)
    phase     = 2 * np.pi * np.cumsum(cfg.f_tone + geo_fine["delta_f"]) * dt_fine
    iq_fine   = np.exp(1j * phase)

    iq  = iq_fine[::cfg.interp_oversample]
    t   = t_fine[::cfg.interp_oversample]

    snr_lin   = 10 ** (cfg.snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_lin))
    noise     = noise_std * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))

    geo = compute_geometry(cfg, t)
    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)


# =============================================================================
# Estimators
# =============================================================================

def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w < 2:
        return x
    h = np.hanning(w); h /= h.sum()
    return np.convolve(x, h, mode='same')


def estimate_doppler_phase_diff(iq, fs, smooth_window=None):
    if smooth_window is None:
        smooth_window = SimConfig.PD_SMOOTH_WINDOW
    f_inst = np.angle(iq[1:] * np.conj(iq[:-1])) / (2 * np.pi) * fs
    f_inst = _hann_smooth(f_inst, smooth_window)
    return np.arange(len(f_inst)) / fs, f_inst


def estimate_doppler_stft(iq, fs, window_dur=None, hop_dur=None, freq_zoom=5000.0):
    if window_dur is None: window_dur = SimConfig.STFT_WINDOW_DUR
    if hop_dur    is None: hop_dur    = SimConfig.STFT_HOP_DUR
    win_samp = int(window_dur * fs)
    hop_samp = int(hop_dur * fs)
    N_fft    = win_samp * 4
    win      = np.hanning(win_samp)
    n_frames = (len(iq) - win_samp) // hop_samp + 1
    Sxx      = np.zeros((N_fft, n_frames))
    for i in range(n_frames):
        seg = iq[i*hop_samp : i*hop_samp + win_samp] * win
        Sxx[:, i] = np.abs(np.fft.fftshift(np.fft.fft(seg, N_fft)))**2
    freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs))
    t_stft    = (np.arange(n_frames) * hop_samp + win_samp // 2) / fs
    peak_freq = freq_axis[np.argmax(Sxx, axis=0)]
    mask      = np.abs(freq_axis) <= freq_zoom
    return dict(t=t_stft, freq_axis=freq_axis, Sxx=Sxx, peak_freq=peak_freq, mask=mask)


def recover_kinematics(t_est, delta_f_est, fc, c, smooth_window=None):
    """
    Invert Δf → r_dot, then differentiate → r_ddot.
    Edge samples within smooth_window//2 of each boundary are NaN
    (Hann convolution zero-padding artifact).
    """
    if smooth_window is None:
        smooth_window = SimConfig.RDOT_SMOOTH_WINDOW
    r_dot_est  = -delta_f_est * c / fc
    dt         = np.diff(t_est)
    r_ddot_raw = np.concatenate([[0], np.diff(r_dot_est) / dt])
    r_ddot_est = _hann_smooth(r_ddot_raw, smooth_window)
    trim = smooth_window // 2
    r_dot_est[:trim]  = np.nan;  r_dot_est[-trim:]  = np.nan
    r_ddot_est[:trim] = np.nan;  r_ddot_est[-trim:] = np.nan
    return dict(t=t_est, r_dot=r_dot_est, r_ddot=r_ddot_est)


# =============================================================================
# Plot colours
# =============================================================================

COLORS = dict(
    gt   = '#f0f0f0',   # near-white  — ground truth
    pd   = '#ff6b35',   # vivid orange — estimated (phase-diff)
    tx   = '#ffaa00',   # amber        — transmitter
    rx   = '#4dabf7',   # sky blue     — receiver
    stft = '#74c0fc',
)


# =============================================================================
# Static figures
# =============================================================================

def plot_trajectory_fig(result, out_path):
    fig, (ax_2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['scenario_name']} — Trajectory", fontweight='bold')
    t = result["t"]

    ax_2d.plot(result["tx_x"], result["tx_y"], color=COLORS["tx"], lw=2, label="Tx")
    ax_2d.plot(result["rx_x"], result["rx_y"], color=COLORS["rx"], lw=2, label="Rx")
    for xs, ys, col in [(result["tx_x"], result["tx_y"], COLORS["tx"]),
                        (result["rx_x"], result["rx_y"], COLORS["rx"])]:
        mid = len(xs) // 2
        ax_2d.annotate("", xy=(xs[mid+5], ys[mid+5]), xytext=(xs[mid], ys[mid]),
                       arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax_2d.scatter([result["tx_x"][0], result["rx_x"][0]],
                  [result["tx_y"][0], result["rx_y"][0]],
                  color=[COLORS["tx"], COLORS["rx"]], s=60, zorder=5)
    ax_2d.set_xlabel("x [m]"); ax_2d.set_ylabel("y [m]")
    ax_2d.set_title("2-D trajectory (top-down)")
    ax_2d.legend(fontsize=8, loc='upper left'); ax_2d.grid(True, alpha=0.4)
    ax_2d.set_aspect('equal', adjustable='datalim')

    ax_r.plot(t, result["r"], color='purple', lw=1.5)
    ax_r.set_xlabel("Time [s]"); ax_r.set_ylabel("Range r(t) [m]")
    ax_r.set_title("Range over time"); ax_r.grid(True, alpha=0.4)

    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_rdot_rddot_fig(result, est_pd, out_path):
    cfg = result["cfg"]
    t   = result["t"]
    kin = recover_kinematics(est_pd["t"], est_pd["delta_f"], cfg.fc, cfg.c)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Radial kinematics", fontweight='bold')

    axes[0].plot(t, result["r_dot"],  color=COLORS["gt"], lw=1.8, label='Ground truth ṙ')
    axes[0].plot(kin["t"], kin["r_dot"], color=COLORS["pd"], lw=1.0, alpha=0.9,
                 label='Estimated ṙ (phase-diff)')
    axes[0].axhline(0, color='gray', lw=0.8, ls='--')
    axes[0].set_ylabel("ṙ [m/s]"); axes[0].set_title("Radial velocity ṙ(t)")
    axes[0].legend(fontsize=8, loc='upper right'); axes[0].grid(True, alpha=0.4)

    axes[1].plot(t, result["r_ddot"], color=COLORS["gt"], lw=1.8, label='Ground truth r̈')
    axes[1].plot(kin["t"], kin["r_ddot"], color=COLORS["pd"], lw=1.0, alpha=0.9,
                 label='Estimated r̈ (phase-diff)')
    axes[1].axhline(0, color='gray', lw=0.8, ls='--')
    axes[1].set_ylabel("r̈ [m/s²]"); axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Radial acceleration r̈(t)  [peak at CPA]")
    axes[1].legend(fontsize=8, loc='upper right'); axes[1].grid(True, alpha=0.4)

    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_doppler_fig(result, est_pd, out_path):
    cfg  = result["cfg"]
    t    = result["t"]
    gt   = result["delta_f"]
    zoom = max(500.0, 3.0 * np.abs(gt).max())
    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)

    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Doppler estimation", fontweight='bold')

    ax_pd.plot(est_pd["t"], est_pd["delta_f"], color=COLORS["pd"],
               lw=0.8, alpha=0.85, label='Phase-diff Δf')
    ax_pd.plot(t, gt, color=COLORS["gt"], lw=1.5, ls='--', label='Ground truth')
    ax_pd.set_ylabel("Δf [Hz]"); ax_pd.set_title("Instantaneous frequency — phase differentiation")
    ax_pd.legend(fontsize=8, loc='upper right'); ax_pd.grid(True, alpha=0.4)

    mask   = stft["mask"]
    Sxx_dB = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    extent = [stft["t"][0], stft["t"][-1], stft["freq_axis"][mask][0], stft["freq_axis"][mask][-1]]
    ax_stft.imshow(Sxx_dB, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno', vmin=np.percentile(Sxx_dB, 20))
    ax_stft.plot(stft["t"], stft["peak_freq"], color=COLORS["stft"],
                 lw=1.0, ls='--', label='STFT peak')
    ax_stft.plot(t, gt, color=COLORS["gt"], lw=1.5, ls='--', label='Ground truth')
    ax_stft.set_ylabel("Δf [Hz]"); ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (zoomed ±{zoom:.0f} Hz)")
    ax_stft.legend(fontsize=8, loc='upper right')

    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# =============================================================================
# Animation
# =============================================================================

def make_animation(result: dict, out_path: Path):
    fps      = SimConfig.ANIMATION_FPS
    n_frames = SimConfig.ANIMATION_N_FRAMES

    cfg = result["cfg"]
    t   = result["t"]
    N   = len(t)

    idx    = np.linspace(0, N - 1, n_frames, dtype=int)
    t_anim = t[idx]

    tx_x_a = result["tx_x"][idx]
    tx_y_a = result["tx_y"][idx]
    rx_x_a = result["rx_x"][idx]
    rx_y_a = result["rx_y"][idx]
    los_a  = result["los_angle"][idx]   # ground-truth bearing
    r_a    = result["r"][idx]

    # Estimated bearing via back-solving dx from r_dot_est
    # r_dot = vdx * dx / r  →  dx_est = r_dot_est * r / rx_vx  (rx_vx known via GPS)
    t_pd, delta_f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    r_dot_est_full   = -delta_f_pd * cfg.c / cfg.fc
    r_dot_est_a      = np.interp(t_anim, t_pd, r_dot_est_full)
    dy               = cfg.tx_y - cfg.rx_y
    v_ref            = cfg.rx_vx if abs(cfg.rx_vx) > 0.5 else 1.0
    dx_est           = r_dot_est_a * r_a / v_ref
    los_est_a        = np.arctan2(dy, dx_est)   # arctan2 gives correct quadrant

    # ---- Layout ----
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('#1a1a2e')
    gs     = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax2d   = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection='polar')

    ax2d.set_facecolor('#16213e')
    for sp in ax2d.spines.values():
        sp.set_edgecolor('#555')
    ax_pol.set_facecolor('#16213e')
    ax_pol.tick_params(colors='#aaa')
    ax_pol.spines['polar'].set_color('#555')

    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color='white')
    ax2d.set_ylabel("y [m]", color='white')
    ax2d.tick_params(colors='white')
    ax2d.grid(True, alpha=0.25, color='#555')
    ax2d.set_title("Vehicle positions", color='white', fontsize=10)

    # Polar: unit-radius circle — angle = bearing, r = 1 always.
    # This makes the angular sweep visible regardless of range scale.
    # Range is displayed as text overlay instead.
    ax_pol.set_theta_zero_location('E')   # 0° = East = Tx directly ahead
    ax_pol.set_theta_direction(1)         # counter-clockwise
    ax_pol.set_ylim(0, 1.3)
    ax_pol.set_yticks([])                 # hide radial ticks — radius is meaningless here
    ax_pol.set_title("Tx bearing\n(from Rx, 0°=ahead)", color='white', fontsize=9, pad=12)

    # Cardinal labels
    for angle, label in [(0, 'Ahead'), (np.pi/2, 'Left'), (np.pi, 'Behind'), (-np.pi/2, 'Right')]:
        ax_pol.text(angle, 1.22, label, ha='center', va='center',
                    color='#888', fontsize=7)

    # Static faint path traces
    ax2d.plot(result["tx_x"], result["tx_y"], color=COLORS["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=COLORS["rx"], alpha=0.15, lw=1)

    # Legend via dummy artists (explicit loc → no 'best' slowness)
    ax2d.plot([], [], 'o', color=COLORS["tx"], ms=7, label='Tx')
    ax2d.plot([], [], 'o', color=COLORS["rx"], ms=7, label='Rx')
    ax2d.legend(fontsize=8, facecolor='#222', labelcolor='white', loc='lower right')

    # Polar legend — gt = white, est = orange (matches all other plots)
    ax_pol.plot([], [], 'o', color=COLORS["gt"],  ms=6, label='True bearing')
    ax_pol.plot([], [], 'o', color=COLORS["pd"],  ms=6, label='Est. bearing')
    ax_pol.legend(fontsize=7, facecolor='#222', labelcolor='white',
                  loc='upper left', bbox_to_anchor=(1.05, 1.12))

    trail_len = max(5, n_frames // 15)

    tx_trail, = ax2d.plot([], [], color=COLORS["tx"], lw=1.5, alpha=0.5)
    rx_trail, = ax2d.plot([], [], color=COLORS["rx"], lw=1.5, alpha=0.5)
    tx_dot,   = ax2d.plot([], [], 'o', color=COLORS["tx"], ms=10, zorder=5)
    rx_dot,   = ax2d.plot([], [], 'o', color=COLORS["rx"], ms=10, zorder=5)
    los_line, = ax2d.plot([], [], '--', color='white', lw=0.8, alpha=0.5)
    time_txt  = ax2d.text(0.02, 0.95, '', transform=ax2d.transAxes,
                          color='white', fontsize=9, va='top')

    # Polar: needle from origin to unit circle + small dot at tip
    pol_gt_line,  = ax_pol.plot([], [], '-', color=COLORS["gt"],  lw=2.0)
    pol_est_line, = ax_pol.plot([], [], '-', color=COLORS["pd"],  lw=2.0, alpha=0.9)
    pol_gt_dot,   = ax_pol.plot([], [], 'o', color=COLORS["gt"],  ms=7)
    pol_est_dot,  = ax_pol.plot([], [], 'o', color=COLORS["pd"],  ms=7)
    range_txt     = ax_pol.text(np.pi * 1.25, 1.5, '', color='white',
                                fontsize=8, ha='center', va='center')

    all_artists = [tx_trail, rx_trail, tx_dot, rx_dot, los_line,
                   pol_gt_line, pol_est_line, pol_gt_dot, pol_est_dot,
                   time_txt, range_txt]

    def init():
        for a in all_artists:
            if hasattr(a, 'set_data'):
                a.set_data([], [])
            else:
                a.set_text('')
        return all_artists

    def update(i):
        sl = slice(max(0, i - trail_len), i + 1)
        tx_trail.set_data(tx_x_a[sl], tx_y_a[sl])
        rx_trail.set_data(rx_x_a[sl], rx_y_a[sl])
        tx_dot.set_data([tx_x_a[i]], [tx_y_a[i]])
        rx_dot.set_data([rx_x_a[i]], [rx_y_a[i]])
        los_line.set_data([rx_x_a[i], tx_x_a[i]], [rx_y_a[i], tx_y_a[i]])
        time_txt.set_text(f"t = {t_anim[i]:.2f} s")

        # Needle: [origin, tip] at unit radius
        pol_gt_line.set_data( [los_a[i],     los_a[i]],     [0, 1.0])
        pol_est_line.set_data([los_est_a[i], los_est_a[i]], [0, 1.0])
        pol_gt_dot.set_data( [los_a[i]],     [1.0])
        pol_est_dot.set_data([los_est_a[i]], [1.0])
        range_txt.set_text(f"r = {r_a[i]:.0f} m")

        return all_artists

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         blit=True, interval=1000 / fps)

    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(title=result["scenario_name"]))
        anim.save(str(out_path.with_suffix(".mp4")), writer=writer,
                  dpi=SimConfig.ANIMATION_DPI_MP4,
                  savefig_kwargs=dict(facecolor=fig.get_facecolor()))
        print(f"    Animation → {out_path.with_suffix('.mp4')}")
    except Exception as e:
        print(f"    ffmpeg unavailable ({e}), falling back to GIF...")
        writer = PillowWriter(fps=fps)
        anim.save(str(out_path.with_suffix(".gif")), writer=writer,
                  dpi=SimConfig.ANIMATION_DPI_GIF,
                  savefig_kwargs=dict(facecolor=fig.get_facecolor()))
        print(f"    Animation → {out_path.with_suffix('.gif')}")

    plt.close(fig)


# =============================================================================
# Save all outputs for one scenario
# =============================================================================

def save_all(result: dict, out_root: str = "plots"):
    name    = result["scenario_name"]
    slug    = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    t_pd, delta_f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    est_pd = {"t": t_pd, "delta_f": delta_f_pd}

    print(f"  Saving static plots...")
    plot_trajectory_fig(result, out_dir / "trajectory.png")
    plot_rdot_rddot_fig(result, est_pd, out_dir / "rdot_rddot.png")
    plot_doppler_fig(result, est_pd, out_dir / "doppler.png")

    if SimConfig.RENDER_ANIMATION:
        print(f"  Rendering animation...")
        make_animation(result, out_dir / "animation")
    else:
        print(f"  Skipping animation (RENDER_ANIMATION=False)")

    print(f"  All outputs → {out_dir}/")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    scenarios = [
        # ("Co-located",     scenario_colocated()),
        # ("Same-direction", scenario_same_direction()),
        ("Oncoming",       scenario_oncoming()),
    ]

    for name, cfg in scenarios:
        print(f"\n=== {name} ===")
        result = generate_iq(cfg)
        result["scenario_name"] = name
        cpa_t = cfg.tx_x0 / max(abs(cfg.rx_vx - cfg.tx_vx), 1e-3)
        print(f"  Duration:       {cfg.duration:.1f} s  ({len(result['iq']):,} samples)")
        print(f"  Rx velocity:    {cfg.rx_vx:.1f} m/s   Tx velocity: {cfg.tx_vx:.1f} m/s")
        print(f"  Δf range:       [{result['delta_f'].min():.2f}, {result['delta_f'].max():.2f}] Hz")
        print(f"  r̈ peak (GT):    {np.abs(result['r_ddot']).max():.1f} m/s²")
        print(f"  Range CPA:      {result['r'].min():.1f} m  (at t ≈ {cpa_t:.2f} s)")
        save_all(result, "plots")