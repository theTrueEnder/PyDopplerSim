"""
Baseband IQ Doppler simulator — full pipeline:
  geometry → IQ generation → Doppler estimation → r_dot/r_ddot recovery → plots + animation

Coordinate convention:
  - Both vehicles travel along x-axis
  - tx_y > 0: lateral lane offset
  - rx always at y = 0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    fc:       float = 5.8e9
    fs:       float = 1e6
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


def scenario_colocated() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.tx_x0, cfg.tx_y, cfg.tx_vx = cfg.rx_x0, cfg.rx_y, cfg.rx_vx
    cfg.duration = 5.0
    return cfg

def scenario_same_direction() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.tx_x0, cfg.tx_y, cfg.tx_vx = 50.0, 3.7, 28.0
    cfg.duration = 10.0
    return cfg

def scenario_oncoming() -> ScenarioConfig:
    cfg = ScenarioConfig()
    cfg.tx_x0, cfg.tx_y, cfg.tx_vx = 400.0, 3.7, -30.0
    cfg.duration = 8.0   # CPA at ~t=3.3 s; vehicles well separated before and after
    return cfg


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_geometry(cfg: ScenarioConfig, t: np.ndarray) -> dict:
    tx_x = cfg.tx_x0 + cfg.tx_vx * t
    rx_x = cfg.rx_x0 + cfg.rx_vx * t

    dx   = tx_x - rx_x
    dy   = cfg.tx_y - cfg.rx_y   # constant
    r    = np.sqrt(dx**2 + dy**2)
    safe_r = np.where(r < 1e-6, 1e-6, r)

    vdx    = cfg.tx_vx - cfg.rx_vx
    r_dot  = (dx * vdx) / safe_r
    r_ddot = (vdx**2 * safe_r - dx * vdx * r_dot) / safe_r**2

    # LoS unit vector angle (for polar plot): angle of (tx - rx) in world frame
    los_angle = np.arctan2(dy, dx)   # scalar dy is fine; dx varies

    return dict(
        r=r, r_dot=r_dot, r_ddot=r_ddot,
        delta_f=-r_dot * cfg.fc / cfg.c,
        tx_x=tx_x, rx_x=rx_x,
        tx_y=np.full_like(t, cfg.tx_y),
        rx_y=np.full_like(t, cfg.rx_y),
        los_angle=los_angle,
    )


# ---------------------------------------------------------------------------
# IQ generation
# ---------------------------------------------------------------------------

def generate_iq(cfg: ScenarioConfig) -> dict:
    N_fine  = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine  = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]

    geo_fine  = compute_geometry(cfg, t_fine)
    inst_freq = cfg.f_tone + geo_fine["delta_f"]
    phase     = 2 * np.pi * np.cumsum(inst_freq) * dt_fine
    iq_fine   = np.exp(1j * phase)

    iq  = iq_fine[::cfg.interp_oversample]
    t   = t_fine[::cfg.interp_oversample]

    snr_lin   = 10 ** (cfg.snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_lin))
    noise     = noise_std * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))

    geo = compute_geometry(cfg, t)
    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)


# ---------------------------------------------------------------------------
# Doppler estimators
# ---------------------------------------------------------------------------

def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w < 2:
        return x
    h = np.hanning(w); h /= h.sum()
    return np.convolve(x, h, mode='same')


def estimate_doppler_phase_diff(iq, fs, smooth_window=501):
    """FM discriminator → instantaneous Δf [Hz]"""
    f_inst = np.angle(iq[1:] * np.conj(iq[:-1])) / (2 * np.pi) * fs
    f_inst = _hann_smooth(f_inst, smooth_window)
    t = np.arange(len(f_inst)) / fs
    return t, f_inst


def estimate_doppler_stft(iq, fs, window_dur=0.05, hop_dur=0.005, freq_zoom=5000.0):
    win_samp = int(window_dur * fs)
    hop_samp = int(hop_dur * fs)
    N_fft    = win_samp * 4
    win      = np.hanning(win_samp)
    n_frames = (len(iq) - win_samp) // hop_samp + 1

    Sxx = np.zeros((N_fft, n_frames))
    for i in range(n_frames):
        seg = iq[i*hop_samp : i*hop_samp + win_samp] * win
        Sxx[:, i] = np.abs(np.fft.fftshift(np.fft.fft(seg, N_fft)))**2

    freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, 1/fs))
    t_stft    = (np.arange(n_frames) * hop_samp + win_samp // 2) / fs
    peak_freq = freq_axis[np.argmax(Sxx, axis=0)]
    mask      = np.abs(freq_axis) <= freq_zoom
    return dict(t=t_stft, freq_axis=freq_axis, Sxx=Sxx, peak_freq=peak_freq, mask=mask)


# ---------------------------------------------------------------------------
# Kinematic recovery from estimated Δf
# ---------------------------------------------------------------------------

def recover_kinematics(
    t_est: np.ndarray,
    delta_f_est: np.ndarray,
    fc: float,
    c: float,
    smooth_window: int = 501,
) -> dict:
    """
    Invert Doppler equation to get estimated r_dot, then numerically
    differentiate with smoothing to get r_ddot.

        r_dot_est  = -Δf_est * c / fc
        r_ddot_est = d(r_dot_est)/dt  ← finite-difference + Hann smoothing

    The double-smoothing (once on Δf, once on r_ddot) is intentional:
    differentiation is a high-pass operation that amplifies noise, so
    the second smooth keeps r_ddot usable for classification.
    """
    r_dot_est  = -delta_f_est * c / fc

    # Central-difference derivative; edges use forward/backward diff
    dt        = np.diff(t_est)
    dr        = np.diff(r_dot_est)
    r_ddot_raw = dr / dt                          # length N-1
    # Pad to same length as r_dot_est using edge values
    r_ddot_raw = np.concatenate([[r_ddot_raw[0]], r_ddot_raw])
    r_ddot_est = _hann_smooth(r_ddot_raw, smooth_window)

    return dict(r_dot=r_dot_est, r_ddot=r_ddot_est)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = dict(gt='black', pd='steelblue', stft='tomato', tx='darkorange', rx='royalblue')


def plot_trajectory_fig(result, out_path):
    cfg = result["cfg"]
    t   = result["t"]
    fig, (ax_2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['scenario_name']} — Trajectory", fontweight='bold')

    ax_2d.plot(result["tx_x"], result["tx_y"], color=COLORS["tx"], lw=2, label="Tx")
    ax_2d.plot(result["rx_x"], result["rx_y"], color=COLORS["rx"], lw=2, label="Rx")
    for xs, ys, col in [(result["tx_x"], result["tx_y"], COLORS["tx"]),
                        (result["rx_x"], result["rx_y"], COLORS["rx"])]:
        mid = len(xs) // 2
        ax_2d.annotate("", xy=(xs[mid+5], ys[mid+5]), xytext=(xs[mid], ys[mid]),
                       arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax_2d.scatter([result["tx_x"][0], result["rx_x"][0]],
                  [result["tx_y"][0], result["rx_y"][0]],
                  color=[COLORS["tx"], COLORS["rx"]], s=60, zorder=5, marker='o')
    ax_2d.set_xlabel("x [m]"); ax_2d.set_ylabel("y [m]")
    ax_2d.set_title("2-D trajectory (top-down)")
    ax_2d.legend(fontsize=8); ax_2d.grid(True, alpha=0.4)
    ax_2d.set_aspect('equal', adjustable='datalim')

    ax_r.plot(t, result["r"], color='purple', lw=1.5)
    ax_r.set_xlabel("Time [s]"); ax_r.set_ylabel("Range r(t) [m]")
    ax_r.set_title("Range over time"); ax_r.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rdot_rddot_fig(result, est_pd, out_path):
    """Plot ground-truth AND estimated r_dot / r_ddot from phase-diff."""
    t   = result["t"]
    cfg = result["cfg"]

    # Recover kinematics from phase-diff estimate
    kin = recover_kinematics(est_pd["t"], est_pd["delta_f"], cfg.fc, cfg.c)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Radial kinematics", fontweight='bold')

    ax = axes[0]
    ax.plot(t, result["r_dot"], color=COLORS["gt"], lw=1.8, label='Ground truth ṙ')
    ax.plot(kin["t"] if "t" in kin else est_pd["t"], kin["r_dot"],
            color=COLORS["pd"], lw=1.0, alpha=0.85, label='Estimated ṙ (phase-diff)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_ylabel("ṙ [m/s]"); ax.set_title("Radial velocity ṙ(t)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t, result["r_ddot"], color=COLORS["gt"], lw=1.8, label='Ground truth r̈')
    ax.plot(est_pd["t"], kin["r_ddot"],
            color=COLORS["pd"], lw=1.0, alpha=0.85, label='Estimated r̈ (phase-diff)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_ylabel("r̈ [m/s²]"); ax.set_xlabel("Time [s]")
    ax.set_title("Radial acceleration r̈(t)  [peak at CPA]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_doppler_fig(result, est_pd, out_path):
    cfg = result["cfg"]
    t   = result["t"]
    gt  = result["delta_f"]
    zoom = max(500.0, 3.0 * np.abs(gt).max())

    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)

    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Doppler estimation", fontweight='bold')

    ax_pd.plot(est_pd["t"], est_pd["delta_f"], color=COLORS["pd"],
               lw=0.8, alpha=0.85, label='Phase-diff Δf')
    ax_pd.plot(t, gt, color=COLORS["gt"], lw=1.5, ls='--', label='Ground truth')
    ax_pd.set_ylabel("Δf [Hz]")
    ax_pd.set_title("Instantaneous frequency — phase differentiation")
    ax_pd.legend(fontsize=8); ax_pd.grid(True, alpha=0.4)

    mask   = stft["mask"]
    f_zoom = stft["freq_axis"][mask]
    Sxx_dB = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    extent = [stft["t"][0], stft["t"][-1], f_zoom[0], f_zoom[-1]]
    ax_stft.imshow(Sxx_dB, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno', vmin=np.percentile(Sxx_dB, 20))
    ax_stft.plot(stft["t"], stft["peak_freq"], color='cyan', lw=1.0, ls='--', label='STFT peak')
    ax_stft.plot(t, gt, color='white', lw=1.5, ls='--', label='Ground truth')
    ax_stft.set_ylabel("Δf [Hz]"); ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (zoomed ±{zoom:.0f} Hz)")
    ax_stft.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def make_animation(result: dict, out_path: Path, fps: int = 20, n_frames: int = 200):
    """
    Two-panel animation:
      Left : 2-D top-down position of Tx and Rx over time
      Right: Polar plot — ground-truth LoS vector (black) and
             estimated LoS angle recovered from phase-diff r_dot (blue).

    Estimating the LoS *angle* from r_dot alone is underdetermined (r_dot = |v_rel| cos θ,
    but we don't know |v_rel| from the receiver's perspective in a blind scenario).
    Here we take a semi-blind approach: the receiver knows its own velocity (GPS/IMU)
    and fc, so it computes r_dot_est from Δf_est. We then show r_dot_est as the
    *radial component* on the polar plot (radius = |r_dot|, angle = estimated LoS),
    where the estimated LoS angle is recovered via:
        cos(θ) = r_dot_est / |v_rel_assumed|
    using v_rel_assumed = rx_vx (self-velocity, known from GPS) as a lower bound.
    This makes the polar plot physically interpretable.
    """
    cfg = result["cfg"]
    t   = result["t"]
    N   = len(t)

    # Subsample to n_frames
    idx    = np.linspace(0, N - 1, n_frames, dtype=int)
    t_anim = t[idx]

    # Ground-truth quantities at animation frames
    tx_x_a = result["tx_x"][idx]
    tx_y_a = result["tx_y"][idx]
    rx_x_a = result["rx_x"][idx]
    rx_y_a = result["rx_y"][idx]
    los_a  = result["los_angle"][idx]       # ground-truth LoS angle
    r_a    = result["r"][idx]
    r_dot_a = result["r_dot"][idx]

    # Estimated r_dot from phase-diff (interpolate to animation frames)
    t_pd, delta_f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    r_dot_pd_full = -delta_f_pd * cfg.c / cfg.fc
    r_dot_pd_a    = np.interp(t_anim, t_pd, r_dot_pd_full)

    # Estimated LoS angle: cos(θ) = r_dot / v_rel_x, clipped to [-1, 1]
    # v_rel_x is not known blind, but Rx knows its own speed (GPS).
    # We use rx_vx as the reference — gives the "minimum angle" estimate.
    v_ref = abs(cfg.rx_vx) if abs(cfg.rx_vx) > 1.0 else 1.0
    cos_theta_est = np.clip(r_dot_pd_a / v_ref, -1.0, 1.0)
    # Sign of dy determines which half-plane: use ground truth dy sign (known from lane model)
    los_est_a = np.arccos(cos_theta_est) * np.sign(cfg.tx_y - cfg.rx_y + 1e-9)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('#1a1a2e')
    gs   = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])

    ax2d  = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection='polar')

    for ax in [ax2d]:
        ax.set_facecolor('#16213e')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
    ax_pol.set_facecolor('#16213e')

    # 2-D axes limits (fixed)
    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = (all_x.max() - all_x.min()) * 0.05
    x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
    y_vals = [cfg.tx_y, cfg.rx_y]
    y_min  = min(y_vals) - 10
    y_max  = max(y_vals) + 10

    ax2d.set_xlim(x_min, x_max)
    ax2d.set_ylim(y_min, y_max)
    ax2d.set_xlabel("x [m]", color='white'); ax2d.set_ylabel("y [m]", color='white')
    ax2d.tick_params(colors='white'); ax2d.grid(True, alpha=0.25, color='#555')
    ax2d.set_title("Vehicle positions", color='white', fontsize=10)

    # Polar axes
    ax_pol.set_theta_zero_location('E')
    ax_pol.set_theta_direction(1)
    ax_pol.tick_params(colors='#aaa')
    ax_pol.yaxis.label.set_color('white')
    ax_pol.set_title("LoS direction\n(from Rx toward Tx)", color='white', fontsize=9, pad=12)
    r_max_pol = r_a.max() * 1.1
    ax_pol.set_ylim(0, r_max_pol)
    ax_pol.set_rlabel_position(45)

    # Static full-path traces (faint)
    ax2d.plot(result["tx_x"], result["tx_y"], color=COLORS["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=COLORS["rx"], alpha=0.15, lw=1)

    # Animated artists — 2D
    trail_len = max(5, n_frames // 20)
    tx_trail,  = ax2d.plot([], [], color=COLORS["tx"], lw=1.5, alpha=0.6)
    rx_trail,  = ax2d.plot([], [], color=COLORS["rx"], lw=1.5, alpha=0.6)
    tx_dot,    = ax2d.plot([], [], 'o', color=COLORS["tx"], ms=9, label='Tx', zorder=5)
    rx_dot,    = ax2d.plot([], [], 'o', color=COLORS["rx"], ms=9, label='Rx', zorder=5)
    los_line,  = ax2d.plot([], [], '--', color='white', lw=0.8, alpha=0.6)
    time_txt   = ax2d.text(0.02, 0.95, '', transform=ax2d.transAxes,
                           color='white', fontsize=9, va='top')
    ax2d.legend(fontsize=8, facecolor='#222', labelcolor='white', loc='lower right')

    # Animated artists — polar
    pol_gt,  = ax_pol.plot([], [], '-o', color=COLORS["gt"], ms=5, lw=1.5, label='GT LoS')
    pol_est, = ax_pol.plot([], [], '-o', color=COLORS["pd"], ms=5, lw=1.5, alpha=0.8, label='Est. LoS')
    ax_pol.legend(fontsize=7, facecolor='#222', labelcolor='white',
                  loc='lower left', bbox_to_anchor=(-0.15, -0.12))

    def init():
        for artist in [tx_trail, rx_trail, tx_dot, rx_dot, los_line, pol_gt, pol_est]:
            artist.set_data([], [])
        time_txt.set_text('')
        return tx_trail, rx_trail, tx_dot, rx_dot, los_line, pol_gt, pol_est, time_txt

    def update(i):
        sl = slice(max(0, i - trail_len), i + 1)

        tx_trail.set_data(tx_x_a[sl], tx_y_a[sl])
        rx_trail.set_data(rx_x_a[sl], rx_y_a[sl])
        tx_dot.set_data([tx_x_a[i]], [tx_y_a[i]])
        rx_dot.set_data([rx_x_a[i]], [rx_y_a[i]])
        los_line.set_data([rx_x_a[i], tx_x_a[i]], [rx_y_a[i], tx_y_a[i]])
        time_txt.set_text(f"t = {t_anim[i]:.2f} s")

        # Polar: plot from origin (0,0) to (los_angle, r)
        pol_gt.set_data([los_a[i]], [r_a[i]])
        pol_est.set_data([los_est_a[i]], [r_a[i]])   # same r — only angle differs

        return tx_trail, rx_trail, tx_dot, rx_dot, los_line, pol_gt, pol_est, time_txt

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         blit=True, interval=1000 / fps)

    # Try MP4 first, fall back to GIF
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(title=result["scenario_name"]))
        anim.save(str(out_path.with_suffix(".mp4")), writer=writer, dpi=120,
                  savefig_kwargs=dict(facecolor=fig.get_facecolor()))
        print(f"    Animation → {out_path.with_suffix('.mp4')}")
    except Exception as e:
        print(f"    ffmpeg unavailable ({e}), falling back to GIF...")
        writer = PillowWriter(fps=fps)
        anim.save(str(out_path.with_suffix(".gif")), writer=writer, dpi=100,
                  savefig_kwargs=dict(facecolor=fig.get_facecolor()))
        print(f"    Animation → {out_path.with_suffix('.gif')}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

def save_all(result: dict, out_root: str = "plots"):
    name  = result["scenario_name"]
    slug  = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    t_pd, delta_f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    est_pd = {"t": t_pd, "delta_f": delta_f_pd}

    print(f"  Saving static plots...")
    plot_trajectory_fig(result, out_dir / "trajectory.png")
    plot_rdot_rddot_fig(result, est_pd, out_dir / "rdot_rddot.png")
    plot_doppler_fig(result, est_pd, out_dir / "doppler.png")

    print(f"  Rendering animation...")
    make_animation(result, out_dir / "animation")

    print(f"  All outputs → {out_dir}/")


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
        result["scenario_name"] = name
        print(f"  Duration:   {cfg.duration:.1f} s  ({len(result['iq']):,} samples)")
        print(f"  Δf range:   [{result['delta_f'].min():.2f}, {result['delta_f'].max():.2f}] Hz")
        print(f"  r̈ peak:     {np.abs(result['r_ddot']).max():.4f} m/s²")
        print(f"  Range CPA:  {result['r'].min():.1f} m")
        save_all(result, "plots")