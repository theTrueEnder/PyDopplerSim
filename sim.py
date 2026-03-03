"""
Baseband IQ Doppler simulator — full pipeline.

Usage
-----
Edit the TOP-LEVEL SimConfig block below, then run.
"""

import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter
from pathlib import Path


# =============================================================================
# ★  TOP-LEVEL CONFIG — edit this block
# =============================================================================

class SimConfig:
    # ---------- rendering ------------------------------------------------
    RENDER_ANIMATION   = True
    # Set to None to rely on PATH, or provide the full path to ffmpeg.exe
    # e.g. r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    FFMPEG_PATH: str | None = None
    
    ANIMATION_FPS      = 10 # 20
    ANIMATION_N_FRAMES = 100
    ANIMATION_DPI_MP4  = 80 # 120
    ANIMATION_DPI_GIF  = 80

    # ---------- RF / sampling --------------------------------------------
    FC      = 5.8e9    # carrier frequency [Hz]
    FS      = 1e6      # sample rate [Hz]
    SNR_DB  = 20.0     # signal-to-noise ratio [dB]
    F_TONE  = 0.0      # optional baseband tone offset [Hz]; 0 = pure CW

    # ---------- receiver (always travels in +x) --------------------------
    RX_VX   = 30.0     # receiver velocity [m/s]  (~108 km/h)
    RX_X0   = 0.0      # receiver initial x position [m]

    # ---------- scenario-specific Tx params ------------------------------
    COLOC_TX_Y      = 0.5

    SAME_TX_X0      = 50.0
    SAME_TX_Y       = 3.7
    SAME_TX_VX      = 28.0
    SAME_DURATION   = 10.0

    ONCO_TX_X0      = 400.0
    ONCO_TX_Y       = 3.7
    ONCO_TX_VX      = -30.0
    ONCO_DURATION   = 13.4   # CPA at t = TX_X0 / (RX_VX - TX_VX)

    # ---------- estimation -----------------------------------------------
    PD_SMOOTH_WINDOW    = 501   # Hann window for phase-diff Δf smoother [samples @ fs]
    # r_ddot is estimated by decimating r_dot to RDOT_DECIM_HZ before differentiating.
    # This avoids amplifying sample-rate noise by ~(fs/decim_hz) factor.
    RDOT_DECIM_HZ       = 100   # Hz — r_dot is decimated to this rate before d/dt
    RDOT_SMOOTH_WINDOW  = 21    # Hann window for r_ddot smoother [samples @ RDOT_DECIM_HZ]
    STFT_WINDOW_DUR     = 0.05
    STFT_HOP_DUR        = 0.005
    INTERP_OVERSAMPLE   = 8


# =============================================================================
# Scenario config
# =============================================================================

# Apply ffmpeg path override before any animation code runs
if SimConfig.FFMPEG_PATH:
    matplotlib.rcParams['animation.ffmpeg_path'] = SimConfig.FFMPEG_PATH


@dataclass
class ScenarioConfig:
    fc: float = 5.8e9;  fs: float = 1e6;  duration: float = 5.0
    tx_x0: float = 100.0;  tx_y: float = 3.7;  tx_vx: float = 30.0
    rx_x0: float = 0.0;    rx_y: float = 0.0;  rx_vx: float = 30.0
    snr_db: float = 20.0;  f_tone: float = 0.0;  interp_oversample: int = 8

    def __post_init__(self):
        self.c = 299_792_458.0


def _base() -> ScenarioConfig:
    return ScenarioConfig(
        fc=SimConfig.FC, fs=SimConfig.FS, snr_db=SimConfig.SNR_DB,
        f_tone=SimConfig.F_TONE, rx_x0=SimConfig.RX_X0, rx_vx=SimConfig.RX_VX,
        interp_oversample=SimConfig.INTERP_OVERSAMPLE,
    )

def scenario_colocated() -> ScenarioConfig:
    cfg = _base(); cfg.tx_x0 = cfg.rx_x0; cfg.tx_y = SimConfig.COLOC_TX_Y
    cfg.tx_vx = cfg.rx_vx; cfg.duration = 5.0; return cfg

def scenario_same_direction() -> ScenarioConfig:
    cfg = _base(); cfg.tx_x0 = SimConfig.SAME_TX_X0; cfg.tx_y = SimConfig.SAME_TX_Y
    cfg.tx_vx = SimConfig.SAME_TX_VX; cfg.duration = SimConfig.SAME_DURATION; return cfg

def scenario_oncoming() -> ScenarioConfig:
    cfg = _base(); cfg.tx_x0 = SimConfig.ONCO_TX_X0; cfg.tx_y = SimConfig.ONCO_TX_Y
    cfg.tx_vx = SimConfig.ONCO_TX_VX; cfg.duration = SimConfig.ONCO_DURATION; return cfg


# =============================================================================
# Geometry
# =============================================================================

def compute_geometry(cfg: ScenarioConfig, t: np.ndarray) -> dict:
    tx_x   = cfg.tx_x0 + cfg.tx_vx * t
    rx_x   = cfg.rx_x0 + cfg.rx_vx * t
    dx     = tx_x - rx_x
    dy     = cfg.tx_y - cfg.rx_y
    r      = np.sqrt(dx**2 + dy**2)
    safe_r = np.where(r < 1e-6, 1e-6, r)
    vdx    = cfg.tx_vx - cfg.rx_vx
    r_dot  = (dx * vdx) / safe_r
    r_ddot = (vdx**2 * safe_r - dx * vdx * r_dot) / safe_r**2
    return dict(
        r=r, r_dot=r_dot, r_ddot=r_ddot,
        delta_f=-r_dot * cfg.fc / cfg.c,
        tx_x=tx_x, rx_x=rx_x,
        tx_y=np.full_like(t, cfg.tx_y),
        rx_y=np.full_like(t, cfg.rx_y),
        los_angle=np.arctan2(dy, dx),
    )


# =============================================================================
# IQ generation
# =============================================================================

def generate_iq(cfg: ScenarioConfig) -> dict:
    N_fine  = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine  = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]
    geo_f   = compute_geometry(cfg, t_fine)
    phase   = 2 * np.pi * np.cumsum(cfg.f_tone + geo_f["delta_f"]) * dt_fine
    iq_fine = np.exp(1j * phase)
    iq = iq_fine[::cfg.interp_oversample]
    t  = t_fine[::cfg.interp_oversample]
    snr_lin   = 10 ** (cfg.snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_lin))
    noise     = noise_std * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    geo = compute_geometry(cfg, t)
    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)


# =============================================================================
# Estimators
# =============================================================================

def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w < 2: return x
    h = np.hanning(w); h /= h.sum()
    return np.convolve(x, h, mode='same')


def estimate_doppler_phase_diff(iq, fs, smooth_window=None):
    if smooth_window is None: smooth_window = SimConfig.PD_SMOOTH_WINDOW
    f_inst = np.angle(iq[1:] * np.conj(iq[:-1])) / (2 * np.pi) * fs
    f_inst = _hann_smooth(f_inst, smooth_window)
    return np.arange(len(f_inst)) / fs, f_inst


def estimate_doppler_stft(iq, fs, window_dur=None, hop_dur=None, freq_zoom=5000.0):
    if window_dur is None: window_dur = SimConfig.STFT_WINDOW_DUR
    if hop_dur    is None: hop_dur    = SimConfig.STFT_HOP_DUR
    win_samp = int(window_dur * fs); hop_samp = int(hop_dur * fs)
    N_fft = win_samp * 4; win = np.hanning(win_samp)
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


def recover_kinematics(t_est, delta_f_est, fc, c):
    """
    r_dot  : invert Doppler equation directly at full sample rate.
    r_ddot : decimate r_dot to RDOT_DECIM_HZ *before* differentiating.

    Why decimate first?
    -------------------
    Finite-difference amplifies noise by 1/dt. At fs=1 MHz, dt=1µs, so even
    1 m/s of r_dot noise becomes ~1e6 m/s² after one difference step.
    Decimating to 100 Hz makes dt=10 ms, cutting amplification by 10,000×.
    Doppler trajectories change on 0.1–1 s timescales so 100 Hz is sufficient.

    NaN handling
    ------------
    The Hann smoother zero-pads edges, producing artifacts in the first and
    last sw_pd//2 samples of r_dot. We mark those NaN, then build the
    decimated grid from *only* fully-valid (all-finite) bins so that no
    NaN bleeds through nanmean into the differentiation step.
    """
    sw_pd    = SimConfig.PD_SMOOTH_WINDOW
    decim_hz = SimConfig.RDOT_DECIM_HZ
    sw_ddot  = SimConfig.RDOT_SMOOTH_WINDOW

    r_dot_full = -delta_f_est * c / fc

    # Mark smoother edge artifacts as NaN
    trim = sw_pd // 2
    r_dot_full[:trim]  = np.nan
    r_dot_full[-trim:] = np.nan

    # Decimate: only keep bins where ALL samples are finite
    fs_orig    = 1.0 / (t_est[1] - t_est[0])
    decim_step = max(1, int(round(fs_orig / decim_hz)))
    n_bins     = len(r_dot_full) // decim_step

    t_d_list     = []
    r_dot_d_list = []
    for i in range(n_bins):
        sl  = slice(i * decim_step, (i + 1) * decim_step)
        seg = r_dot_full[sl]
        if np.all(np.isfinite(seg)):                 # skip any bin touching NaN edge
            t_d_list.append(t_est[sl].mean())
            r_dot_d_list.append(seg.mean())

    t_d     = np.array(t_d_list)
    r_dot_d = np.array(r_dot_d_list)

    # Differentiate on the clean, uniformly-spaced decimated grid
    dt_d       = np.diff(t_d)
    r_ddot_raw = np.concatenate([[np.nan], np.diff(r_dot_d) / dt_d])
    r_ddot_d   = _hann_smooth(r_ddot_raw, sw_ddot)

    # Trim ddot smoother edges
    trim_d = sw_ddot // 2
    r_ddot_d[:trim_d]  = np.nan
    r_ddot_d[-trim_d:] = np.nan

    return dict(
        t_dot=t_est,  r_dot=r_dot_full,
        t_ddot=t_d,   r_ddot=r_ddot_d,
    )


# =============================================================================
# Colours — two palettes: dark bg (animation) and light bg (static plots)
# =============================================================================

# Static plots (light background)
C_STATIC = dict(
    gt   = '#1a1a1a',   # near-black  — ground truth
    pd   = '#e05a00',   # dark orange — estimated
    tx   = '#c07000',   # dark amber
    rx   = '#1565c0',   # dark blue
    stft = '#1565c0',
)

# Animation (dark background)
C_DARK = dict(
    gt   = '#f0f0f0',   # near-white
    pd   = '#ff6b35',   # vivid orange
    tx   = '#ffaa00',   # amber
    rx   = '#4dabf7',   # sky blue
    stft = '#74c0fc',
)


# =============================================================================
# Static figures
# =============================================================================

def plot_trajectory_fig(result, out_path):
    C = C_STATIC
    fig, (ax_2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['scenario_name']} — Trajectory", fontweight='bold')
    t = result["t"]

    ax_2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], lw=2, label="Tx")
    ax_2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], lw=2, label="Rx")
    for xs, ys, col in [(result["tx_x"], result["tx_y"], C["tx"]),
                        (result["rx_x"], result["rx_y"], C["rx"])]:
        mid = len(xs) // 2
        ax_2d.annotate("", xy=(xs[mid+5], ys[mid+5]), xytext=(xs[mid], ys[mid]),
                       arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax_2d.scatter([result["tx_x"][0], result["rx_x"][0]],
                  [result["tx_y"][0], result["rx_y"][0]],
                  color=[C["tx"], C["rx"]], s=60, zorder=5)
    ax_2d.set_xlabel("x [m]"); ax_2d.set_ylabel("y [m]")
    ax_2d.set_title("2-D trajectory (top-down)")
    ax_2d.legend(fontsize=8, loc='upper left'); ax_2d.grid(True, alpha=0.4)
    # Do NOT use equal aspect — x spans hundreds of metres, y spans ~4 m.
    # Instead fix y to a sensible window around the lanes.
    cfg = result["cfg"]
    y_c = (cfg.tx_y + cfg.rx_y) / 2
    ax_2d.set_ylim(y_c - 15, y_c + 15)

    ax_r.plot(t, result["r"], color='purple', lw=1.5)
    ax_r.set_xlabel("Time [s]"); ax_r.set_ylabel("Range r(t) [m]")
    ax_r.set_title("Range over time"); ax_r.grid(True, alpha=0.4)

    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_rdot_rddot_fig(result, est_pd, out_path):
    C   = C_STATIC
    cfg = result["cfg"]
    t   = result["t"]
    kin = recover_kinematics(est_pd["t"], est_pd["delta_f"], cfg.fc, cfg.c)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"{result['scenario_name']} — Radial kinematics", fontweight='bold')

    ax = axes[0]
    ax.plot(t, result["r_dot"], color=C["gt"], lw=1.8, label='Ground truth ṙ')
    ax.plot(kin["t_dot"], kin["r_dot"], color=C["pd"], lw=1.0, alpha=0.9,
            label='Estimated ṙ (phase-diff)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ṙ [m/s]")
    ax.set_title("Radial velocity ṙ(t)")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t, result["r_ddot"], color=C["gt"], lw=1.8, label='Ground truth r̈')
    ax.plot(kin["t_ddot"], kin["r_ddot"], color=C["pd"], lw=1.0, alpha=0.9,
            label=f'Estimated r̈ (decimated to {SimConfig.RDOT_DECIM_HZ} Hz, then d/dt)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("r̈ [m/s²]")
    ax.set_title("Radial acceleration r̈(t)  [peak at CPA]")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.4)

    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_doppler_fig(result, est_pd, out_path):
    C    = C_STATIC
    cfg  = result["cfg"]
    t    = result["t"]
    gt   = result["delta_f"]
    zoom = max(500.0, 3.0 * np.abs(gt).max())
    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)

    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Doppler estimation", fontweight='bold')

    ax_pd.plot(est_pd["t"], est_pd["delta_f"], color=C["pd"], lw=0.8, alpha=0.85,
               label='Phase-diff Δf')
    ax_pd.plot(t, gt, color=C["gt"], lw=1.5, ls='--', label='Ground truth')
    ax_pd.set_ylabel("Δf [Hz]"); ax_pd.set_title("Instantaneous frequency — phase differentiation")
    ax_pd.legend(fontsize=8, loc='upper right'); ax_pd.grid(True, alpha=0.4)

    mask   = stft["mask"]
    Sxx_dB = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    extent = [stft["t"][0], stft["t"][-1],
              stft["freq_axis"][mask][0], stft["freq_axis"][mask][-1]]
    ax_stft.imshow(Sxx_dB, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno', vmin=np.percentile(Sxx_dB, 20))
    ax_stft.plot(stft["t"], stft["peak_freq"], color=C["stft"],
                 lw=1.0, ls='--', label='STFT peak')
    ax_stft.plot(t, gt, color='#cccccc', lw=1.5, ls='--', label='Ground truth')
    ax_stft.set_ylabel("Δf [Hz]"); ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (zoomed ±{zoom:.0f} Hz)")
    ax_stft.legend(fontsize=8, loc='upper right')

    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# =============================================================================
# Animation
# =============================================================================

def make_animation(result: dict, out_path: Path):
    """
    Render animation by driving the writer directly in a for-loop rather than
    using FuncAnimation. This avoids FuncAnimation's per-frame figure-diffing
    overhead and lets us pre-compute all frame data as numpy arrays upfront so
    zero numpy work happens inside the render loop.

    Speed levers (all in SimConfig):
      ANIMATION_N_FRAMES  — fewer frames = linearly faster
      ANIMATION_DPI_*     — lower DPI = quadratically faster (pixel count)
      ANIMATION_FPS       — only affects playback speed, not render time
    """
    C   = C_DARK
    fps = SimConfig.ANIMATION_FPS
    n_f = SimConfig.ANIMATION_N_FRAMES
    cfg = result["cfg"]
    t   = result["t"]

    # ---- Pre-compute ALL per-frame data as arrays -------------------------
    idx    = np.linspace(0, len(t) - 1, n_f, dtype=int)
    t_anim = t[idx]
    tx_x_a = result["tx_x"][idx];  tx_y_a = result["tx_y"][idx]
    rx_x_a = result["rx_x"][idx];  rx_y_a = result["rx_y"][idx]
    los_a  = result["los_angle"][idx]
    r_a    = result["r"][idx]

    t_pd, delta_f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    r_dot_est_a = np.interp(t_anim, t_pd, -delta_f_pd * cfg.c / cfg.fc)
    vdx   = cfg.tx_vx - cfg.rx_vx
    dy    = cfg.tx_y - cfg.rx_y
    v_ref = vdx if abs(vdx) > 0.5 else cfg.rx_vx
    los_est_a = np.arctan2(dy, r_dot_est_a * r_a / v_ref)

    trail = max(5, n_f // 15)

    # ---- Build figure once; all artists are mutated in the loop -----------
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('#1a1a2e')
    gs     = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax2d   = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection='polar')

    ax2d.set_facecolor('#16213e')
    for sp in ax2d.spines.values(): sp.set_edgecolor('#555')
    ax_pol.set_facecolor('#16213e')
    ax_pol.tick_params(colors='#aaa')
    ax_pol.spines['polar'].set_color('#555')

    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color='white'); ax2d.set_ylabel("y [m]", color='white')
    ax2d.tick_params(colors='white'); ax2d.grid(True, alpha=0.25, color='#555')
    ax2d.set_title("Vehicle positions", color='white', fontsize=10)

    ax_pol.set_theta_zero_location('E'); ax_pol.set_theta_direction(1)
    ax_pol.set_ylim(0, 1.3); ax_pol.set_yticks([])
    ax_pol.set_title("Tx bearing\n(from Rx, 0°=ahead)", color='white', fontsize=9, pad=12)
    for angle, lbl in [(0,'Ahead'),(np.pi/2,'Left'),(np.pi,'Behind'),(-np.pi/2,'Right')]:
        ax_pol.text(angle, 1.22, lbl, ha='center', va='center', color='#888', fontsize=7)

    # Static traces
    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], alpha=0.15, lw=1)
    ax2d.plot([], [], 'o', color=C["tx"], ms=7, label='Tx')
    ax2d.plot([], [], 'o', color=C["rx"], ms=7, label='Rx')
    ax2d.legend(fontsize=8, facecolor='#222', labelcolor='white', loc='lower right')
    ax_pol.plot([], [], '-', color=C["gt"], lw=2, label='True')
    ax_pol.plot([], [], '-', color=C["pd"], lw=2, label='Est.')
    ax_pol.legend(fontsize=7, facecolor='#222', labelcolor='white',
                  loc='upper left', bbox_to_anchor=(1.05, 1.12))

    # Mutable artists
    tx_trail, = ax2d.plot([], [], color=C["tx"], lw=1.5, alpha=0.5)
    rx_trail, = ax2d.plot([], [], color=C["rx"], lw=1.5, alpha=0.5)
    tx_dot,   = ax2d.plot([], [], 'o', color=C["tx"], ms=10, zorder=5)
    rx_dot,   = ax2d.plot([], [], 'o', color=C["rx"], ms=10, zorder=5)
    los_line, = ax2d.plot([], [], '--', color='white', lw=0.8, alpha=0.5)
    time_txt  = ax2d.text(0.02, 0.95, '', transform=ax2d.transAxes,
                          color='white', fontsize=9, va='top')
    pol_gt_ln,   = ax_pol.plot([], [], '-', color=C["gt"], lw=2.5)
    pol_est_ln,  = ax_pol.plot([], [], '-', color=C["pd"], lw=2.5, alpha=0.9)
    pol_gt_dot,  = ax_pol.plot([], [], 'o', color=C["gt"], ms=8)
    pol_est_dot, = ax_pol.plot([], [], 'o', color=C["pd"], ms=8)
    range_txt    = ax_pol.text(np.pi * 1.25, 1.5, '', color='white',
                               fontsize=8, ha='center', va='center')

    plt.tight_layout()

    # ---- Direct writer loop — no FuncAnimation overhead ------------------
    def _render(writer, file_path, dpi):
        with writer.saving(fig, str(file_path), dpi=dpi):
            for i in range(n_f):
                sl = slice(max(0, i - trail), i + 1)
                tx_trail.set_data(tx_x_a[sl], tx_y_a[sl])
                rx_trail.set_data(rx_x_a[sl], rx_y_a[sl])
                tx_dot.set_data([tx_x_a[i]], [tx_y_a[i]])
                rx_dot.set_data([rx_x_a[i]], [rx_y_a[i]])
                los_line.set_data([rx_x_a[i], tx_x_a[i]], [rx_y_a[i], tx_y_a[i]])
                time_txt.set_text(f"t = {t_anim[i]:.2f} s")
                pol_gt_ln.set_data( [los_a[i],     los_a[i]],     [0, 1.0])
                pol_est_ln.set_data([los_est_a[i], los_est_a[i]], [0, 1.0])
                pol_gt_dot.set_data( [los_a[i]],     [1.0])
                pol_est_dot.set_data([los_est_a[i]], [1.0])
                range_txt.set_text(f"r = {r_a[i]:.0f} m")
                writer.grab_frame()

    try:
        mp4_path = out_path.with_suffix(".mp4")
        w = FFMpegWriter(fps=fps, metadata=dict(title=result["scenario_name"]),
                         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                                     '-preset', 'fast', '-crf', '23'])
        _render(w, mp4_path, SimConfig.ANIMATION_DPI_MP4)
        print(f"    Animation → {mp4_path}")
    except Exception as e:
        print(f"    ffmpeg unavailable ({e}), falling back to GIF...")
        gif_path = out_path.with_suffix(".gif")
        w = PillowWriter(fps=fps)
        _render(w, gif_path, SimConfig.ANIMATION_DPI_GIF)
        print(f"    Animation → {gif_path}")

    plt.close(fig)


# =============================================================================
# Save all outputs
# =============================================================================

def save_all(result: dict, out_root: str = "plots"):
    name    = result["scenario_name"]
    slug    = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result["cfg"]
    t_pd, df_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    est_pd = {"t": t_pd, "delta_f": df_pd}
    print(f"  Saving static plots...")
    plot_trajectory_fig(result,      out_dir / "trajectory.png")
    plot_rdot_rddot_fig(result, est_pd, out_dir / "rdot_rddot.png")
    plot_doppler_fig(result,    est_pd, out_dir / "doppler.png")
    if SimConfig.RENDER_ANIMATION:
        print(f"  Rendering animation...")
        make_animation(result, out_dir / "animation")
    else:
        print(f"  Skipping animation (RENDER_ANIMATION=False)")
    print(f"  → {out_dir}/")


# =============================================================================
# Main
# =============================================================================

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
        vdx  = cfg.tx_vx - cfg.rx_vx
        cpa_t = cfg.tx_x0 / max(abs(vdx), 1e-3) if abs(vdx) > 0.1 else float('inf')
        print(f"  Rx: {cfg.rx_vx:.1f} m/s   Tx: {cfg.tx_vx:.1f} m/s   Δv: {vdx:.1f} m/s")
        print(f"  Δf range:    [{result['delta_f'].min():.2f}, {result['delta_f'].max():.2f}] Hz")
        print(f"  r̈ peak (GT): {np.nanmax(np.abs(result['r_ddot'])):.1f} m/s²")
        print(f"  Range CPA:   {result['r'].min():.1f} m  (at t ≈ {cpa_t:.2f} s)")
        save_all(result, "plots")