"""
Baseband IQ Doppler simulator — full pipeline.

Pipeline stages
---------------
1. Geometry   : compute r(t), ṙ(t), r̈(t) from 2-D kinematic model
2. IQ gen     : integrate instantaneous phase → complex baseband signal + AWGN
3. Estimation : recover Δf(t) via phase-differentiation and STFT peak-tracking
4. Kinematics : invert Doppler eq → ṙ_est; decimate + differentiate → r̈_est
5. Output     : static PNG plots + optional MP4/GIF animation

Usage
-----
Edit the SimConfig class below, then run:
    python sim.py
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — write files, no display window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np


# =============================================================================
# Logging setup — module-level logger, INFO by default
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@contextmanager
def timed(label: str):
    """Context manager that logs wall-clock duration of a code block."""
    t0 = time.perf_counter()
    log.info(f"START  {label}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info(f"DONE   {label}  ({dt:.2f}s)")


# =============================================================================
# ★  TOP-LEVEL CONFIG — edit this block
# =============================================================================

class SimConfig:
    # ---------- rendering ------------------------------------------------
    # RENDER_FORMATS: which animation formats to produce.
    # Options: "mp4", "gif", "both", "none"  (case-insensitive)
    RENDER_FORMATS     = "both"
    ANIMATION_FPS      = 20
    ANIMATION_N_FRAMES = 100    # fewer frames = faster; 100 is good for GIF
    ANIMATION_DPI_MP4  = 120
    ANIMATION_DPI_GIF  = 80

    # Full path to ffmpeg binary. Set to None to rely on system PATH.
    # Find your path with `where ffmpeg` (Windows) or `which ffmpeg` (Unix).
    # Example: r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    FFMPEG_PATH: str | None = None

    # ---------- RF / sampling --------------------------------------------
    FC      = 5.8e9    # carrier frequency [Hz]  — controls Hz-per-(m/s) sensitivity
    FS      = 1e6      # IQ sample rate [Hz]
    SNR_DB  = 20.0     # signal-to-noise ratio [dB]  (10 → noisy, 30 → clean)
    F_TONE  = 0.0      # optional baseband tone offset [Hz]; 0 = pure CW at DC

    # ---------- receiver (always travels in +x) --------------------------
    RX_VX   = 30.0     # receiver speed [m/s]  (~108 km/h)
    RX_X0   = 0.0      # receiver initial x position [m]

    # ---------- scenario: Co-located -------------------------------------
    # Tx rides in the same vehicle as Rx → near-zero relative velocity.
    # A tiny lateral offset (0.5 m) keeps both dots visible in trajectory plots.
    COLOC_TX_Y        = 0.5     # [m]

    # ---------- scenario: Same-direction ---------------------------------
    # Tx is in an adjacent lane travelling the same direction but slightly slower.
    # Rx will overtake Tx during the simulation.
    SAME_TX_X0        = 50.0    # Tx starts this far ahead of Rx [m]
    SAME_TX_Y         = 3.7     # lateral lane separation [m]
    SAME_TX_VX        = 28.0    # Tx speed [m/s]  (slower than Rx → Rx overtakes)
    SAME_DURATION     = 10.0    # [s]

    # ---------- scenario: Oncoming ---------------------------------------
    # Tx is in the opposing lane heading directly toward Rx.
    # CPA (closest point of approach) occurs at t = TX_X0 / (RX_VX - TX_VX).
    # ONCO_DURATION should be long enough to capture well past CPA.
    ONCO_TX_X0        = 400.0   # initial separation [m]
    ONCO_TX_Y         = 3.7     # opposing lane offset [m]
    ONCO_TX_VX        = -30.0   # Tx speed [m/s]  (negative = approaching)
    ONCO_DURATION     = 13.4    # [s]  → CPA at t ≈ 400/60 ≈ 6.67 s

    # ---------- estimation parameters ------------------------------------
    # Phase-differentiator smoother: wide window suppresses noise but blurs
    # fast Doppler transitions.  Tune down if CPA peak is smeared.
    PD_SMOOTH_WINDOW    = 501   # Hann FIR length [samples @ FS]

    # r̈ is estimated by decimating ṙ to a low rate before differentiating.
    # Differencing at full FS would amplify noise by ~FS × (noise on ṙ).
    # At 100 Hz the dt denominator is 10 ms instead of 1 µs → 10,000× less noise.
    RDOT_DECIM_HZ       = 100   # ṙ decimation rate before d/dt [Hz]
    RDOT_SMOOTH_WINDOW  = 21    # Hann FIR for r̈ smoother [samples @ RDOT_DECIM_HZ]

    STFT_WINDOW_DUR     = 0.05  # STFT analysis window [s]  → freq resolution = 1/T Hz
    STFT_HOP_DUR        = 0.005 # STFT hop size [s]  → temporal resolution of spectrogram
    INTERP_OVERSAMPLE   = 8     # phase integration oversampling factor (accuracy vs speed)


# Apply ffmpeg path override before any animation writers are instantiated
if SimConfig.FFMPEG_PATH:
    matplotlib.rcParams['animation.ffmpeg_path'] = SimConfig.FFMPEG_PATH

# Parse render format flag once at import time
_fmt = SimConfig.RENDER_FORMATS.lower().strip()
_RENDER_MP4 = _fmt in ("mp4", "both")
_RENDER_GIF = _fmt in ("gif", "both")


# =============================================================================
# Scenario config dataclass
# =============================================================================

@dataclass
class ScenarioConfig:
    """All parameters needed to define one simulation run."""
    fc: float = 5.8e9;  fs: float = 1e6;  duration: float = 5.0
    tx_x0: float = 100.0;  tx_y: float = 3.7;  tx_vx: float = 30.0
    rx_x0: float = 0.0;    rx_y: float = 0.0;  rx_vx: float = 30.0
    snr_db: float = 20.0;  f_tone: float = 0.0;  interp_oversample: int = 8

    def __post_init__(self):
        self.c = 299_792_458.0  # speed of light [m/s]


def _base() -> ScenarioConfig:
    """Populate a ScenarioConfig from the global SimConfig values."""
    return ScenarioConfig(
        fc=SimConfig.FC, fs=SimConfig.FS, snr_db=SimConfig.SNR_DB,
        f_tone=SimConfig.F_TONE, rx_x0=SimConfig.RX_X0, rx_vx=SimConfig.RX_VX,
        interp_oversample=SimConfig.INTERP_OVERSAMPLE,
    )

def scenario_colocated() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0 = cfg.rx_x0          # start at same x position
    cfg.tx_y  = SimConfig.COLOC_TX_Y
    cfg.tx_vx = cfg.rx_vx          # identical velocity → Δv ≈ 0
    cfg.duration = 5.0
    return cfg

def scenario_same_direction() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0    = SimConfig.SAME_TX_X0
    cfg.tx_y     = SimConfig.SAME_TX_Y
    cfg.tx_vx    = SimConfig.SAME_TX_VX
    cfg.duration = SimConfig.SAME_DURATION
    return cfg

def scenario_oncoming() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0    = SimConfig.ONCO_TX_X0
    cfg.tx_y     = SimConfig.ONCO_TX_Y
    cfg.tx_vx    = SimConfig.ONCO_TX_VX   # negative → approaching
    cfg.duration = SimConfig.ONCO_DURATION
    return cfg


# =============================================================================
# Stage 1 — Geometry
# =============================================================================

def compute_geometry(cfg: ScenarioConfig, t: np.ndarray) -> dict:
    """
    Compute ground-truth kinematic quantities at each time sample.

    Both vehicles travel in straight lines along the x-axis at constant speed,
    separated by a fixed lateral offset (dy).  All quantities are derived
    analytically — no numerical differentiation needed here.

    Returned quantities
    -------------------
    r        : range = |tx_pos - rx_pos|  [m]
    r_dot    : radial velocity = d(r)/dt  [m/s]   (+ve = moving apart)
    r_ddot   : radial acceleration = d²(r)/dt²  [m/s²]
    delta_f  : Doppler shift = -ṙ · fc / c  [Hz]
    los_angle: bearing from Rx to Tx = arctan2(dy, dx)  [rad]
               0 = ahead, π/2 = left, π = behind, -π/2 = right
    """
    # World-frame positions
    tx_x   = cfg.tx_x0 + cfg.tx_vx * t
    rx_x   = cfg.rx_x0 + cfg.rx_vx * t

    # Relative position vector (tx − rx)
    dx     = tx_x - rx_x          # changes with time
    dy     = cfg.tx_y - cfg.rx_y  # constant (no lateral motion)
    r      = np.sqrt(dx**2 + dy**2)
    safe_r = np.where(r < 1e-6, 1e-6, r)   # guard against exact co-location

    # Relative velocity in x only (dy is constant → vdy = 0)
    vdx    = cfg.tx_vx - cfg.rx_vx

    # ṙ = d/dt √(dx²+dy²) = (dx·vdx) / r
    r_dot  = (dx * vdx) / safe_r

    # r̈ = d/dt [ṙ] via quotient rule:
    #   r̈ = (vdx²·r − dx·vdx·ṙ) / r²
    # Physical interpretation: peaks at CPA when the LoS rotates fastest.
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
# Stage 2 — IQ generation
# =============================================================================

def generate_iq(cfg: ScenarioConfig) -> dict:
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
    """
    # Build fine time grid for accurate phase integration
    N_fine  = int(cfg.duration * cfg.fs * cfg.interp_oversample)
    t_fine  = np.linspace(0, cfg.duration, N_fine, endpoint=False)
    dt_fine = t_fine[1] - t_fine[0]

    # Instantaneous frequency on fine grid: tone + Doppler
    geo_f     = compute_geometry(cfg, t_fine)
    inst_freq = cfg.f_tone + geo_f["delta_f"]   # [Hz]

    # Integrate frequency → phase, then form complex baseband signal
    phase   = 2 * np.pi * np.cumsum(inst_freq) * dt_fine   # [rad]
    iq_fine = np.exp(1j * phase)                            # unit-amplitude CW

    # Decimate to output sample rate by striding
    iq = iq_fine[::cfg.interp_oversample]
    t  = t_fine[::cfg.interp_oversample]

    # Add complex AWGN
    snr_lin   = 10 ** (cfg.snr_db / 10)
    noise_std = np.sqrt(1.0 / (2.0 * snr_lin))
    noise     = noise_std * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))

    # Ground-truth geometry at output sample rate
    geo = compute_geometry(cfg, t)
    return dict(t=t, iq=iq + noise, iq_clean=iq, cfg=cfg, **geo)


# =============================================================================
# Stage 3 — Doppler estimators
# =============================================================================

def _hann_smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Convolve x with a normalised Hann window of length w (linear-phase FIR)."""
    if w < 2:
        return x
    h = np.hanning(w)
    h /= h.sum()
    return np.convolve(x, h, mode='same')


def estimate_doppler_phase_diff(iq: np.ndarray, fs: float,
                                smooth_window: int | None = None) -> tuple:
    """
    FM discriminator (lag-1 autocorrelation argument).

    The instantaneous frequency is proportional to the rate of phase change:

        f_inst[n] = angle(iq[n] · conj(iq[n−1])) · fs / (2π)

    This gives sample-rate time resolution but is noisy — the Hann smoother
    trades temporal resolution for noise suppression.

    Returns (t, delta_f) both at sample rate fs (length N-1).
    """
    if smooth_window is None:
        smooth_window = SimConfig.PD_SMOOTH_WINDOW

    # Angle of complex lag-1 product = phase increment per sample
    f_inst = np.angle(iq[1:] * np.conj(iq[:-1])) / (2 * np.pi) * fs
    f_inst = _hann_smooth(f_inst, smooth_window)
    t      = np.arange(len(f_inst)) / fs
    return t, f_inst


def estimate_doppler_stft(iq: np.ndarray, fs: float,
                          window_dur: float | None = None,
                          hop_dur: float | None = None,
                          freq_zoom: float = 5000.0) -> dict:
    """
    Short-Time Fourier Transform spectrogram with peak frequency tracking.

    Design choices
    --------------
    - Hann analysis window (good sidelobe rejection for close-in peaks)
    - 4× zero-padding → smoother peak interpolation without changing resolution
    - freq_zoom: only rows within ±freq_zoom Hz are returned for display;
      auto-set to 3× max ground-truth |Δf| in the plotting code.

    Returns dict with: t, freq_axis, Sxx (power), peak_freq, mask.
    """
    if window_dur is None: window_dur = SimConfig.STFT_WINDOW_DUR
    if hop_dur    is None: hop_dur    = SimConfig.STFT_HOP_DUR

    win_samp = int(window_dur * fs)
    hop_samp = int(hop_dur * fs)
    N_fft    = win_samp * 4          # zero-pad for interpolated peak
    win      = np.hanning(win_samp)

    n_frames = (len(iq) - win_samp) // hop_samp + 1
    Sxx      = np.zeros((N_fft, n_frames))

    for i in range(n_frames):
        seg        = iq[i*hop_samp : i*hop_samp + win_samp] * win
        Sxx[:, i]  = np.abs(np.fft.fftshift(np.fft.fft(seg, N_fft)))**2

    freq_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, 1.0 / fs))
    t_stft    = (np.arange(n_frames) * hop_samp + win_samp // 2) / fs
    peak_freq = freq_axis[np.argmax(Sxx, axis=0)]
    mask      = np.abs(freq_axis) <= freq_zoom

    return dict(t=t_stft, freq_axis=freq_axis, Sxx=Sxx,
                peak_freq=peak_freq, mask=mask)


# =============================================================================
# Stage 4 — Kinematic recovery
# =============================================================================

def recover_kinematics(t_est: np.ndarray, delta_f_est: np.ndarray,
                       fc: float, c: float) -> dict:
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

    Returns dict with:
        t_dot   : time axis for ṙ  [full rate]
        r_dot   : estimated ṙ      [m/s, NaN at edges]
        t_ddot  : time axis for r̈  [decimated rate]
        r_ddot  : estimated r̈      [m/s², NaN at edges]
    """
    sw_pd    = SimConfig.PD_SMOOTH_WINDOW
    decim_hz = SimConfig.RDOT_DECIM_HZ
    sw_ddot  = SimConfig.RDOT_SMOOTH_WINDOW

    # --- ṙ at full sample rate ---
    r_dot_full = -delta_f_est * c / fc

    # Mark smoother edge transients as NaN
    trim = sw_pd // 2
    r_dot_full[:trim]  = np.nan
    r_dot_full[-trim:] = np.nan

    # --- Decimate ṙ to low rate, skipping any NaN-containing bin ---
    fs_orig    = 1.0 / (t_est[1] - t_est[0])
    decim_step = max(1, int(round(fs_orig / decim_hz)))
    n_bins     = len(r_dot_full) // decim_step

    t_d, r_dot_d = [], []
    for i in range(n_bins):
        sl  = slice(i * decim_step, (i + 1) * decim_step)
        seg = r_dot_full[sl]
        if np.all(np.isfinite(seg)):           # discard any bin touching NaN edge
            t_d.append(t_est[sl].mean())
            r_dot_d.append(seg.mean())

    t_d     = np.array(t_d)
    r_dot_d = np.array(r_dot_d)

    # --- Differentiate on clean decimated grid ---
    dt_d       = np.diff(t_d)
    r_ddot_raw = np.concatenate([[np.nan], np.diff(r_dot_d) / dt_d])
    r_ddot_d   = _hann_smooth(r_ddot_raw, sw_ddot)

    # Trim smoother edges on r̈
    trim_d = sw_ddot // 2
    r_ddot_d[:trim_d]  = np.nan
    r_ddot_d[-trim_d:] = np.nan

    return dict(t_dot=t_est, r_dot=r_dot_full,
                t_ddot=t_d,  r_ddot=r_ddot_d)


# =============================================================================
# Colour palettes
# =============================================================================

# Static plots use a light matplotlib background
C_STATIC = dict(
    gt='#1a1a1a',    # near-black  — ground truth
    pd='#e05a00',    # dark orange — phase-diff estimate
    tx='#c07000',    # dark amber  — transmitter
    rx='#1565c0',    # dark blue   — receiver
    stft='#1565c0',
)

# Animation uses a dark background
C_DARK = dict(
    gt='#f0f0f0',    # near-white  — ground truth
    pd='#ff6b35',    # vivid orange — estimate
    tx='#ffaa00',    # amber       — transmitter
    rx='#4dabf7',    # sky blue    — receiver
    stft='#74c0fc',
)


# =============================================================================
# Stage 5a — Static figures
# =============================================================================

def plot_trajectory_fig(result: dict, out_path: Path) -> None:
    """
    Two-panel figure:
      Left : 2-D top-down trajectory of both vehicles with direction arrows.
             x-axis = world position, y-axis = lane position.
             y-limits are fixed to ±15 m around the lane midpoint so both
             parallel lines are visible regardless of x-span.
      Right: Range r(t) — shows CPA as the minimum.
    """
    C   = C_STATIC
    cfg = result["cfg"]
    t   = result["t"]

    fig, (ax_2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['scenario_name']} — Trajectory", fontweight='bold')

    ax_2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], lw=2, label="Tx")
    ax_2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], lw=2, label="Rx")

    # Direction arrows at mid-trajectory
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
    # Fix y to ±15 m around lane midpoint — do NOT use equal aspect (x spans
    # hundreds of metres, y spans ~4 m, equal aspect would squash the lanes).
    y_c = (cfg.tx_y + cfg.rx_y) / 2
    ax_2d.set_ylim(y_c - 15, y_c + 15)

    ax_r.plot(t, result["r"], color='purple', lw=1.5)
    ax_r.set_xlabel("Time [s]"); ax_r.set_ylabel("Range r(t) [m]")
    ax_r.set_title("Range over time"); ax_r.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rdot_rddot_fig(result: dict, est_pd: dict, out_path: Path) -> None:
    """
    Two-panel figure comparing ground-truth vs estimated radial kinematics.

      Top   : ṙ(t) — radial velocity.  Zero-crossing at CPA.
      Bottom: r̈(t) — radial acceleration.  Peaks at CPA; magnitude scales
              as v_rel² / r_min, so oncoming at highway speed with a narrow
              lane separation produces large (~1000 m/s²) peaks.

    The estimated r̈ uses a separate (decimated) time axis from ṙ, so
    sharex=False is intentional.
    """
    C   = C_STATIC
    cfg = result["cfg"]
    t   = result["t"]
    kin = recover_kinematics(est_pd["t"], est_pd["delta_f"], cfg.fc, cfg.c)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"{result['scenario_name']} — Radial kinematics", fontweight='bold')

    ax = axes[0]
    ax.plot(t,           result["r_dot"],  color=C["gt"], lw=1.8, label='Ground truth ṙ')
    ax.plot(kin["t_dot"], kin["r_dot"],    color=C["pd"], lw=1.0, alpha=0.9,
            label='Estimated ṙ (phase-diff)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ṙ [m/s]")
    ax.set_title("Radial velocity ṙ(t)")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t,             result["r_ddot"], color=C["gt"], lw=1.8, label='Ground truth r̈')
    ax.plot(kin["t_ddot"], kin["r_ddot"],   color=C["pd"], lw=1.0, alpha=0.9,
            label=f'Estimated r̈  (decimated to {SimConfig.RDOT_DECIM_HZ} Hz then d/dt)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("r̈ [m/s²]")
    ax.set_title("Radial acceleration r̈(t)  [peak magnitude = v_rel² / r_min at CPA]")
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_doppler_fig(result: dict, est_pd: dict, out_path: Path) -> None:
    """
    Two-panel Doppler estimation figure.

      Top   : Phase-diff instantaneous frequency vs ground truth.
      Bottom: STFT spectrogram (zoomed to ±3× max |Δf|) with STFT peak
              track and ground-truth overlay.

    The STFT frequency zoom is computed automatically from the ground-truth
    range so the signal occupies a sensible fraction of the display.
    """
    C    = C_STATIC
    cfg  = result["cfg"]
    t    = result["t"]
    gt   = result["delta_f"]

    # Auto-zoom STFT display to 3× max true Doppler (minimum ±500 Hz)
    zoom = max(500.0, 3.0 * np.abs(gt).max())
    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)

    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Doppler estimation", fontweight='bold')

    ax_pd.plot(est_pd["t"], est_pd["delta_f"], color=C["pd"],
               lw=0.8, alpha=0.85, label='Phase-diff Δf')
    ax_pd.plot(t, gt, color=C["gt"], lw=1.5, ls='--', label='Ground truth')
    ax_pd.set_ylabel("Δf [Hz]")
    ax_pd.set_title("Instantaneous frequency — phase differentiation")
    ax_pd.legend(fontsize=8, loc='upper right'); ax_pd.grid(True, alpha=0.4)

    mask   = stft["mask"]
    Sxx_dB = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    extent = [stft["t"][0], stft["t"][-1],
              stft["freq_axis"][mask][0], stft["freq_axis"][mask][-1]]
    ax_stft.imshow(Sxx_dB, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno', vmin=np.percentile(Sxx_dB, 20))
    ax_stft.plot(stft["t"], stft["peak_freq"], color=C["stft"],
                 lw=1.0, ls='--', label='STFT peak')
    # Use light grey for GT overlay on dark spectrogram background
    ax_stft.plot(t, gt, color='#cccccc', lw=1.5, ls='--', label='Ground truth')
    ax_stft.set_ylabel("Δf [Hz]"); ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (zoomed ±{zoom:.0f} Hz)")
    ax_stft.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Stage 5b — Animation
# =============================================================================

def _build_animation_frame_data(result: dict) -> dict:
    """
    Pre-compute every quantity needed for animation frames as numpy arrays.
    All numpy work is done here so the render loop contains only artist
    mutations and grab_frame() calls — keeping it as fast as possible.

    Bearing estimation (semi-blind)
    --------------------------------
    The receiver knows its own velocity (GPS) and fc, so it can compute ṙ_est
    from Δf_est.  From ṙ_est we back-solve the LoS angle:

        ṙ = vdx · dx / r   →   dx_est = ṙ_est · r / vdx

    vdx = tx_vx − rx_vx.  The receiver knows rx_vx but not tx_vx in a truly
    blind scenario; here we use the true vdx (ideal case).  In practice, the
    receiver would need to estimate or assume tx_vx — that uncertainty is a
    key research question for the thesis.
    """
    cfg = result["cfg"]
    t   = result["t"]
    n_f = SimConfig.ANIMATION_N_FRAMES

    # Sub-sample ground-truth arrays to n_f evenly-spaced animation frames
    idx    = np.linspace(0, len(t) - 1, n_f, dtype=int)
    t_anim = t[idx]

    # Phase-diff Doppler estimate → ṙ_est → bearing estimate
    t_pd, delta_f_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    r_dot_est_a      = np.interp(t_anim, t_pd, -delta_f_pd * cfg.c / cfg.fc)

    vdx   = cfg.tx_vx - cfg.rx_vx
    dy    = cfg.tx_y - cfg.rx_y
    v_ref = vdx if abs(vdx) > 0.5 else cfg.rx_vx   # fallback for co-located (vdx ≈ 0)
    r_a   = result["r"][idx]
    dx_est    = r_dot_est_a * r_a / v_ref
    los_est_a = np.arctan2(dy, dx_est)              # arctan2 handles all four quadrants

    return dict(
        t_anim  = t_anim,
        tx_x_a  = result["tx_x"][idx],
        tx_y_a  = result["tx_y"][idx],
        rx_x_a  = result["rx_x"][idx],
        rx_y_a  = result["rx_y"][idx],
        los_a   = result["los_angle"][idx],
        los_est_a = los_est_a,
        r_a     = r_a,
    )


def _build_animation_figure(result: dict) -> tuple:
    """
    Create the animation figure and all mutable artists.
    Returns (fig, artists_dict).

    Layout: two panels side by side.
      Left : 2-D top-down vehicle positions with trail and LoS line.
      Right: Polar compass showing Tx bearing (unit-radius needle).
             Radius is fixed at 1.0 — only the angle matters.
             The actual range is shown as a text overlay.
    """
    C   = C_DARK
    cfg = result["cfg"]

    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('#1a1a2e')
    gs     = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax2d   = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection='polar')

    # --- 2-D panel styling ---
    ax2d.set_facecolor('#16213e')
    for sp in ax2d.spines.values(): sp.set_edgecolor('#555')
    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color='white'); ax2d.set_ylabel("y [m]", color='white')
    ax2d.tick_params(colors='white'); ax2d.grid(True, alpha=0.25, color='#555')
    ax2d.set_title("Vehicle positions", color='white', fontsize=10)

    # Faint full-path ghost traces so you can see where they came from/are going
    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], alpha=0.15, lw=1)

    # Dummy artists for legend (explicit loc avoids 'best' location search)
    ax2d.plot([], [], 'o', color=C["tx"], ms=7, label='Tx')
    ax2d.plot([], [], 'o', color=C["rx"], ms=7, label='Rx')
    ax2d.legend(fontsize=8, facecolor='#222', labelcolor='white', loc='lower right')

    # --- Polar panel styling ---
    ax_pol.set_facecolor('#16213e')
    ax_pol.tick_params(colors='#aaa')
    ax_pol.spines['polar'].set_color('#555')
    ax_pol.set_theta_zero_location('E')   # 0° = East = Tx directly ahead (+x)
    ax_pol.set_theta_direction(1)         # counter-clockwise (standard math convention)
    ax_pol.set_ylim(0, 1.3)
    ax_pol.set_yticks([])                 # radial ticks meaningless (unit radius)
    ax_pol.set_title("Tx bearing\n(from Rx, 0°=ahead)", color='white', fontsize=9, pad=12)

    # Cardinal direction labels
    for angle, lbl in [(0, 'Ahead'), (np.pi/2, 'Left'),
                       (np.pi, 'Behind'), (-np.pi/2, 'Right')]:
        ax_pol.text(angle, 1.22, lbl, ha='center', va='center', color='#888', fontsize=7)

    # Polar legend
    ax_pol.plot([], [], '-', color=C["gt"], lw=2, label='True')
    ax_pol.plot([], [], '-', color=C["pd"], lw=2, label='Est.')
    ax_pol.legend(fontsize=7, facecolor='#222', labelcolor='white',
                  loc='upper left', bbox_to_anchor=(1.05, 1.12))

    # --- Mutable artists updated each frame ---
    tx_trail,    = ax2d.plot([], [], color=C["tx"], lw=1.5, alpha=0.5)
    rx_trail,    = ax2d.plot([], [], color=C["rx"], lw=1.5, alpha=0.5)
    tx_dot,      = ax2d.plot([], [], 'o', color=C["tx"], ms=10, zorder=5)
    rx_dot,      = ax2d.plot([], [], 'o', color=C["rx"], ms=10, zorder=5)
    los_line,    = ax2d.plot([], [], '--', color='white', lw=0.8, alpha=0.5)
    time_txt     = ax2d.text(0.02, 0.95, '', transform=ax2d.transAxes,
                             color='white', fontsize=9, va='top')
    pol_gt_ln,   = ax_pol.plot([], [], '-', color=C["gt"], lw=2.5)
    pol_est_ln,  = ax_pol.plot([], [], '-', color=C["pd"], lw=2.5, alpha=0.9)
    pol_gt_dot,  = ax_pol.plot([], [], 'o', color=C["gt"], ms=8)
    pol_est_dot, = ax_pol.plot([], [], 'o', color=C["pd"], ms=8)
    range_txt    = ax_pol.text(np.pi * 1.25, 1.5, '', color='white',
                               fontsize=8, ha='center', va='center')

    plt.tight_layout()

    artists = dict(
        tx_trail=tx_trail, rx_trail=rx_trail,
        tx_dot=tx_dot, rx_dot=rx_dot, los_line=los_line,
        time_txt=time_txt, pol_gt_ln=pol_gt_ln, pol_est_ln=pol_est_ln,
        pol_gt_dot=pol_gt_dot, pol_est_dot=pol_est_dot, range_txt=range_txt,
    )
    return fig, artists


def _render_to_file(fig, artists: dict, fd: dict, file_path: Path,
                    writer, dpi: int) -> None:
    """
    Core render loop: mutate artists, grab frame, repeat.
    Driving the writer directly (rather than via FuncAnimation) avoids
    per-frame overhead from FuncAnimation's internal diffing machinery.
    """
    n_f   = SimConfig.ANIMATION_N_FRAMES
    trail = max(5, n_f // 15)   # how many frames to keep in the position trail

    with writer.saving(fig, str(file_path), dpi=dpi):
        for i in range(n_f):
            sl = slice(max(0, i - trail), i + 1)

            # 2-D panel: trails, current dots, LoS line, timestamp
            artists["tx_trail"].set_data(fd["tx_x_a"][sl], fd["tx_y_a"][sl])
            artists["rx_trail"].set_data(fd["rx_x_a"][sl], fd["rx_y_a"][sl])
            artists["tx_dot"].set_data([fd["tx_x_a"][i]], [fd["tx_y_a"][i]])
            artists["rx_dot"].set_data([fd["rx_x_a"][i]], [fd["rx_y_a"][i]])
            artists["los_line"].set_data(
                [fd["rx_x_a"][i], fd["tx_x_a"][i]],
                [fd["rx_y_a"][i], fd["tx_y_a"][i]])
            artists["time_txt"].set_text(f"t = {fd['t_anim'][i]:.2f} s")

            # Polar panel: unit-radius needle from origin to bearing angle
            artists["pol_gt_ln"].set_data( [fd["los_a"][i],     fd["los_a"][i]],     [0, 1.0])
            artists["pol_est_ln"].set_data([fd["los_est_a"][i], fd["los_est_a"][i]], [0, 1.0])
            artists["pol_gt_dot"].set_data( [fd["los_a"][i]],     [1.0])
            artists["pol_est_dot"].set_data([fd["los_est_a"][i]], [1.0])
            artists["range_txt"].set_text(f"r = {fd['r_a'][i]:.0f} m")

            writer.grab_frame()


def make_animation(result: dict, out_path: Path) -> None:
    """
    Render animation to MP4 and/or GIF depending on SimConfig.RENDER_FORMATS.
    The figure and frame data are built once and reused across both formats.
    """
    if not (_RENDER_MP4 or _RENDER_GIF):
        log.info("  Animation skipped (RENDER_FORMATS='none')")
        return

    fps = SimConfig.ANIMATION_FPS

    with timed("  pre-compute frame data"):
        fd = _build_animation_frame_data(result)

    with timed("  build figure"):
        fig, artists = _build_animation_figure(result)

    if _RENDER_MP4:
        mp4_path = out_path.with_suffix(".mp4")
        with timed(f"  render MP4  ({SimConfig.ANIMATION_N_FRAMES} frames, "
                   f"dpi={SimConfig.ANIMATION_DPI_MP4})"):
            try:
                w = FFMpegWriter(
                    fps=fps,
                    metadata=dict(title=result["scenario_name"]),
                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                                '-preset', 'fast', '-crf', '23'],
                )
                _render_to_file(fig, artists, fd, mp4_path, w, SimConfig.ANIMATION_DPI_MP4)
                log.info(f"  Saved → {mp4_path}")
            except Exception as e:
                log.warning(f"  MP4 render failed: {e}")

    if _RENDER_GIF:
        gif_path = out_path.with_suffix(".gif")
        with timed(f"  render GIF  ({SimConfig.ANIMATION_N_FRAMES} frames, "
                   f"dpi={SimConfig.ANIMATION_DPI_GIF})"):
            try:
                w = PillowWriter(fps=fps)
                _render_to_file(fig, artists, fd, gif_path, w, SimConfig.ANIMATION_DPI_GIF)
                log.info(f"  Saved → {gif_path}")
            except Exception as e:
                log.warning(f"  GIF render failed: {e}")

    plt.close(fig)


# =============================================================================
# Save all outputs for one scenario
# =============================================================================

def save_all(result: dict, out_root: str = "plots") -> None:
    name    = result["scenario_name"]
    slug    = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    t_pd, df_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    est_pd = {"t": t_pd, "delta_f": df_pd}

    with timed("  static plots"):
        plot_trajectory_fig(result,       out_dir / "trajectory.png")
        plot_rdot_rddot_fig(result, est_pd, out_dir / "rdot_rddot.png")
        plot_doppler_fig(result,    est_pd, out_dir / "doppler.png")

    make_animation(result, out_dir / "animation")
    log.info(f"  All outputs → {out_dir}/")


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
        log.info(f"{'='*50}")
        log.info(f"Scenario: {name}")

        with timed("  IQ generation"):
            result = generate_iq(cfg)
        result["scenario_name"] = name

        vdx   = cfg.tx_vx - cfg.rx_vx
        cpa_t = cfg.tx_x0 / max(abs(vdx), 1e-3) if abs(vdx) > 0.1 else float('inf')
        log.info(f"  Rx: {cfg.rx_vx:.1f} m/s   Tx: {cfg.tx_vx:.1f} m/s   Δv: {vdx:.1f} m/s")
        log.info(f"  Samples: {len(result['iq']):,}   Duration: {cfg.duration:.1f} s")
        log.info(f"  Δf range:    [{result['delta_f'].min():.2f}, {result['delta_f'].max():.2f}] Hz")
        log.info(f"  r̈ peak (GT): {np.nanmax(np.abs(result['r_ddot'])):.1f} m/s²")
        log.info(f"  Range CPA:   {result['r'].min():.1f} m  (at t ≈ {cpa_t:.2f} s)")

        save_all(result, "plots")

    log.info("Done.")