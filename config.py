"""
Configuration module for PyDopplerSim.

Exports SimConfig class and scenario builders.
"""

__all__ = [
    "SimConfig",
    "ScenarioConfig",
    "scenario_colocated",
    "scenario_same_direction",
    "scenario_oncoming",
]

# =============================================================================
# ★  TOP-LEVEL CONFIG — edit this block
# =============================================================================


class SimConfig:
    # ---------- rendering ------------------------------------------------
    # RENDER_FORMATS: which animation formats to produce.
    # Options: "mp4", "gif", "both", "none"  (case-insensitive)
    RENDER_FORMATS = "both"
    # ANIMATION_VARIANTS: which animations to produce.
    # Options: "positions", "spectrogram", "both"  (case-insensitive)
    ANIMATION_VARIANTS = "both"
    ANIMATION_FPS = 20
    ANIMATION_N_FRAMES = 100  # fewer frames = faster; 100 is good for GIF
    ANIMATION_DPI_MP4 = 120
    ANIMATION_DPI_GIF = 80

    # Full path to ffmpeg binary. Set to None to rely on system PATH.
    # Find your path with `where ffmpeg` (Windows) or `which ffmpeg` (Unix).
    # Example: r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    FFMPEG_PATH: str | None = None
    FFMPEG_PATH = r"C:\Users\manic\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    # ---------- RF / sampling --------------------------------------------
    FC = 915e6  # carrier frequency [Hz]  — controls Hz-per-(m/s) sensitivity
    FS = 1e6  # IQ sample rate [Hz]
    SNR_DB = 20.0  # signal-to-noise ratio [dB]  (10 → noisy, 30 → clean)
    F_TONE = 0.0  # optional baseband tone offset [Hz]; 0 = pure CW at DC

    # ---------- receiver (always travels in +x) --------------------------
    RX_VX = 30.0  # receiver speed [m/s]  (~108 km/h)
    RX_X0 = 0.0  # receiver initial x position [m]
    RX_KNOWS_VELOCITY = False  # RX reconstruction may use its own vx when True

    # ---------- scenario: Co-located -------------------------------------
    # Tx rides in the same vehicle as Rx → near-zero relative velocity.
    # A tiny lateral offset (0.5 m) keeps both dots visible in trajectory plots.
    COLOC_TX_Y = 0.5  # [m]

    # ---------- scenario: Same-direction ---------------------------------
    # Tx is in an adjacent lane travelling the same direction but slightly slower.
    # Rx will overtake Tx during the simulation.
    SAME_TX_X0 = 50.0  # Tx starts this far ahead of Rx [m]
    SAME_TX_Y = 3.7  # lateral lane separation [m]
    SAME_TX_VX = 20.0  # Tx speed [m/s]  (slower than Rx → Rx overtakes)
    SAME_DURATION = 10.0  # [s]

    # ---------- scenario: Oncoming ---------------------------------------
    # Tx is in the opposing lane heading directly toward Rx.
    # CPA (closest point of approach) occurs at t = TX_X0 / (RX_VX - TX_VX).
    # ONCO_DURATION should be long enough to capture well past CPA.
    ONCO_TX_X0 = 400.0  # initial separation [m]
    ONCO_TX_Y = 3.7  # opposing lane offset [m]
    ONCO_TX_VX = -30.0  # Tx speed [m/s]  (negative = approaching)
    ONCO_DURATION = 13.4  # [s]  → CPA at t ≈ 400/60 ≈ 6.67 s

    # ---------- estimation parameters ------------------------------------
    # Phase-differentiator smoother: wide window suppresses noise but blurs
    # fast Doppler transitions.  Tune down if CPA peak is smeared.
    PD_SMOOTH_WINDOW = 501  # Hann FIR length [samples @ FS]

    # r̈ is estimated by decimating ṙ to a low rate before differentiating.
    # Differencing at full FS would amplify noise by ~FS × (noise on ṙ).
    # At 100 Hz the dt denominator is 10 ms instead of 1 µs → 10,000× less noise.
    RDOT_DECIM_HZ = 100  # ṙ decimation rate before d/dt [Hz]
    RDOT_SMOOTH_WINDOW = 21  # Hann FIR for r̈ smoother [samples @ RDOT_DECIM_HZ]

    STFT_WINDOW_DUR = 0.05  # STFT analysis window [s]  → freq resolution = 1/T Hz
    STFT_HOP_DUR = 0.005  # STFT hop size [s]  → temporal resolution of spectrogram
    INTERP_OVERSAMPLE = 8  # phase integration oversampling factor (accuracy vs speed)


# Apply ffmpeg path override before any animation writers are instantiated
import matplotlib

if SimConfig.FFMPEG_PATH:
    matplotlib.rcParams["animation.ffmpeg_path"] = SimConfig.FFMPEG_PATH

# Parse render format flag once at import time
_fmt = SimConfig.RENDER_FORMATS.lower().strip()
RENDER_MP4 = _fmt in ("mp4", "both")
RENDER_GIF = _fmt in ("gif", "both")


# =============================================================================
# Scenario config dataclass
# =============================================================================

from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """All parameters needed to define one simulation run."""

    fc: float = 5.8e9
    fs: float = 1e6
    duration: float = 5.0
    tx_x0: float = 100.0
    tx_y: float = 3.7
    tx_vx: float = 30.0
    rx_x0: float = 0.0
    rx_y: float = 0.0
    rx_vx: float = 30.0
    rx_knows_velocity: bool = False
    snr_db: float = 20.0
    f_tone: float = 0.0
    interp_oversample: int = 8

    def __post_init__(self):
        self.c = 299_792_458.0  # speed of light [m/s]


def _base() -> ScenarioConfig:
    """Populate a ScenarioConfig from the global SimConfig values."""
    return ScenarioConfig(
        fc=SimConfig.FC,
        fs=SimConfig.FS,
        snr_db=SimConfig.SNR_DB,
        f_tone=SimConfig.F_TONE,
        rx_x0=SimConfig.RX_X0,
        rx_vx=SimConfig.RX_VX,
        rx_knows_velocity=SimConfig.RX_KNOWS_VELOCITY,
        interp_oversample=SimConfig.INTERP_OVERSAMPLE,
    )


def scenario_colocated() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0 = cfg.rx_x0  # start at same x position
    cfg.tx_y = SimConfig.COLOC_TX_Y
    cfg.tx_vx = cfg.rx_vx  # identical velocity → Δv ≈ 0
    cfg.duration = 5.0
    return cfg


def scenario_same_direction() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0 = SimConfig.SAME_TX_X0
    cfg.tx_y = SimConfig.SAME_TX_Y
    cfg.tx_vx = SimConfig.SAME_TX_VX
    cfg.duration = SimConfig.SAME_DURATION
    return cfg


def scenario_oncoming() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0 = SimConfig.ONCO_TX_X0
    cfg.tx_y = SimConfig.ONCO_TX_Y
    cfg.tx_vx = SimConfig.ONCO_TX_VX  # negative → approaching
    cfg.duration = SimConfig.ONCO_DURATION
    return cfg
