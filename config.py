"""
config.py — Single source of truth for all simulation and DSP parameters.

Two classes live here:
  SimConfig     : top-level knobs (RF params, scenario geometry, render flags).
                  Edit this before running sim.py.
  ScenarioConfig: dataclass populated from SimConfig; one instance per scenario.
                  Passed through the entire pipeline so every stage is self-contained.
"""

from dataclasses import dataclass


# =============================================================================
# SimConfig — edit this block to configure a run
# =============================================================================

class SimConfig:

    # ---------- rendering ------------------------------------------------
    # "mp4" | "gif" | "both" | "none"
    RENDER_FORMATS      = "mp4"
    ANIMATION_FPS       = 20
    ANIMATION_N_FRAMES  = 100       # fewer = faster; 100 good for GIF
    ANIMATION_DPI_MP4   = 120
    ANIMATION_DPI_GIF   = 80

    # Full path to ffmpeg binary, or None to rely on system PATH.
    # Find with `where ffmpeg` (Windows) or `which ffmpeg` (Unix).
    FFMPEG_PATH: str | None = None
    # ---------- RF / sampling --------------------------------------------
    FC      = 5.8e9     # carrier frequency [Hz]
    FS      = 1e6       # IQ sample rate [Hz]
    SNR_DB  = 20.0      # signal-to-noise ratio [dB]
    F_TONE  = 0.0       # baseband tone offset [Hz]; 0 = pure CW at DC

    # ---------- receiver -------------------------------------------------
    RX_VX   = 30.0      # receiver speed [m/s]
    RX_X0   = 0.0       # receiver initial x position [m]

    # ---------- scenario: Co-located -------------------------------------
    # Tx rides in same vehicle → near-zero relative velocity.
    # Small lateral offset keeps both dots visible in trajectory plots.
    COLOC_TX_Y      = 0.5       # [m]

    # ---------- scenario: Same-direction ---------------------------------
    # Adjacent lane, same direction, slightly slower → Rx overtakes.
    SAME_TX_X0      = 50.0      # Tx starts ahead [m]
    SAME_TX_Y       = 3.7       # lane separation [m]
    SAME_TX_VX      = 28.0      # Tx speed [m/s]
    SAME_DURATION   = 10.0      # [s]

    # ---------- scenario: Oncoming ---------------------------------------
    # Opposing lane, head-on approach.
    # CPA at t = ONCO_TX_X0 / (RX_VX − ONCO_TX_VX).
    ONCO_TX_X0      = 400.0     # initial separation [m]
    ONCO_TX_Y       = 3.7       # opposing lane offset [m]
    ONCO_TX_VX      = -30.0     # Tx speed [m/s]  (negative = approaching)
    ONCO_DURATION   = 13.4      # [s]  → CPA at t ≈ 6.67 s

    # ---------- DSP: phase-diff estimator --------------------------------
    # Hann smoother applied to raw FM-discriminator output.
    # Wide window → lower noise floor but blurs fast transitions near CPA.
    PD_SMOOTH_WINDOW    = 501   # [samples @ FS]

    # ---------- DSP: kinematic recovery ----------------------------------
    # ṙ is decimated before differentiating to avoid amplifying sample-rate
    # noise.  See kinematics.py for a detailed explanation.
    RDOT_DECIM_HZ       = 100   # decimation target rate [Hz]
    RDOT_SMOOTH_WINDOW  = 21    # Hann smoother on r̈ [samples @ RDOT_DECIM_HZ]

    # ---------- DSP: STFT ------------------------------------------------
    STFT_WINDOW_DUR     = 0.05  # analysis window [s] → freq resolution = 1/T Hz
    STFT_HOP_DUR        = 0.005 # hop size [s]

    # ---------- simulation numerics -------------------------------------
    INTERP_OVERSAMPLE   = 8     # phase-integration oversampling (accuracy vs speed)


# =============================================================================
# ScenarioConfig — one instance per simulation run
# =============================================================================

@dataclass
class ScenarioConfig:
    """
    All parameters needed to fully define one scenario.
    Passed through geometry, IQ generation, DSP, and plotting unchanged
    so every stage is self-contained and testable in isolation.
    """
    # RF
    fc:   float = 5.8e9
    fs:   float = 1e6
    f_tone: float = 0.0
    snr_db: float = 20.0
    interp_oversample: int = 8

    # Duration
    duration: float = 5.0

    # Transmitter
    tx_x0: float = 100.0
    tx_y:  float = 3.7
    tx_vx: float = 30.0

    # Receiver
    rx_x0: float = 0.0
    rx_y:  float = 0.0
    rx_vx: float = 30.0

    def __post_init__(self):
        self.c = 299_792_458.0  # speed of light [m/s]


# =============================================================================
# Factory helpers — build ScenarioConfig instances from SimConfig
# =============================================================================

def _base() -> ScenarioConfig:
    """Populate shared RF/receiver fields from SimConfig."""
    return ScenarioConfig(
        fc=SimConfig.FC, fs=SimConfig.FS,
        f_tone=SimConfig.F_TONE, snr_db=SimConfig.SNR_DB,
        rx_x0=SimConfig.RX_X0, rx_vx=SimConfig.RX_VX,
        interp_oversample=SimConfig.INTERP_OVERSAMPLE,
    )


def scenario_colocated() -> ScenarioConfig:
    cfg = _base()
    cfg.tx_x0    = cfg.rx_x0
    cfg.tx_y     = SimConfig.COLOC_TX_Y
    cfg.tx_vx    = cfg.rx_vx        # identical velocity → Δv ≈ 0
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
    cfg.tx_vx    = SimConfig.ONCO_TX_VX    # negative = approaching
    cfg.duration = SimConfig.ONCO_DURATION
    return cfg