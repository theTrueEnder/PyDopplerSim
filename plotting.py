"""
Plotting module for PyDopplerSim.

Provides static plots and animation for simulation results.
"""

__all__ = [
    "plot_trajectory_fig",
    "plot_rdot_rddot_fig",
    "plot_doppler_fig",
    "plot_tx_derivation_fig",
    "make_animation",
    "save_all",
    "C_STATIC",
    "C_DARK",
]

import logging
import time
import warnings
from pathlib import Path
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np

# Import our modules
from config import SimConfig
from estimation import estimate_doppler_phase_diff, estimate_doppler_stft
from kinematics import recover_kinematics

# Setup logging
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


def _estimate_tx_positions(result: dict, t_est: np.ndarray, delta_f_est: np.ndarray) -> dict:
    """
    Report RX-minimal TX position observability.

    Deprecated: TX path derivation is not part of RX-minimal reconstruction.
    This compatibility helper will be removed in a future release.

    Doppler gives range-rate, not range, bearing, or TX position. This helper
    intentionally does not use simulator-only truth such as TX velocity, lane
    offset, or range to back-fill an RX estimate.
    """
    warnings.warn(
        "_estimate_tx_positions() is deprecated; TX path derivation is not "
        "observable in RX-minimal reconstruction.",
        DeprecationWarning,
        stacklevel=2,
    )
    t = result["t"]

    tx_x_est = np.full_like(t, np.nan, dtype=float)
    tx_y_est = np.full_like(t, np.nan, dtype=float)
    los_est = np.full_like(t, np.nan, dtype=float)
    valid_mask = np.zeros_like(t, dtype=bool)

    return dict(
        tx_x_est=tx_x_est,
        tx_y_est=tx_y_est,
        los_est=los_est,
        valid_mask=valid_mask,
        observable=False,
        reason=(
            "TX position, bearing, and range are not observable from Doppler "
            "alone or from Doppler plus RX velocity."
        ),
    )


def _add_truth_notice(ax, color: str = "#333333") -> None:
    """Label simulator-only ground-truth views."""
    ax.text(
        0.02,
        0.97,
        "Ground truth diagnostic: RX-minimal reconstruction does not know TX path/range.",
        transform=ax.transAxes,
        fontsize=8,
        color=color,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
            edgecolor="#cccccc",
        ),
    )


def _mask_phase_diff_edges(
    t_est: np.ndarray, delta_f_est: np.ndarray, smooth_window: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mask phase-diff smoothing edge samples for display."""
    if smooth_window is None:
        smooth_window = SimConfig.PD_SMOOTH_WINDOW
    t_plot = np.asarray(t_est)
    df_plot = np.asarray(delta_f_est, dtype=float).copy()
    trim = min(smooth_window // 2, len(df_plot) // 2)
    if trim:
        df_plot[:trim] = np.nan
        df_plot[-trim:] = np.nan
    return t_plot, df_plot


# =============================================================================
# Colour palettes
# =============================================================================

# Static plots use a light matplotlib background
C_STATIC = dict(
    gt="#1a1a1a",  # near-black  — ground truth
    pd="#e05a00",  # dark orange — phase-diff estimate
    tx="#c07000",  # dark amber  — transmitter
    rx="#1565c0",  # dark blue   — receiver
    stft="#1565c0",
    tx_est="#888888",  # gray for estimated TX position
)

# Animation uses a dark background
C_DARK = dict(
    gt="#f0f0f0",  # near-white  — ground truth
    pd="#ff6b35",  # vivid orange — estimate
    tx="#ffaa00",  # amber       — transmitter
    rx="#4dabf7",  # sky blue    — receiver
    stft="#74c0fc",
    tx_est="#666666",  # gray for estimated TX position
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
    C = C_STATIC
    cfg = result["cfg"]
    t = result["t"]

    fig, (ax_2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['scenario_name']} — Trajectory", fontweight="bold")

    ax_2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], lw=2, label="Tx")
    ax_2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], lw=2, label="Rx")

    # Direction arrows at mid-trajectory
    for xs, ys, col in [
        (result["tx_x"], result["tx_y"], C["tx"]),
        (result["rx_x"], result["rx_y"], C["rx"]),
    ]:
        mid = len(xs) // 2
        ax_2d.annotate(
            "",
            xy=(xs[mid + 5], ys[mid + 5]),
            xytext=(xs[mid], ys[mid]),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
        )

    ax_2d.scatter(
        [result["tx_x"][0], result["rx_x"][0]],
        [result["tx_y"][0], result["rx_y"][0]],
        color=[C["tx"], C["rx"]],
        s=60,
        zorder=5,
    )

    ax_2d.set_xlabel("x [m]")
    ax_2d.set_ylabel("y [m]")
    ax_2d.set_title("2-D trajectory (top-down)")
    ax_2d.legend(fontsize=8, loc="lower right")
    ax_2d.grid(True, alpha=0.4)
    _add_truth_notice(ax_2d)
    # Fix y to ±15 m around lane midpoint — do NOT use equal aspect (x spans
    # hundreds of metres, y spans ~4 m, equal aspect would squash the lanes).
    y_c = (cfg.tx_y + cfg.rx_y) / 2
    ax_2d.set_ylim(y_c - 15, y_c + 15)

    ax_r.plot(t, result["r"], color="purple", lw=1.5)
    ax_r.set_xlabel("Time [s]")
    ax_r.set_ylabel("Range r(t) [m]")
    ax_r.set_title("Range over time")
    ax_r.grid(True, alpha=0.4)
    _add_truth_notice(ax_r)

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
    C = C_STATIC
    cfg = result["cfg"]
    t = result["t"]
    kin = recover_kinematics(
        est_pd["t"],
        est_pd["delta_f"],
        cfg.fc,
        cfg.c,
        rx_knows_velocity=getattr(cfg, "rx_knows_velocity", False),
        rx_vx=cfg.rx_vx if getattr(cfg, "rx_knows_velocity", False) else None,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"{result['scenario_name']} — Radial kinematics", fontweight="bold")

    ax = axes[0]
    ax.plot(t, result["r_dot"], color=C["gt"], lw=1.8, label="Ground truth ṙ")
    ax.plot(
        kin["t_dot"],
        kin["r_dot"],
        color=C["pd"],
        lw=1.0,
        alpha=0.9,
        label="Estimated ṙ (phase-diff)",
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ṙ [m/s]")
    ax.set_title("Radial velocity ṙ(t)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t, result["r_ddot"], color=C["gt"], lw=1.8, label="Ground truth r̈")
    ax.plot(
        kin["t_ddot"],
        kin["r_ddot"],
        color=C["pd"],
        lw=1.0,
        alpha=0.9,
        label=f"Estimated r̈  (decimated to {SimConfig.RDOT_DECIM_HZ} Hz then d/dt)",
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("r̈ [m/s²]")
    ax.set_title("Radial acceleration r̈(t)  [peak magnitude = v_rel² / r_min at CPA]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.4)

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
    C = C_STATIC
    cfg = result["cfg"]
    t = result["t"]
    gt = result["delta_f"]

    # Auto-zoom STFT display to 3× max true Doppler (minimum ±500 Hz)
    zoom = max(500.0, 3.0 * np.abs(gt).max())
    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)

    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Doppler estimation", fontweight="bold")

    t_pd_plot, df_pd_plot = _mask_phase_diff_edges(est_pd["t"], est_pd["delta_f"])
    ax_pd.plot(
        t_pd_plot,
        df_pd_plot,
        color=C["pd"],
        lw=0.8,
        alpha=0.85,
        label="Phase-diff Δf (edge-masked)",
    )
    ax_pd.plot(t, gt, color=C["gt"], lw=1.5, ls="--", label="Ground truth")
    ax_pd.set_ylabel("Δf [Hz]")
    ax_pd.set_title("Instantaneous frequency — phase differentiation")
    ax_pd.legend(fontsize=8, loc="upper right")
    ax_pd.grid(True, alpha=0.4)

    mask = stft["mask"]
    Sxx_dB = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    extent = [
        stft["t"][0],
        stft["t"][-1],
        stft["freq_axis"][mask][0],
        stft["freq_axis"][mask][-1],
    ]
    ax_stft.imshow(
        Sxx_dB,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=np.percentile(Sxx_dB, 20),
    )
    ax_stft.plot(
        stft["t"],
        stft["peak_freq"],
        color=C["stft"],
        lw=1.0,
        ls="--",
        label="STFT peak",
    )
    # Use light grey for GT overlay on dark spectrogram background
    ax_stft.plot(t, gt, color="#cccccc", lw=1.5, ls="--", label="Ground truth")
    ax_stft.set_ylabel("Δf [Hz]")
    ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (zoomed ±{zoom:.0f} Hz)")
    ax_stft.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# TX Path Derivation (T11)
# =============================================================================


def plot_tx_derivation_fig(result: dict, est_pd: dict, out_path: Path) -> None:
    """
    Plot simulator TX trajectory and RX-minimal observability.

    TX position is not reconstructed without explicit future assumptions.
    """
    warnings.warn(
        "plot_tx_derivation_fig() is deprecated and is no longer generated by "
        "default.",
        DeprecationWarning,
        stacklevel=2,
    )
    C = C_STATIC
    cfg = result["cfg"]
    est_tx = _estimate_tx_positions(result, est_pd["t"], est_pd["delta_f"])

    # The estimate is only drawn when the Doppler inversion is observable.

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(
        f"{result['scenario_name']} — TX Path Derivation (Deprecated)",
        fontweight="bold",
    )

    # Plot true trajectories
    ax.plot(result["tx_x"], result["tx_y"], color=C["tx"], lw=2, label="True TX")
    ax.plot(result["rx_x"], result["rx_y"], color=C["rx"], lw=2, label="RX")

    if est_tx["observable"] and np.any(est_tx["valid_mask"]):
        ax.plot(
            est_tx["tx_x_est"][est_tx["valid_mask"]],
            est_tx["tx_y_est"][est_tx["valid_mask"]],
            color=C["tx_est"],
            lw=1.5,
            alpha=0.8,
            linestyle="--",
            label="Estimated TX",
        )
    else:
        ax.text(
            0.02,
            0.95,
            est_tx["reason"],
            transform=ax.transAxes,
            fontsize=9,
            color=C["tx_est"],
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            ),
        )

    # Mark start positions
    ax.scatter(
        [result["tx_x"][0], result["rx_x"][0]],
        [result["tx_y"][0], result["rx_y"][0]],
        color=[C["tx"], C["rx"]],
        s=60,
        zorder=5,
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("TX path observability")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.4)
    _add_truth_notice(ax)

    # Set y limits
    y_c = (cfg.tx_y + cfg.rx_y) / 2
    ax.set_ylim(y_c - 15, y_c + 15)

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

    This animation shows simulator ground truth. RX-minimal reconstruction can
    recover range-rate from Doppler, but cannot infer range, bearing, or TX
    position from Doppler plus optional RX velocity alone.
    """
    t = result["t"]
    n_f = SimConfig.ANIMATION_N_FRAMES

    # Sub-sample ground-truth arrays to n_f evenly-spaced animation frames
    idx = np.linspace(0, len(t) - 1, n_f, dtype=int)
    t_anim = t[idx]

    # Estimated TX state is intentionally unavailable in RX-minimal mode.
    r_a = result["r"][idx]
    unavailable = np.full_like(t_anim, np.nan, dtype=float)

    return dict(
        t_anim=t_anim,
        tx_x_a=result["tx_x"][idx],
        tx_y_a=result["tx_y"][idx],
        rx_x_a=result["rx_x"][idx],
        rx_y_a=result["rx_y"][idx],
        los_a=result["los_angle"][idx],
        los_est_a=unavailable,
        r_a=r_a,
        tx_x_est_a=unavailable,
        tx_y_est_a=unavailable,
        tx_est_observable=False,
        observability_note=(
            "Ground truth diagnostic. RX-minimal reconstruction cannot infer "
            "range, bearing, or TX path."
        ),
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
    C = C_DARK
    cfg = result["cfg"]

    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax2d = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection="polar")

    # --- 2-D panel styling ---
    ax2d.set_facecolor("#16213e")
    for sp in ax2d.spines.values():
        sp.set_edgecolor("#555")
    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color="white")
    ax2d.set_ylabel("y [m]", color="white")
    ax2d.tick_params(colors="white")
    ax2d.grid(True, alpha=0.25, color="#555")
    ax2d.set_title("Vehicle positions", color="white", fontsize=10)
    ax2d.text(
        0.02,
        0.96,
        "Ground truth diagnostic: TX path/range are not RX-observable.",
        transform=ax2d.transAxes,
        color="white",
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(
            facecolor="#111",
            alpha=0.75,
            edgecolor="#666",
            boxstyle="round,pad=0.3",
        ),
    )

    # Faint full-path ghost traces so you can see where they came from/are going
    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], alpha=0.15, lw=1)

    # Dummy artists for legend (explicit loc avoids 'best' location search)
    ax2d.plot([], [], "o", color=C["tx"], ms=7, label="Tx")
    ax2d.plot([], [], "o", color=C["rx"], ms=7, label="Rx")
    ax2d.plot(
        [],
        [],
        "o",
        color=C["tx_est"],
        ms=7,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Tx estimate unavailable",
    )
    ax2d.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="lower right")

    # --- Polar panel styling ---
    ax_pol.set_facecolor("#16213e")
    ax_pol.tick_params(colors="#aaa")
    ax_pol.spines["polar"].set_color("#555")
    ax_pol.set_theta_zero_location("E")  # 0° = East = Tx directly ahead (+x)
    ax_pol.set_theta_direction(1)  # counter-clockwise (standard math convention)
    ax_pol.set_ylim(0, 1.3)
    ax_pol.set_yticks([])  # radial ticks meaningless (unit radius)
    ax_pol.set_title(
        "True Tx bearing\n(ground truth, 0 deg=ahead)", color="white", fontsize=9, pad=12
    )

    # Cardinal direction labels
    for angle, lbl in [
        (0, "Ahead"),
        (np.pi / 2, "Left"),
        (np.pi, "Behind"),
        (-np.pi / 2, "Right"),
    ]:
        ax_pol.text(
            angle, 1.22, lbl, ha="center", va="center", color="#888", fontsize=7
        )

    # Polar legend
    ax_pol.plot([], [], "-", color=C["gt"], lw=2, label="True")
    ax_pol.plot([], [], "-", color=C["pd"], lw=2, label="RX estimate unavailable")
    ax_pol.legend(
        fontsize=7,
        facecolor="#222",
        labelcolor="white",
        loc="upper left",
        bbox_to_anchor=(1.05, 1.12),
    )

    # --- Mutable artists updated each frame ---
    (tx_trail,) = ax2d.plot([], [], color=C["tx"], lw=1.5, alpha=0.5)
    (rx_trail,) = ax2d.plot([], [], color=C["rx"], lw=1.5, alpha=0.5)
    (tx_est_trail,) = ax2d.plot(
        [], [], color=C["tx_est"], lw=1.0, alpha=0.4, linestyle="--"
    )
    (tx_dot,) = ax2d.plot([], [], "o", color=C["tx"], ms=10, zorder=5)
    (rx_dot,) = ax2d.plot([], [], "o", color=C["rx"], ms=10, zorder=5)
    (tx_est_dot,) = ax2d.plot(
        [],
        [],
        "o",
        color=C["tx_est"],
        ms=8,
        markerfacecolor="none",
        markeredgewidth=2,
        zorder=5,
    )
    (los_line,) = ax2d.plot([], [], "--", color="white", lw=0.8, alpha=0.5)
    time_txt = ax2d.text(
        0.02, 0.95, "", transform=ax2d.transAxes, color="white", fontsize=9, va="top"
    )
    (pol_gt_ln,) = ax_pol.plot([], [], "-", color=C["gt"], lw=2.5)
    (pol_est_ln,) = ax_pol.plot([], [], "-", color=C["pd"], lw=2.5, alpha=0.9)
    (pol_gt_dot,) = ax_pol.plot([], [], "o", color=C["gt"], ms=8)
    (pol_est_dot,) = ax_pol.plot([], [], "o", color=C["pd"], ms=8)
    range_txt = ax_pol.text(
        np.pi * 1.25, 1.5, "", color="white", fontsize=8, ha="center", va="center"
    )

    plt.tight_layout()

    artists = dict(
        tx_trail=tx_trail,
        rx_trail=rx_trail,
        tx_est_trail=tx_est_trail,
        tx_dot=tx_dot,
        rx_dot=rx_dot,
        tx_est_dot=tx_est_dot,
        los_line=los_line,
        time_txt=time_txt,
        pol_gt_ln=pol_gt_ln,
        pol_est_ln=pol_est_ln,
        pol_gt_dot=pol_gt_dot,
        pol_est_dot=pol_est_dot,
        range_txt=range_txt,
    )
    return fig, artists


def _render_to_file(
    fig, artists: dict, fd: dict, file_path: Path, writer, dpi: int
) -> None:
    """
    Core render loop: mutate artists, grab frame, repeat.
    Driving the writer directly (rather than via FuncAnimation) avoids
    per-frame overhead from FuncAnimation's internal diffing machinery.
    """
    n_f = SimConfig.ANIMATION_N_FRAMES
    trail = max(5, n_f // 15)  # how many frames to keep in the position trail

    with writer.saving(fig, str(file_path), dpi=dpi):
        for i in range(n_f):
            sl = slice(max(0, i - trail), i + 1)

            # 2-D panel: trails, current dots, LoS line, timestamp
            artists["tx_trail"].set_data(fd["tx_x_a"][sl], fd["tx_y_a"][sl])
            artists["rx_trail"].set_data(fd["rx_x_a"][sl], fd["rx_y_a"][sl])
            artists["tx_est_trail"].set_data(fd["tx_x_est_a"][sl], fd["tx_y_est_a"][sl])
            artists["tx_dot"].set_data([fd["tx_x_a"][i]], [fd["tx_y_a"][i]])
            artists["rx_dot"].set_data([fd["rx_x_a"][i]], [fd["rx_y_a"][i]])
            artists["tx_est_dot"].set_data([fd["tx_x_est_a"][i]], [fd["tx_y_est_a"][i]])
            artists["los_line"].set_data(
                [fd["rx_x_a"][i], fd["tx_x_a"][i]], [fd["rx_y_a"][i], fd["tx_y_a"][i]]
            )
            artists["time_txt"].set_text(f"t = {fd['t_anim'][i]:.2f} s")

            # Polar panel: unit-radius needle from origin to bearing angle
            artists["pol_gt_ln"].set_data([fd["los_a"][i], fd["los_a"][i]], [0, 1.0])
            artists["pol_est_ln"].set_data(
                [fd["los_est_a"][i], fd["los_est_a"][i]], [0, 1.0]
            )
            artists["pol_gt_dot"].set_data([fd["los_a"][i]], [1.0])
            artists["pol_est_dot"].set_data([fd["los_est_a"][i]], [1.0])
            artists["range_txt"].set_text(f"GT r = {fd['r_a'][i]:.0f} m")

            writer.grab_frame()


def _build_positions_animation_figure(result: dict) -> tuple:
    """Create the positions-only animation figure."""
    C = C_DARK
    cfg = result["cfg"]

    fig, ax2d = plt.subplots(1, 1, figsize=(9, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax2d.set_facecolor("#16213e")
    for sp in ax2d.spines.values():
        sp.set_edgecolor("#555")

    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color="white")
    ax2d.set_ylabel("y [m]", color="white")
    ax2d.tick_params(colors="white")
    ax2d.grid(True, alpha=0.25, color="#555")
    ax2d.set_title("Vehicle positions", color="white", fontsize=10)
    ax2d.text(
        0.02,
        0.96,
        "Ground truth diagnostic: TX path/range are not RX-observable.",
        transform=ax2d.transAxes,
        color="white",
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(
            facecolor="#111",
            alpha=0.75,
            edgecolor="#666",
            boxstyle="round,pad=0.3",
        ),
    )

    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], alpha=0.15, lw=1)
    ax2d.plot([], [], "o", color=C["tx"], ms=7, label="Tx")
    ax2d.plot([], [], "o", color=C["rx"], ms=7, label="Rx")
    ax2d.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="lower right")

    (tx_trail,) = ax2d.plot([], [], color=C["tx"], lw=1.5, alpha=0.5)
    (rx_trail,) = ax2d.plot([], [], color=C["rx"], lw=1.5, alpha=0.5)
    (tx_dot,) = ax2d.plot([], [], "o", color=C["tx"], ms=10, zorder=5)
    (rx_dot,) = ax2d.plot([], [], "o", color=C["rx"], ms=10, zorder=5)
    (los_line,) = ax2d.plot([], [], "--", color="white", lw=0.8, alpha=0.5)
    time_txt = ax2d.text(
        0.02, 0.88, "", transform=ax2d.transAxes, color="white", fontsize=9, va="top"
    )

    plt.tight_layout()
    return fig, dict(
        tx_trail=tx_trail,
        rx_trail=rx_trail,
        tx_dot=tx_dot,
        rx_dot=rx_dot,
        los_line=los_line,
        time_txt=time_txt,
    )


def _render_positions_to_file(
    fig, artists: dict, fd: dict, file_path: Path, writer, dpi: int
) -> None:
    """Render the positions-only animation."""
    n_f = SimConfig.ANIMATION_N_FRAMES
    trail = max(5, n_f // 15)

    with writer.saving(fig, str(file_path), dpi=dpi):
        for i in range(n_f):
            sl = slice(max(0, i - trail), i + 1)
            artists["tx_trail"].set_data(fd["tx_x_a"][sl], fd["tx_y_a"][sl])
            artists["rx_trail"].set_data(fd["rx_x_a"][sl], fd["rx_y_a"][sl])
            artists["tx_dot"].set_data([fd["tx_x_a"][i]], [fd["tx_y_a"][i]])
            artists["rx_dot"].set_data([fd["rx_x_a"][i]], [fd["rx_y_a"][i]])
            artists["los_line"].set_data(
                [fd["rx_x_a"][i], fd["tx_x_a"][i]], [fd["rx_y_a"][i], fd["tx_y_a"][i]]
            )
            artists["time_txt"].set_text(
                f"t = {fd['t_anim'][i]:.2f} s\nGT r = {fd['r_a'][i]:.0f} m"
            )
            writer.grab_frame()


def _build_spectrogram_frame_data(result: dict) -> dict:
    """Precompute position and STFT data for the spectrogram animation."""
    fd = _build_animation_frame_data(result)
    cfg = result["cfg"]
    zoom = max(500.0, 3.0 * np.nanmax(np.abs(result["delta_f"])))
    stft = estimate_doppler_stft(result["iq"], cfg.fs, freq_zoom=zoom)
    mask = stft["mask"]
    sxx_db = 10 * np.log10(stft["Sxx"][mask, :] + 1e-12)
    if sxx_db.size:
        vmin = float(np.nanpercentile(sxx_db, 20))
        vmax = float(np.nanpercentile(sxx_db, 99))
    else:
        vmin, vmax = -120.0, 0.0
    fd.update(
        stft_t=stft["t"],
        stft_freq=stft["freq_axis"][mask],
        stft_db=sxx_db,
        stft_peak=stft["peak_freq"],
        stft_vmin=vmin,
        stft_vmax=vmax,
    )
    return fd


def _build_spectrogram_animation_figure(result: dict, fd: dict) -> tuple:
    """Create a positions plus live spectrogram animation figure."""
    C = C_DARK
    cfg = result["cfg"]

    fig, (ax2d, ax_spec) = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[1.25, 1])
    fig.patch.set_facecolor("#1a1a2e")

    ax2d.set_facecolor("#16213e")
    for sp in ax2d.spines.values():
        sp.set_edgecolor("#555")
    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color="white")
    ax2d.set_ylabel("y [m]", color="white")
    ax2d.tick_params(colors="white")
    ax2d.grid(True, alpha=0.25, color="#555")
    ax2d.set_title("Vehicle positions (ground truth)", color="white", fontsize=10)
    ax2d.text(
        0.02,
        0.96,
        "Ground truth diagnostic: TX path/range are not RX-observable.",
        transform=ax2d.transAxes,
        color="white",
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(
            facecolor="#111",
            alpha=0.75,
            edgecolor="#666",
            boxstyle="round,pad=0.3",
        ),
    )
    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], alpha=0.15, lw=1)
    ax2d.plot([], [], "o", color=C["tx"], ms=7, label="Tx")
    ax2d.plot([], [], "o", color=C["rx"], ms=7, label="Rx")
    ax2d.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="lower right")

    ax_spec.set_facecolor("#16213e")
    for sp in ax_spec.spines.values():
        sp.set_edgecolor("#555")
    ax_spec.tick_params(colors="white")
    ax_spec.set_xlabel("time [s]", color="white")
    ax_spec.set_ylabel("frequency [Hz]", color="white")
    ax_spec.set_title("Live STFT spectrogram (RX observable)", color="white", fontsize=10)

    spec = fd["stft_db"]
    if spec.size:
        extent = [
            fd["stft_t"][0],
            fd["stft_t"][-1],
            fd["stft_freq"][0],
            fd["stft_freq"][-1],
        ]
        initial_spec = np.full_like(spec, np.nan)
    else:
        extent = [0.0, result["t"][-1], -1.0, 1.0]
        initial_spec = np.zeros((2, 2))
    im = ax_spec.imshow(
        initial_spec,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=fd["stft_vmin"],
        vmax=fd["stft_vmax"],
    )
    (peak_ln,) = ax_spec.plot([], [], color=C["stft"], lw=1.5, label="STFT peak")
    ax_spec.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="upper right")

    (tx_trail,) = ax2d.plot([], [], color=C["tx"], lw=1.5, alpha=0.5)
    (rx_trail,) = ax2d.plot([], [], color=C["rx"], lw=1.5, alpha=0.5)
    (tx_dot,) = ax2d.plot([], [], "o", color=C["tx"], ms=10, zorder=5)
    (rx_dot,) = ax2d.plot([], [], "o", color=C["rx"], ms=10, zorder=5)
    (los_line,) = ax2d.plot([], [], "--", color="white", lw=0.8, alpha=0.5)
    time_txt = ax2d.text(
        0.02, 0.88, "", transform=ax2d.transAxes, color="white", fontsize=9, va="top"
    )

    plt.tight_layout()
    return fig, dict(
        tx_trail=tx_trail,
        rx_trail=rx_trail,
        tx_dot=tx_dot,
        rx_dot=rx_dot,
        los_line=los_line,
        time_txt=time_txt,
        spec_im=im,
        peak_ln=peak_ln,
    )


def _render_spectrogram_to_file(
    fig, artists: dict, fd: dict, file_path: Path, writer, dpi: int
) -> None:
    """Render the positions plus live spectrogram animation."""
    n_f = SimConfig.ANIMATION_N_FRAMES
    trail = max(5, n_f // 15)

    with writer.saving(fig, str(file_path), dpi=dpi):
        for i in range(n_f):
            sl = slice(max(0, i - trail), i + 1)
            artists["tx_trail"].set_data(fd["tx_x_a"][sl], fd["tx_y_a"][sl])
            artists["rx_trail"].set_data(fd["rx_x_a"][sl], fd["rx_y_a"][sl])
            artists["tx_dot"].set_data([fd["tx_x_a"][i]], [fd["tx_y_a"][i]])
            artists["rx_dot"].set_data([fd["rx_x_a"][i]], [fd["rx_y_a"][i]])
            artists["los_line"].set_data(
                [fd["rx_x_a"][i], fd["tx_x_a"][i]], [fd["rx_y_a"][i], fd["tx_y_a"][i]]
            )
            artists["time_txt"].set_text(
                f"t = {fd['t_anim'][i]:.2f} s\nGT r = {fd['r_a'][i]:.0f} m"
            )

            if fd["stft_db"].size:
                col = np.searchsorted(fd["stft_t"], fd["t_anim"][i], side="right")
                visible = np.full_like(fd["stft_db"], np.nan)
                visible[:, :col] = fd["stft_db"][:, :col]
                artists["spec_im"].set_data(visible)
                artists["peak_ln"].set_data(fd["stft_t"][:col], fd["stft_peak"][:col])

            writer.grab_frame()


def _writer_for_format(fmt: str, result: dict):
    """Build an animation writer for mp4 or gif."""
    if fmt == "mp4":
        return FFMpegWriter(
            fps=SimConfig.ANIMATION_FPS,
            metadata=dict(title=result["scenario_name"]),
            extra_args=[
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
            ],
        )
    return PillowWriter(fps=SimConfig.ANIMATION_FPS)


def _render_animation_variant(
    result: dict,
    base_path: Path,
    label: str,
    fd: dict,
    fig,
    artists: dict,
    render_fn,
) -> None:
    """Render one animation variant to configured output formats."""
    fmt = SimConfig.RENDER_FORMATS.lower().strip()
    outputs = []
    if fmt in ("mp4", "both"):
        outputs.append(("mp4", SimConfig.ANIMATION_DPI_MP4))
    if fmt in ("gif", "both"):
        outputs.append(("gif", SimConfig.ANIMATION_DPI_GIF))

    for out_fmt, dpi in outputs:
        file_path = base_path.with_suffix(f".{out_fmt}")
        with timed(
            f"  render {label} {out_fmt.upper()}  "
            f"({SimConfig.ANIMATION_N_FRAMES} frames, dpi={dpi})"
        ):
            try:
                writer = _writer_for_format(out_fmt, result)
                render_fn(fig, artists, fd, file_path, writer, dpi)
                log.info(f"  Saved → {file_path}")
            except Exception as e:
                log.warning(f"  {label} {out_fmt.upper()} render failed: {e}")


def _deprecated_bearing_make_animation(result: dict, out_path: Path) -> None:
    """
    Render animation to MP4 and/or GIF depending on SimConfig.RENDER_FORMATS.
    The figure and frame data are built once and reused across both formats.
    """
    fmt = SimConfig.RENDER_FORMATS.lower().strip()
    render_mp4 = fmt in ("mp4", "both")
    render_gif = fmt in ("gif", "both")

    if not (render_mp4 or render_gif):
        log.info("  Animation skipped (RENDER_FORMATS='none')")
        return

    fps = SimConfig.ANIMATION_FPS

    with timed("  pre-compute frame data"):
        fd = _build_animation_frame_data(result)

    with timed("  build figure"):
        fig, artists = _build_animation_figure(result)

    if render_mp4:
        mp4_path = out_path.with_suffix(".mp4")
        with timed(
            f"  render MP4  ({SimConfig.ANIMATION_N_FRAMES} frames, "
            f"dpi={SimConfig.ANIMATION_DPI_MP4})"
        ):
            try:
                w = FFMpegWriter(
                    fps=fps,
                    metadata=dict(title=result["scenario_name"]),
                    extra_args=[
                        "-vcodec",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                    ],
                )
                _render_to_file(
                    fig, artists, fd, mp4_path, w, SimConfig.ANIMATION_DPI_MP4
                )
                log.info(f"  Saved → {mp4_path}")
            except Exception as e:
                log.warning(f"  MP4 render failed: {e}")

    if render_gif:
        gif_path = out_path.with_suffix(".gif")
        with timed(
            f"  render GIF  ({SimConfig.ANIMATION_N_FRAMES} frames, "
            f"dpi={SimConfig.ANIMATION_DPI_GIF})"
        ):
            try:
                w = PillowWriter(fps=fps)
                _render_to_file(
                    fig, artists, fd, gif_path, w, SimConfig.ANIMATION_DPI_GIF
                )
                log.info(f"  Saved → {gif_path}")
            except Exception as e:
                log.warning(f"  GIF render failed: {e}")

    plt.close(fig)


def make_animation(result: dict, out_path: Path) -> None:
    """
    Render configured positions and/or spectrogram animation variants.

    The deprecated bearing-panel renderer is kept only as a private
    compatibility reference and is not used by default outputs.
    """
    fmt = SimConfig.RENDER_FORMATS.lower().strip()
    if fmt not in ("mp4", "gif", "both"):
        log.info("  Animation skipped (RENDER_FORMATS='none')")
        return

    variants = getattr(SimConfig, "ANIMATION_VARIANTS", "positions").lower().strip()
    render_positions = variants in ("positions", "both")
    render_spectrogram = variants in ("spectrogram", "both")
    if not (render_positions or render_spectrogram):
        log.info(f"  Animation skipped (ANIMATION_VARIANTS={variants!r})")
        return

    if render_positions:
        with timed("  pre-compute positions frame data"):
            fd = _build_animation_frame_data(result)

        with timed("  build positions animation figure"):
            fig, artists = _build_positions_animation_figure(result)

        _render_animation_variant(
            result, out_path, "positions", fd, fig, artists, _render_positions_to_file
        )
        plt.close(fig)

    if render_spectrogram:
        with timed("  pre-compute spectrogram frame data"):
            fd = _build_spectrogram_frame_data(result)

        with timed("  build spectrogram animation figure"):
            fig, artists = _build_spectrogram_animation_figure(result, fd)

        spec_path = out_path.with_name(f"{out_path.name}_spectrogram")
        _render_animation_variant(
            result,
            spec_path,
            "spectrogram",
            fd,
            fig,
            artists,
            _render_spectrogram_to_file,
        )
        plt.close(fig)


# =============================================================================
# Save all outputs for one scenario
# =============================================================================


def save_all(result: dict, out_root: str = "plots") -> None:
    """Save all outputs for one scenario."""
    name = result["scenario_name"]
    slug = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    t_pd, df_pd = estimate_doppler_phase_diff(result["iq"], cfg.fs)
    est_pd = {"t": t_pd, "delta_f": df_pd}

    with timed("  static plots"):
        plot_trajectory_fig(result, out_dir / "trajectory.png")
        plot_rdot_rddot_fig(result, est_pd, out_dir / "rdot_rddot.png")
        plot_doppler_fig(result, est_pd, out_dir / "doppler.png")

    make_animation(result, out_dir / "animation")
    log.info(f"  All outputs → {out_dir}/")
