"""
plotting/animation.py — Vehicle position + polar bearing animation.

Two-panel layout (dark background):
  Left : 2-D top-down vehicle positions with position trail and LoS line.
  Right: Polar compass showing Tx bearing relative to Rx.
         - Radius fixed at 1.0 (only angle matters; range shown as text).
         - Ground truth needle (white) and estimated needle (orange).
         - 0° = ahead, 90° = left, 180° = behind.

Bearing estimation
------------------
The Rx knows its own velocity via GPS and measures ṙ_est from Δf_est.
From:   ṙ = vdx · dx / r   →   dx_est = ṙ_est · r / vdx
Then:   bearing_est = arctan2(dy, dx_est)

vdx = tx_vx − rx_vx: the receiver knows rx_vx but not tx_vx in a blind
scenario.  Here we use the true vdx (oracle case).  In practice, estimating
tx_vx is a key open problem addressed by the thesis.

Render strategy
---------------
We pre-compute all per-frame arrays before entering the writer loop so that
the render loop contains only artist.set_data() calls and grab_frame().
This avoids any numpy work per frame and is significantly faster than
FuncAnimation, which re-evaluates figure layout each frame.
"""

import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter, PillowWriter
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SimConfig
from dsp.estimators import phase_diff


# ---------------------------------------------------------------------------
# Dark-background colour palette
# ---------------------------------------------------------------------------

C = dict(
    gt       = '#f0f0f0',   # near-white  — ground truth
    smoothed = '#ff6b35',   # vivid orange — smoothed estimate
    tx       = '#ffaa00',   # amber        — transmitter
    rx       = '#4dabf7',   # sky blue     — receiver
)


# ---------------------------------------------------------------------------
# Pre-compute frame data (all numpy, done once before render loop)
# ---------------------------------------------------------------------------

def _build_frame_data(result: dict) -> dict:
    """
    Sub-sample ground-truth arrays and compute estimated bearing at each
    animation frame.  Everything returned is a plain numpy array so the
    render loop does zero computation.
    """
    cfg = result["cfg"]
    t   = result["t"]
    n_f = SimConfig.ANIMATION_N_FRAMES

    idx    = np.linspace(0, len(t) - 1, n_f, dtype=int)
    t_anim = t[idx]
    r_a    = result["r"][idx]

    # Estimated bearing via back-solved dx from ṙ_est
    pd_est      = phase_diff(result["iq"], cfg.fs)
    r_dot_est_a = np.interp(t_anim, pd_est["t"], -pd_est["smoothed"] * cfg.c / cfg.fc)

    vdx   = cfg.tx_vx - cfg.rx_vx
    dy    = cfg.tx_y - cfg.rx_y
    v_ref = vdx if abs(vdx) > 0.5 else cfg.rx_vx
    los_est_a = np.arctan2(dy, r_dot_est_a * r_a / v_ref)

    return dict(
        t_anim    = t_anim,
        tx_x_a    = result["tx_x"][idx],
        tx_y_a    = result["tx_y"][idx],
        rx_x_a    = result["rx_x"][idx],
        rx_y_a    = result["rx_y"][idx],
        los_a     = result["los_angle"][idx],
        los_est_a = los_est_a,
        r_a       = r_a,
    )


# ---------------------------------------------------------------------------
# Build the matplotlib figure and all mutable artists
# ---------------------------------------------------------------------------

def _build_figure(result: dict) -> tuple[plt.Figure, dict]:
    """
    Construct animation figure with all artists in their initial (empty) state.
    Returns (fig, artists_dict).
    """
    cfg = result["cfg"]

    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('#1a1a2e')
    gs     = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax2d   = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection='polar')

    # 2-D panel
    ax2d.set_facecolor('#16213e')
    for sp in ax2d.spines.values():
        sp.set_edgecolor('#555')
    all_x = np.concatenate([result["tx_x"], result["rx_x"]])
    pad_x = max((all_x.max() - all_x.min()) * 0.05, 5.0)
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(min(cfg.tx_y, cfg.rx_y) - 10, max(cfg.tx_y, cfg.rx_y) + 10)
    ax2d.set_xlabel("x [m]", color='white')
    ax2d.set_ylabel("y [m]", color='white')
    ax2d.tick_params(colors='white')
    ax2d.grid(True, alpha=0.25, color='#555')
    ax2d.set_title("Vehicle positions", color='white', fontsize=10)

    # Faint full-path ghost traces
    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], alpha=0.15, lw=1)
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], alpha=0.15, lw=1)

    # Legend via dummy artists (explicit loc avoids slow 'best' search)
    ax2d.plot([], [], 'o', color=C["tx"], ms=7, label='Tx')
    ax2d.plot([], [], 'o', color=C["rx"], ms=7, label='Rx')
    ax2d.legend(fontsize=8, facecolor='#222', labelcolor='white', loc='lower right')

    # Polar panel
    ax_pol.set_facecolor('#16213e')
    ax_pol.tick_params(colors='#aaa')
    ax_pol.spines['polar'].set_color('#555')
    ax_pol.set_theta_zero_location('E')   # 0° = ahead (+x direction)
    ax_pol.set_theta_direction(1)         # counter-clockwise
    ax_pol.set_ylim(0, 1.3)
    ax_pol.set_yticks([])                 # radial axis meaningless (unit radius)
    ax_pol.set_title("Tx bearing\n(0°=ahead)", color='white', fontsize=9, pad=12)
    for angle, lbl in [(0, 'Ahead'), (np.pi/2, 'Left'),
                       (np.pi, 'Behind'), (-np.pi/2, 'Right')]:
        ax_pol.text(angle, 1.22, lbl, ha='center', va='center', color='#888', fontsize=7)

    ax_pol.plot([], [], '-', color=C["gt"],       lw=2, label='True')
    ax_pol.plot([], [], '-', color=C["smoothed"], lw=2, label='Est. (smoothed)')
    ax_pol.legend(fontsize=7, facecolor='#222', labelcolor='white',
                  loc='upper left', bbox_to_anchor=(1.05, 1.12))

    # Mutable artists (mutated each frame)
    tx_trail,    = ax2d.plot([], [], color=C["tx"], lw=1.5, alpha=0.5)
    rx_trail,    = ax2d.plot([], [], color=C["rx"], lw=1.5, alpha=0.5)
    tx_dot,      = ax2d.plot([], [], 'o', color=C["tx"], ms=10, zorder=5)
    rx_dot,      = ax2d.plot([], [], 'o', color=C["rx"], ms=10, zorder=5)
    los_line,    = ax2d.plot([], [], '--', color='white', lw=0.8, alpha=0.5)
    time_txt     = ax2d.text(0.02, 0.95, '', transform=ax2d.transAxes,
                             color='white', fontsize=9, va='top')
    pol_gt_ln,   = ax_pol.plot([], [], '-', color=C["gt"],       lw=2.5)
    pol_est_ln,  = ax_pol.plot([], [], '-', color=C["smoothed"], lw=2.5, alpha=0.9)
    pol_gt_dot,  = ax_pol.plot([], [], 'o', color=C["gt"],       ms=8)
    pol_est_dot, = ax_pol.plot([], [], 'o', color=C["smoothed"], ms=8)
    range_txt    = ax_pol.text(np.pi * 1.25, 1.5, '', color='white',
                               fontsize=8, ha='center', va='center')

    plt.tight_layout()

    artists = dict(
        tx_trail=tx_trail, rx_trail=rx_trail, tx_dot=tx_dot, rx_dot=rx_dot,
        los_line=los_line, time_txt=time_txt,
        pol_gt_ln=pol_gt_ln, pol_est_ln=pol_est_ln,
        pol_gt_dot=pol_gt_dot, pol_est_dot=pol_est_dot,
        range_txt=range_txt,
    )
    return fig, artists


# ---------------------------------------------------------------------------
# Core render loop
# ---------------------------------------------------------------------------

def _render(fig: plt.Figure, artists: dict, fd: dict,
            file_path: Path, writer, dpi: int) -> None:
    """
    Mutate artists frame-by-frame and write to file.
    All data is pre-computed in fd; this loop does only set_data + grab_frame.
    """
    n_f   = SimConfig.ANIMATION_N_FRAMES
    trail = max(5, n_f // 15)

    with writer.saving(fig, str(file_path), dpi=dpi):
        for i in range(n_f):
            sl = slice(max(0, i - trail), i + 1)

            artists["tx_trail"].set_data(fd["tx_x_a"][sl], fd["tx_y_a"][sl])
            artists["rx_trail"].set_data(fd["rx_x_a"][sl], fd["rx_y_a"][sl])
            artists["tx_dot"].set_data([fd["tx_x_a"][i]], [fd["tx_y_a"][i]])
            artists["rx_dot"].set_data([fd["rx_x_a"][i]], [fd["rx_y_a"][i]])
            artists["los_line"].set_data(
                [fd["rx_x_a"][i], fd["tx_x_a"][i]],
                [fd["rx_y_a"][i], fd["tx_y_a"][i]])
            artists["time_txt"].set_text(f"t = {fd['t_anim'][i]:.2f} s")

            # Polar needles: [angle, angle] paired with [0, 1] draws a line
            # from the origin to the unit circle at the given bearing
            artists["pol_gt_ln"].set_data( [fd["los_a"][i],     fd["los_a"][i]],     [0, 1.0])
            artists["pol_est_ln"].set_data([fd["los_est_a"][i], fd["los_est_a"][i]], [0, 1.0])
            artists["pol_gt_dot"].set_data( [fd["los_a"][i]],     [1.0])
            artists["pol_est_dot"].set_data([fd["los_est_a"][i]], [1.0])
            artists["range_txt"].set_text(f"r = {fd['r_a'][i]:.0f} m")

            writer.grab_frame()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_animation(result: dict, out_path: Path) -> None:
    """
    Render animation to MP4 and/or GIF per SimConfig.RENDER_FORMATS.
    Figure and frame data are built once and reused across both formats.
    """
    fmt = SimConfig.RENDER_FORMATS.lower().strip()
    do_mp4 = fmt in ("mp4", "both")
    do_gif = fmt in ("gif", "both")

    if not (do_mp4 or do_gif):
        return

    import logging
    log = logging.getLogger(__name__)

    fd          = _build_frame_data(result)
    fig, artists = _build_figure(result)
    fps          = SimConfig.ANIMATION_FPS

    if do_mp4:
        mp4_path = out_path.with_suffix(".mp4")
        try:
            w = FFMpegWriter(
                fps=fps,
                metadata=dict(title=result["scenario_name"]),
                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                            '-preset', 'fast', '-crf', '23'],
            )
            _render(fig, artists, fd, mp4_path, w, SimConfig.ANIMATION_DPI_MP4)
            log.info(f"    Saved → {mp4_path}")
        except Exception as e:
            log.warning(f"    MP4 failed: {e}")

    if do_gif:
        gif_path = out_path.with_suffix(".gif")
        try:
            w = PillowWriter(fps=fps)
            _render(fig, artists, fd, gif_path, w, SimConfig.ANIMATION_DPI_GIF)
            log.info(f"    Saved → {gif_path}")
        except Exception as e:
            log.warning(f"    GIF failed: {e}")

    plt.close(fig)