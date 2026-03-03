"""
plotting/static.py — Static PNG figures for one simulation result.

Each figure follows the three-line convention:
  ── Ground truth   (near-black, thick dashed)   — simulation only
  ── Raw estimate   (light orange, thin)          — direct algorithm output
  ── Smoothed est.  (dark orange, medium)         — after Hann FIR

Figures produced
----------------
  trajectory.png   : 2-D top-down vehicle paths + range r(t)
  rdot_rddot.png   : ṙ(t) and r̈(t) with all three lines
  doppler.png      : phase-diff Δf(t) + STFT spectrogram with peak track
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SimConfig
from dsp.estimators import phase_diff, stft_peak
from dsp.kinematics import recover_kinematics


# ---------------------------------------------------------------------------
# Colour palette (light background)
# ---------------------------------------------------------------------------

C = dict(
    gt       = '#1a1a1a',   # near-black      — ground truth
    raw      = '#f4a261',   # light orange    — raw estimate
    smoothed = '#e05a00',   # dark orange     — smoothed estimate
    tx       = '#c07000',   # amber           — transmitter path
    rx       = '#1565c0',   # dark blue       — receiver path
    stft_gt  = '#cccccc',   # light grey      — GT overlay on dark spectrogram
    stft_pk  = '#1565c0',   # blue            — STFT peak track
)

# Line styles for the three-signal convention
LS = dict(
    gt       = dict(color=C["gt"],       lw=1.8, ls='--', zorder=3),
    raw      = dict(color=C["raw"],      lw=0.8, alpha=0.7, zorder=1),
    smoothed = dict(color=C["smoothed"], lw=1.4, alpha=0.95, zorder=2),
)


def _run_estimators(result: dict) -> tuple[dict, dict]:
    """
    Run phase-diff and kinematics on a result dict.
    Returns (pd_est, kin_est) so each plot function can use them.
    """
    cfg = result["cfg"]
    pd  = phase_diff(result["iq"], cfg.fs)
    kin = recover_kinematics(
        pd["t"], pd["raw"].copy(), pd["smoothed"].copy(), cfg.fc, cfg.c
    )
    return pd, kin


# ---------------------------------------------------------------------------
# Figure 1: Trajectory
# ---------------------------------------------------------------------------

def plot_trajectory(result: dict, out_path: Path) -> None:
    """
    Left : 2-D top-down positions of Tx (amber) and Rx (blue).
           y-axis fixed to ±15 m around lane midpoint so both parallel
           tracks are visible regardless of x-span.
    Right: Range r(t) — minimum is the CPA range.
    """
    cfg = result["cfg"]
    t   = result["t"]

    fig, (ax2d, ax_r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{result['scenario_name']} — Trajectory", fontweight='bold')

    ax2d.plot(result["tx_x"], result["tx_y"], color=C["tx"], lw=2, label="Tx")
    ax2d.plot(result["rx_x"], result["rx_y"], color=C["rx"], lw=2, label="Rx")

    for xs, ys, col in [(result["tx_x"], result["tx_y"], C["tx"]),
                        (result["rx_x"], result["rx_y"], C["rx"])]:
        mid = len(xs) // 2
        ax2d.annotate("", xy=(xs[mid+5], ys[mid+5]), xytext=(xs[mid], ys[mid]),
                      arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    ax2d.scatter([result["tx_x"][0], result["rx_x"][0]],
                 [result["tx_y"][0], result["rx_y"][0]],
                 color=[C["tx"], C["rx"]], s=60, zorder=5)

    ax2d.set_xlabel("x [m]"); ax2d.set_ylabel("y [m]")
    ax2d.set_title("2-D trajectory (top-down)")
    ax2d.legend(fontsize=8, loc='upper left')
    ax2d.grid(True, alpha=0.4)
    y_c = (cfg.tx_y + cfg.rx_y) / 2
    ax2d.set_ylim(y_c - 15, y_c + 15)

    ax_r.plot(t, result["r"], color='purple', lw=1.5)
    ax_r.set_xlabel("Time [s]"); ax_r.set_ylabel("Range r(t) [m]")
    ax_r.set_title("Range over time"); ax_r.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: ṙ and r̈ — three lines each
# ---------------------------------------------------------------------------

def plot_rdot_rddot(result: dict, out_path: Path) -> None:
    """
    Top   : Radial velocity ṙ(t).
             Ground truth | raw phase-diff-derived | smoothed.
             Zero-crossing at CPA is the key discriminating event.

    Bottom: Radial acceleration r̈(t).
             Same three lines.  Peak magnitude at CPA ≈ vrel² / r_min.
             Note: r̈ uses a separate (decimated) time axis, so sharex=False.
    """
    cfg      = result["cfg"]
    t        = result["t"]
    pd, kin  = _run_estimators(result)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"{result['scenario_name']} — Radial kinematics", fontweight='bold')

    # Legend proxy artists (created once, used on first subplot)
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], label='Ground truth', **LS["gt"]),
        mlines.Line2D([], [], label='Raw estimate',  **LS["raw"]),
        mlines.Line2D([], [], label='Smoothed estimate', **LS["smoothed"]),
    ]

    # ṙ subplot
    ax = axes[0]
    ax.plot(t,          result["r_dot"],      **LS["gt"])
    ax.plot(kin["t_dot"], kin["r_dot_raw"],   **LS["raw"])
    ax.plot(kin["t_dot"], kin["r_dot_smoothed"], **LS["smoothed"])
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ṙ [m/s]")
    ax.set_title("Radial velocity ṙ(t)  [zero-crossing = CPA]")
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.4)

    # r̈ subplot
    ax = axes[1]
    ax.plot(t,                   result["r_ddot"],      **LS["gt"])
    ax.plot(kin["t_ddot_raw"],   kin["r_ddot_raw"],     **LS["raw"])
    ax.plot(kin["t_ddot_smooth"], kin["r_ddot_smoothed"], **LS["smoothed"])
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel("Time [s]"); ax.set_ylabel("r̈ [m/s²]")
    ax.set_title(f"Radial acceleration r̈(t)  [r̈_CPA ≈ vrel² / r_min = "
                 f"{(cfg.tx_vx - cfg.rx_vx)**2 / max(cfg.tx_y - cfg.rx_y, 0.01):.0f} m/s²]")
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Doppler estimation — phase-diff + STFT
# ---------------------------------------------------------------------------

def plot_doppler(result: dict, out_path: Path) -> None:
    """
    Top   : Phase-differentiator Δf(t) — three lines.
    Bottom: STFT power spectrogram (auto-zoomed) with:
              - STFT raw peak track  (blue dashed)
              - STFT smoothed peak   (cyan solid)
              - Ground truth Δf      (light grey dashed)
    """
    cfg  = result["cfg"]
    t    = result["t"]
    gt   = result["delta_f"]

    pd   = phase_diff(result["iq"], cfg.fs)

    # Auto-zoom STFT display: 3× max true Doppler, minimum ±500 Hz
    zoom = max(500.0, 3.0 * np.abs(gt).max())
    stft = stft_peak(result["iq"], cfg.fs, freq_zoom=zoom)

    fig, (ax_pd, ax_stft) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{result['scenario_name']} — Doppler estimation", fontweight='bold')

    # --- Phase-diff panel ---
    ax_pd.plot(t,        gt,              **LS["gt"],       label='Ground truth')
    ax_pd.plot(pd["t"],  pd["raw"],       **LS["raw"],      label='Raw (phase-diff)')
    ax_pd.plot(pd["t"],  pd["smoothed"],  **LS["smoothed"], label='Smoothed')
    ax_pd.set_ylabel("Δf [Hz]")
    ax_pd.set_title("Doppler shift — phase differentiation")
    ax_pd.legend(fontsize=8, loc='upper right')
    ax_pd.grid(True, alpha=0.4)

    # --- STFT spectrogram panel ---
    Sxx_dB = 10 * np.log10(stft["Sxx"] + 1e-12)
    extent = [stft["t"][0], stft["t"][-1],
              stft["freq_axis"][0], stft["freq_axis"][-1]]
    ax_stft.imshow(Sxx_dB, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno', vmin=np.percentile(Sxx_dB, 20))
    ax_stft.plot(stft["t"], stft["raw"],      color='#4dabf7', lw=1.0, ls='--',
                 label='STFT raw peak')
    ax_stft.plot(stft["t"], stft["smoothed"], color='#74c0fc', lw=1.5,
                 label='STFT smoothed peak')
    ax_stft.plot(t,         gt,               color=C["stft_gt"], lw=1.5, ls='--',
                 label='Ground truth')
    ax_stft.set_ylabel("Δf [Hz]"); ax_stft.set_xlabel("Time [s]")
    ax_stft.set_title(f"STFT spectrogram (±{zoom:.0f} Hz zoom)")
    ax_stft.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: save all three figures
# ---------------------------------------------------------------------------

def save_static_plots(result: dict, out_dir: Path) -> None:
    plot_trajectory(result,  out_dir / "trajectory.png")
    plot_rdot_rddot(result,  out_dir / "rdot_rddot.png")
    plot_doppler(result,     out_dir / "doppler.png")