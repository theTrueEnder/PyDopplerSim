"""
sim.py — Thin orchestrator.  Edit SimConfig in config.py, then run this.

All this file does:
  1. Define which scenarios to run.
  2. Call generate_iq() for each.
  3. Call save_all() to produce plots + animation.
  4. Log summary statistics.

All real logic lives in the imported modules.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — write files, no display window
import numpy as np

from config import SimConfig, scenario_colocated, scenario_same_direction, scenario_oncoming
from iq_gen import generate_iq
from plotting.static import save_static_plots
from plotting.animation import save_animation

# Apply ffmpeg override before any writers are instantiated
if SimConfig.FFMPEG_PATH:
    matplotlib.rcParams['animation.ffmpeg_path'] = SimConfig.FFMPEG_PATH


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@contextmanager
def timed(label: str):
    """Log wall-clock duration of the wrapped block."""
    t0 = time.perf_counter()
    log.info(f"START  {label}")
    try:
        yield
    finally:
        log.info(f"DONE   {label}  ({time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# Per-scenario output
# ---------------------------------------------------------------------------

def save_all(result: dict, out_root: str = "plots") -> None:
    name    = result["scenario_name"]
    slug    = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    with timed(f"  [{name}] static plots"):
        save_static_plots(result, out_dir)

    with timed(f"  [{name}] animation"):
        save_animation(result, out_dir / "animation")

    log.info(f"  [{name}] outputs → {out_dir}/")


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
        log.info("=" * 55)
        log.info(f"Scenario: {name}")

        with timed(f"  [{name}] IQ generation"):
            result = generate_iq(cfg)
        result["scenario_name"] = name

        # Summary statistics
        vdx   = cfg.tx_vx - cfg.rx_vx
        cpa_t = cfg.tx_x0 / max(abs(vdx), 1e-3) if abs(vdx) > 0.1 else float('inf')
        log.info(f"  Rx {cfg.rx_vx:.1f} m/s | Tx {cfg.tx_vx:.1f} m/s | Δv {vdx:.1f} m/s")
        log.info(f"  Samples: {len(result['iq']):,} @ {cfg.fs/1e6:.1f} MHz | "
                 f"Duration: {cfg.duration:.1f} s")
        log.info(f"  Δf range: [{result['delta_f'].min():.1f}, "
                 f"{result['delta_f'].max():.1f}] Hz")
        log.info(f"  r̈ peak (GT): {np.nanmax(np.abs(result['r_ddot'])):.1f} m/s²")
        log.info(f"  CPA range:  {result['r'].min():.1f} m  at t ≈ {cpa_t:.2f} s")

        save_all(result)

    log.info("=" * 55)
    log.info("All scenarios complete.")
