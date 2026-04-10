#!/usr/bin/env python
"""
Main orchestration module for PyDopplerSim.

Provides CLI interface for running scenarios with optional save/load of IQ data.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Import our modules
from config import (
    SimConfig,
    ScenarioConfig,
    scenario_colocated,
    scenario_same_direction,
    scenario_oncoming,
)
from iq_gen import generate_iq, save_iq_wav, load_iq_wav
from plotting import save_all

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


SCENARIOS = {
    "colocated": scenario_colocated,
    "same-direction": scenario_same_direction,
    "oncoming": scenario_oncoming,
}


def run_scenario(
    name: str,
    cfg: ScenarioConfig,
    output_dir: str,
    save_wav: Path = None,
    iq_data: tuple = None,
) -> dict:
    """Run a single scenario."""
    log.info(f"{'=' * 50}")
    log.info(f"Scenario: {name}")

    if iq_data is not None:
        # Use loaded IQ data
        iq, t, metadata = iq_data
        result = {
            "t": t,
            "iq": iq,
            "iq_clean": iq,  # No clean version when loaded
            "cfg": cfg,
        }
        # Regenerate geometry for the loaded data
        from geometry import compute_geometry

        geo = compute_geometry(cfg, t)
        result.update(geo)
        log.info(f"  Loaded IQ: {len(iq):,} samples")
    else:
        # Generate IQ
        log.info("  IQ generation")
        result = generate_iq(cfg)
        result["scenario_name"] = name

        vdx = cfg.tx_vx - cfg.rx_vx
        cpa_t = cfg.tx_x0 / max(abs(vdx), 1e-3) if abs(vdx) > 0.1 else float("inf")
        log.info(
            f"  Rx: {cfg.rx_vx:.1f} m/s   Tx: {cfg.tx_vx:.1f} m/s   Δv: {vdx:.1f} m/s"
        )
        log.info(f"  Samples: {len(result['iq']):,}   Duration: {cfg.duration:.1f} s")
        log.info(
            f"  Δf range:    [{result['delta_f'].min():.2f}, {result['delta_f'].max():.2f}] Hz"
        )
        log.info(f"  r̈ peak (GT): {np.nanmax(np.abs(result['r_ddot'])):.1f} m/s²")
        log.info(f"  Range CPA:   {result['r'].min():.1f} m  (at t ≈ {cpa_t:.2f} s)")

    result["scenario_name"] = name

    # Save WAV if requested
    if save_wav is not None:
        wav_path = save_wav.parent / f"{save_wav.stem}_{name.replace('-', '_')}.wav"
        log.info(f"  Saving IQ to {wav_path}")
        save_iq_wav(result["iq"], result["t"], wav_path, cfg)

    # Save plots and animation
    save_all(result, output_dir)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PyDopplerSim - Baseband IQ Doppler simulator"
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="plots",
        help="Output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--save-wav",
        type=Path,
        help="Save IQ to WAV file (specify directory or full path)",
    )
    parser.add_argument(
        "--load-wav", type=Path, help="Load IQ from WAV file instead of generating"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--render-formats",
        choices=["mp4", "gif", "both", "none"],
        help="Override render formats from config",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Override render formats if specified
    if args.render_formats:
        SimConfig.RENDER_FORMATS = args.render_formats.upper()

    # Load IQ if requested
    iq_data = None
    if args.load_wav:
        log.info(f"Loading IQ from {args.load_wav}")
        iq_data = load_iq_wav(args.load_wav)

    # Determine scenarios to run
    if args.scenario == "all":
        scenarios = list(SCENARIOS.items())
    else:
        scenarios = [(args.scenario, SCENARIOS[args.scenario])]

    # Run scenarios
    for name, scenario_fn in scenarios:
        cfg = scenario_fn()
        run_scenario(name, cfg, args.output_dir, args.save_wav, iq_data)

    log.info("Done.")


if __name__ == "__main__":
    main()
