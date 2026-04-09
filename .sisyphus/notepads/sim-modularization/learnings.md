# PyDopplerSim Modularization - Learnings

## Project Summary
Refactored a single-file Doppler radar simulator into a modular Python package with 8 new modules.

## Key Patterns Discovered
1. **Pipeline Architecture**: Geometry → IQ Generation → Estimation → Kinematics → Plotting
2. **Config-driven scenarios**: Use dataclasses with consistent field names
3. **WAV format for IQ**: 16-bit interleaved I/Q is standard for SDR replay

## Technical Decisions
- **WAV save/load**: Uses Python's `wave` module with companion `.iqmeta` file for metadata
- **Waypoint interpolation**: Linear by default, no scipy dependency required
- **Backward compatibility**: Original `sim.py` re-exports from new modules with deprecation warning

## Files Created
1. `config.py` - SimConfig, ScenarioConfig, scenario builders
2. `geometry.py` - compute_geometry() function
3. `paths.py` - WaypointPath class with preset builders
4. `iq_gen.py` - generate_iq() + WAV save/load
5. `estimation.py` - Phase-diff and STFT estimators
6. `kinematics.py` - recover_kinematics() function
7. `plotting.py` - All plotting + animation + TX derivation
8. `main.py` - CLI orchestrator

## Bugs Fixed During Implementation
1. Fixed `wave.getnframesize()` → `wave.getsampwidth()` in load_iq_wav
2. Fixed shape mismatch in plot_tx_derivation_fig by interpolating r_dot_est to match result length

## Usage
```bash
# Run scenarios
python main.py --scenario same-direction

# Save IQ to WAV
python main.py --scenario oncoming --save-wav output/

# Replay from WAV
python main.py --load-wav data.wav --scenario oncoming
```

## Notes
- Animation rendering requires ffmpeg (optional - falls back gracefully if not available)
- All verification tests passed: imports, scenarios, WAV round-trip, waypoints, TX derivation visualization