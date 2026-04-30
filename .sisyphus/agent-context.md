# PyDopplerSim - Agent Context

> Quick reference for AI agents working with this codebase.

## Project Purpose

Simulates Doppler radar signals with mobile TX and RX. Generates realistic IQ baseband samples with:
- Time-varying Doppler shift based on relative motion
- AWGN noise
- Phase integration for accurate frequency tracking

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | `SimConfig` (global), `ScenarioConfig` (per-run), scenario builders |
| `geometry.py` | `compute_geometry()` - analytical r, ṙ, r̈, Δf |
| `iq_gen.py` | `generate_iq()` - creates IQ samples, `save/load_iq_wav()` |
| `estimation.py` | `estimate_doppler_phase_diff()`, `estimate_doppler_stft()` |
| `kinematics.py` | `recover_kinematics()` - inverse: Δf → ṙ → r̈ |
| `paths.py` | `WaypointPath` class for custom trajectories |
| `plotting.py` | All visualization functions |
| `main.py` | CLI entry point |

## Data Flow

```
ScenarioConfig → geometry.compute_geometry() → iq_gen.generate_iq()
                                                      ↓
                                              IQ samples (complex)
                                                      ↓
                              estimation.estimate_doppler_phase_diff()
                                                      ↓
                                              delta_f estimate
                                                      ↓
                              kinematics.recover_kinematics()
                                                      ↓
                                              r_dot, r_ddot estimates
                                                      ↓
                                              plotting.save_all()
```

## Important Conventions

### Sign Conventions
- **r_dot**: positive = moving apart, negative = approaching
- **delta_f**: positive = blue shift (approaching), negative = red shift (receding)
- **Coordinate system**: +x = forward, +y = left of center

### Output Dict Keys (from generate_iq)
```
t, iq, iq_clean, cfg,
r, r_dot, r_ddot, delta_f,
tx_x, tx_y, rx_x, rx_y, los_angle
```

### Estimation Edge Handling
- Phase-diff: NaN at edges (first/last `PD_SMOOTH_WINDOW//2` samples)
- Kinematics: NaN propagates through decimation

## Common Operations

```python
# Run a scenario
from config import scenario_oncoming
from iq_gen import generate_iq

cfg = scenario_oncoming()
result = generate_iq(cfg)

# Access results
print(result['r'])           # Range array
print(result['r_dot'])       # Radial velocity
print(result['delta_f'])     # Doppler shift
print(result['iq'])          # Complex IQ samples (noisy)
print(result['iq_clean'])    # Clean IQ samples

# Custom path
from paths import WaypointPath
path = WaypointPath([(0, 0, 0), (5, 100, 10), (10, 200, 0)])
result = generate_iq(cfg, path_provider=path)
```

## CLI Usage

```bash
# Run all scenarios
python main.py

# Specific scenario
python main.py --scenario oncoming

# Save/load IQ
python main.py --scenario same-direction --save-wav output/
python main.py --load-wav data.wav --scenario oncoming
```

## Test Running

```bash
pytest tests/ -v                    # All tests
pytest tests/ -m "not slow"         # Skip slow tests
pytest tests/ --cov                 # With coverage
```

## Configuration

Edit `SimConfig` class in `config.py`:
- `FC` = 5.8e9 (carrier frequency Hz)
- `FS` = 1e6 (sample rate Hz)
- `SNR_DB` = 20 (dB)
- `RX_VX` = 30 (m/s)

## Gotchas

1. **r_dot near zero**: At CPA, r_dot approaches zero, causing numerical instability in bearing estimation
2. **Phase integration**: Uses oversampling (×8) for accuracy near CPA
3. **NaN at edges**: Estimation produces NaN where the smoother window extends beyond data
4. **ffmpeg**: Animation requires ffmpeg in PATH or set `FFMPEG_PATH` in config

## Dependencies

- numpy
- matplotlib
- pytest (dev)
- pytest-cov (dev)
- hypothesis (dev)