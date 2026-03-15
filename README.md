# Mobile RF Sim

Mobile RF Doppler simulator for two moving vehicles (Tx/Rx). It generates IQ,
runs estimators, recovers radial kinematics, and produces plots and animations.
The goal is a clear, end-to-end pipeline for Doppler/kinematics experiments
plus GNURadio integration.

## Goals
- Provide a reproducible Doppler/kinematics testbed with realistic motion geometry
- Keep DSP components pure numpy and testable in isolation
- Export GNURadio-compatible blocks for flowgraph use
- Produce clear visual outputs for validation and comparison

## What Is Included
- Geometry: range, radial velocity, radial acceleration, LOS angle
- IQ generation with phase integration and AWGN
- Estimators: phase differentiation and STFT peak tracking
- Kinematics recovery from Doppler (decimate + differentiate + smoothing)
- Plotting: static PNGs and an animation (gif/mp4)

## Quick Start
1. Edit `config.py` (scenario, RF, DSP, and rendering settings)
2. Run `python sim.py`
3. View outputs in `plots/<scenario>/`

## Outputs
- `trajectory.png`
- `rdot_rddot.png` (ground truth, raw, smoothed)
- `rdot_rddot_smooth_gt.png` (ground truth + smoothed)
- `doppler.png`
- `animation.gif` or `animation.mp4` (based on render settings)

## Repo Layout
```
config.py
geometry.py
iq_gen.py
dsp/
  estimators.py
  kinematics.py
  gnuradio/
    phase_diff_block.py
    kinematics_block.py
plotting/
  static.py
  animation.py
sim.py
```

## Notes
- Matplotlib uses the "Agg" backend for headless rendering
- Set `FFMPEG_PATH` in `config.py` if ffmpeg is not on your PATH
