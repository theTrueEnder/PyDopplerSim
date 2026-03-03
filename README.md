# Mobile RF Sim
RF simulator that models Doppler behavior of simultaneous mobile reciever and transmitters

Structure:
```
doppler_sim/
│
├── config.py          # SimConfig dataclass + ScenarioConfig — single source of truth
│
├── geometry.py        # Pure geometry: r, r_dot, r_ddot, los_angle from trajectories
│
├── iq_gen.py          # IQ signal generation (phase integration + AWGN)
│
├── dsp/
│   ├── __init__.py
│   ├── estimators.py  # Raw estimators: phase_diff(), stft_peak() → (t, delta_f)
│   ├── kinematics.py  # Kinematic recovery: delta_f → r_dot, r_ddot (decimate + diff)
│   └── gnuradio/
│       ├── __init__.py
│       ├── phase_diff_block.py   # gr.sync_block wrapping estimators.phase_diff
│       └── kinematics_block.py   # gr.decim_block wrapping kinematics.recover
│
├── plotting/
│   ├── __init__.py
│   ├── static.py      # trajectory, rdot_rddot, doppler figures (3-line versions)
│   └── animation.py   # vehicle animation + polar bearing plot
│
└── sim.py             # Thin orchestrator: run scenarios, call everything above
```