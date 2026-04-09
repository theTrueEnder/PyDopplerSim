# PyDopplerSim Modularization Plan

## TL;DR

> **Quick Summary**: Refactor single-file simulator into modular structure (config, geometry, iq_gen, estimation, kinematics, plotting, paths, main) with WAV-based IQ save/load, waypoint-based custom paths, and TX path derivation visualization.

> **Deliverables**:
> - 7 new Python modules + 1 main orchestrator
> - WAV file save/load for IQ data
> - WaypointPath class for custom trajectories
> - TX estimated trajectory visualization (static + animated)

> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: config → geometry + paths → iq_gen → estimation → kinematics → plotting

---

## Context

### Original Request
Upgrade PyDopplerSim with:
1. Separate functionality into modules (Geometry → IQ → Kinematics → Plotting)
2. IQ interface: save IQ to 16-bit WAV, load for replay
3. Custom paths: waypoint-based trajectory definitions
4. TX path derivation: visualize estimated vs actual TX trajectory

### Interview Summary
**Key Decisions**:
- IQ Format: 16-bit WAV (SDR standard, interleaved I/Q)
- Path Format: Waypoint list with interpolation
- Config Format: Python module (maintain current style)
- TX Derivation: Both static overlaid plot + animated estimated position

### Gap Analysis (Self-Completed)
**Identified Issues to Handle**:
- Waypoint interpolation method (linear default, spline option)
- Edge cases: vdx ≈ 0 (colocated), insufficient waypoints
- Backward compatibility for existing scenario functions
- WAV sample rate handling

---

## Work Objectives

### Core Objective
Transform the monolithic `sim.py` into a modular package while adding new features:
- IQ save/replay via WAV files
- Custom trajectory paths via waypoints
- TX path estimation visualization

### Concrete Deliverables
- `config.py` - All configuration classes and scenario builders
- `geometry.py` - Compute r(t), ṙ(t), r̈(t), bearing from positions
- `iq_gen.py` - IQ generation + WAV save/load functions
- `estimation.py` - Phase-diff and STFT Doppler estimators
- `kinematics.py` - Recover ṙ and r̈ from Doppler estimates
- `paths.py` - WaypointPath class for custom trajectories
- `plotting.py` - All static plots and animation
- `main.py` - Orchestration + CLI

### Definition of Done
- [x] `python -c "from sim import generate_iq"` works (backward compat)
- [x] All modules importable independently
- [x] `save_iq_wav()` produces valid 16-bit WAV
- [x] `load_iq_wav()` restores IQ data with metadata
- [x] WaypointPath generates valid trajectories
- [x] Static plot shows estimated vs actual TX path
- [x] Animation shows estimated TX position as ghost marker

### Must Have
- Backward compatibility: existing code using `sim.py` continues to work
- All current scenarios (colocated, same-direction, oncoming) function identically
- WAV files are standard SDR-compatible format

### Must NOT Have
- No breaking changes to public API
- No changes to physics/Doppler equations
- No new external dependencies beyond scipy (for spline interpolation)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO (no test framework in project)
- **Automated tests**: None (manual verification via running sim)
- **Agent-Executed QA**: Every task includes verification scenarios that run the code and verify outputs

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - can start immediately):
├── T1: Create config.py (extract SimConfig, ScenarioConfig, scenarios)
├── T2: Create geometry.py (extract compute_geometry)
├── T3: Create paths.py (WaypointPath class for custom trajectories)
├── T4: Create iq_gen.py skeleton (import geometry, add save/load stubs)
└── T5: Create estimation.py skeleton (stub functions)

Wave 2 (Core modules - after T1, T2):
├── T6: Implement iq_gen.py full (generate_iq + WAV save/load)
├── T7: Implement estimation.py full (phase_diff + STFT)
├── T8: Implement kinematics.py (extract recover_kinematics)
└── T9: Update iq_gen to use new geometry module

Wave 3 (Plotting + Integration - after T2, T6, T8):
├── T10: Create plotting.py (extract all plotting functions)
├── T11: Add TX path derivation to plotting (static overlay)
├── T12: Add TX estimation to animation (ghost marker)
├── T13: Create main.py (orchestration + CLI)
└── T14: Add backward-compat import shim to sim.py

Wave 4 (Final - after Wave 3):
├── T15: Verify all scenarios run correctly
├── T16: Test WAV save/load round-trip
├── T17: Test waypoint paths work
└── T18: Verify output quality (plots + animation)
```

---

## TODOs

- [x] 1. Create config.py — Extract SimConfig, ScenarioConfig, scenario builders

  **What to do**:
  - Copy SimConfig class from sim.py lines 60-126
  - Copy ScenarioConfig dataclass and _base() function
  - Copy scenario_colocated(), scenario_same_direction(), scenario_oncoming()
  - Keep ffmpeg path handling
  - Create __all__ exports

  **Must NOT do**:
  - Change any default values (maintain backward compatibility)
  - Add new config parameters not in original

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple extraction, minimal logic changes
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2-T5)
  - **Blocks**: T9, T14 (depends on config)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `sim.py:60-126` - Original SimConfig class
  - `sim.py:133-179` - ScenarioConfig and builders

  **Acceptance Criteria**:
  - [ ] config.py file created with all config classes
  - [ ] `python -c "from config import SimConfig, ScenarioConfig"` works
  - [ ] All 3 scenario functions return identical ScenarioConfig to original

- [x] 2. Create geometry.py — Extract compute_geometry function

  **What to do**:
  - Copy compute_geometry() function from sim.py
  - Keep all physics/math exactly the same
  - Add type hints
  - Create __all__ exports

  **Must NOT do**:
  - Change Doppler equation or geometry calculations
  - Add any dependencies beyond numpy

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct extraction, math is well-tested
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3-T5)
  - **Blocks**: T4 (iq_gen uses geometry)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `sim.py:182-231` - Original compute_geometry function

  **Acceptance Criteria**:
  - [ ] geometry.py created with compute_geometry
  - [ ] Output matches original for same input (test with scenario config)

- [x] 3. Create paths.py — WaypointPath class for custom trajectories

  **What to do**:
  - Create WaypointPath class with:
    - `__init__(waypoints: list[tuple[x, y, t]])` - waypoints as (x, y, time)
    - `interpolate(t: np.ndarray)` - returns (x, y) at requested times
    - Support linear interpolation (default)
    - Optional: cubic spline via scipy (if available)
  - Add preset builders:
    - `parallel_path(x0, y, vx, duration)` - current behavior
    - `angled_path(x0, y0, angle_deg, speed, duration)` - angled approach
    - `curved_path(waypoints)` - from waypoint list
  - Create __all__ exports

  **Must NOT do**:
  - Break existing parallel path behavior (used in scenarios)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: New feature requiring design decisions
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T4, T5)
  - **Blocks**: T6 (iq_gen uses paths)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `sim.py:204-209` - Current parallel path logic (tx_x = tx_x0 + tx_vx * t)

  **Acceptance Criteria**:
  - [ ] WaypointPath class created
  - [ ] `linear_interpolate` produces same results as parallel path for straight lines
  - [ ] `python -c "from paths import WaypointPath"` works

- [x] 4. Create iq_gen.py skeleton — Stub functions ready for implementation

  **What to do**:
  - Create placeholder for generate_iq (will implement in T6)
  - Add stub for save_iq_wav(iq, t, filepath, cfg)
  - Add stub for load_iq_wav(filepath) -> (iq, t, metadata)
  - Import geometry module (will fail until T2 done - expected)

  **Must NOT do**:
  - Implement actual functions yet (T6)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Skeleton creation, no complex logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T3, T5)
  - **Blocks**: T6 (will implement full functions)
  - **Blocked By**: None (can start immediately)

  **References**:
  - None yet - will reference sim.py lines 234-282 when implementing T6

  **Acceptance Criteria**:
  - [ ] iq_gen.py file created with function stubs
  - [ ] `from iq_gen import generate_iq, save_iq_wav, load_iq_wav` imports work

- [x] 5. Create estimation.py skeleton — Stub Doppler estimators

  **What to do**:
  - Create placeholder for estimate_doppler_phase_diff
  - Create placeholder for estimate_doppler_stft
  - Copy _hann_smooth helper function
  - Add stub for _phase_diff_smoother = SimConfig.PD_SMOOTH_WINDOW

  **Must NOT do**:
  - Implement actual estimators yet (T7)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Skeleton creation
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T4)
  - **Blocks**: T7 (implement full functions)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `sim.py:289-295` - _hann_smooth
  - `sim.py:298-319` - estimate_doppler_phase_diff
  - `sim.py:322-359` - estimate_doppler_stft

  **Acceptance Criteria**:
  - [ ] estimation.py created with stub functions
  - [ ] Imports work: `from estimation import estimate_doppler_phase_diff`

- [x] 6. Implement iq_gen.py full — Generate IQ + WAV save/load

  **What to do**:
  - Implement generate_iq(cfg) using geometry module
  - Implement save_iq_wav(iq, t, filepath, cfg):
    - Convert complex IQ to interleaved float32
    - Write 16-bit WAV via scipy.io.wavfile or wave module
    - Store metadata (fc, fs, duration) in WAV info chunk
  - Implement load_iq_wav(filepath) -> (iq, t, metadata):
    - Read 16-bit WAV
    - De-interleave to complex IQ
    - Extract metadata
  - Keep AWGN model exactly as-is
  - Keep phase integration oversampling exactly as-is

  **Must NOT do**:
  - Change IQ generation algorithm
  - Change noise model or SNR calculations

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Need careful WAV format handling
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7, T8, T9)
  - **Blocks**: T15, T16 (testing)
  - **Blocked By**: T2 (geometry), T3 (paths), T4 (skeleton)

  **References**:
  - `sim.py:234-282` - Original generate_iq implementation
  - scipy.io.wavfile.write - WAV writing

  **Acceptance Criteria**:
  - [ ] generate_iq produces identical output to original
  - [ ] save_iq_wav creates valid 16-bit WAV file
  - [ ] load_iq_wav restores data correctly
  - [ ] Round-trip error < 1e-6

- [x] 7. Implement estimation.py full — Phase-diff and STFT estimators

  **What to do**:
  - Implement estimate_doppler_phase_diff(iq, fs, smooth_window)
  - Implement estimate_doppler_stft(iq, fs, window_dur, hop_dur, freq_zoom)
  - Copy _hann_smooth from original
  - Use config.PD_SMOOTH_WINDOW as default
  - Keep STFT implementation exactly as-is (4x zero-padding, etc.)

  **Must NOT do**:
  - Change algorithm or parameters
  - Change window functions or FFT settings

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Signal processing accuracy critical
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T8, T9)
  - **Blocks**: None
  - **Blocked By**: T5 (skeleton), T2 (geometry needed for testing)

  **References**:
  - `sim.py:289-359` - Original estimator implementations

  **Acceptance Criteria**:
  - [ ] Output matches original for same input
  - [ ] STFT spectrogram produces identical results

- [x] 8. Implement kinematics.py — Extract recover_kinematics

  **What to do**:
  - Copy recover_kinematics function from sim.py
  - Copy _hann_smooth (or import from estimation)
  - Keep decim_hz = 100 Hz and smooth_window = 21
  - Keep edge handling (NaN at trim regions)
  - Add type hints

  **Must NOT do**:
  - Change decimation or differentiation algorithm
  - Change smoothing parameters

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Math accuracy critical
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T7, T9)
  - **Blocks**: T15 (testing)
  - **Blocked By**: T2 (geometry), T5 (skeleton)

  **References**:
  - `sim.py:362-432` - Original recover_kinematics

  **Acceptance Criteria**:
  - [ ] Output matches original for same input
  - [ ] NaN handling at edges preserved

- [x] 9. Update iq_gen.py — Integrate with paths and config

  **What to do**:
  - Update generate_iq to accept custom paths via WaypointPath
  - If cfg has waypoints, use WaypointPath instead of parallel motion
  - Backward compat: if no waypoints, use existing parallel logic
  - This is the integration point for custom paths

  **Must NOT do**:
  - Break existing parallel path behavior

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T6, T7, T8)
  - **Blocks**: T17 (testing waypoints)
  - **Blocked By**: T3 (paths), T6 (implementation)

  **References**:
  - `paths.py` - WaypointPath class

  **Acceptance Criteria**:
  - [ ] generate_iq works with both old-style config and WaypointPath
  - [ ] Parallel paths produce identical output to original

- [x] 10. Create plotting.py — Extract all plotting functions

  **What to do**:
  - Copy all plotting functions:
    - plot_trajectory_fig
    - plot_rdot_rddot_fig
    - plot_doppler_fig
    - _build_animation_frame_data
    - _build_animation_figure
    - _render_to_file
    - make_animation
    - save_all
  - Copy color palettes (C_STATIC, C_DARK)
  - Import from config, estimation, kinematics as needed
  - Keep matplotlib non-interactive backend

  **Must NOT do**:
  - Change plotting logic or aesthetics
  - Change animation framerate or DPI defaults

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct extraction, no logic changes
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11-T14)
  - **Blocks**: T15 (testing)
  - **Blocked By**: T7 (estimation), T8 (kinematics), T2 (geometry)

  **References**:
  - `sim.py:435-820` - All plotting code

  **Acceptance Criteria**:
  - [ ] plotting.py created with all functions
  - [ ] Imports work: `from plotting import plot_trajectory_fig`
  - [ ] Output identical to original (visual comparison)

- [x] 11. Add TX path derivation — Static trajectory overlay plot

  **What to do**:
  - Create new function: plot_tx_derivation_fig(result, est, out_path)
  - Plot true TX trajectory (tx_x, tx_y)
  - Plot estimated TX trajectory (derived from bearing estimation)
  - Compute estimated position:
    - From Doppler: ṙ_est = -Δf * c / fc
    - From bearing: los_est = arctan2(dy, dx_est)
    - Solve for (tx_x_est, tx_y_est): use known rx position + estimated range + bearing
  - Color code: true path (amber), estimated path (ghost/gray)
  - Add legend and time labels
  - Add to save_all() in plotting.py

  **Must NOT do**:
  - Change estimation algorithm (just visualize it)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Need careful geometry for path reconstruction
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T12-T14)
  - **Blocks**: T15 (testing)
  - **Blocked By**: T7 (estimation), T10 (plotting)

  **References**:
  - `sim.py:605-651` - Bearing estimation in _build_animation_frame_data

  **Acceptance Criteria**:
  - [ ] New plot shows both true and estimated TX trajectories
  - [ ] Divergence visible at edges (where estimation is noisy)
  - [ ] Plot saved to output directory

- [x] 12. Add TX estimation to animation — Ghost marker for estimated position

  **What to do**:
  - Update _build_animation_frame_data to include estimated TX position
  - Compute tx_x_est, tx_y_est same as static plot
  - Add new artists to animation:
    - tx_est_dot (hollow/ghost marker)
    - tx_est_trail (fainter trail)
  - Update _render_to_file to animate both true and estimated

  **Must NOT do**:
  - Change animation framerate or visual style

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Animation updates
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T11, T13, T14)
  - **Blocks**: T15 (testing)
  - **Blocked By**: T10 (plotting), T11 (static derivation)

  **References**:
  - `sim.py:715-728` - Animation artists setup

  **Acceptance Criteria**:
  - [ ] Animation shows ghost marker for estimated TX position
  - [ ] Trail for estimated position visible
  - [ ] Both MP4 and GIF include new markers

- [x] 13. Create main.py — Orchestration and CLI

  **What to do**:
  - Create main() function:
    - Parse args: --scenario (colocated/same/oncoming), --output-dir, --save-wav, --load-wav
    - If --load-wav: load IQ and skip generation
    - If --save-wav: save IQ after generation
    - Run scenarios, generate plots + animation
  - Add argparser with:
    - --scenario: which scenario(s) to run
    - --output-dir: override output directory
    - --save-wav: save IQ to WAV file
    - --load-wav: load IQ from WAV instead of generating
    - --waypoints: JSON file with custom waypoints (future)
  - Import all modules and wire together

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: CLI design decisions
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10-T12, T14)
  - **Blocks**: T15 (testing)
  - **Blocked By**: T6 (iq_gen), T7 (estimation), T10 (plotting)

  **References**:
  - `sim.py:845-876` - Original main block

  **Acceptance Criteria**:
  - [ ] `python main.py --scenario same-direction` runs correctly
  - [ ] `python main.py --save-wav` saves WAV file
  - [ ] `python main.py --load-wav data.wav` loads and replays

- [x] 14. Add backward-compat import shim to sim.py

  **What to do**:
  - Update sim.py to re-export all public APIs from new modules
  - Keep SimConfig in sim.py for backward compat (but mark deprecated)
  - Add: `from config import *`, `from geometry import *`, etc.
  - Add deprecation warning for direct sim.py imports
  - Original code should work: `from sim import generate_iq`

  **Must NOT do**:
  - Remove any functionality
  - Break existing imports

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple import shim
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10-T13)
  - **Blocks**: T15, F1 (testing)
  - **Blocked By**: All previous tasks

  **References**:
  - None - this is the backward-compat layer

  **Acceptance Criteria**:
  - [ ] `from sim import generate_iq` works
  - [ ] `from sim import SimConfig` works (with deprecation warning)
  - [ ] `python sim.py` runs identical to before

- [x] 15. Verify all scenarios run correctly

  **What to do**:
  - Run: python main.py --scenario colocated
  - Run: python main.py --scenario same-direction
  - Run: python main.py --scenario oncoming
  - Verify plots generated in correct directories
  - Verify outputs match original (use original as baseline)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration testing
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (sequential test runs)
  - **Parallel Group**: Wave 4
  - **Blocks**: F2 (final verification)
  - **Blocked By**: All Wave 1-3 tasks

  **Acceptance Criteria**:
  - [ ] All 3 scenarios complete without error
  - [ ] Plot files created in expected locations
  - [ ] Output visually matches original

- [x] 16. Test WAV save/load round-trip

  **What to do**:
  - Run: python main.py --scenario same-direction --save-wav output/iq_test.wav
  - Verify WAV file created with correct size
  - Run: python main.py --load-wav output/iq_test.wav
  - Compare loaded IQ to original (should be within 1e-6)
  - Verify metadata (fc, fs) stored correctly

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Round-trip verification
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T17, T18)
  - **Blocks**: F3
  - **Blocked By**: T6 (iq_gen full)

  **Acceptance Criteria**:
  - [ ] WAV file created and readable
  - [ ] Round-trip error < 1e-6
  - [ ] Metadata correctly preserved

- [x] 17. Test waypoint paths work

  **What to do**:
  - Create test script with custom waypoints:
    ```python
    from paths import WaypointPath
    import numpy as np
    waypoints = [(0, 0, 0), (100, 10, 2), (200, 5, 4)]  # (x, y, t)
    path = WaypointPath(waypoints)
    t_test = np.array([0, 1, 2, 3, 4])
    x, y = path.interpolate(t_test)
    ```
  - Run with custom path via main.py
  - Verify trajectory plot shows curved/angled path

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: New feature testing
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T16, T18)
  - **Blocks**: F4
  - **Blocked By**: T3 (paths), T9 (integration), T13 (main)

  **Acceptance Criteria**:
  - [ ] WaypointPath produces valid interpolated path
  - [ ] Custom path generates correct plots
  - [ ] No errors when running with waypoints

- [x] 18. Verify output quality (plots + animation)

  **What to do**:
  - Check static trajectory plots: correct colors, labels, ranges
  - Check TX derivation plot: shows both paths clearly
  - Check animation: smooth playback, correct markers
  - Verify no matplotlib warnings or errors

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Visual QA
  - **Skills**: []
  - **Skills Evaluated but Omitted**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with T16, T17)
  - **Blocks**: F5
  - **Blocked By**: T11, T12 (TX derivation)

  **Acceptance Criteria**:
  - [ ] All plots render without errors
  - [ ] Animation plays smoothly
  - [ ] Visual quality matches original

---

## Final Verification Wave

- [x] F1. **Import Verification** — `unspecified-high`
  Run `python -c "from sim import *"` and verify no errors. Test `from sim import generate_iq, SimConfig`. Verify backward-compat shim works.
  Output: `Imports [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Scenario Test** — `unspecified-high`
  Run all 3 scenarios via `python main.py`. Verify plots generate in correct directories. Compare output to baseline.
  Output: `Scenarios [3/3] | Plots [N] | VERDICT: APPROVE/REJECT`

- [x] F3. **WAV Round-Trip** — `unspecified-high`
  Save IQ to WAV, load back, compare arrays (tolerance: 1e-6). Verify metadata preserved.
  Output: `Round-trip error [X] | Metadata [OK/FAIL] | VERDICT: APPROVE/REJECT`

- [x] F4. **Waypoint Test** — `unspecified-high`
  Create custom waypoint path, run simulation, verify trajectory plot shows correct curved path.
  Output: `Waypoints parsed | Trajectory valid | VERDICT: APPROVE/REJECT`

- [x] F5. **TX Derivation Visual** — `unspecified-high`
  Check static plot shows both true and estimated TX paths. Check animation shows ghost marker.
  Output: `Static [OK/FAIL] | Animation [OK/FAIL] | VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- **1**: `refactor: extract config to config.py` - config.py
- **2**: `refactor: extract geometry to geometry.py` - geometry.py
- **3**: `feat: add WaypointPath class` - paths.py
- **4**: `feat: add WAV save/load to iq_gen` - iq_gen.py
- **5**: `refactor: extract estimation to estimation.py` - estimation.py
- **6**: `refactor: extract kinematics to kinematics.py` - kinematics.py
- **7**: `feat: add TX path derivation visualization` - plotting.py
- **8**: `refactor: extract plotting to plotting.py` - plotting.py
- **9**: `feat: add main.py orchestrator` - main.py
- **10**: `refactor: add backward-compat shim to sim.py` - sim.py

---

## Success Criteria

### Verification Commands
```bash
python -c "from sim import generate_iq, scenario_same_direction; print('Import OK')"
python main.py  # Should run all scenarios
python -c "from iq_gen import save_iq_wav, load_iq_wav; print('WAV functions OK')"
python -c "from paths import WaypointPath; print('WaypointPath OK')"
```

### Final Checklist
- [x] All modules import without errors
- [x] All 3 scenarios produce identical output to original
- [x] WAV save/load preserves IQ data
- [x] Waypoint-based paths work correctly
- [x] TX estimated path visualization displays correctly
- [x] Animation shows estimated TX position