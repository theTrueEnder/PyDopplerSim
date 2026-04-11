# 3D Expansion Plan for PyDopplerSim

## TL;DR

> **Quick Summary**: Transition PyDopplerSim from 2D (x-y plane) to full 3D (x-y-z) while maintaining 100% backward compatibility with existing 2D scenarios. Add new 3D scenarios (overfly, circular, vertical), 3D trajectory visualization, and ensure mathematical accuracy using numerical gradient methods.

> **Deliverables**:
> - Extended ScenarioConfig with Z position/velocity parameters
> - 3D geometry computation using numerical gradients
> - Three new 3D scenario builders
> - 3D trajectory plot function
> - Updated CLI with new scenario names
> - New 3D test suite
> - Backward-compatible 2D operation

> **Estimated Effort**: Large  
> **Parallel Execution**: YES - 4 waves  
> **Critical Path**: Phase 1 → Phase 3 → Phase 5 → Phase 6

---

## Context

### Original Request
User wants to shift simulator from 2D to 3D:
- Full 3D velocities (vx, vy, vz) for both tx and rx
- 2D scenarios work in 3D environment (colocated, same-direction, oncoming)
- Add new 3D scenarios: overfly, circular flight, vertical approach
- 3D trajectory plots
- New CLI scenario names
- 100% backward compatible

### Interview Summary
**Key Discussions**:
- Full 3D velocities confirmed (not just adding z to existing 2D)
- 2D scenarios must work unchanged in 3D environment
- New 3D scenarios should be added beyond 2D equivalents
- 3D visualization required for trajectories
- Backward compatibility is critical

### Gap Analysis Findings (Metis Review)

**Critical Insights**:
1. **DO NOT adapt analytic r_ddot formula** for 3D - use numerical gradient (compute_geometry_from_samples already does this correctly)
2. **safe_r threshold** (1e-6) handles zero-range cases
3. **compute_geometry_from_samples** already handles arbitrary trajectories - extend it for z
4. **Los angle** needs both azimuth and elevation for 3D
5. **TX derivation** cannot uniquely invert 3D from r_dot alone - document limitation
6. **Keep compute_geometry** unchanged for 2D, create separate 3D path

**Gap Analysis Summary**:
- Critical gaps: 1 (r_ddot formula - resolved by using gradient method)
- Edge cases: 6 identified (zero range, pure vertical, circular, etc.)
- Backward compatibility risks: 4 identified (result dict keys, tests, etc.)
- Modules requiring changes: 7 (config, geometry, paths, iq_gen, plotting, main, tests)

---

## Work Objectives

### Core Objective
Implement full 3D support for PyDopplerSim while maintaining exact backward compatibility with existing 2D scenarios and ensuring mathematical accuracy.

### Concrete Deliverables
1. Extended `ScenarioConfig` with `tx_z0`, `tx_vz`, `rx_z0`, `rx_vz` (default=0.0)
2. Updated `compute_geometry_from_samples` to handle z-coordinates
3. New scenario builders: `scenario_overfly`, `scenario_circular`, `scenario_vertical_approach`
4. New `plot_trajectory_3d()` function using mpl_toolkits.mplot3d
5. Updated CLI with new 3D scenario names
6. New test file `tests/test_geometry_3d.py`
7. Verification that all 2D scenarios produce identical output

### Definition of Done
- [ ] 2D scenarios (colocated, same-direction, oncoming) produce bit-identical output
- [ ] 3D scenarios run without exceptions
- [ ] 3D trajectory plots render correctly
- [ ] All tests pass (both 2D and 3D)
- [ ] No breaking changes to existing API

### Must Have
- Z parameters in ScenarioConfig with defaults preserving 2D behavior
- 3D geometry computed via numerical gradient (not analytic formula)
- New 3D scenarios functional
- Backward compatibility verified

### Must NOT Have
- Modified existing test files (create new ones)
- Analytic 3D r_ddot formula (use gradient)
- Breaking changes to result dictionary keys for 2D scenarios
- TX derivation showing incorrect results for 3D (document limitation)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (pytest + hypothesis)
- **Automated tests**: YES (tests-after for new 3D features)
- **Framework**: pytest

### QA Policy
Every task includes agent-executed QA scenarios:
- **Backend/Math**: Use Python (numpy) to verify computed values against expected formulas
- **CLI**: Run commands and verify exit codes and output files
- **Plots**: Verify files are created with valid content

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - configuration + geometry base):
├── T1: Add Z parameters to ScenarioConfig (config.py) [quick]
├── T2: Verify backward compat of config changes [quick]
├── T3: Update geometry.py compute_geometry_from_samples for z [deep]
└── T4: Add tests for 3D geometry math [unspecified-high]

Wave 2 (3D Scenarios + Paths):
├── T5: Create scenario_overfly builder [unspecified-high]
├── T6: Create scenario_circular builder [unspecified-high]
├── T7: Create scenario_vertical_approach builder [unspecified-high]
├── T8: Update paths.py for 3D waypoints [unspecified-high]
└── T9: Test all new scenarios run without error [unspecified-high]

Wave 3 (Visualization):
├── T10: Add plot_trajectory_3d() function [visual-engineering]
├── T11: Update save_all for 3D scenarios [unspecified-high]
├── T12: Update animation for 3D support [visual-engineering]
└── T13: Test 3D plots render correctly [unspecified-high]

Wave 4 (CLI + Integration):
├── T14: Add 3D scenario names to CLI [quick]
├── T15: Verify CLI backward compatibility [unspecified-high]
├── T16: Run full test suite [quick]
└── T17: Regression test 2D scenarios unchanged [unspecified-high]

Wave FINAL (Verification):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review
├── F3: Full integration test
└── F4: Backward compatibility verification
```

### Dependency Matrix

- T1 → T2, T3 (config must work before geometry uses it)
- T3 → T4 (geometry must work before testing)
- T3 → T5, T6, T7 (geometry needed for scenarios)
- T8 → T5, T6 (paths needed for complex 3D scenarios)
- T5, T6, T7 → T9 (scenarios tested after creation)
- T3, T9 → T10 (3D plots need geometry and scenarios)
- T10 → T11 → T12 (save then animation)
- T1, T12, T14 → T15 (CLI needs config + animation)
- T4, T9, T13, T15 → T16 (full tests need all components)
- T16 → T17 (regression after full tests)

---

## TODOs

- [ ] 1. **Add Z parameters to ScenarioConfig**

  **What to do**:
  - Add fields `tx_z0: float = 0.0`, `tx_vz: float = 0.0`, `rx_z0: float = 0.0`, `rx_vz: float = 0.0` to ScenarioConfig dataclass
  - Add to __post_init__ if needed
  - Update _base() function to include z parameters

  **Must NOT do**:
  - Change existing x, y parameter defaults
  - Remove any existing fields

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass field additions with defaults
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Parallel Group**: Wave 1 (with T2, T3, T4)
  - **Blocks**: T2, T3, T4

  **References**:
  - `config.py:102-121` - Current ScenarioConfig structure
  - `config.py:123-133` - _base() function pattern

  **Acceptance Criteria**:
  - [ ] ScenarioConfig has tx_z0, tx_vz, rx_z0, rx_vz fields
  - [ ] Defaults are 0.0 (backward compatible)
  - [ ] python -m py_compile config.py passes

- [ ] 2. **Verify backward compatibility of config changes**

  **What to do**:
  - Run existing 2D scenarios to verify they work with new config
  - Verify ScenarioConfig fields are accessible as before

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Parallel Group**: Wave 1 (with T1, T3, T4)

  **Acceptance Criteria**:
  - [ ] Existing scenarios can be created with new config
  - [ ] cfg.tx_y still works (scalar, not array)

- [ ] 3. **Update geometry.py compute_geometry_from_samples for z**

  **What to do**:
  - Modify compute_geometry_from_samples to accept and process z-coordinates
  - Add vdz = np.gradient(dz, t, edge_order=2)
  - Update r_dot = (dx*vdx + dy*vdy + dz*vdz) / safe_r
  - Keep delta_f formula unchanged (scalar, works with scalar r_dot)
  - Add tx_z, rx_z to return dict

  **Must NOT do**:
  - Modify compute_geometry (2D analytic) - keep separate
  - Try to create analytic 3D r_ddot formula - use gradient

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Mathematical implementation requiring careful vector math
  - **Skills**: None required (numpy math)

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Parallel Group**: Wave 1 (with T1, T2, T4)

  **References**:
  - `geometry.py:74-109` - compute_geometry_from_samples pattern
  - `iq_gen.py:96-100` - how geometry is called

  **Acceptance Criteria**:
  - [ ] Function accepts z arrays (tx_z, rx_z)
  - [ ] r_dot includes dz*vdz term
  - [ ] r_ddot computed via gradient
  - [ ] Returns tx_z, rx_z in result dict
  - [ ] 2D calls (z=0) produce same r_dot as compute_geometry

  **QA Scenarios**:
  ```
  Scenario: Pure vertical motion (3D)
    Tool: Python/numpy
    Preconditions: dx=0, dy=0, only dz changes
    Steps:
      1. Create tx_z = np.linspace(100, 0, 100), rx_z = np.zeros(100)
      2. Call compute_geometry_from_samples
      3. Verify r_dot = dz*vdz/r (vertical velocity component)
    Expected Result: r_dot is negative when descending, positive when ascending
    Evidence: .sisyphus/evidence/task-3-vertical-motion.txt
  ```

- [ ] 4. **Add tests for 3D geometry math**

  **What to do**:
  - Create tests/test_geometry_3d.py
  - Test 3D range calculation
  - Test 3D r_dot with all velocity components
  - Test vertical motion scenario
  - Test circular path (r_ddot not zero at CPA)

  **Must NOT do**:
  - Modify existing test_geometry.py

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)
  - **Parallel Group**: Wave 1 (with T1, T2, T3)

  **Acceptance Criteria**:
  - [ ] Test file created with 5+ test cases
  - [ ] All tests pass
  - [ ] python -m py_compile passes

- [ ] 5. **Create scenario_overfly builder**

  **What to do**:
  - Add scenario_overfly() to config.py
  - Tx starts ahead and above Rx, descends through Rx altitude
  - Should produce classic "tennis ball" r(t) curve with minimum at zenith pass

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Parallel Group**: Wave 2 (with T6, T7, T8, T9)

  **References**:
  - `config.py:136-160` - existing scenario builder pattern

  **Acceptance Criteria**:
  - [ ] Function added to config.py and __all__
  - [ ] Creates ScenarioConfig with non-zero tx_z0, tx_vz
  - [ ] r(t) has minimum at zenith pass

- [ ] 6. **Create scenario_circular builder**

  **What to do**:
  - Add scenario_circular() to config.py
  - Use WaypointPath with circular x-y trajectory
  - Rx stationary or moving linearly

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Parallel Group**: Wave 2 (with T5, T7, T8, T9)

  **Acceptance Criteria**:
  - [ ] Function added to config.py and __all__
  - [ ] Produces circular/elliptical r(t) pattern

- [ ] 7. **Create scenario_vertical_approach builder**

  **What to do**:
  - Add scenario_vertical_approach() to config.py
  - Tx descends vertically toward stationary Rx (or vice versa)
  - Tests pure vertical motion component

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Parallel Group**: Wave 2 (with T5, T6, T8, T9)

  **Acceptance Criteria**:
  - [ ] Function added to config.py and __all__
  - [ ] r_dot derived purely from vertical component

- [ ] 8. **Update paths.py for 3D waypoints**

  **What to do**:
  - Add z coordinate to WaypointPath class
  - Create circular_path_3d(), angled_path_3d() if needed
  - Update interpolate() to return (x, y, z) tuples

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Parallel Group**: Wave 2 (with T5, T6, T7, T9)

  **References**:
  - `paths.py:24-57` - WaypointPath structure

  **Acceptance Criteria**:
  - [ ] WaypointPath supports z coordinates
  - [ ] interpolate() returns z array

- [ ] 9. **Test all new scenarios run without error**

  **What to do**:
  - Run each 3D scenario via CLI
  - Verify no exceptions thrown

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Parallel Group**: Wave 2 (with T5, T6, T7, T8)

  **Acceptance Criteria**:
  - [ ] overfly scenario runs
  - [ ] circular scenario runs
  - [ ] vertical_approach scenario runs

- [ ] 10. **Add plot_trajectory_3d() function**

  **What to do**:
  - Create new plotting function using mpl_toolkits.mplot3d
  - 3D scatter plot of tx and rx trajectories
  - Color-coded by time (optional)
  - Return figure for saving

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with T11, T12, T13)

  **References**:
  - `plotting.py:125-200` - plot_trajectory_fig pattern

  **Acceptance Criteria**:
  - [ ] Function created in plotting.py
  - [ ] Produces valid 3D matplotlib figure
  - [ ] Added to __all__

- [ ] 11. **Update save_all for 3D scenarios**

  **What to do**:
  - Modify save_all to call plot_trajectory_3d for 3D scenarios
  - Detect 3D based on tx_z/rx_z being non-zero

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with T10, T12, T13)

  **Acceptance Criteria**:
  - [ ] 3D scenarios produce 3D trajectory plot
  - [ ] 2D scenarios unchanged (2D trajectory plot)

- [ ] 12. **Update animation for 3D support**

  **What to do**:
  - Extend make_animation or create 3D variant
  - Add elevation indicator to 2D projection (text overlay)
  - Keep backward compatibility with 2D animations

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with T10, T11, T13)

  **References**:
  - `plotting.py:468-609` - current animation implementation

  **Acceptance Criteria**:
  - [ ] Animation works for 3D scenarios
  - [ ] 2D animation unchanged

- [ ] 13. **Test 3D plots render correctly**

  **What to do**:
  - Run 3D scenarios and verify output files exist
  - Check plot files are non-zero size

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Parallel Group**: Wave 3 (with T10, T11, T12)

  **Acceptance Criteria**:
  - [ ] trajectory_3d.png exists and non-empty
  - [ ] No rendering errors

- [ ] 14. **Add 3D scenario names to CLI**

  **What to do**:
  - Add overfly, circular, vertical_approach to SCENARIOS dict in main.py
  - Add to CLI --scenario choices

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 4)
  - **Parallel Group**: Wave 4 (with T15, T16, T17)

  **References**:
  - `main.py:35-39` - SCENARIOS dict
  - `main.py:104-109` - argparse choices

  **Acceptance Criteria**:
  - [ ] --scenario overfly works
  - [ ] --scenario circular works
  - [ ] --scenario vertical_approach works

- [ ] 15. **Verify CLI backward compatibility**

  **What to do**:
  - Run existing 2D scenarios with CLI
  - Verify output matches expected format

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 4)
  - **Parallel Group**: Wave 4 (with T14, T16, T17)

  **Acceptance Criteria**:
  - [ ] --scenario colocated works
  - [ ] --scenario same-direction works
  - [ ] --scenario oncoming works

- [ ] 16. **Run full test suite**

  **What to do**:
  - Run pytest on all test files
  - Fix any failures

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 4)
  - **Parallel Group**: Wave 4 (with T14, T15, T17)

  **Acceptance Criteria**:
  - [ ] pytest tests/ passes
  - [ ] All tests pass

- [ ] 17. **Regression test 2D scenarios unchanged**

  **What to do**:
  - Compare output of 2D scenarios before/after changes
  - Verify trajectory plots identical

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 4)
  - **Parallel Group**: Wave 4 (with T14, T15, T16)

  **Acceptance Criteria**:
  - [ ] 2D trajectory plots unchanged
  - [ ] Numeric outputs identical

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  - Verify all "Must Have" implemented
  - Verify all "Must NOT Have" avoided
  - Check evidence files exist

- [ ] F2. **Code Quality Review** — `unspecified-high`
  - python -m py_compile on all modified files
  - No syntax errors
  - No obvious code smells

- [ ] F3. **Full Integration Test** — `unspecified-high`
  - Run all scenarios (2D and 3D)
  - Verify no crashes

- [ ] F4. **Backward Compatibility Verification** — `deep`
  - Compare 2D scenario outputs before/after
  - Document any differences

---

## Commit Strategy

- **1**: `feat(config): add z parameters to ScenarioConfig` - config.py
- **2**: `feat(geometry): add z support to compute_geometry_from_samples` - geometry.py
- **3**: `feat(config): add 3D scenario builders` - config.py
- **4**: `feat(paths): add z to WaypointPath` - paths.py
- **5**: `feat(plotting): add 3D trajectory plot` - plotting.py
- **6**: `feat(main): add 3D scenarios to CLI` - main.py
- **7**: `test: add 3D geometry tests` - tests/test_geometry_3d.py
- **8**: `test: verify backward compatibility` - regression test

---

## Success Criteria

### Verification Commands
```bash
# 2D scenarios (must be unchanged)
python main.py --scenario colocated --output-dir /tmp/test_colocated
python main.py --scenario same-direction --output-dir /tmp/test_same
python main.py --scenario oncoming --output-dir /tmp/test_oncoming

# New 3D scenarios
python main.py --scenario overfly --output-dir /tmp/test_overfly
python main.py --scenario circular --output-dir /tmp/test_circular
python main.py --scenario vertical_approach --output-dir /tmp/test_vertical

# Tests
python -m pytest tests/ -v
```

### Final Checklist
- [ ] All 2D scenarios work identically
- [ ] All 3D scenarios run without error
- [ ] 3D trajectory plots render
- [ ] All tests pass
- [ ] No breaking changes