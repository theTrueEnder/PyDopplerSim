# PyDopplerSim Testing Framework Plan

## TL;DR

> Add pytest-based unit testing with Hypothesis property-based testing for analytical modules, plus GitHub Actions CI.

**Deliverables:**
- tests/ directory with test modules  
- pytest configuration (pyproject.toml)
- Hypothesis profile for property-based tests
- GitHub Actions workflow

**Estimated:** Medium effort

---

## Context

**Request:** Create unit testing framework for PyDopplerSim, focusing on analytical aspects.

**User Preferences:**
- Framework: pytest
- Location: tests/ directory
- Coverage: All core modules (geometry, estimation, kinematics, iq_gen, paths)
- Property testing: Yes (Hypothesis)
- CI: Yes (GitHub Actions)

---

## Work Objectives

### Core Deliverables
1. Tests for geometry.compute_geometry() - verify analytical solutions
2. Tests for estimation module - phase_diff and STFT estimators
3. Tests for kinematics.recover_kinematics() - numerical differentiation
4. Tests for iq_gen save/load round-trip
5. Tests for paths.WaypointPath interpolation
6. Hypothesis property tests for edge cases
7. GitHub Actions CI workflow

### Must Have
- Mathematical correctness verified against known solutions
- Property-based tests discover edge cases
- CI runs tests automatically
- Test markers for slow tests

### Must NOT Have  
- Tests requiring matplotlib display (use Agg backend)
- Tests requiring ffmpeg (skip if unavailable)
- Slow tests blocking CI

---

## Execution Strategy

### Wave 1 (Setup):
- T1: Install pytest, pytest-cov, hypothesis
- T2: Create pytest config in pyproject.toml
- T3: Create tests/conftest.py with fixtures
- T4: Add Hypothesis profile settings

### Wave 2 (Core tests):
- T5: Test geometry.compute_geometry() - analytical verification
- T6: Test estimation._hann_smooth() - known filter response
- T7: Test estimation.estimate_doppler_phase_diff() - synthetic IQ
- T8: Test estimation.estimate_doppler_stft() - spectral peaks

### Wave3 (Advanced tests):
- T9: Test kinematics.recover_kinematics() - round-trip
- T10: Test iq_gen.generate_iq() - signal properties
- T11: Test iq_gen save/load round-trip - data integrity
- T12: Test paths.WaypointPath - interpolation accuracy

### Wave4 (Property tests):
- T13: Hypothesis - geometry edge cases (vdx=0, dy=0)
- T14: Hypothesis - estimation stability at various SNR
- T15: Hypothesis - kinematics numerical stability

### Wave5 (CI):
- T16: Create .github/workflows/test.yml
- T17: Add pytest markers (@slow, @requires_ffmpeg)
- T18: Verify all tests pass locally

---

## TODOs

- [ ] 1. Install test dependencies (pytest, pytest-cov, hypothesis)
- [ ] 2. Create pytest configuration in pyproject.toml
- [ ] 3. Create tests/conftest.py with fixtures
- [ ] 4. Add Hypothesis profile
- [ ] 5. Test geometry.compute_geometry() - analytical verification
- [ ] 6. Test estimation._hann_smooth()
- [ ] 7. Test estimation.estimate_doppler_phase_diff()
- [ ] 8. Test estimation.estimate_doppler_stft()
- [ ] 9. Test kinematics.recover_kinematics()
- [ ] 10. Test iq_gen.generate_iq()
- [ ] 11. Test iq_gen save/load round-trip
- [ ] 12. Test paths.WaypointPath
- [ ] 13. Hypothesis test - geometry edge cases
- [ ] 14. Hypothesis test - estimation stability
- [ ] 15. Hypothesis test - kinematics numerical stability
- [ ] 16. Create GitHub Actions workflow
- [ ] 17. Add pytest markers
- [ ] 18. Verify all tests pass

---

## Final Verification

- pytest runs without errors
- Coverage report generated
- CI workflow valid

---

## Success Criteria

pytest tests/ -v
pytest tests/ --cov --cov-report=html