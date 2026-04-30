"""
Pytest configuration and fixtures for PyDopplerSim tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import scenario builders
from config import (
    ScenarioConfig,
    scenario_colocated,
    scenario_same_direction,
    scenario_oncoming,
)


# =============================================================================
# Fixtures for ScenarioConfig (all 3 scenarios)
# =============================================================================


def _fast_cfg(cfg: ScenarioConfig) -> ScenarioConfig:
    """Shrink scenarios for unit tests; full-rate runs belong in slow tests."""
    cfg.fs = 20_000.0
    cfg.duration = min(cfg.duration, 0.2)
    cfg.interp_oversample = 2
    return cfg


@pytest.fixture
def scenario_colocated_cfg() -> ScenarioConfig:
    """Co-located scenario: Tx and Rx at same position, moving together."""
    return _fast_cfg(scenario_colocated())


@pytest.fixture
def scenario_same_direction_cfg() -> ScenarioConfig:
    """Same-direction scenario: Tx ahead, moving slower than Rx."""
    return _fast_cfg(scenario_same_direction())


@pytest.fixture
def scenario_oncoming_cfg() -> ScenarioConfig:
    """Oncoming scenario: Tx approaching from opposite direction."""
    return _fast_cfg(scenario_oncoming())


@pytest.fixture(params=["colocated", "same_direction", "oncoming"])
def any_scenario_cfg(request) -> ScenarioConfig:
    """Parameterized fixture for all three scenarios."""
    scenarios = {
        "colocated": scenario_colocated,
        "same_direction": scenario_same_direction,
        "oncoming": scenario_oncoming,
    }
    return _fast_cfg(scenarios[request.param]())


# =============================================================================
# Fixture for fixed random seed
# =============================================================================


@pytest.fixture
def fixed_seed():
    """Fixed random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def rng(fixed_seed):
    """Random number generator with fixed seed."""
    return np.random.default_rng(fixed_seed)


# =============================================================================
# Fixture for temp WAV directory
# =============================================================================


@pytest.fixture
def temp_wav_dir():
    """Create a temporary directory for WAV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Hypothesis profile settings (CI profile with 100 examples)
# =============================================================================


def pytest_configure(config):
    """Register custom Hypothesis profiles."""
    config.addinivalue_line(
        "markers", "hypothesis_profile:ci"
    )


@pytest.fixture(scope="session", autouse=True)
def hypothesis_ci_profile():
    """Apply Hypothesis CI profile with 100 examples."""
    try:
        from hypothesis import settings

        settings.register_profile(
            "ci",
            max_examples=100,
            deadline=None,  # Disable deadline to avoid issues with slow tests
            database=None,  # Disable example database for CI
        )
        settings.load_profile("ci")
    except ImportError:
        # Hypothesis not installed, skip
        pass
