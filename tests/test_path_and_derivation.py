"""
Integration tests for custom paths and TX derivation helpers.
"""

import numpy as np

from config import scenario_colocated, scenario_same_direction
from iq_gen import generate_iq
from paths import angled_path
from plotting import _estimate_tx_positions


def test_path_provider_geometry_matches_interpolated_path():
    cfg = scenario_same_direction()
    path = angled_path(
        x0=cfg.tx_x0,
        y0=cfg.tx_y,
        angle_deg=10.0,
        speed=cfg.tx_vx,
        duration=cfg.duration,
    )

    np.random.seed(42)
    result = generate_iq(cfg, path_provider=path)
    tx_x_expected, tx_y_expected = path.interpolate(result["t"])

    np.testing.assert_allclose(result["tx_x"], tx_x_expected, atol=1e-6)
    np.testing.assert_allclose(result["tx_y"], tx_y_expected, atol=1e-6)
    assert np.ptp(result["tx_y"]) > 1.0


def test_tx_derivation_marks_colocated_as_unobservable():
    cfg = scenario_colocated()
    result = generate_iq(cfg)
    est = _estimate_tx_positions(
        result,
        t_est=result["t"][1:],
        delta_f_est=np.zeros(len(result["t"]) - 1),
    )

    assert est["observable"] is False
    assert not np.any(est["valid_mask"])
    assert np.all(np.isnan(est["tx_x_est"]))
    assert np.all(np.isnan(est["tx_y_est"]))


def test_tx_derivation_uses_direct_dx_without_range_reprojection():
    cfg = scenario_same_direction()
    result = generate_iq(cfg)

    dx_true = result["tx_x"] - result["rx_x"]
    delta_f_true = -result["r_dot"][1:] * cfg.fc / cfg.c
    est = _estimate_tx_positions(result, result["t"][1:], delta_f_true)

    expected_tx_x = result["rx_x"] + dx_true
    expected_tx_y = np.full_like(result["t"], cfg.tx_y)

    valid = est["valid_mask"]
    np.testing.assert_allclose(est["tx_x_est"][valid], expected_tx_x[valid], atol=1e-4)
    np.testing.assert_allclose(est["tx_y_est"][valid], expected_tx_y[valid], atol=1e-8)
