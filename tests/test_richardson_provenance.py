"""Richardson order_p: single resolution point, no silent defaults, HDF5 records the value actually used.

Also covers the apparent boundary-arrival velocity of the report (distance from the saved d_hit_eff,
no N/2 fallback for legacy files).
"""
import math
import warnings

import numpy as np
import pytest

from bav_dqs.core.models.dirac_scheme import INTEGRATOR_ORDER
from bav_dqs.utils.helpers.resolution import richardson_extrapolate
from bav_dqs.utils.plugins.dirac_simulation import parse_richardson_cfg
from bav_dqs.utils.runtime import run_dirac_simulation as runner


# --- parse_richardson_cfg: the only place where order_p is read -------------------------------------
@pytest.mark.parametrize("cfg", [{}, {"richardson": None}, {"richardson": {"enabled": False, "order_p": 2}}])
def test_disabled_richardson_carries_no_order(cfg):
    assert parse_richardson_cfg(cfg) == {"enabled": False}


def test_enabled_richardson_requires_explicit_order_p():
    with pytest.raises(KeyError, match="richardson.order_p"):
        parse_richardson_cfg({"richardson": {"enabled": True}})


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1", True, None])
def test_invalid_order_p_is_rejected_not_truncated(bad):
    with pytest.raises(ValueError, match="integer >= 1"):
        parse_richardson_cfg({"richardson": {"enabled": True, "order_p": bad}})


@pytest.mark.parametrize("p", [1, 2, 2.0])
def test_valid_order_p_is_returned_as_int(p):
    out = parse_richardson_cfg({"richardson": {"enabled": True, "order_p": p}})
    assert out == {"enabled": True, "order_p": int(p)}
    assert isinstance(out["order_p"], int)


# --- HDF5 attributes come from the same resolved config ----------------------------------------------
def test_attributes_when_disabled_use_the_documented_sentinel():
    assert runner.richardson_attributes({"enabled": False}) == {
        "richardson_enabled": False, "richardson_order_p": -1, "richardson_order_consistent": True,
    }


@pytest.mark.parametrize("p", [1, 2, 3])
def test_attributes_record_the_order_actually_configured(p):
    attrs = runner.richardson_attributes(parse_richardson_cfg({"richardson": {"enabled": True, "order_p": p}}))
    assert attrs["richardson_enabled"] is True
    assert attrs["richardson_order_p"] == p
    assert attrs["richardson_order_consistent"] is (p == INTEGRATOR_ORDER)


def test_attributes_never_fall_back_to_a_default_order():
    """An unvalidated config (enabled but no order_p) must fail, not be recorded as 0 or 1."""
    with pytest.raises(KeyError):
        runner.richardson_attributes({"enabled": True})


@pytest.mark.parametrize("p", [1, 2])
def test_extrapolation_uses_the_same_p_that_is_recorded(p):
    rng = np.random.default_rng(0)
    occ_full, occ_half = rng.random((10, 6)), rng.random((10, 6))
    rich_cfg = parse_richardson_cfg({"richardson": {"enabled": True, "order_p": p}})
    occ_rich, gap_l, gap_r = runner._apply_richardson_if_enabled(
        rich_cfg, {"edge_window": 1}, occ_full, occ_half, n_safe=8)
    recorded = runner.richardson_attributes(rich_cfg)["richardson_order_p"]
    np.testing.assert_allclose(occ_rich, richardson_extrapolate(occ_full[:8], occ_half[:8], recorded))
    assert occ_rich.shape == (8, 6) and gap_l.shape == (8,) and gap_r.shape == (8,)


def test_disabled_richardson_computes_nothing():
    z = np.zeros((4, 4))
    assert runner._apply_richardson_if_enabled({"enabled": False}, {"edge_window": 1}, z, z, 4) == (None, None, None)


def test_inconsistent_order_warns_but_does_not_abort():
    with pytest.warns(UserWarning, match="does not match"):
        msg = runner._warn_if_richardson_order_inconsistent({"enabled": True, "order_p": INTEGRATOR_ORDER + 1})
    assert msg is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert runner._warn_if_richardson_order_inconsistent({"enabled": True, "order_p": INTEGRATOR_ORDER}) is None
        assert runner._warn_if_richardson_order_inconsistent({"enabled": False}) is None


# --- apparent boundary-arrival velocity (report) -----------------------------------------------------
def _velocity(rows):
    from bav_dqs.utils.runtime.generate_plots import ScientificReport
    return ScientificReport.causal_velocity(None, rows).set_index("n_qubits")


def test_velocity_uses_saved_distance_of_the_side_that_was_hit():
    # N=4, right-edge hit: source at 2, right edge at 3 -> d_eff = 1 (v1 used N/2 = 2)
    v = _velocity([{"n_qubits": 4, "first_hit_full": 7, "dt_full": 0.02, "d_hit_eff": 1.0}])
    assert v.loc[4, "d_eff"] == 1.0
    assert v.loc[4, "v_hit"] == pytest.approx(1.0 / (7 * 0.02))


def test_velocity_is_undefined_without_hit_or_saved_distance():
    v = _velocity([
        {"n_qubits": 20, "first_hit_full": -1, "dt_full": 0.02, "d_hit_eff": float("nan")},  # NO HIT
        {"n_qubits": 8, "first_hit_full": 0, "dt_full": 0.02, "d_hit_eff": 3.0},              # t_hit = 0
        {"n_qubits": 6, "first_hit_full": 27, "dt_full": 0.02, "d_hit_eff": float("nan")},    # legacy file
    ])
    assert all(math.isnan(x) for x in v["v_hit"])
    assert (v["v_hit"].dropna() >= 0).all()
