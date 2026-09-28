import numpy as np
import pandas as pd
import pytest

from unpacked_applause.event_study import abnormal_returns, abnormal_volume, car, locate_t0


def synthetic_panel(n=400, beta=1.2, alpha=0.0005, shock_at=300, shocks=(0.03, 0.01), seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0, 0.01, n)
    ret = alpha + beta * mkt + rng.normal(0, 0.005, n)
    for i, s in enumerate(shocks):
        ret[shock_at + i] += s
    close = days.tz_localize("UTC") + pd.Timedelta(hours=20)
    return pd.DataFrame({"ret": ret, "mkt": mkt, "logvol": np.log1p(np.full(n, 1e6)),
                         "close_utc": close}, index=days)


def test_market_model_recovers_beta_and_injected_car():
    panel = synthetic_panel()
    res = abnormal_returns(panel, 300, "market")
    assert res["beta"] == pytest.approx(1.2, abs=0.1)
    assert car(res["ar"], (0, 1)) == pytest.approx(0.04, abs=0.012)
    assert car(res["ar"], (-10, -1)) == pytest.approx(0.0, abs=0.03)
    assert res["n_est"] == 221


def test_market_adjusted_and_mean_models_run():
    panel = synthetic_panel()
    for model in ("market_adj", "mean"):
        res = abnormal_returns(panel, 300, model)
        assert np.isfinite(res["sigma"]) and res["sigma"] > 0
    assert abnormal_returns(panel, 300, "ff3") is None  # no factor columns in this panel


def test_estimation_window_must_fit_before_t0():
    assert abnormal_returns(synthetic_panel(), 100, "market") is None


def test_car_is_nan_when_window_runs_past_the_data():
    panel = synthetic_panel(n=310, shock_at=300)
    res = abnormal_returns(panel, 300, "market")
    assert np.isfinite(car(res["ar"], (0, 5)))
    assert np.isnan(car(res["ar"], (2, 20)))


def test_locate_t0_uses_session_close():
    days = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"])  # Thu, Fri, Mon
    closes = pd.Series(days.tz_localize("UTC") + pd.Timedelta(hours=21), index=days)
    during = pd.Timestamp("2024-01-04 18:00", tz="UTC")  # 1 pm ET, market open
    evening = pd.Timestamp("2024-01-05 23:30", tz="UTC")  # after Friday close
    assert locate_t0(closes, during) == 0
    assert locate_t0(closes, evening) == 2  # next Monday
    assert locate_t0(closes, pd.Timestamp("2024-01-09", tz="UTC")) is None


def test_abnormal_volume_detects_spike():
    panel = synthetic_panel()
    panel.loc[panel.index[300], "logvol"] = np.log1p(3e6)
    raw, _ = abnormal_volume(panel, 300)
    assert raw == pytest.approx((np.log1p(3e6) - np.log1p(1e6)) / 2, rel=1e-6)
