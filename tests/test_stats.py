import numpy as np

from unpacked_applause.stats import (
    benjamini_hochberg, caar_tests, design, permutation_pvalue, wild_cluster_bootstrap,
)
import pandas as pd


def test_benjamini_hochberg_matches_hand_calculation():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_caar_tests_detects_shift():
    rng = np.random.default_rng(0)
    car = rng.normal(0.02, 0.02, 200)
    out = caar_tests(car, car / 0.02, np.full(200, 0.5))
    assert out["p_t"] < 1e-6 and out["ci_lo"] > 0 and out["p_sign"] < 1e-3


def _clustered_data(effect, seed):
    rng = np.random.default_rng(seed)
    firms = np.repeat(np.arange(9), 16)
    x = rng.normal(size=len(firms))
    y = effect * x + rng.normal(size=9)[firms] + rng.normal(size=len(firms))
    return pd.DataFrame({"y": y, "x": x, "company": firms.astype(str)})


def test_wild_cluster_bootstrap_power_and_size():
    strong = _clustered_data(0.8, 1)
    X, _ = design(strong, "x")
    assert wild_cluster_bootstrap(strong["y"], X, strong["company"], 0, n_boot=999)["p_wcb"] < 0.01
    # Under the null the p-values should not pile up near zero.
    ps = []
    for seed in range(20):
        null = _clustered_data(0.0, seed + 100)
        X, _ = design(null, "x")
        ps.append(wild_cluster_bootstrap(null["y"], X, null["company"], 0, n_boot=499)["p_wcb"])
    assert np.mean(np.array(ps) < 0.05) <= 0.2


def test_permutation_pvalue():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    rho, p = permutation_pvalue(x, x + rng.normal(scale=0.5, size=80), n_perm=999)
    assert rho > 0.7 and p < 0.01
