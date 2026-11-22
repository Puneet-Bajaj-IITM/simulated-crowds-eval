import numpy as np
import pandas as pd

from crowdeval.core import metrics


def test_js_divergence_bounds():
    same = np.array([5, 3, 2])
    assert metrics.js_divergence(same, same) == 0
    assert np.isclose(metrics.js_divergence(np.array([1, 0]), np.array([0, 1])), 1.0)


def test_bins_shared_between_integer_and_continuous_answers():
    human, k_h = metrics.bin_codes(np.array([1.0, 2.0, 6.0, np.nan]), "Q157")
    sim, k_s = metrics.bin_codes(np.array([1.4, 2.6, 5.9, 3.5]), "Q157")
    assert k_h == k_s == 6
    assert list(human) == [0, 1, 5, -1]
    assert list(sim) == [0, 2, 5, 3]


def test_wide_scale_uses_decile_bins():
    codes, k = metrics.bin_codes(np.array([0.0, 49.9, 100.0]), "Q156_1")
    assert k == 10 and list(codes) == [0, 4, 9]


def test_population_metrics_constant_prediction_has_zero_spread():
    truth = pd.DataFrame({"Q157": [1.0, 2.0, 3.0, 4.0]})
    pred = pd.DataFrame({"Q157": [2.0, 2.0, 2.0, 2.0]})
    mask = pd.DataFrame({"Q157": [True] * 4})
    out = metrics.population_metrics(pred, truth, mask)
    assert out["spread_ratio"] == 0
    assert out["js_divergence"] > 0
