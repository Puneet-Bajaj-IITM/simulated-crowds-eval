import numpy as np
import pandas as pd
import pytest

from crowdeval.new_questions import mega_effects, mega_panel, mega_scoring as ms
from crowdeval.core.panel import BASELINE

OUTCOMES = pd.DataFrame(
    {"study": ["a", "a", "b"], "low": [1.0, 1.0, 0.0], "high": [5.0, 5.0, 100.0]},
    index=["a/x", "a/y", "b/z"],
)


def frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=OUTCOMES.index, index=pd.RangeIndex(1, len(rows) + 1))


def test_accuracy_counts_each_study_equally_and_each_outcome_in_the_secondary_weighting():
    truth = frame([[1, 5, 0], [3, 5, 100], [5, np.nan, 50]])
    pred = frame([[2, 5, 0], [3, 1, 50], [5, 5, 50]])
    mask = truth.notna()
    errors = ms.errors(pred, truth, mask, OUTCOMES)
    per_outcome = 1 - np.array([np.mean([0.25, 0, 0]), np.mean([0, 1.0]), np.mean([0, 0.5, 0])])
    result = ms.accuracy(errors, ms.study_groups(truth.columns), ms.point_weights(len(truth)))
    assert result["accuracy"][0] == pytest.approx(np.mean([per_outcome[:2].mean(), per_outcome[2]]))
    assert result["accuracy_outcome_weighted"][0] == pytest.approx(per_outcome.mean())


def test_chance_uses_scale_points_for_integer_answers_and_the_continuum_otherwise():
    truth = frame([[1, 2.5, 0], [3, 2.5, 50]])
    chance = ms.chance_errors(truth, truth.notna(), OUTCOMES)
    points = np.arange(1, 6)
    assert chance[1, 0] == pytest.approx(np.abs(points - 3).mean() / 4)
    assert chance[0, 1] == pytest.approx((1.5**2 + 2.5**2) / (2 * 4**2))
    assert chance[0, 2] == pytest.approx(np.abs(np.arange(0, 101) - 0).mean() / 100)


def test_weighted_moments_equal_moments_of_the_resampled_people():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 3))
    values[::7, 1] = np.nan
    weights = ms.bootstrap_weights(50, 4, seed=2)
    assert (weights.sum(axis=1) == 50).all()
    for draw in range(4):
        resampled = np.repeat(values, weights[draw].astype(int), axis=0)
        np.testing.assert_allclose(ms.outcome_means(values, weights)[draw], np.nanmean(resampled, axis=0))
        np.testing.assert_allclose(ms.outcome_sds(values, weights)[draw], np.nanstd(resampled, axis=0, ddof=1))


def test_panel_rescales_against_the_median_chance_and_own_prior():
    rng = np.random.default_rng(3)
    truth = frame(rng.integers(1, 6, size=(60, 3)).astype(float).tolist())
    truth["b/z"] = rng.integers(0, 101, 60).astype(float)
    approaches = {BASELINE: truth * 0 + truth.median(), "perfect": truth.copy(), "noisy": truth + rng.normal(0, 1, truth.shape)}
    priors = {BASELINE: BASELINE, "perfect": BASELINE, "noisy": BASELINE}
    mask = mega_panel.scored_mask(truth, approaches)
    scorer = mega_panel.Scorer(mega_panel.Inputs(truth, approaches, priors, OUTCOMES), mask)
    table, context = mega_panel.build(scorer, draws=50)
    rows = table.set_index("approach")
    assert rows.loc["perfect", "accuracy"] == pytest.approx(1.0)
    assert rows.loc["perfect", "gain_share"] == pytest.approx(1.0)
    assert rows.loc[BASELINE, "gain_points"] == 0.0
    expected = (rows.loc["perfect", "accuracy"] - context["chance"]) / (1 - context["chance"])
    assert rows.loc["perfect", "chance_corrected"] == pytest.approx(expected)
    base = rows.loc[BASELINE, "accuracy"]
    assert rows.loc["noisy", "coverage"] == pytest.approx((rows.loc["noisy", "accuracy"] - base) / (1 - base))
    assert rows.loc["perfect", "spread_ratio"] == pytest.approx(1.0)
    assert rows.loc["perfect", "accuracy_lo"] <= 1.0 <= rows.loc["perfect", "accuracy_hi"] + 1e-12


def test_effects_compare_two_condition_levels():
    labels = np.array(["a", "b"] * 30, dtype=object)
    truth = pd.DataFrame({"s/y": np.where(labels == "a", 4.0, 2.0) + np.tile([0.0, 0.5, 1.0], 20)})
    conditions = pd.DataFrame({"s/y": labels}, dtype=object)
    flipped = pd.DataFrame({"s/y": np.where(labels == "a", 2.0, 4.0) + np.tile([0.0, 0.5, 1.0], 20)})
    table = mega_effects.effect_table(truth, {"same": truth, "flipped": flipped}, conditions, truth.notna(), ["s/y"])
    row = table.iloc[0]
    assert row["human_significant"] and row["same_same_direction"] and not row["flipped_same_direction"]
    assert row["same_size_ratio"] == pytest.approx(1.0) and row["flipped_raw_ratio"] == pytest.approx(-1.0)
