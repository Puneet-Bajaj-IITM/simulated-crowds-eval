"""Leak safety for the mega-study models: no target, or copy of one, is ever a model input."""

import numpy as np
import pandas as pd
import pytest

from crowdeval.new_questions import mega_baselines, mega_data, mega_leaks, mega_outcomes
from crowdeval.new_questions.mega_conditions import SINGLE_CELL

PEOPLE = 200
TARGET = "study/outcome"


def synthetic(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.Index(np.arange(1, PEOPLE + 1))
    features = pd.DataFrame(rng.normal(size=(PEOPLE, 5)), index=index, columns=[f"QID{i}" for i in range(5)])
    truth = pd.DataFrame({TARGET: rng.integers(1, 8, PEOPLE).astype(float)}, index=index)
    conditions = pd.DataFrame(SINGLE_CELL, index=index, columns=truth.columns, dtype=object)
    return features, truth, conditions


def test_guard_passes_unrelated_features():
    features, truth, _ = synthetic()
    assert mega_leaks.leaking_columns(features, truth) == []


def test_guard_catches_a_target_by_name_and_by_copy():
    features, truth, _ = synthetic()
    named = features.assign(**{TARGET: 0.0})
    assert (TARGET, TARGET, 1.0) in mega_leaks.leaking_columns(named, truth)
    copied = features.assign(QID99=truth[TARGET] * 3 + 1)
    copied.iloc[::4, -1] = np.nan
    assert [f for f, _, _ in mega_leaks.leaking_columns(copied, truth)] == ["QID99"]


@pytest.mark.parametrize("model", [mega_baselines.trained_on_behavior, mega_baselines.statistical_twin])
def test_models_refuse_features_that_copy_a_target(model):
    features, truth, conditions = synthetic()
    with pytest.raises(mega_leaks.LeakError):
        model(truth, conditions, features.assign(QID99=truth[TARGET]))


def test_suspiciously_perfect_scores_stop_the_run():
    per_outcome = pd.DataFrame({"median": [0.8, 0.995], "model": [0.995, 0.995]}, index=["a", "b"])
    with pytest.raises(mega_leaks.SuspiciousScoreError, match="'a'"):
        mega_leaks.assert_not_suspicious(per_outcome, "median")
    mega_leaks.assert_not_suspicious(per_outcome.loc[["b"]], "median")


def test_no_target_is_in_the_real_model_inputs():
    """The regression test for the mega-study: real features against every real target."""
    truth = mega_data.load_released(mega_outcomes.included(), (mega_data.TWIN,)).truth
    features = mega_baselines.wave1_3_features()
    variables = {mega_outcomes.variable_of(c) for c in truth.columns}
    assert set(features.columns).isdisjoint(truth.columns)
    assert set(features.columns).isdisjoint(variables)
    assert mega_leaks.leaking_columns(features, truth) == []
    assert mega_leaks.largest_correlation(features, truth)[2] < mega_leaks.LEAK_CORRELATION


def test_real_features_are_finite_wave_1_3_answers():
    from crowdeval import config, data

    features = mega_baselines.wave1_3_features()
    source = pd.read_csv(data.fetch(config.HUMAN_WAVE1_3_FILE), nrows=0).columns
    assert set(features.columns) <= set(source)
    assert np.isfinite(features.to_numpy(dtype=float)[features.notna().to_numpy()]).all()
