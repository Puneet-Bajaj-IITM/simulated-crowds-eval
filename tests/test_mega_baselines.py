import numpy as np
import pandas as pd
import pytest

from crowdeval import config
from crowdeval.models import baselines
from crowdeval.new_questions import mega_baselines

PEOPLE = 240
OUTCOMES = ("s/a", "s/b")
LEVELS = ("control", "treated")
TREATMENT_SHIFT = 3.0
PERSON = 17


def synthetic(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.Index(np.arange(100, 100 + PEOPLE))
    features = pd.DataFrame(rng.normal(size=(PEOPLE, 4)), index=index, columns=[f"QID{i}" for i in range(4)])
    level = rng.choice(LEVELS, PEOPLE)
    shift = np.where(level == LEVELS[1], TREATMENT_SHIFT, 0.0)
    truth = pd.DataFrame({
        OUTCOMES[0]: np.round(2 + features["QID0"].to_numpy() + shift),
        OUTCOMES[1]: rng.integers(1, 6, PEOPLE).astype(float),
    }, index=index)
    truth.iloc[::9, 1] = np.nan
    conditions = pd.DataFrame({c: level for c in OUTCOMES}, index=index, dtype=object).where(truth.notna())
    return truth, conditions, features


def test_folds_cover_everyone_once():
    truth, _, _ = synthetic()
    fold_of = mega_baselines.folds(truth.index)
    assert sorted(set(fold_of)) == list(range(config.CV_FOLDS))
    for fold, (_, test) in enumerate(baselines.person_folds(truth.index)):
        assert (fold_of[truth.index.get_indexer(test)] == fold).all()


def test_median_is_learned_within_the_cell_from_other_folds():
    truth, conditions, _ = synthetic()
    predicted = mega_baselines.guess_the_average(truth, conditions)
    fold_of = mega_baselines.folds(truth.index)
    column = OUTCOMES[0]
    for i in (0, 5, PERSON):
        same_cell = (conditions[column] == conditions[column].iloc[i]).to_numpy() & (fold_of != fold_of[i])
        assert predicted[column].iloc[i] == np.median(truth[column].to_numpy()[same_cell])
    assert predicted[OUTCOMES[1]].isna().equals(truth[OUTCOMES[1]].isna())


def _own_answer_changed(truth: pd.DataFrame) -> pd.DataFrame:
    changed = truth.copy()
    changed.iloc[PERSON] = changed.iloc[PERSON] + 50
    return changed


@pytest.mark.parametrize("model", ["median", "twin", "behavior"])
def test_a_persons_own_answers_never_reach_their_prediction(model):
    truth, conditions, features = synthetic()
    run = {
        "median": lambda t: mega_baselines.guess_the_average(t, conditions),
        "twin": lambda t: mega_baselines.statistical_twin(t, conditions, features),
        "behavior": lambda t: mega_baselines.trained_on_behavior(t, conditions, features),
    }[model]
    before, after = run(truth), run(_own_answer_changed(truth))
    assert before.iloc[PERSON].notna().any()
    pd.testing.assert_series_equal(before.iloc[PERSON], after.iloc[PERSON])


def test_statistical_twin_keeps_the_condition_difference():
    truth, conditions, features = synthetic()
    predicted = mega_baselines.statistical_twin(truth, conditions, features)[OUTCOMES[0]]
    treated = conditions[OUTCOMES[0]] == LEVELS[1]
    assert predicted[treated].mean() - predicted[~treated].mean() > TREATMENT_SHIFT / 2


def test_cells_with_too_few_answers_fall_back_to_the_cell_median():
    truth, conditions, features = synthetic()
    tiny = truth.copy()
    tiny.iloc[baselines.MIN_TRAIN_ROWS:, 0] = np.nan
    cells = conditions.where(tiny.notna())
    predicted = mega_baselines.trained_on_behavior(tiny, cells, features)
    expected = mega_baselines.guess_the_average(tiny, cells)
    pd.testing.assert_series_equal(predicted[OUTCOMES[0]], expected[OUTCOMES[0]])
