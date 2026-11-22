import json

import numpy as np
import pandas as pd

from crowdeval import config, data
from crowdeval.models import baselines

PRICING_SUBQUESTION = "QID9_1"
PRICING_BASE = "QID9"


def test_sub_question_ids_reduce_to_their_base():
    assert baselines.base_question_id(PRICING_SUBQUESTION) == PRICING_BASE
    assert baselines.base_question_id("QID287") == "QID287"


def test_without_holdout_drops_every_sub_question_of_a_held_out_item():
    columns = pd.Index(["QID9_1", "QID9_40", "QID90_1", "QID287_2", "age"])
    held_out = {baselines.base_question_id(PRICING_SUBQUESTION), "QID287"}
    assert baselines.without_holdout(columns, held_out) == ["QID90_1", "age"]


def test_behavior_features_contain_no_holdout_column():
    mapping = json.loads(data.fetch(config.HOLDOUT_MAPPING_FILE).read_text())
    held_out_columns = {entry["catalog_csv_column"] for entry in mapping}
    held_out_bases = {baselines.base_question_id(entry["QuestionID"]) for entry in mapping}
    features = baselines.behavior_features()
    assert held_out_columns.isdisjoint(features.columns)
    assert not [c for c in features.columns if baselines.base_question_id(c) in held_out_bases]


def test_parallel_behavior_model_matches_a_sequential_fit():
    """Same folds, rows and model as fitting each column in a plain loop, and nothing predicted for unanswered cells."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(3)
    index = pd.Index(np.arange(200), name=config.ID_COLUMN)
    features = pd.DataFrame(rng.normal(size=(200, 6)), index=index, columns=[f"QID{i}_1" for i in range(6)])
    truth = pd.DataFrame({"t1": features.iloc[:, 0] * 2 + rng.normal(size=200), "t2": rng.integers(1, 6, 200).astype(float)})
    truth.iloc[::7, 1] = np.nan
    parallel = baselines.trained_on_behavior(truth, ["t1", "t2"], features)
    for col in truth.columns:
        expected = pd.Series(np.nan, index=index)
        for train, test in baselines.person_folds(index):
            y = truth.loc[train, col].dropna()
            model = HistGradientBoostingRegressor(loss=baselines.GBM_LOSS, max_iter=baselines.GBM_MAX_ITER,
                                                  learning_rate=baselines.GBM_LEARNING_RATE, random_state=config.RANDOM_SEED)
            model.fit(features.loc[y.index].to_numpy(), y.to_numpy())
            answered = test[truth.loc[test, col].notna().to_numpy()]
            expected[answered] = model.predict(features.loc[answered].to_numpy())
        np.testing.assert_allclose(parallel[col].to_numpy(), expected.to_numpy(), rtol=1e-6)
