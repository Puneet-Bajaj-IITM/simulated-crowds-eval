"""Non-LLM rows: guess the population answer, and a model trained on other people's behavior.

Both are cross-validated across people: every prediction for a person comes from
a model or statistic computed without that person.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from crowdeval import config, data

PID_COLUMN = "pid"
QUESTION_ID = re.compile(r"^(QID\d+)")
GBM_LOSS = "absolute_error"
GBM_MAX_ITER = 200
GBM_LEARNING_RATE = 0.05
MIN_TRAIN_ROWS = 30
MIN_FEATURE_VALUES = 2
NEIGHBOURS = 50
SPARE_CORES = 2
BEHAVIOR_JOBS = max(1, (os.cpu_count() or 1) - SPARE_CORES)


def person_folds(index: pd.Index) -> list[tuple[pd.Index, pd.Index]]:
    splitter = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    return [(index[tr], index[te]) for tr, te in splitter.split(index)]


def guess_the_average(truth: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each person gets the median answer of the people outside their fold."""
    out = pd.DataFrame(np.nan, index=truth.index, columns=columns)
    for train, test in person_folds(truth.index):
        medians = truth.loc[train, columns].median()
        out.loc[test, columns] = np.tile(medians.to_numpy(), (len(test), 1))
    return out


def base_question_id(question_id: str) -> str:
    """QID9_1 -> QID9; the mapping names some holdout items by sub-question."""
    match = QUESTION_ID.match(question_id)
    return match.group(1) if match else question_id


def holdout_question_ids() -> set[str]:
    mapping = json.loads(data.fetch(config.HOLDOUT_MAPPING_FILE).read_text())
    return {base_question_id(entry["QuestionID"]) for entry in mapping}


def without_holdout(columns: pd.Index, held_out: set[str]) -> list[str]:
    return [c for c in columns if (m := QUESTION_ID.match(c)) is None or m.group(1) not in held_out]


def behavior_features() -> pd.DataFrame:
    """Every numeric wave 1–3 answer except the holdout questions, indexed by person."""
    raw = pd.read_csv(data.fetch(config.HUMAN_WAVE1_3_FILE), low_memory=False)
    raw = raw.set_index(PID_COLUMN)
    keep = without_holdout(raw.columns, holdout_question_ids())
    numeric = raw[keep].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any()]
    numeric.index = numeric.index.astype(int)
    return numeric


def statistical_twin(truth: pd.DataFrame, columns: list[str], features: pd.DataFrame) -> pd.DataFrame:
    """Per person, the median answer of their nearest neighbours (from other folds) on wave 1–3 answers."""
    x_all = features.reindex(truth.index)
    out = pd.DataFrame(np.nan, index=truth.index, columns=columns)
    for train, test in person_folds(truth.index):
        x_train = x_all.loc[train]
        keep = x_train.columns[x_train.notna().any()]
        centre, scale = x_train[keep].mean(), x_train[keep].std(ddof=0).replace(0, 1).fillna(1)
        standardize = lambda frame: ((frame[keep] - centre) / scale).fillna(0.0).to_numpy()
        finder = NearestNeighbors(n_neighbors=NEIGHBOURS).fit(standardize(x_train))
        _, positions = finder.kneighbors(standardize(x_all.loc[test]))
        for col in columns:
            answers = truth.loc[train, col].to_numpy(dtype=float)[positions]
            answered = truth.loc[test, col].notna().to_numpy()
            with np.errstate(all="ignore"):
                medians = np.nanmedian(np.where(np.isnan(answers).all(axis=1, keepdims=True), 0, answers), axis=1)
            has_neighbour_answer = ~np.isnan(answers).all(axis=1)
            out.loc[test, col] = np.where(answered & has_neighbour_answer, medians, np.nan)
    return out


def _fit_predict(x: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, predict_rows: np.ndarray) -> np.ndarray:
    """One single-threaded model; runs in a worker process, so rows are passed as positions into shared x."""
    x_fit = x[fit_rows]
    usable = (~np.isnan(x_fit)).sum(axis=0) >= MIN_FEATURE_VALUES
    model = HistGradientBoostingRegressor(
        loss=GBM_LOSS, max_iter=GBM_MAX_ITER, learning_rate=GBM_LEARNING_RATE, random_state=config.RANDOM_SEED
    )
    model.fit(x_fit[:, usable], y[fit_rows])
    return model.predict(x[predict_rows][:, usable])


def _fit_all(truth: pd.DataFrame, features: pd.DataFrame, tasks: list[tuple[str, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Run (column, fit rows, predict rows) tasks in parallel, one thread per model."""
    x = features.reindex(truth.index).to_numpy(dtype=float)
    values = {col: truth[col].to_numpy(dtype=float) for col in {t[0] for t in tasks}}
    predictions = Parallel(n_jobs=BEHAVIOR_JOBS)(
        delayed(_fit_predict)(x, values[col], fit, predict) for col, fit, predict in tasks
    )
    out = pd.DataFrame(np.nan, index=truth.index, columns=list(dict.fromkeys(t[0] for t in tasks)))
    for (col, _, predict), predicted in zip(tasks, predictions):
        out.iloc[predict, out.columns.get_loc(col)] = predicted
    return out


def behavior_in_sample(truth: pd.DataFrame, columns: list[str], features: pd.DataFrame) -> pd.DataFrame:
    """The behavior model trained on everyone, predicting the same people it saw: its "recall"."""
    tasks = []
    for col in columns:
        answered = np.flatnonzero(truth[col].notna().to_numpy())
        if len(answered) >= MIN_TRAIN_ROWS:
            tasks.append((col, answered, answered))
    return _fit_all(truth, features, tasks).reindex(columns=columns)


def trained_on_behavior(truth: pd.DataFrame, columns: list[str], features: pd.DataFrame) -> pd.DataFrame:
    """One gradient-boosted model per holdout column and fold, predicting the conditional median."""
    position = pd.Series(np.arange(len(truth)), index=truth.index)
    tasks = []
    for train, test in person_folds(truth.index):
        for col in columns:
            answered = truth[col].notna()
            fit = position[train[answered.loc[train].to_numpy()]].to_numpy()
            predict = position[test[answered.loc[test].to_numpy()]].to_numpy()
            if len(fit) >= MIN_TRAIN_ROWS and len(predict):
                tasks.append((col, fit, predict))
    return _fit_all(truth, features, tasks).reindex(columns=columns)
