"""Non-LLM rows for the mega-study, all cross-validated across people within condition cells.

Inputs are wave 1–3 answers only. The randomized condition only chooses which
cell's people a prediction is learned from; it is never a feature.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from crowdeval import config, data
from crowdeval.models import baselines
from crowdeval.new_questions import mega_leaks

Cell = tuple[str, np.ndarray]


def wave1_3_features() -> pd.DataFrame:
    """Every numeric wave 1–3 answer, indexed by person. No wave 1–3 question is a mega-study target.

    Free-text words that parse as numbers ("infinity") are not answers on a scale, so non-finite values are missing.
    """
    raw = pd.read_csv(data.fetch(config.HUMAN_WAVE1_3_FILE), low_memory=False).set_index(baselines.PID_COLUMN)
    numeric = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.notna().any()]
    numeric.index = numeric.index.astype(int)
    return numeric


def folds(index: pd.Index) -> np.ndarray:
    """Fold number of each person, from the plan's 5-fold split over everyone scored."""
    out = np.empty(len(index), dtype=int)
    position = pd.Series(np.arange(len(index)), index=index)
    for fold, (_, test) in enumerate(baselines.person_folds(index)):
        out[position[test].to_numpy()] = fold
    return out


def cells(truth: pd.DataFrame, conditions: pd.DataFrame) -> Iterator[Cell]:
    """(outcome, positions of the people who answered it in one condition level)."""
    for column in truth.columns:
        labels = conditions[column].to_numpy(dtype=object)
        answered = truth[column].notna().to_numpy()
        for label in pd.unique(labels[answered]):
            yield column, np.flatnonzero(answered & (labels == label))


def _splits(rows: np.ndarray, fold_of: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for fold in range(config.CV_FOLDS):
        held_out = fold_of[rows] == fold
        if held_out.any():
            yield rows[~held_out], rows[held_out]


def guess_the_average(truth: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """The median answer of other-fold people in the person's condition cell."""
    fold_of, out = folds(truth.index), np.full(truth.shape, np.nan)
    values = truth.to_numpy(dtype=float)
    for column, rows in cells(truth, conditions):
        j = truth.columns.get_loc(column)
        for fit, predict in _splits(rows, fold_of):
            out[predict, j] = np.median(values[fit, j]) if len(fit) else np.nan
    return pd.DataFrame(out, index=truth.index, columns=truth.columns)


def trained_on_behavior(truth: pd.DataFrame, conditions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One gradient-boosted model per condition cell and fold; small cells get the cell's median."""
    mega_leaks.assert_no_leak(features, truth)
    fold_of, values = folds(truth.index), truth.to_numpy(dtype=float)
    tasks, small = [], []
    for column, rows in cells(truth, conditions):
        for fit, predict in _splits(rows, fold_of):
            (tasks if len(fit) >= baselines.MIN_TRAIN_ROWS else small).append((column, fit, predict))
    out = baselines._fit_all(truth, features, tasks).reindex(columns=truth.columns)
    for column, fit, predict in small:
        j = truth.columns.get_loc(column)
        out.iloc[predict, j] = np.median(values[fit, j]) if len(fit) else np.nan
    return out


def _standardized(features: pd.DataFrame, fit_rows: np.ndarray) -> np.ndarray:
    fit = features.iloc[fit_rows]
    keep = fit.columns[fit.notna().any()]
    centre, scale = fit[keep].mean(), fit[keep].std(ddof=0).replace(0, 1).fillna(1)
    return ((features[keep] - centre) / scale).fillna(0.0).to_numpy()


def statistical_twin(truth: pd.DataFrame, conditions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """The median answer of the nearest other-fold people who answered the same outcome in the same cell."""
    mega_leaks.assert_no_leak(features, truth)
    x = features.reindex(truth.index)
    fold_of, values = folds(truth.index), truth.to_numpy(dtype=float)
    all_cells = list(cells(truth, conditions))
    out = np.full(truth.shape, np.nan)
    for fold in range(config.CV_FOLDS):
        z = _standardized(x, np.flatnonzero(fold_of != fold))
        for column, rows in all_cells:
            j = truth.columns.get_loc(column)
            fit, predict = rows[fold_of[rows] != fold], rows[fold_of[rows] == fold]
            if not len(fit) or not len(predict):
                continue
            finder = NearestNeighbors(n_neighbors=min(baselines.NEIGHBOURS, len(fit))).fit(z[fit])
            _, neighbours = finder.kneighbors(z[predict])
            out[predict, j] = np.median(values[fit, j][neighbours], axis=1)
    return pd.DataFrame(out, index=truth.index, columns=truth.columns)
