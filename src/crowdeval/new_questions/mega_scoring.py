"""Scores on the mega-study outcomes, with a bootstrap over people.

Every estimate is a function of person weights: all ones for the point estimate,
multinomial resample counts for a bootstrap draw. Resampling a person moves all
of their answers together, and every approach shares the same draws.
"""

import numpy as np
import pandas as pd

from crowdeval import config
from crowdeval.new_questions import mega_outcomes

DISCRETE_TOLERANCE = 1e-9


def errors(pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame, outcomes: pd.DataFrame) -> np.ndarray:
    """|pred − truth| / range per answer, NaN outside the mask."""
    spans = mega_outcomes.spans(outcomes, truth.columns)
    diff = np.abs(pred[truth.columns].to_numpy(dtype=float) - truth.to_numpy(dtype=float)) / spans
    return np.where(mask.to_numpy(), diff, np.nan)


def is_discrete(values: np.ndarray) -> bool:
    present = values[~np.isnan(values)]
    return bool(np.all(np.abs(present - np.round(present)) < DISCRETE_TOLERANCE))


def chance_errors(truth: pd.DataFrame, mask: pd.DataFrame, outcomes: pd.DataFrame) -> np.ndarray:
    """Expected normalized error of a uniformly random answer: over integer points if humans answer in integers, else continuous."""
    out = np.full(truth.shape, np.nan)
    for j, column in enumerate(truth.columns):
        low, high = outcomes.loc[column, [mega_outcomes.LOW, mega_outcomes.HIGH]].astype(float)
        span, y = high - low, truth[column].to_numpy(dtype=float)
        if is_discrete(y):
            points = np.arange(low, high + 1)
            expected = np.abs(points[None, :] - y[:, None]).mean(axis=1) / span
        else:
            expected = ((y - low) ** 2 + (high - y) ** 2) / (2 * span**2)
        out[:, j] = np.where(mask[column].to_numpy(), expected, np.nan)
    return out


def study_groups(columns: pd.Index) -> list[np.ndarray]:
    studies = np.array([mega_outcomes.study_of(c) for c in columns])
    return [np.flatnonzero(studies == s) for s in sorted(set(studies))]


def bootstrap_weights(people: int, draws: int, seed: int = config.RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(people, np.full(people, 1 / people), size=draws).astype(float)


def point_weights(people: int) -> np.ndarray:
    return np.ones((1, people))


def outcome_means(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """draws × outcomes: weighted mean over the people who have a value."""
    present = ~np.isnan(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (weights @ np.where(present, values, 0.0)) / (weights @ present.astype(float))


def outcome_sds(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """draws × outcomes: the sample SD (ddof 1) of the resampled values."""
    present = ~np.isnan(values)
    filled = np.where(present, values, 0.0)
    n = weights @ present.astype(float)
    s1, s2 = weights @ filled, weights @ filled**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.clip((s2 - s1**2 / n) / (n - 1), 0.0, None))


def by_study(per_outcome: np.ndarray, groups: list[np.ndarray], how: str = "mean") -> np.ndarray:
    """draws × studies: mean (or median) over each study's outcomes."""
    reduce = np.nanmedian if how == "median" else np.nanmean
    return np.column_stack([reduce(per_outcome[:, g], axis=1) for g in groups])


def study_weighted(per_outcome: np.ndarray, groups: list[np.ndarray], how: str = "mean") -> np.ndarray:
    return np.nanmean(by_study(per_outcome, groups, how), axis=1)


def accuracy(cell_errors: np.ndarray, groups: list[np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    per_outcome = 1 - outcome_means(cell_errors, weights)
    return {"accuracy": study_weighted(per_outcome, groups), "accuracy_outcome_weighted": np.nanmean(per_outcome, axis=1)}


def population(pred: np.ndarray, truth: np.ndarray, groups: list[np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    """Spread ratio (median within study) and mean error in human SDs (mean within study)."""
    sd_h, sd_s = outcome_sds(truth, weights), outcome_sds(pred, weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        usable = sd_h > 0
        spread = np.where(usable, sd_s / sd_h, np.nan)
        gap = np.where(usable, np.abs(outcome_means(pred, weights) - outcome_means(truth, weights)) / sd_h, np.nan)
    return {"spread_ratio": study_weighted(spread, groups, "median"), "mean_error_sd": study_weighted(gap, groups)}


def rescaled(score: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Share of the gap from a reference score to perfect (1) that a score closes."""
    return (score - reference) / (1 - reference)
