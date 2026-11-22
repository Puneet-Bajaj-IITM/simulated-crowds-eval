"""Population-level metrics: does the simulated crowd look like the real one?

Every metric is computed per column, then aggregated within each task, then
averaged across tasks, so each of the 17 tasks counts equally (as in the
authors' accuracy score).
"""

import numpy as np
import pandas as pd

from crowdeval import config
from crowdeval.core.scoring import COLUMN_RANGES, COLUMN_TASKS

MAX_CATEGORIES = 10


def bin_codes(values: np.ndarray, column: str) -> tuple[np.ndarray, int]:
    """Map answers to 0..k-1 categories (−1 for missing).

    The scheme depends only on the column's answer scale, so human and simulated
    answers always share bins: scales of up to 10 points use one category per
    point (non-integer predictions are rounded), wider scales use 10 equal bins.
    """
    low, high = COLUMN_RANGES[column]
    is_integer_scale = high - low + 1 <= MAX_CATEGORIES
    codes = np.full(values.shape, -1, dtype=np.int64)
    present = ~np.isnan(values)
    if is_integer_scale:
        k = int(high - low + 1)
        codes[present] = np.clip(np.round(values[present]) - low, 0, k - 1).astype(np.int64)
        return codes, k
    k = config.JSD_BINS
    scaled = (values[present] - low) / (high - low) * k
    codes[present] = np.clip(np.floor(scaled), 0, k - 1).astype(np.int64)
    return codes, k


def js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> np.ndarray:
    """Jensen–Shannon divergence (base 2, 0–1) between rows of two count matrices."""
    p = p_counts / np.clip(p_counts.sum(axis=-1, keepdims=True), 1, None)
    q = q_counts / np.clip(q_counts.sum(axis=-1, keepdims=True), 1, None)
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(a > 0, a * np.log2(a / b), 0.0)
        return terms.sum(axis=-1)

    return (kl(p, m) + kl(q, m)) / 2


def _task_average(per_column: pd.Series, how: str) -> float:
    tasks = pd.Series({c: COLUMN_TASKS[c] for c in per_column.index})
    grouped = per_column.groupby(tasks)
    by_task = grouped.median() if how == "median" else grouped.mean()
    return float(by_task.mean())


def population_metrics(pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame) -> dict[str, float]:
    """Mean error (human SD units), spread ratio, and JS divergence, task-averaged."""
    mean_error, spread, jsd = {}, {}, {}
    for col in mask.columns:
        keep = mask[col].to_numpy()
        human = truth[col].to_numpy(dtype=float)[keep]
        sim = pred[col].to_numpy(dtype=float)[keep]
        sd_human = human.std(ddof=1) if len(human) > 1 else 0.0
        if len(human) < 2 or sd_human == 0:
            continue
        mean_error[col] = abs(sim.mean() - human.mean()) / sd_human
        spread[col] = sim.std(ddof=1) / sd_human
        h_codes, k = bin_codes(human, col)
        s_codes, _ = bin_codes(sim, col)
        jsd[col] = float(js_divergence(np.bincount(h_codes, minlength=k), np.bincount(s_codes, minlength=k)))
    return {
        "mean_error_sd": _task_average(pd.Series(mean_error), "mean"),
        "spread_ratio": _task_average(pd.Series(spread), "median"),
        "js_divergence": _task_average(pd.Series(jsd), "mean"),
    }


def stability(samples: list[pd.DataFrame], truth: pd.DataFrame, mask: pd.DataFrame) -> dict[str, float]:
    """How much the population estimate moves between repeated samples.

    Per column: SD across samples of the simulated mean, divided by
    |average simulated mean − human mean| (above 1: resampling moves the answer
    more than its error), and the same SD in human-SD units. Median within
    task, then mean across tasks.
    """
    ratio, drift = {}, {}
    for col in mask.columns:
        keep = mask[col].to_numpy()
        human = truth[col].to_numpy(dtype=float)[keep]
        if len(human) < 2 or human.std(ddof=1) == 0:
            continue
        means = np.array([s[col].to_numpy(dtype=float)[keep].mean() for s in samples])
        between = means.std(ddof=1)
        gap = abs(means.mean() - human.mean())
        drift[col] = between / human.std(ddof=1)
        if gap > 0:
            ratio[col] = between / gap
    return {
        "stability_ratio": _task_average(pd.Series(ratio), "median"),
        "between_sample_sd": _task_average(pd.Series(drift), "median"),
    }


def gain_over_average(accuracy: float, baseline: float, ceiling: float) -> float:
    """Share of the person-specific signal recovered: 0 = no better than guessing the average, 1 = as good as the person themself."""
    return (accuracy - baseline) / (ceiling - baseline)
