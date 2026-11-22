"""Rank agreement: does a simulator order people, and questions, the way humans do?."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crowdeval.core.scoring import COLUMN_TASKS

MIN_PAIRS = 10
MIN_ITEMS = 3


def _task_mean(values: pd.Series) -> float:
    tasks = pd.Series({c: COLUMN_TASKS[c] for c in values.index})
    return float(values.groupby(tasks).mean().mean())


def person_order(pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame) -> float:
    """Spearman's rho between simulated and human answers across people, per column, task-averaged."""
    rhos = {}
    for col in mask.columns:
        keep = mask[col].to_numpy()
        sim, human = pred[col].to_numpy(dtype=float)[keep], truth[col].to_numpy(dtype=float)[keep]
        if len(sim) < MIN_PAIRS or np.ptp(sim) == 0 or np.ptp(human) == 0:
            continue
        rhos[col] = spearmanr(sim, human).statistic
    return _task_mean(pd.Series(rhos)) if rhos else float("nan")


def item_order(pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame) -> float:
    """Spearman's rho between simulated and human column means within each task, averaged over tasks."""
    sim_means = pred[mask.columns].where(mask).mean()
    human_means = truth[mask.columns].where(mask).mean()
    tasks = pd.Series({c: COLUMN_TASKS[c] for c in mask.columns})
    rhos = []
    for _, cols in tasks.groupby(tasks).groups.items():
        sim, human = sim_means[cols].dropna(), human_means[cols].dropna()
        shared = sim.index.intersection(human.index)
        if len(shared) < MIN_ITEMS or np.ptp(sim[shared]) == 0 or np.ptp(human[shared]) == 0:
            continue
        rhos.append(spearmanr(sim[shared], human[shared]).statistic)
    return float(np.mean(rhos)) if rhos else float("nan")
