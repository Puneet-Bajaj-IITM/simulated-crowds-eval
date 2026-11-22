"""Chance and prior corrections, information coverage, and the two F1 scores.

All scores use the main accuracy rule (scoring.py). "Coverage" rescales a score so that
0 = a reference point (chance, or a method's own prior) and 1 = the ceiling.
"""

import numpy as np
import pandas as pd

from crowdeval.core import scoring

DEPARTURE_SHARE = 0.05
CHANCE_REPEATS = 20


def scale_points(column: str) -> np.ndarray:
    low, high = scoring.COLUMN_RANGES[column]
    return np.arange(low, high + 1, dtype=float)


def chance_errors(truth: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Expected normalized error of a uniformly random answer on each cell's scale."""
    out = pd.DataFrame(np.nan, index=mask.index, columns=mask.columns)
    for col in mask.columns:
        points = scale_points(col)
        span = points[-1] - points[0]
        values = truth[col].to_numpy(dtype=float)
        expected = np.abs(points[None, :] - values[:, None]).mean(axis=1) / span
        out[col] = np.where(mask[col].to_numpy(), expected, np.nan)
    return out


def chance_accuracy(truth: pd.DataFrame, mask: pd.DataFrame) -> float:
    return float(scoring.accuracy_from_errors(chance_errors(truth, mask)).mean())


def coverage(score: float, reference: float, ceiling: float) -> float:
    """Share of the gap from a reference point to the ceiling that a score closes."""
    return (score - reference) / (ceiling - reference)


def harmonic_f1(a: float, b: float) -> float:
    if not (a > 0 and b > 0):
        return 0.0
    return 2 * a * b / (a + b)


def _spans(columns: pd.Index) -> np.ndarray:
    return np.array([scoring.COLUMN_RANGES[c][1] - scoring.COLUMN_RANGES[c][0] for c in columns], dtype=float)


def _hits(pred: np.ndarray, prior: np.ndarray, truth: np.ndarray, spans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    departs = np.abs(pred - prior) / spans >= DEPARTURE_SHARE
    better = np.abs(pred - truth) < np.abs(prior - truth)
    return departs, departs & better


def _f1(hits: np.ndarray, departs: np.ndarray, person_departs: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    method_departures = (departs & valid).sum()
    person_departures = (person_departs & valid).sum()
    precision = (hits & valid).sum() / method_departures if method_departures else 0.0
    recall = (hits & person_departs & valid).sum() / person_departures if person_departures else 0.0
    return float(precision), float(recall), harmonic_f1(precision, recall)


def _chance_prediction(departs: np.ndarray, pred: np.ndarray, prior: np.ndarray, lows: np.ndarray,
                       highs: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Depart on as many randomly chosen cells per column, with the method's departure sizes and random signs."""
    out = prior.copy()
    for j in range(pred.shape[1]):
        cells = np.flatnonzero(valid[:, j])
        chosen_rows = cells[departs[cells, j]]
        if len(chosen_rows) == 0:
            continue
        sizes = np.abs(pred[chosen_rows, j] - prior[chosen_rows, j])
        targets = rng.choice(cells, size=len(chosen_rows), replace=False)
        signs = rng.choice([-1.0, 1.0], size=len(targets))
        up, down = prior[targets, j] + sizes, prior[targets, j] - sizes
        candidate = np.where(signs > 0, up, down)
        fallback = np.where(signs > 0, down, up)
        inside = (candidate >= lows[j]) & (candidate <= highs[j])
        out[targets, j] = np.clip(np.where(inside, candidate, fallback), lows[j], highs[j])
    return out


def individuality(pred: pd.DataFrame, prior: pd.DataFrame, truth: pd.DataFrame, data_prior: pd.DataFrame,
                  mask: pd.DataFrame, seed: int) -> dict[str, float]:
    """Precision/recall of departures from the prior, and F1 corrected against a matched random predictor."""
    cols = mask.columns
    spans = _spans(cols)
    lows = np.array([scoring.COLUMN_RANGES[c][0] for c in cols], dtype=float)
    highs = np.array([scoring.COLUMN_RANGES[c][1] for c in cols], dtype=float)
    p, q, t, m = (f[cols].to_numpy(dtype=float) for f in (pred, prior, truth, data_prior))
    valid = mask.to_numpy() & ~np.isnan(q) & ~np.isnan(m)
    person_departs = np.abs(t - m) / spans >= DEPARTURE_SHARE
    departs, hits = _hits(p, q, t, spans)
    precision, recall, f1 = _f1(hits, departs, person_departs, valid)
    rng = np.random.default_rng(seed)
    chance = []
    for _ in range(CHANCE_REPEATS):
        random_pred = _chance_prediction(departs & valid, p, q, lows, highs, valid, rng)
        r_departs, r_hits = _hits(random_pred, q, t, spans)
        chance.append(_f1(r_hits, r_departs, person_departs, valid)[2])
    f1_chance = float(np.mean(chance))
    kappa = (f1 - f1_chance) / (1 - f1_chance) if f1_chance < 1 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "f1_chance": f1_chance, "f1_corrected": kappa}
