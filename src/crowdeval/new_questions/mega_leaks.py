"""Guards that keep mega-study answers out of model inputs and stop on implausibly perfect scores."""

import numpy as np
import pandas as pd

LEAK_CORRELATION = 0.95
MIN_OVERLAP = 30
VARIANCE_TOLERANCE = 1e-9
SUSPICIOUS_ACCURACY = 0.99


class LeakError(ValueError):
    """A model input is, or copies, a prediction target."""


class SuspiciousScoreError(ValueError):
    """An approach is near-perfect where guessing the average is not."""


def _pairwise_correlations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|r| between y and each column of x on the rows both have, and how many rows that is."""
    present = ~np.isnan(x)
    n = present.sum(axis=0).astype(float)
    xs = np.where(present, x, 0.0)
    ys = np.where(present, y[:, None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx, sy = xs.sum(axis=0), ys.sum(axis=0)
        sxx, syy = (xs**2).sum(axis=0), (ys**2).sum(axis=0)
        cov = (xs * ys).sum(axis=0) - sx * sy / n
        var_x, var_y = sxx - sx**2 / n, syy - sy**2 / n
        constant = (var_x <= VARIANCE_TOLERANCE * np.maximum(sxx, 1.0)) | (var_y <= VARIANCE_TOLERANCE * np.maximum(syy, 1.0))
        r = np.abs(cov / np.sqrt(var_x * var_y))
    return np.where(constant, 0.0, np.nan_to_num(r, nan=0.0)), n


def _correlations(features: pd.DataFrame, targets: pd.DataFrame):
    """Yield (target, |r| per feature) on the people both cover, with too-small overlaps set to 0."""
    x_all = features.reindex(targets.index).to_numpy(dtype=float)
    for target in targets.columns:
        y = targets[target].to_numpy(dtype=float)
        rows = ~np.isnan(y)
        if rows.sum() < MIN_OVERLAP:
            continue
        r, n = _pairwise_correlations(x_all[rows], y[rows])
        yield target, np.where(n >= MIN_OVERLAP, r, 0.0)


def leaking_columns(features: pd.DataFrame, targets: pd.DataFrame) -> list[tuple[str, str, float]]:
    """(feature, target, |r|) for every feature named as a target or copying one on the people both cover."""
    found = [(c, c, 1.0) for c in sorted(set(features.columns) & set(targets.columns))]
    for target, r in _correlations(features, targets):
        for position in np.flatnonzero(r >= LEAK_CORRELATION):
            found.append((str(features.columns[position]), target, float(r[position])))
    return found


def largest_correlation(features: pd.DataFrame, targets: pd.DataFrame) -> tuple[str, str, float]:
    """The feature–target pair with the largest |r|, for the record."""
    best = ("", "", 0.0)
    for target, r in _correlations(features, targets):
        position = int(np.argmax(r))
        if r[position] > best[2]:
            best = (str(features.columns[position]), target, float(r[position]))
    return best


def assert_no_leak(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    found = leaking_columns(features, targets)
    if found:
        raise LeakError(f"{len(found)} feature/target pairs leak, e.g. {found[:5]}")


def assert_not_suspicious(per_outcome: pd.DataFrame, baseline: str) -> None:
    """per_outcome: outcomes × approaches accuracy. Near-perfect where the baseline is not means a bug until explained."""
    reference = per_outcome[baseline] < SUSPICIOUS_ACCURACY
    flagged = {
        approach: list(per_outcome.index[(per_outcome[approach] >= SUSPICIOUS_ACCURACY) & reference])
        for approach in per_outcome.columns
        if approach != baseline
    }
    flagged = {k: v for k, v in flagged.items() if v}
    if flagged:
        raise SuspiciousScoreError(f"near-perfect scores where the median is not: {flagged}")
