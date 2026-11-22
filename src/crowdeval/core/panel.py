"""Build one results panel: every approach scored on the same people and the same answers.

An approach may have several repeated samples. Each sample is scored on its own
and the metric values are averaged; samples are never pooled into one prediction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from crowdeval import config
from crowdeval.core import metrics, scoring

BASELINE = "Guess the average"
CEILING = "Humans (retest)"
METRIC_KEYS = ("accuracy", "accuracy_pct_ceiling", "gain_over_average", "mean_error_sd", "spread_ratio", "js_divergence")

Samples = list[pd.DataFrame]


@dataclass(frozen=True)
class PanelInput:
    truth: pd.DataFrame
    retest: pd.DataFrame
    approaches: dict[str, pd.DataFrame | Samples]


def as_samples(value: pd.DataFrame | Samples) -> Samples:
    return value if isinstance(value, list) else [value]


def scoring_frames(inp: PanelInput) -> tuple[pd.DataFrame, dict[str, Samples], pd.DataFrame]:
    """Truth and all approaches on the scoring scale, plus the cells every frame answered."""
    cuts = scoring.anchoring_thresholds(inp.truth)
    truth = scoring.to_scoring_scale(inp.truth, cuts)
    rows = {k: [scoring.to_scoring_scale(f, cuts) for f in as_samples(v)] for k, v in inp.approaches.items()}
    rows[CEILING] = [scoring.to_scoring_scale(inp.retest, cuts)]
    every = [f for frames in rows.values() for f in frames]
    columns = scoring.scored_columns(truth, *every)
    mask = scoring.valid_mask(columns, truth, *every)
    return truth, rows, mask.loc[:, mask.any()]


def _estimates(truth: pd.DataFrame, rows: dict[str, Samples], mask: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for name, frames in rows.items():
        per_sample = []
        for frame in frames:
            values = {"accuracy": float(scoring.person_accuracy(frame, truth, mask).mean())}
            if name != CEILING:
                values.update(metrics.population_metrics(frame, truth, mask))
            per_sample.append(values)
        out[name] = {k: float(np.mean([v[k] for v in per_sample])) for k in per_sample[0]}
    ceiling, base = out[CEILING]["accuracy"], out[BASELINE]["accuracy"]
    for values in out.values():
        values["accuracy_pct_ceiling"] = values["accuracy"] / ceiling
        values["gain_over_average"] = metrics.gain_over_average(values["accuracy"], base, ceiling)
    return out


def _resample(frames: Samples, sample: np.ndarray, relabel: pd.Index) -> Samples:
    return [f.loc[sample].set_axis(relabel) for f in frames]


def _draw_estimates(truth: pd.DataFrame, rows: dict[str, Samples], mask: pd.DataFrame,
                    samples: list[np.ndarray]) -> list[dict[str, dict[str, float]]]:
    out = []
    for sample in samples:
        relabel = pd.RangeIndex(len(sample))
        out.append(_estimates(
            truth.loc[sample].set_axis(relabel),
            {n: _resample(f, sample, relabel) for n, f in rows.items()},
            mask.loc[sample].set_axis(relabel),
        ))
    return out


def build_panel(inp: PanelInput, bootstrap: int = config.BOOTSTRAP_SAMPLES, n_jobs: int = 1) -> pd.DataFrame:
    """Point estimates and bootstrap 95% intervals (people resampled with replacement).

    The resamples are drawn up front in a fixed order, so any n_jobs gives the same intervals.
    """
    truth, rows, mask = scoring_frames(inp)
    point = _estimates(truth, rows, mask)

    rng = np.random.default_rng(config.RANDOM_SEED)
    people = mask.index.to_numpy()
    samples = [rng.choice(people, size=len(people), replace=True) for _ in range(bootstrap)]
    chunks = [list(c) for c in np.array_split(np.arange(bootstrap), max(1, min(n_jobs, bootstrap)))]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_draw_estimates)(truth, rows, mask, [samples[i] for i in chunk]) for chunk in chunks if chunk
    )
    draws: dict[str, dict[str, list[float]]] = {n: {k: [] for k in METRIC_KEYS} for n in rows}
    for est in (e for part in results for e in part):
        for name, values in est.items():
            for key, value in values.items():
                draws[name][key].append(value)

    alpha = (1 - config.CONFIDENCE) / 2
    records = []
    for name, frames in rows.items():
        record = {"approach": name, "people": int(mask.any(axis=1).sum()), "samples": len(frames)}
        for key in METRIC_KEYS:
            if key not in point[name]:
                continue
            record[key] = point[name][key]
            record[f"{key}_lo"], record[f"{key}_hi"] = np.nanquantile(np.asarray(draws[name][key], float), [alpha, 1 - alpha])
        records.append(record)
    return pd.DataFrame(records)
