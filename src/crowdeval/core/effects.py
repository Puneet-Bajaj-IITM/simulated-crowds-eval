"""Between-subject bias experiments: does the simulator reproduce each effect's direction and size?

Each person saw one condition of each experiment (verified: no person answered
both columns of any pair). An effect is Cohen's d of condition A minus
condition B on the raw answer scale. Condition labels follow the authors'
questionnaire and evaluation/within_between_subjects.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_95 = 1.959964


@dataclass(frozen=True)
class Experiment:
    name: str
    column_a: str
    column_b: str


EXPERIMENTS = (
    Experiment("Base rate (70 vs. 30 engineers)", "Q156_1", "FORM A _1"),
    Experiment("Framing (gain vs. loss frame)", "Q157", "Q158"),
    Experiment("Conjunction / Linda (conjunction vs. single)", "Q160_3", "Q159_3"),
    Experiment("Outcome bias (success vs. failure)", "Q161", "Q162"),
    Experiment("Anchoring, African countries (high vs. low)", "Q166", "Q164"),
    Experiment("Anchoring, redwood (high vs. low)", "Q170", "Q168"),
    Experiment("Sunk cost (sunk vs. no sunk cost)", "Q182", "Q181"),
    Experiment("Absolute vs. relative savings (calculator vs. jacket)", "Q183", "Q184"),
    Experiment("WTA vs. WTP (certainty)", "Q190", "Q189"),
    Experiment("WTP certainty vs. non-certainty", "Q189", "Q191"),
    Experiment("Allais (form 1 vs. form 2)", "Q192", "Q193"),
    Experiment("Myside (Ford in Germany vs. German car in US)", "Q194", "Q195"),
)


def cohens_d(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Cohen's d (pooled SD) and its large-sample standard error."""
    a, b = a.dropna().to_numpy(float), b.dropna().to_numpy(float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan, np.nan
    d = (a.mean() - b.mean()) / pooled
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    return float(d), float(se)


def mean_difference(a: pd.Series, b: pd.Series) -> float:
    """Unstandardized effect: mean of condition A minus mean of condition B."""
    return float(a.dropna().mean() - b.dropna().mean())


def effect_table(frames: dict[str, pd.DataFrame], human_key: str) -> pd.DataFrame:
    """One row per experiment: human d (with significance) and each simulator's d and size ratio."""
    rows = []
    human = frames[human_key]
    for exp in EXPERIMENTS:
        if exp.column_a not in human.columns or exp.column_b not in human.columns:
            continue
        d_h, se_h = cohens_d(human[exp.column_a], human[exp.column_b])
        row = {
            "experiment": exp.name,
            "human_d": d_h,
            "human_significant": bool(abs(d_h) > Z_95 * se_h) if not np.isnan(d_h) else False,
        }
        raw_h = mean_difference(human[exp.column_a], human[exp.column_b])
        for key, frame in frames.items():
            if key == human_key:
                continue
            d_s, _ = cohens_d(frame[exp.column_a], frame[exp.column_b])
            row[f"{key}_d"] = d_s
            row[f"{key}_same_direction"] = bool(np.sign(d_s) == np.sign(d_h)) if not np.isnan(d_s) else False
            row[f"{key}_size_ratio"] = d_s / d_h if d_h else np.nan
            row[f"{key}_raw_ratio"] = mean_difference(frame[exp.column_a], frame[exp.column_b]) / raw_h if raw_h else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def effect_summary(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per simulator, over the experiments where the human effect is significant."""
    real = table[table["human_significant"]]
    return pd.DataFrame(
        {
            "approach": keys,
            "human_effects": [len(real)] * len(keys),
            "same_direction": [int(real[f"{k}_same_direction"].sum()) for k in keys],
            "median_size_ratio": [float(real[f"{k}_size_ratio"].median()) for k in keys],
            "median_raw_ratio_exploratory": [float(real[f"{k}_raw_ratio"].median()) for k in keys],
        }
    )
