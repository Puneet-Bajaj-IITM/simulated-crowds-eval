"""Condition effects on the mega-study's two-level randomized outcomes.

Same statistics as effects.py: Cohen's d of level A minus level B (levels in sorted
order; direction agreement and size ratios do not depend on that order).
"""

import numpy as np
import pandas as pd

from crowdeval.core import effects


def effect_table(truth: pd.DataFrame, predictions: dict[str, pd.DataFrame], conditions: pd.DataFrame,
                 mask: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        labels = conditions[outcome].where(mask[outcome])
        level_a, level_b = sorted(labels.dropna().unique())
        in_a, in_b = labels == level_a, labels == level_b
        human = truth[outcome]
        d_h, se_h = effects.cohens_d(human[in_a], human[in_b])
        raw_h = effects.mean_difference(human[in_a], human[in_b])
        row = {
            "outcome": outcome,
            "level_a": level_a,
            "level_b": level_b,
            "n_a": int(in_a.sum()),
            "n_b": int(in_b.sum()),
            "human_d": d_h,
            "human_significant": bool(abs(d_h) > effects.Z_95 * se_h) if not np.isnan(d_h) else False,
        }
        for name, frame in predictions.items():
            values = frame[outcome]
            d_s, _ = effects.cohens_d(values[in_a], values[in_b])
            raw_s = effects.mean_difference(values[in_a], values[in_b])
            row[f"{name}_d"] = d_s
            row[f"{name}_same_direction"] = bool(np.sign(d_s) == np.sign(d_h)) if not np.isnan(d_s) else False
            row[f"{name}_size_ratio"] = d_s / d_h if d_h else np.nan
            row[f"{name}_raw_ratio"] = raw_s / raw_h if raw_h else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def effect_summary(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return effects.effect_summary(table, keys)
