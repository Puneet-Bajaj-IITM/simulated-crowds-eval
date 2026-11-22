"""The generalization table: every approach scored on the same answers, with bootstrap intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crowdeval import config
from crowdeval.new_questions import mega_outcomes, mega_scoring as ms
from crowdeval.core.panel import BASELINE

METRICS = (
    "accuracy",
    "accuracy_outcome_weighted",
    "gain_points",
    "gain_share",
    "chance_corrected",
    "coverage",
    "spread_ratio",
    "mean_error_sd",
)


@dataclass(frozen=True)
class Inputs:
    truth: pd.DataFrame
    approaches: dict[str, pd.DataFrame]
    priors: dict[str, str]
    outcomes: pd.DataFrame


def scored_mask(truth: pd.DataFrame, approaches: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cells the human answered and every approach predicted."""
    mask = truth.notna()
    for frame in approaches.values():
        mask &= frame[truth.columns].notna()
    return mask


class Scorer:
    """Per-answer errors computed once; estimates for any set of person weights."""

    def __init__(self, inp: Inputs, mask: pd.DataFrame):
        self.inp, self.mask = inp, mask
        self.groups = ms.study_groups(inp.truth.columns)
        self.truth = inp.truth.where(mask).to_numpy(dtype=float)
        self.chance = ms.chance_errors(inp.truth, mask, inp.outcomes)
        self.errors = {k: ms.errors(f, inp.truth, mask, inp.outcomes) for k, f in inp.approaches.items()}
        self.values = {k: f[inp.truth.columns].where(mask).to_numpy(dtype=float) for k, f in inp.approaches.items()}

    def estimates(self, weights: np.ndarray) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
        chance = ms.accuracy(self.chance, self.groups, weights)["accuracy"]
        out = {}
        for name in self.inp.approaches:
            est = ms.accuracy(self.errors[name], self.groups, weights)
            est.update(ms.population(self.values[name], self.truth, self.groups, weights))
            out[name] = est
        base = out[BASELINE]["accuracy"]
        for name, est in out.items():
            est["gain_points"] = est["accuracy"] - base
            est["gain_share"] = ms.rescaled(est["accuracy"], base)
            est["chance_corrected"] = ms.rescaled(est["accuracy"], chance)
            est["coverage"] = ms.rescaled(est["accuracy"], out[self.inp.priors[name]]["accuracy"])
        return out, chance

    def per_outcome_accuracy(self) -> pd.DataFrame:
        ones = ms.point_weights(len(self.mask))
        return pd.DataFrame(
            {k: 1 - ms.outcome_means(e, ones)[0] for k, e in self.errors.items()}, index=self.inp.truth.columns
        )

    def per_study_accuracy(self) -> pd.DataFrame:
        ones = ms.point_weights(len(self.mask))
        studies = sorted({mega_outcomes.study_of(c) for c in self.inp.truth.columns})
        table = pd.DataFrame(
            {k: ms.by_study(1 - ms.outcome_means(e, ones), self.groups)[0] for k, e in self.errors.items()},
            index=pd.Index(studies, name="study"),
        )
        table.insert(0, "outcomes", [len(g) for g in self.groups])
        table.insert(1, "people", [int(self.mask.iloc[:, g].any(axis=1).sum()) for g in self.groups])
        return table


def build(scorer: Scorer, draws: int = config.BOOTSTRAP_SAMPLES) -> tuple[pd.DataFrame, dict]:
    """Point estimates and bootstrap 95% percentile intervals for every approach and metric."""
    mask, people = scorer.mask, len(scorer.mask)
    point, chance = scorer.estimates(ms.point_weights(people))
    boot, _ = scorer.estimates(ms.bootstrap_weights(people, draws))
    alpha = (1 - config.CONFIDENCE) / 2
    records = []
    for name in scorer.inp.approaches:
        record = {"approach": name, "own_prior": scorer.inp.priors[name]}
        for key in METRICS:
            record[key] = float(point[name][key][0])
            record[f"{key}_lo"], record[f"{key}_hi"] = np.nanquantile(boot[name][key], [alpha, 1 - alpha])
        records.append(record)
    context = {
        "people": int(mask.any(axis=1).sum()),
        "outcomes": int(mask.any(axis=0).sum()),
        "studies": len(scorer.groups),
        "answers": int(mask.to_numpy().sum()),
        "chance": float(chance[0]),
        "bootstrap_draws": draws,
    }
    return pd.DataFrame(records), context
