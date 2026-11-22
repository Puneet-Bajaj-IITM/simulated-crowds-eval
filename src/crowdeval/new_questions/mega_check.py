"""Pipeline check: reproduce the authors' per-outcome accuracy and the published averages.

Uses the authors' ranges and stored answer codes, on every ranged outcome a run has,
exactly as their scripts score it: 1 − mean |human − model| / range over the pairs.
"""

import functools

import numpy as np
import pandas as pd

from crowdeval.new_questions import mega_data, mega_outcomes

PUBLISHED_ACCURACY = {mega_data.TWIN: 0.748, mega_data.PRIOR: 0.734, mega_data.TWIN_TEMPERATURE_ZERO: 0.752}
PUBLISHED_DECIMALS = 3
AUTHORS_VARIABLE = "variable name"
AUTHORS_ACCURACY = "accuracy between humans vs. their twins"
AUTHORS_HUMAN_D = "effect size based on human"
AUTHORS_TWIN_D = "effect size based on twin"
MATCH_TOLERANCE = 1e-9


def reproduced_accuracy(run: str) -> pd.Series:
    table = mega_outcomes.ranged(mega_outcomes.authors_ranges())
    studies = sorted(table[mega_outcomes.STUDY].unique())
    human = mega_data.outcome_values(run, mega_data.HUMAN, studies)
    model = mega_data.outcome_values(run, mega_data.MODEL, studies).reindex(human.index)
    out = {}
    for outcome in table.index.intersection(human.columns).intersection(model.columns):
        pair = human[outcome].notna() & model[outcome].notna()
        if pair.any():
            span = table.loc[outcome, mega_outcomes.HIGH] - table.loc[outcome, mega_outcomes.LOW]
            out[outcome] = 1 - (human.loc[pair, outcome] - model.loc[pair, outcome]).abs().mean() / span
    return pd.Series(out, dtype=float)


def released_accuracy(run: str, studies: list[str]) -> pd.Series:
    frames = []
    for study in studies:
        summary = mega_data.read_released(study, run, mega_data.AUTHORS_SUMMARY_FILE)
        keys = [mega_outcomes.key(study, v) for v in summary[AUTHORS_VARIABLE]]
        frames.append(pd.Series(summary[AUTHORS_ACCURACY].to_numpy(dtype=float), index=keys))
    return pd.concat(frames)


def effect_check(effects: pd.DataFrame, twin_column: str) -> pd.DataFrame:
    """Our human and twin Cohen's d against the authors' released ones (their condition code), up to level order."""
    released = []
    for study in sorted({mega_outcomes.study_of(o) for o in effects["outcome"]}):
        summary = mega_data.read_released(study, mega_data.TWIN, mega_data.AUTHORS_SUMMARY_FILE)
        summary.index = [mega_outcomes.key(study, v) for v in summary[AUTHORS_VARIABLE]]
        released.append(summary[[AUTHORS_HUMAN_D, AUTHORS_TWIN_D]])
    theirs = pd.concat(released).reindex(effects["outcome"])
    ours_h, ours_t = effects["human_d"].to_numpy(), effects[twin_column].to_numpy()
    theirs_h, theirs_t = theirs[AUTHORS_HUMAN_D].to_numpy(), theirs[AUTHORS_TWIN_D].to_numpy()
    same_size = (np.abs(np.abs(ours_h) - np.abs(theirs_h)) <= MATCH_TOLERANCE) & (
        np.abs(np.abs(ours_t) - np.abs(theirs_t)) <= MATCH_TOLERANCE)
    same_relative_sign = np.sign(ours_h * ours_t) == np.sign(theirs_h * theirs_t)
    return pd.DataFrame([{"outcomes": len(effects), "matching_authors": int((same_size & same_relative_sign).sum())}])


def pipeline_check(runs: tuple[str, ...]) -> pd.DataFrame:
    """Per run: agreement with the authors' per-outcome accuracy, and the outcome average two ways.

    The paper's comparisons drop outcomes not available for every run (the creativity
    rating), so the published averages are over the outcomes every run shares.
    """
    accuracy = {run: reproduced_accuracy(run) for run in runs}
    shared = functools.reduce(pd.Index.intersection, [s.index for s in accuracy.values()])
    rows = []
    for run, ours in accuracy.items():
        theirs = released_accuracy(run, sorted({mega_outcomes.study_of(k) for k in ours.index})).reindex(ours.index)
        difference = (ours - theirs).abs()
        published = PUBLISHED_ACCURACY.get(run, np.nan)
        mean_shared = float(ours[shared].mean())
        rows.append({
            "run": run,
            "outcomes": len(ours),
            "matching_authors": int((difference <= MATCH_TOLERANCE).sum()),
            "max_abs_difference": float(difference.max()),
            "mean_accuracy_all_outcomes": float(ours.mean()),
            "shared_outcomes": len(shared),
            "mean_accuracy_shared_outcomes": mean_shared,
            "published": published,
            "reproduces_published": bool(round(mean_shared, PUBLISHED_DECIMALS) == published) if not np.isnan(published) else None,
        })
    return pd.DataFrame(rows)
