"""Each person's randomized condition for each mega-study outcome.

Ported from the condition definitions in the authors' mega_study_evaluation/<study>/
mega_study_evaluation.py (TianyiPeng/Twin-2K-500-Mega-Study, Apache-2.0). A condition
cell is one outcome within one condition level; outcomes without a condition are one cell.
"""

import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from crowdeval.new_questions import mega_outcomes

SINGLE_CELL = "all"

STUDY_CONDITION = {
    "accuracy_nudges": "Condition",
    "affective_priming": "cond",
    "digital_certification": "item",
    "idea_evaluation": "condition",
    "privacy": "Group",
    "targeting_fairness": "Segment",
}
OBEDIENT_TASKS = {2: "s_{}", 3: "absurd_{}"}
OBEDIENT_STORIES = (1, 2, 3)
OUTCOME_CONDITION = {
    f"obedient_twins/task {task} - S{story}": column.format(story)
    for task, column in OBEDIENT_TASKS.items()
    for story in OBEDIENT_STORIES
}

DEFAULTS_SHOWN = {
    "default_eric/target_behavior_green": ("OPTOUT_Green", "OPTIN_Green"),
    "default_eric/target_behavior_organ": ("OPTOUT_Organ", "OPTIN_Organ"),
}
OPT_OUT, OPT_IN = "opt-out", "opt-in"

HIRING_STUDY = "hiring_algorithms"
HIRING_JOB = re.compile(r"^job(\d)_item\d$")
ALGORITHM_FIRST = "FL_7_DO_AlgorithmicHiring"
TEAM_FIRST = "FL_7_DO_HiringTeam"
FLAG_SET = "1"
ALGORITHM, TEAM = "algo", "team"
JOBS_FOLLOWING_FLAG = frozenset({1, 3})

RawAnswers = Callable[[str], pd.DataFrame]


def _shown(column: pd.Series) -> np.ndarray:
    return (column.notna() & column.fillna("").str.strip().ne("")).to_numpy()


def _defaults(answers: pd.DataFrame, outcome: str) -> np.ndarray:
    opt_out, opt_in = DEFAULTS_SHOWN[outcome]
    return np.select([_shown(answers[opt_out]), _shown(answers[opt_in])], [OPT_OUT, OPT_IN], default=None)


def _hiring(answers: pd.DataFrame, job: int) -> np.ndarray:
    algorithm_first = (answers[ALGORITHM_FIRST] == FLAG_SET).to_numpy()
    team_first = (answers[TEAM_FIRST] == FLAG_SET).to_numpy()
    first, second = (ALGORITHM, TEAM) if job in JOBS_FOLLOWING_FLAG else (TEAM, ALGORITHM)
    return np.select([algorithm_first, team_first], [first, second], default=None)


def _hiring_job(outcome: str) -> int | None:
    match = HIRING_JOB.match(mega_outcomes.variable_of(outcome))
    return int(match.group(1)) if mega_outcomes.study_of(outcome) == HIRING_STUDY and match else None


def has_condition(outcome: str) -> bool:
    return (
        mega_outcomes.study_of(outcome) in STUDY_CONDITION
        or outcome in OUTCOME_CONDITION
        or outcome in DEFAULTS_SHOWN
        or _hiring_job(outcome) is not None
    )


def labels_for(outcome: str, answers: pd.DataFrame) -> np.ndarray:
    """Condition label per row of `answers` for one outcome that has a randomized condition."""
    study = mega_outcomes.study_of(outcome)
    if study in STUDY_CONDITION:
        return answers[STUDY_CONDITION[study]].to_numpy(dtype=object)
    if outcome in OUTCOME_CONDITION:
        return answers[OUTCOME_CONDITION[outcome]].to_numpy(dtype=object)
    if outcome in DEFAULTS_SHOWN:
        return _defaults(answers, outcome)
    job = _hiring_job(outcome)
    if job is None:
        raise KeyError(f"{outcome} has no randomized condition")
    return _hiring(answers, job)


def conditions(truth: pd.DataFrame, raw_answers: RawAnswers) -> pd.DataFrame:
    """People × outcomes: the condition label of every answered cell, NaN where the person did not answer."""
    out = pd.DataFrame(SINGLE_CELL, index=truth.index, columns=truth.columns, dtype=object)
    conditioned = [c for c in truth.columns if has_condition(c)]
    for study in sorted({mega_outcomes.study_of(c) for c in conditioned}):
        answers = raw_answers(study).reindex(truth.index)
        for outcome in (c for c in conditioned if mega_outcomes.study_of(c) == study):
            out[outcome] = labels_for(outcome, answers)
    out = out.where(truth.notna())
    unlabelled = truth.notna() & out.isna()
    if unlabelled.any().any():
        raise ValueError(f"answers without a condition: {unlabelled.sum()[unlabelled.sum() > 0].to_dict()}")
    return out


def two_level_outcomes(cells: pd.DataFrame) -> list[str]:
    return [c for c in cells.columns if cells[c].dropna().nunique() == 2]
