"""Which mega-study outcomes are scored, and on what scale.

Ranges come from the authors' post_metric_calculation/dv_variables_summary.csv
(TianyiPeng/Twin-2K-500-Mega-Study, Apache-2.0), copied into crowdeval.upstream.
"""

from importlib import resources

import numpy as np
import pandas as pd

RANGES_FILE = "mega_study_ranges.csv"
KEY_SEPARATOR = "/"
STUDY, LOW, HIGH = "study", "low", "high"

STUDY_FOLDERS = {
    "Accuracy Nudge": "accuracy_nudges",
    "Affective Primes": "affective_priming",
    "Consumer Minimalism": "consumer_minimalism",
    "context effects": "context_effects",
    "Defaults": "default_eric",
    "Digital certificates for luxury consumption": "digital_certification",
    "hiring algorithms": "hiring_algorithms",
    "idea evaluation": "idea_evaluation",
    "idea generation": "idea_generation",
    "infotainment news sharing": "infotainment",
    "Junk Fees": "junk_fees",
    "Obedient twins": "obedient_twins",
    "Preferences for redistribution": "preference_redistribution",
    "Privacy": "privacy",
    "Promiscuous donors": "promiscuous_donors",
    "quantitative intuition": "quantitative_intuition",
    "recommendation systems": "recommendation_algorithms",
    "Story Beliefs": "story_beliefs",
    "Targeting fairness": "targeting_fairness",
}

EXCLUDED = frozenset({
    "digital_certification/log_WTP",
    "idea_generation/DAT_perf",
    "idea_generation/SSPT",
    "idea_generation/creativity_rating_byhuman",
    "quantitative_intuition/scenario_open_QI",
})

SEVEN_POINT = (1.0, 7.0)
INFOTAINMENT_PROFILES = range(1, 19)
DONOR_RATINGS = range(1, 11)
DONOR_CODES = {1: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7}
RANGE_FIXES = {
    **{f"infotainment/Profile{i}": SEVEN_POINT for i in INFOTAINMENT_PROFILES},
    "digital_certification/DV1": SEVEN_POINT,
}
RECODES = {f"promiscuous_donors/rating_{i}_1": DONOR_CODES for i in DONOR_RATINGS}


def key(study: str, variable: str) -> str:
    return f"{study}{KEY_SEPARATOR}{variable}"


def study_of(outcome: str) -> str:
    return outcome.split(KEY_SEPARATOR, 1)[0]


def variable_of(outcome: str) -> str:
    return outcome.split(KEY_SEPARATOR, 1)[1]


def authors_ranges() -> pd.DataFrame:
    """All 164 outcomes with the authors' declared ranges (NaN where none), indexed by key."""
    raw = pd.read_csv(resources.files("crowdeval.upstream").joinpath(RANGES_FILE))
    folders = raw["study_name"].map(STUDY_FOLDERS)
    if folders.isna().any():
        raise ValueError(f"unknown study names: {sorted(raw.loc[folders.isna(), 'study_name'].unique())}")
    table = pd.DataFrame({STUDY: folders, LOW: raw["DV_min"], HIGH: raw["DV_max"]})
    table.index = [key(s, v) for s, v in zip(folders, raw["DV_variable"])]
    return table


def ranged(table: pd.DataFrame) -> pd.DataFrame:
    return table[table[LOW].notna() & table[HIGH].notna()]


def included() -> pd.DataFrame:
    """The outcomes of the primary analysis, with the questionnaire's ranges where the authors' differ."""
    table = ranged(authors_ranges())
    table = table[~table.index.isin(EXCLUDED)].copy()
    for outcome, (low, high) in RANGE_FIXES.items():
        table.loc[outcome, [LOW, HIGH]] = (low, high)
    table.loc[list(RECODES), [LOW, HIGH]] = SEVEN_POINT
    return table


def spans(table: pd.DataFrame, columns: pd.Index) -> np.ndarray:
    rows = table.loc[columns]
    return (rows[HIGH] - rows[LOW]).to_numpy(dtype=float)


def recode(frame: pd.DataFrame) -> pd.DataFrame:
    """Map stored answer codes to scale positions (humans and predictions alike); unknown codes are an error."""
    out = frame.copy()
    for column, codes in RECODES.items():
        if column not in out.columns:
            continue
        present = out[column].dropna()
        unknown = sorted(set(present) - set(codes))
        if unknown:
            raise ValueError(f"{column}: unexpected answer codes {unknown}")
        out[column] = out[column].map(codes)
    return out
