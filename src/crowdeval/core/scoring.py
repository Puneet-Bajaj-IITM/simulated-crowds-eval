"""Individual-level accuracy, scored exactly as the Twin-2K-500 authors score it.

Column ranges, task mapping and the anchoring decile transform are ported from
tianyipeng-lab/Digital-Twin-Simulation, evaluation/mad_accuracy_evaluation.py
(Apache-2.0). See NOTICE.
"""

import numpy as np
import pandas as pd

ANCHOR_GROUPS = (("Q164", "Q166"), ("Q168", "Q170"))
DECILE_STEPS = np.arange(10, 100, 10)
DECILE_COUNT = 10


def _column_ranges() -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    ranges.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in range(1, 11)})
    ranges.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)})
    ranges["Q156_1"] = (0, 100)
    ranges["FORM A _1"] = (0, 100)
    five_point = ["157", "158"] + [f"160_{i}" for i in (1, 2, 3)] + [f"159_{i}" for i in (1, 2, 3)]
    ranges.update({f"Q{c}": (1, 6) for c in five_point})
    ranges.update({f"Q{c}": (1, 7) for c in ("161", "162")})
    ranges.update({f"Q{c}": (1, 10) for c in ("164", "166", "168", "170")})
    ranges.update({f"Q{c}": (1, 5) for c in ("171", "172", "173", "174", "175", "176")})
    ranges.update({f"Q{c}": (1, 6) for c in ("177", "178", "179")})
    ranges.update({f"Q{c}": (0, 20) for c in ("181", "182")})
    ranges.update({f"Q{c}": (1, 2) for c in ("183", "184")})
    ranges.update({f"Q{c}": (1, 10) for c in ("189", "190", "191")})
    ranges.update({f"Q{c}": (1, 2) for c in ("192", "193")})
    ranges.update({f"Q{c}": (1, 6) for c in ("194", "195")})
    ranges.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    ranges.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})
    ranges.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1, 5)})
    ranges.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1, 5)})
    ranges["OMISSION BIAS "] = (1, 4)
    ranges["DENOMINATOR NEGLECT "] = (1, 2)
    ranges.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})
    return ranges


def _column_tasks() -> dict[str, str]:
    tasks = {
        **{f"False Cons. self _{i}": "false consensus" for i in range(1, 11)},
        **{f"False cons. others _{i}": "false consensus" for i in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)},
        "Q156_1": "base rate",
        "Form A _1": "base rate",
        "Q157": "framing problem",
        "Q158": "framing problem",
        **{f"Q160_{i}": "conjunction problem (Linda)" for i in (1, 2, 3)},
        **{f"Q159_{i}": "conjunction problem (Linda)" for i in (1, 2, 3)},
        "Q161": "outcome bias",
        "Q162": "outcome bias",
        **{f"Q{c}": "anchoring and adjustment" for c in ("164", "166", "168", "170")},
        **{f"Q17{i}": "less is more" for i in range(1, 10)},
        "Q181": "sunk cost fallacy",
        "Q182": "sunk cost fallacy",
        "Q183": "absolute vs. relative savings",
        "Q184": "absolute vs. relative savings",
        **{f"Q{c}": "WTA/WTP-Thaler" for c in ("189", "190", "191")},
        "Q192": "Allais",
        "Q193": "Allais",
        "Q194": "myside",
        "Q195": "myside",
        **{f"Q198_{i}": "prob matching vs. max" for i in range(1, 11)},
        **{f"Q203_{i}": "prob matching vs. max" for i in range(1, 7)},
        **{f"nonseparabilty bene _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        **{f"nonseparability ris _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        "Omission bias ": "omission",
        "Denominator neglect ": "denominator neglect",
        **{f"{i}_Q295": "pricing" for i in range(1, 41)},
    }
    return {k.upper(): v for k, v in tasks.items()}


COLUMN_RANGES = _column_ranges()
COLUMN_TASKS = _column_tasks()


def _decile(value: float, thresholds: np.ndarray) -> float:
    if pd.isna(value):
        return np.nan
    position = int(np.searchsorted(thresholds, value, side="left"))
    return float(min(position + 1, DECILE_COUNT))


def anchoring_thresholds(reference: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Decile cut points for each anchoring pair, from the human wave 1–3 answers."""
    out = {}
    for group in ANCHOR_GROUPS:
        values = pd.concat([reference[c] for c in group if c in reference.columns]).dropna()
        out[group] = np.percentile(values, DECILE_STEPS)
    return out


def to_scoring_scale(frame: pd.DataFrame, thresholds: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """Return a copy with anchoring answers replaced by their decile (1–10)."""
    out = frame.copy()
    for group, cuts in thresholds.items():
        for col in group:
            if col in out.columns:
                out[col] = out[col].map(lambda v, c=cuts: _decile(v, c))
    return out


def scored_columns(*frames: pd.DataFrame) -> list[str]:
    shared = set(COLUMN_RANGES)
    for frame in frames:
        shared &= set(frame.columns)
    return sorted(shared)


def valid_mask(columns: list[str], *frames: pd.DataFrame) -> pd.DataFrame:
    """True where every frame has an answer for that person and column."""
    mask = pd.DataFrame(True, index=frames[0].index, columns=columns)
    for frame in frames:
        mask &= frame[columns].notna()
    return mask


def normalized_errors(pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """|pred − truth| / range per cell, NaN outside the mask."""
    cols = list(mask.columns)
    spans = pd.Series({c: COLUMN_RANGES[c][1] - COLUMN_RANGES[c][0] for c in cols})
    errors = (pred[cols] - truth[cols]).abs().div(spans, axis=1)
    return errors.where(mask)


def accuracy_from_errors(errors: pd.DataFrame) -> pd.Series:
    """Per person: 1 − mean over tasks of the mean normalized error within each task."""
    tasks = pd.Series({c: COLUMN_TASKS[c] for c in errors.columns})
    by_task = errors.T.groupby(tasks).mean().T
    return 1 - by_task.mean(axis=1, skipna=True).dropna()


def person_accuracy(pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """Per person: the authors' task-weighted accuracy of `pred` against `truth` on the masked cells."""
    return accuracy_from_errors(normalized_errors(pred, truth, mask))
