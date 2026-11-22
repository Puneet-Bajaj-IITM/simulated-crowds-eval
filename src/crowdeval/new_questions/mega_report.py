"""Write the generalization results as CSV and Markdown."""

from pathlib import Path

import pandas as pd

from crowdeval import config

OUT_DIR = config.RESULTS_DIR / "generalization"
PERCENT = 100
MAIN_COLUMNS = (
    ("accuracy", "Accuracy", True),
    ("gain_points", "Gain over the median (points)", True),
    ("gain_share", "Gain (% of gap to perfect)", True),
    ("chance_corrected", "Chance-corrected", True),
    ("coverage", "Information coverage (vs own prior)", True),
    ("spread_ratio", "Spread vs humans", False),
    ("mean_error_sd", "Mean error (human SDs)", False),
)
WEIGHTING_COLUMNS = (("accuracy", "Each study counts equally", True), ("accuracy_outcome_weighted", "Each outcome counts equally", True))

HEADER = """# Generalization to new questions

**Exploratory.** The design (data, people, outcomes, approaches, metrics) was fixed before any score on these data was computed.

The questions are the dataset authors' follow-up mega-study (Peng et al., 2026, *Science Advances* 12(36), eaeh8260): new studies fielded on the Twin-2K-500 panel, none of which any participant had seen in waves 1–4. Source: Hugging Face `{dataset}` at revision `{revision}`. The LLM rows are the authors' released GPT-4.1 predictions; no fresh model calls were made.
"""

NOTES = """## Notes

- New questions have no human retest, so the ceiling is 1 (exact reproduction). Gains and coverage are shares of the gap to perfect, not to a retest ceiling as in Panels A and B, and are not comparable with those panels' values.
- Accuracy is 1 − |prediction − answer| / range, averaged over people within an outcome, then over outcomes within a study, then over studies. "Each outcome counts equally" is the paper's weighting.
- Guess the average, the behavior model and the statistical twin are learned within each randomized condition cell from other cross-validation folds. So they need real answers to each new question from other people (the other four folds, about 80% of each study's sample); the LLM rows use none. Their inputs about the person are wave 1–3 answers only; a guard checked that no input names or copies a target (|r| < 0.95 throughout).
- Own prior: the LLM prior for personas and twins, the median for the rest. The LLM prior's own row is measured against the median.
- Five of the authors' 164 outcomes are excluded (no range, or not a closed-ended answer), and three scales are corrected to the questionnaire.
- Intervals are bootstrap 95% percentile intervals over people ({draws} resamples; each person's answers move together; the same resamples for every approach). Effects are point estimates.
"""


def _fmt(value: float, scale: int) -> str:
    return "—" if pd.isna(value) else (f"{value * scale:.1f}" if scale == PERCENT else f"{value:.2f}")


def _cell(row: pd.Series, key: str, percent: bool) -> str:
    scale = PERCENT if percent else 1
    return f"{_fmt(row[key], scale)} ({_fmt(row[f'{key}_lo'], scale)} to {_fmt(row[f'{key}_hi'], scale)})"


def _table(frame: pd.DataFrame, columns: tuple) -> str:
    lines = ["| Approach | " + " | ".join(label for _, label, _ in columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for _, row in frame.iterrows():
        lines.append(f"| {row['approach']} | " + " | ".join(_cell(row, k, p) for k, _, p in columns) + " |")
    return "\n".join(lines)


def _percent_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    numeric = shown.select_dtypes("number").columns.difference(["outcomes", "people"])
    shown[numeric] = shown[numeric] * PERCENT
    return shown.to_markdown(floatfmt=".1f")


def write(table: pd.DataFrame, context: dict, per_study: pd.DataFrame, per_outcome: pd.DataFrame,
          effects: pd.DataFrame, effect_summary: pd.DataFrame, checks: dict[str, pd.DataFrame]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "generalization.csv", index=False)
    per_study.to_csv(OUT_DIR / "by_study.csv")
    per_outcome.to_csv(OUT_DIR / "by_outcome.csv", index_label="outcome")
    effects.to_csv(OUT_DIR / "effects.csv", index=False)
    effect_summary.to_csv(OUT_DIR / "effect_summary.csv", index=False)
    check, effects_check = checks["accuracy"], checks["effects"]
    check.to_csv(OUT_DIR / "pipeline_check.csv", index=False)
    matched = f"{effects_check['matching_authors'].iloc[0]} of {effects_check['outcomes'].iloc[0]}"
    scope = ", ".join(f"{k}: {v}" for k, v in context.items() if k != "chance")
    md = OUT_DIR / "generalization.md"
    md.write_text(
        HEADER.format(dataset=config.MEGA_DATASET, revision=config.MEGA_REVISION)
        + f"\n{scope}, chance accuracy: {context['chance'] * PERCENT:.1f}\n\n"
        + "## Accuracy and gain over the median\n\nPercentages, except spread and mean error. Brackets: bootstrap 95% intervals.\n\n"
        + _table(table, MAIN_COLUMNS)
        + "\n\n## Accuracy under the two weightings\n\n"
        + _table(table, WEIGHTING_COLUMNS)
        + "\n\n## Accuracy by study (point estimates, %)\n\n"
        + _percent_table(per_study)
        + "\n\n## Condition effects\n\nOver the two-level randomized outcomes whose human effect is significant (|d| > 1.96 SE).\n\n"
        + effect_summary.to_markdown(index=False, floatfmt=".2f")
        + "\n\n## Pipeline check\n\nOur per-outcome accuracy against the authors' released `meta analysis.csv`, on their ranges and stored codes. "
        "The paper's averages drop the outcome not released for every run (the creativity rating), so the published figure is compared with the average over the outcomes all four runs share.\n\n"
        + check.to_markdown(index=False, floatfmt=".4f")
        + f"\n\nCondition effects: our human and twin Cohen's d equal the authors' released `effect size based on human` and "
        f"`based on twin` (up to the order of levels) on {matched} two-level outcomes, which checks the ported condition definitions.\n\n"
        + NOTES.format(draws=context["bootstrap_draws"])
    )
    return md
