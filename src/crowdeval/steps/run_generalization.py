"""Generalization to new questions: the authors' mega-study on the same panel. Exploratory."""

import argparse

import pandas as pd

from crowdeval import config
from crowdeval.core.panel import BASELINE
from crowdeval.new_questions import (
    mega_baselines,
    mega_check,
    mega_conditions,
    mega_data,
    mega_effects,
    mega_leaks,
    mega_outcomes,
    mega_panel,
    mega_report,
)

PRIOR = "LLM prior (GPT-4.1, empty persona)"
PERSONAS = "Prompted personas (GPT-4.1)"
BEHAVIOR = "Trained on behavior"
STAT_TWIN = "Statistical twin (nearest neighbours)"
TWINS = "Digital twins (GPT-4.1)"
TWINS_T0 = "Digital twins (GPT-4.1, temperature 0)"
RELEASED = {
    PRIOR: mega_data.PRIOR,
    PERSONAS: mega_data.PERSONAS,
    TWINS: mega_data.TWIN,
    TWINS_T0: mega_data.TWIN_TEMPERATURE_ZERO,
}
OWN_PRIOR = {
    BASELINE: BASELINE,
    PRIOR: BASELINE,
    PERSONAS: PRIOR,
    BEHAVIOR: BASELINE,
    STAT_TWIN: BASELINE,
    TWINS: PRIOR,
    TWINS_T0: PRIOR,
}
ORDER = (BASELINE, PRIOR, PERSONAS, BEHAVIOR, STAT_TWIN, TWINS, TWINS_T0)
EFFECT_ROWS = (PRIOR, PERSONAS, BEHAVIOR, STAT_TWIN, TWINS, TWINS_T0)


def approaches(released: mega_data.Released, conditions: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    truth = released.truth
    rows = {BASELINE: mega_baselines.guess_the_average(truth, conditions)}
    rows.update({label: released.models[run] for label, run in RELEASED.items()})
    rows[BEHAVIOR] = mega_baselines.trained_on_behavior(truth, conditions, features)
    rows[STAT_TWIN] = mega_baselines.statistical_twin(truth, conditions, features)
    return {name: rows[name] for name in ORDER}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    runs = tuple(RELEASED.values())
    check = mega_check.pipeline_check(runs)
    if not (check["matching_authors"] == check["outcomes"]).all():
        raise RuntimeError(f"pipeline check failed:\n{check}")

    outcomes = mega_outcomes.included()
    released = mega_data.load_released(outcomes, runs)
    conditions = mega_conditions.conditions(released.truth, mega_data.raw_answers)
    features = mega_baselines.wave1_3_features()
    rows = approaches(released, conditions, features)

    mask = mega_panel.scored_mask(released.truth, rows)
    scorer = mega_panel.Scorer(mega_panel.Inputs(released.truth, rows, OWN_PRIOR, outcomes), mask)
    per_outcome = scorer.per_outcome_accuracy()
    mega_leaks.assert_not_suspicious(per_outcome, BASELINE)
    table, context = mega_panel.build(scorer, args.bootstrap)
    feature, target, r = mega_leaks.largest_correlation(features, released.truth)
    context["largest_feature_target_r"] = f"{r:.3f} ({feature} vs {target})"

    two_level = mega_conditions.two_level_outcomes(conditions.where(mask))
    effect_rows = mega_effects.effect_table(
        released.truth, {k: rows[k] for k in EFFECT_ROWS}, conditions, mask, two_level
    )
    summary = mega_effects.effect_summary(effect_rows, list(EFFECT_ROWS))
    effects_check = mega_check.effect_check(effect_rows, f"{TWINS}_d")
    if not (effects_check["matching_authors"] == effects_check["outcomes"]).all():
        raise RuntimeError(f"condition effects differ from the authors' release:\n{effects_check}")
    checks = {"accuracy": check, "effects": effects_check}
    path = mega_report.write(table, context, scorer.per_study_accuracy(), per_outcome, effect_rows, summary, checks)
    print(path.read_text())


if __name__ == "__main__":
    main()
