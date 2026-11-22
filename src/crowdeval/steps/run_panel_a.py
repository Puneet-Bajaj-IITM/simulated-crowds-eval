"""Panel A: the authors' released GPT-4.1-mini outputs, plus our two non-LLM rows, on all participants."""

import argparse

import pandas as pd

from crowdeval import config, data
from crowdeval.core import effects, panel, report
from crowdeval.models import baselines

PERSONAS = "Prompted personas (GPT-4.1-mini)"
BEHAVIOR = "Trained on behavior"
TWINS = "Digital twins (GPT-4.1-mini)"
HUMAN = "Humans"
PANEL_NAME = "panel_a"
BEHAVIOR_CACHE = config.DATA_DIR / "cache" / "behavior_predictions.parquet"


def _behavior_predictions(truth: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cross-validated behavior-model predictions, cached because training takes a while."""
    if BEHAVIOR_CACHE.exists():
        return pd.read_parquet(BEHAVIOR_CACHE).loc[truth.index, columns]
    predictions = baselines.trained_on_behavior(truth, columns, baselines.behavior_features())
    BEHAVIOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(BEHAVIOR_CACHE)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    demographics = data.load_spec(config.SPEC_DEMOGRAPHICS)
    twins = data.load_spec(config.SPEC_TWIN_DEFAULT_TEMP)
    people = demographics.wave1_3.index.intersection(twins.wave1_3.index)
    truth = twins.wave1_3.loc[people]
    retest = twins.wave4.loc[people]
    columns = [c for c in truth.columns if c in panel.scoring.COLUMN_RANGES]

    approaches = {
        panel.BASELINE: baselines.guess_the_average(truth, columns),
        PERSONAS: demographics.simulated.loc[people],
        BEHAVIOR: _behavior_predictions(truth, columns),
        TWINS: twins.simulated.loc[people],
    }
    result = panel.build_panel(panel.PanelInput(truth, retest, approaches), bootstrap=args.bootstrap,
                               n_jobs=baselines.BEHAVIOR_JOBS)

    sims = [PERSONAS, BEHAVIOR, TWINS]
    effect_rows = effects.effect_table({HUMAN: truth, **{k: approaches[k] for k in sims}}, HUMAN)
    summary = effects.effect_summary(effect_rows, sims)
    print(report.write(PANEL_NAME, result, effect_rows, summary).read_text())


if __name__ == "__main__":
    main()
