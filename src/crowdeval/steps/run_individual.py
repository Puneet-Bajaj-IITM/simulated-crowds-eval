"""Personal price model vs every other approach on the 40 pricing decisions."""

import numpy as np
import pandas as pd

from crowdeval import config, data
from crowdeval.core import information, panel, scoring
from crowdeval.llm import simulate
from crowdeval.models import baselines, individual_models, pricing
from crowdeval.steps.run_exploratory import OUT_DIR, STAT_TWIN, _runs_path, _samples, statistical_twin_predictions
from crowdeval.steps.run_panel_a import BEHAVIOR, BEHAVIOR_CACHE, PERSONAS as PERSONAS_A, TWINS as TWINS_A
from crowdeval.steps.run_panel_b import LABELS, load_replies

PERSONAL = "Personal price model"
POPULATION = "Price model, population only"
RETEST = "Humans (retest)"
INFORMATION = {
    panel.BASELINE: "other people's answers",
    POPULATION: "other people's pricing answers, product and price",
    PERSONAL: "as above, plus the person's own other pricing answers (32 of 40)",
    BEHAVIOR: "the person's wave 1–3 answers (holdout items excluded)",
    STAT_TWIN: "the person's wave 1–3 answers (holdout items excluded)",
    PERSONAS_A: "demographics",
    TWINS_A: "wave 1–3 persona text (holdout items excluded)",
    LABELS[simulate.PERSONAS]: "demographics",
    LABELS[simulate.TWIN]: "wave 1–3 persona text (holdout items excluded)",
    RETEST: "the same person answering again",
}
TABLE_COLUMNS = ("panel", "approach", "information", "samples", "people", "accuracy", "accuracy_lo", "accuracy_hi", "coverage")


def _person_accuracy(frames: list[pd.DataFrame], truth: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
    return np.mean([scoring.person_accuracy(f, truth, mask).reindex(truth.index).to_numpy() for f in frames], axis=0)


def _table(name: str, truth: pd.DataFrame, rows: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    columns = list(pricing.PRICING_COLUMNS)
    mask = scoring.valid_mask(columns, truth, *[f for frames in rows.values() for f in frames])
    scored = mask.any(axis=1)
    truth, mask = truth[scored], mask[scored]
    per_person = {k: _person_accuracy(v, truth, mask) for k, v in rows.items()}
    median, ceiling = per_person[panel.BASELINE].mean(), per_person[RETEST].mean()
    draws = np.random.default_rng(config.RANDOM_SEED).integers(0, len(truth), (config.BOOTSTRAP_SAMPLES, len(truth)))
    alpha = (1 - config.CONFIDENCE) / 2
    records = []
    for approach, values in per_person.items():
        lo, hi = np.quantile(values[draws].mean(axis=1), [alpha, 1 - alpha])
        records.append((name, approach, INFORMATION[approach], len(rows[approach]), len(truth), values.mean(), lo, hi,
                        information.coverage(values.mean(), median, ceiling)))
    return pd.DataFrame(records, columns=TABLE_COLUMNS)


def _non_llm_rows(truth_raw: pd.DataFrame, people: pd.Index, models: dict[str, pd.DataFrame], twin: pd.DataFrame) -> dict:
    columns = list(pricing.PRICING_COLUMNS)
    return {
        panel.BASELINE: [baselines.guess_the_average(truth_raw.loc[people], columns)],
        POPULATION: [models["population"].loc[people]],
        PERSONAL: [models["personal"].loc[people]],
        BEHAVIOR: [pd.read_parquet(BEHAVIOR_CACHE).loc[people, columns]],
        STAT_TWIN: [twin.loc[people, columns]],
    }


def build() -> pd.DataFrame:
    demographics = data.load_spec(config.SPEC_DEMOGRAPHICS)
    twins = data.load_spec(config.SPEC_TWIN_DEFAULT_TEMP)
    truth_raw = twins.wave1_3
    columns = list(pricing.PRICING_COLUMNS)
    codes, prices = pricing.offers()
    models = individual_models.price_models(truth_raw[columns], codes, prices)
    twin = statistical_twin_predictions(truth_raw, [c for c in truth_raw.columns if c in scoring.COLUMN_RANGES])

    everyone = truth_raw.index
    panel_a = _non_llm_rows(truth_raw, everyone, models, twin) | {
        PERSONAS_A: [demographics.simulated.loc[everyone, columns]],
        TWINS_A: [twins.simulated.loc[everyone, columns]],
        RETEST: [twins.wave4.loc[everyone, columns]],
    }
    main, _ = load_replies(_runs_path("main"))
    people = pd.Index(sorted(set.intersection(*(set(f.index) for f in main.values()))))
    panel_b = _non_llm_rows(truth_raw, people, models, twin) | {
        LABELS[c]: _samples(main, c, list(people), pd.Index(columns)) for c in simulate.CONDITIONS
    } | {RETEST: [twins.wave4.loc[people, columns]]}
    return pd.concat([
        _table("A", truth_raw.loc[everyone, columns], panel_a),
        _table("B", truth_raw.loc[people, columns], panel_b),
    ], ignore_index=True)


def markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Personal price model (exploratory)",
        "",
        "Accuracy on the 40 pricing decisions only (buy or not), by the main accuracy rule, so it equals the share of "
        "decisions predicted correctly. Brackets are person-bootstrap 95% intervals. Coverage is the share of the gap "
        "from guessing the average to the human retest that an approach closes. The personal model sees 32 of the "
        "person's own 40 pricing answers when predicting the other 8, so it has information no other row has.",
        "",
        "| Panel | Approach | Uses | Samples | Accuracy, % | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.panel} | {r.approach} | {r.information} | {r.samples} | "
                     f"{100 * r.accuracy:.1f} ({100 * r.accuracy_lo:.1f} to {100 * r.accuracy_hi:.1f}) | {r.coverage:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "personal_price.csv", index=False)
    (OUT_DIR / "personal_price.md").write_text(markdown(table))
    print(markdown(table))


if __name__ == "__main__":
    main()
