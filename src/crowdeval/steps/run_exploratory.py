"""Exploratory core table: chance/prior-corrected scores, information coverage,
rank agreement, individuality precision/recall/F1, memory vs generalization F1, and the memory check."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crowdeval import config, data
from crowdeval.core import information, panel, ranks, scoring
from crowdeval.llm import simulate
from crowdeval.models import baselines
from crowdeval.llm.gemini_run import API_MODEL, RUNS_DIR
from crowdeval.steps.run_panel_a import BEHAVIOR, BEHAVIOR_CACHE, PERSONAS as PERSONAS_A, TWINS as TWINS_A
from crowdeval.steps.run_panel_b import LABELS, PUBLISHED_REPLIES, load_replies

OUT_DIR = config.RESULTS_DIR / "exploratory"
IN_SAMPLE_CACHE = config.DATA_DIR / "cache" / "behavior_in_sample.parquet"
STAT_TWIN_CACHE = config.DATA_DIR / "cache" / "statistical_twin.parquet"
STAT_TWIN = "Statistical twin (nearest neighbours)"
LLM_PRIOR = f"LLM prior ({API_MODEL})"
ID_ONLY = f"ID only ({API_MODEL})"
BOOTSTRAP = 500
PERFECT = 1.0


def _runs_path(name: str) -> Path:
    local = RUNS_DIR / f"replies_{name}.jsonl"
    published = config.RESULTS_DIR / "replies" / f"{API_MODEL}_{name}.jsonl.gz"
    return local if local.exists() else published


def _samples(frames: dict, condition: str, people: list[int], columns: pd.Index) -> list[pd.DataFrame]:
    keys = sorted(k for k in frames if k[0] == condition)
    return [frames[k].reindex(index=people, columns=columns) for k in keys]


def _mean_accuracy(frames: list[pd.DataFrame], truth: pd.DataFrame, mask: pd.DataFrame) -> float:
    return float(np.mean([scoring.person_accuracy(f, truth, mask).mean() for f in frames]))


def _scaled(frames: dict[str, list[pd.DataFrame]], truth: pd.DataFrame, extra: list[pd.DataFrame]):
    cuts = scoring.anchoring_thresholds(truth)
    scale = lambda f: scoring.to_scoring_scale(f, cuts)
    return scale(truth), {k: [scale(f) for f in v] for k, v in frames.items()}, [scale(f) for f in extra]


def _in_sample(truth: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if IN_SAMPLE_CACHE.exists():
        return pd.read_parquet(IN_SAMPLE_CACHE).reindex(index=truth.index, columns=columns)
    predictions = baselines.behavior_in_sample(truth, columns, baselines.behavior_features())
    IN_SAMPLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(IN_SAMPLE_CACHE)
    return predictions


def statistical_twin_predictions(truth: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if STAT_TWIN_CACHE.exists():
        return pd.read_parquet(STAT_TWIN_CACHE).reindex(index=truth.index, columns=columns)
    predictions = baselines.statistical_twin(truth, columns, baselines.behavior_features())
    STAT_TWIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(STAT_TWIN_CACHE)
    return predictions


def _row(name, frames, truth, mask, chance, ceiling, prior_acc, prior_frames, data_prior, seed):
    acc = _mean_accuracy(frames, truth, mask)
    indiv = [information.individuality(f, p, truth, data_prior, mask, seed + i) for i, (f, p) in
             enumerate(zip(frames, prior_frames))]
    return {
        "approach": name,
        "accuracy": acc,
        "chance_corrected": information.coverage(acc, chance, ceiling),
        "own_prior_accuracy": prior_acc,
        "information_coverage": information.coverage(acc, prior_acc, ceiling),
        "person_order_rho": float(np.nanmean([ranks.person_order(f, truth, mask) for f in frames])),
        "item_order_rho": float(np.nanmean([ranks.item_order(f, truth, mask) for f in frames])),
        **{f"individuality_{k}": float(np.mean([d[k] for d in indiv])) for k in indiv[0]},
    }


def _memory_generalization(recall_acc, recall_prior_acc, coverage_value) -> dict[str, float]:
    recall = information.coverage(recall_acc, recall_prior_acc, PERFECT)
    return {"memory_R": recall, "generalization_G": coverage_value,
            "memory_generalization_F1": information.harmonic_f1(recall, coverage_value)}


def _paired_gap(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Mean of a − b over people, with a bootstrap 95% interval."""
    diff = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    draws = [rng.choice(diff, size=len(diff)).mean() for _ in range(BOOTSTRAP)]
    return float(diff.mean()), *np.quantile(draws, [0.025, 0.975])


def panel_b() -> tuple[pd.DataFrame, dict]:
    main, _ = load_replies(_runs_path("main"))
    prior, _ = load_replies(_runs_path("prior"))
    recall, _ = load_replies(_runs_path("recall"))
    id_only, _ = load_replies(_runs_path("id_only"))
    human = data.load_spec(config.SPEC_TWIN_DEFAULT_TEMP)
    people = sorted(set.intersection(*(set(f.index) for f in [*main.values(), *prior.values()])))
    truth_raw, retest_raw = human.wave1_3.loc[people], human.wave4.loc[people]
    cols = truth_raw.columns
    columns = [c for c in cols if c in scoring.COLUMN_RANGES]
    llm = {LABELS[c]: _samples(main, c, people, cols) for c in simulate.CONDITIONS}
    groups = {
        panel.BASELINE: [baselines.guess_the_average(truth_raw, columns).reindex(columns=cols)],
        **llm,
        BEHAVIOR: [pd.read_parquet(BEHAVIOR_CACHE).reindex(index=people, columns=cols)],
        STAT_TWIN: [statistical_twin_predictions(human.wave1_3, columns).loc[people].reindex(columns=cols)],
        LLM_PRIOR: _samples(prior, simulate.PRIOR, people, cols),
        ID_ONLY: _samples(id_only, simulate.ID_ONLY, people, cols),
        "retest": [retest_raw],
    }
    in_sample_raw = _in_sample(human.wave1_3, columns).loc[people].reindex(columns=cols)
    recall_raw = _samples(recall, simulate.RECALL, people, cols)
    truth, g, (in_sample, wave4) = _scaled(groups, truth_raw, [in_sample_raw, retest_raw])
    recall_frames = [scoring.to_scoring_scale(f, scoring.anchoring_thresholds(truth_raw)) for f in recall_raw]
    every = [f for v in g.values() for f in v]
    mask = scoring.valid_mask(scoring.scored_columns(truth, *every), truth, *every)
    mask = mask.loc[:, mask.any()]

    ceiling = _mean_accuracy(g["retest"], truth, mask)
    chance = information.chance_accuracy(truth, mask)
    median = g[panel.BASELINE]
    median_acc = _mean_accuracy(median, truth, mask)
    llm_prior_acc = _mean_accuracy(g[LLM_PRIOR], truth, mask)
    rows = [
        _row(panel.BASELINE, median, truth, mask, chance, ceiling, median_acc, median, median[0], 1),
        _row(LLM_PRIOR, g[LLM_PRIOR], truth, mask, chance, ceiling, median_acc, median * 3, median[0], 11),
        _row(ID_ONLY, g[ID_ONLY], truth, mask, chance, ceiling, llm_prior_acc, g[LLM_PRIOR][:1], median[0], 21),
    ]
    for name in llm:
        rows.append(_row(name, g[name], truth, mask, chance, ceiling, llm_prior_acc, g[LLM_PRIOR], median[0], 31))
    rows.append(_row(BEHAVIOR, g[BEHAVIOR], truth, mask, chance, ceiling, median_acc, median, median[0], 41))
    rows.append(_row(STAT_TWIN, g[STAT_TWIN], truth, mask, chance, ceiling, median_acc, median, median[0], 45))
    table = pd.DataFrame(rows)

    recall_mask = scoring.valid_mask(list(mask.columns), wave4, recall_frames[0], *g[LLM_PRIOR])
    twin = LABELS[simulate.TWIN]
    twin_cov = table.loc[table.approach == twin, "information_coverage"].item()
    memory = {twin: _memory_generalization(
        _mean_accuracy(recall_frames, wave4, recall_mask), _mean_accuracy(g[LLM_PRIOR], wave4, recall_mask), twin_cov)}
    behavior_cov = table.loc[table.approach == BEHAVIOR, "information_coverage"].item()
    memory[BEHAVIOR] = _memory_generalization(
        _mean_accuracy([in_sample], truth, mask), median_acc, behavior_cov)
    for name, values in memory.items():
        for key, value in values.items():
            table.loc[table.approach == name, key] = value

    id_acc = scoring.person_accuracy(g[ID_ONLY][0], truth, mask)
    prior_acc = scoring.person_accuracy(g[LLM_PRIOR][0], truth, mask)
    gap = _paired_gap(id_acc, prior_acc)
    refs = {"people": len(people), "chance": chance, "ceiling": ceiling, "median": median_acc,
            "llm_prior": llm_prior_acc, "id_minus_prior": gap}
    return table, refs


def panel_a() -> tuple[pd.DataFrame, dict]:
    demographics = data.load_spec(config.SPEC_DEMOGRAPHICS)
    twins = data.load_spec(config.SPEC_TWIN_DEFAULT_TEMP)
    truth_raw, retest_raw = twins.wave1_3, twins.wave4
    cols = truth_raw.columns
    columns = [c for c in cols if c in scoring.COLUMN_RANGES]
    groups = {
        panel.BASELINE: [baselines.guess_the_average(truth_raw, columns).reindex(columns=cols)],
        PERSONAS_A: [demographics.simulated.loc[truth_raw.index]],
        BEHAVIOR: [pd.read_parquet(BEHAVIOR_CACHE).reindex(index=truth_raw.index, columns=cols)],
        TWINS_A: [twins.simulated],
        STAT_TWIN: [statistical_twin_predictions(truth_raw, columns).reindex(columns=cols)],
        "retest": [retest_raw],
    }
    in_sample_raw = _in_sample(truth_raw, columns).reindex(columns=cols)
    truth, g, (in_sample,) = _scaled(groups, truth_raw, [in_sample_raw])
    every = [f for v in g.values() for f in v]
    mask = scoring.valid_mask(scoring.scored_columns(truth, *every), truth, *every)
    mask = mask.loc[:, mask.any()]
    ceiling = _mean_accuracy(g["retest"], truth, mask)
    chance = information.chance_accuracy(truth, mask)
    median = g[panel.BASELINE]
    median_acc = _mean_accuracy(median, truth, mask)
    rows = [_row(name, g[name], truth, mask, chance, ceiling, median_acc, median, median[0], 51 + i)
            for i, name in enumerate([panel.BASELINE, PERSONAS_A, BEHAVIOR, TWINS_A, STAT_TWIN])]
    table = pd.DataFrame(rows)
    behavior_cov = table.loc[table.approach == BEHAVIOR, "information_coverage"].item()
    for key, value in _memory_generalization(_mean_accuracy([in_sample], truth, mask), median_acc, behavior_cov).items():
        table.loc[table.approach == BEHAVIOR, key] = value
    return table, {"people": len(truth), "chance": chance, "ceiling": ceiling, "median": median_acc}


COLUMNS = (
    ("accuracy", "Accuracy", 100), ("chance_corrected", "Chance-corrected", 100),
    ("information_coverage", "Information coverage (vs own prior)", 100),
    ("person_order_rho", "Person-order ρ", 1), ("item_order_rho", "Item-order ρ", 1),
    ("individuality_precision", "Departure precision", 100), ("individuality_recall", "Departure recall", 100),
    ("individuality_f1_corrected", "Individuality F1 (chance-corrected)", 100),
    ("memory_R", "Memory R", 100), ("generalization_G", "Generalization G", 100),
    ("memory_generalization_F1", "Memory–generalization F1", 100),
)


def _markdown(title: str, table: pd.DataFrame, refs: dict) -> str:
    shown = [(k, label, s) for k, label, s in COLUMNS if k in table.columns and table[k].notna().any()]
    lines = [f"## {title}", "", ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                                         for k, v in refs.items() if not isinstance(v, tuple)), ""]
    lines += ["| Approach | " + " | ".join(lbl for _, lbl, _ in shown) + " |", "|---" * (len(shown) + 1) + "|"]
    for _, row in table.iterrows():
        cells = ["—" if pd.isna(row.get(k)) else f"{row[k] * s:.1f}" if s == 100 else f"{row[k]:.2f}" for k, _, s in shown]
        lines.append(f"| {row['approach']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def statistical_twin_inputs() -> dict[str, panel.PanelInput]:
    """Panel A and Panel B inputs with the statistical twin added."""
    demographics, twins = data.load_spec(config.SPEC_DEMOGRAPHICS), data.load_spec(config.SPEC_TWIN_DEFAULT_TEMP)
    truth, retest = twins.wave1_3, twins.wave4
    columns = [c for c in truth.columns if c in scoring.COLUMN_RANGES]
    stat_twin = statistical_twin_predictions(truth, columns)
    behavior = pd.read_parquet(BEHAVIOR_CACHE)
    approaches = {panel.BASELINE: baselines.guess_the_average(truth, columns), PERSONAS_A: demographics.simulated,
                  BEHAVIOR: behavior, TWINS_A: twins.simulated, STAT_TWIN: stat_twin}

    main_runs, _ = load_replies(_runs_path("main"))
    people = sorted(set.intersection(*(set(f.index) for f in main_runs.values())))
    truth_b, retest_b = truth.loc[people], retest.loc[people]
    llm = {LABELS[c]: _samples(main_runs, c, people, truth.columns) for c in simulate.CONDITIONS}
    approaches_b = {panel.BASELINE: baselines.guess_the_average(truth_b, columns), **llm,
                    BEHAVIOR: behavior.loc[people], STAT_TWIN: stat_twin.loc[people]}
    return {"A": panel.PanelInput(truth, retest, approaches), "B": panel.PanelInput(truth_b, retest_b, approaches_b)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a, refs_a = panel_a()
    b, refs_b = panel_b()
    a.to_csv(OUT_DIR / "core_panel_a.csv", index=False)
    b.to_csv(OUT_DIR / "core_panel_b.csv", index=False)
    gap, lo, hi = refs_b.pop("id_minus_prior")
    text = "\n\n".join([
        "# Exploratory core analyses",
        "Percent columns are percentages. Coverage and F1 scales: 0 = reference point, 100 = ceiling. "
        "Panel A has no GPT-4.1-mini prior run, so its coverage uses the data prior (the median).",
        _markdown("Panel A: released GPT-4.1-mini outputs", a, refs_a),
        _markdown(f"Panel B: fresh {API_MODEL} runs", b, refs_b),
        f"**Training-memory check:** ID-only minus LLM-prior accuracy = {gap * 100:.2f} points "
        f"(95% CI {lo * 100:.2f} to {hi * 100:.2f}).",
    ])
    (OUT_DIR / "core.md").write_text(text + "\n")
    print(text)
    print(json.dumps({"panel_a": refs_a, "panel_b": refs_b}, indent=1, default=float))


if __name__ == "__main__":
    main()
