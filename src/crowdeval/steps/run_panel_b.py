"""Panel B: fresh Gemini runs (personas and twins, repeated samples) on the sampled people."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from crowdeval import config, data
from crowdeval.core import effects, metrics, panel, report
from crowdeval.llm import simulate
from crowdeval.models import baselines
from crowdeval.llm.gemini_run import API_MODEL, RUNS_DIR
from crowdeval.steps.run_panel_a import BEHAVIOR, BEHAVIOR_CACHE, HUMAN

PANEL_NAME = "panel_b"
RUN_NAME = "main"
PUBLISHED_REPLIES = config.RESULTS_DIR / "replies" / f"{API_MODEL}_{RUN_NAME}.jsonl.gz"
LABELS = {simulate.PERSONAS: f"Prompted personas ({API_MODEL})", simulate.TWIN: f"Digital twins ({API_MODEL})"}


def load_replies(path: Path) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[str, int]]:
    """One frame of parsed answers per (condition, sample), indexed by person."""
    answers: dict[tuple[str, int], dict[int, dict[str, float]]] = defaultdict(dict)
    counts = {"replies": 0, "unparsed": 0}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            rec = json.loads(line)
            if not rec.get("reply"):
                continue
            counts["replies"] += 1
            parsed = simulate.parse_response(rec["pid"], rec["reply"])
            counts["unparsed"] += not parsed
            answers[(rec["condition"], rec["sample"])][int(rec["pid"])] = parsed
    frames = {key: pd.DataFrame.from_dict(rows, orient="index").sort_index() for key, rows in answers.items()}
    return frames, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    local = RUNS_DIR / f"replies_{RUN_NAME}.jsonl"
    frames, counts = load_replies(local if local.exists() else PUBLISHED_REPLIES)
    human = data.load_spec(config.SPEC_TWIN_DEFAULT_TEMP)
    people = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    truth, retest = human.wave1_3.loc[people], human.wave4.loc[people]
    columns = [c for c in truth.columns if c in panel.scoring.COLUMN_RANGES]

    def samples_for(condition: str) -> list[pd.DataFrame]:
        keys = sorted(k for k in frames if k[0] == condition)
        return [frames[k].reindex(index=people, columns=truth.columns) for k in keys]

    llm_rows = {LABELS[c]: samples_for(c) for c in simulate.CONDITIONS}
    behavior = pd.read_parquet(BEHAVIOR_CACHE).loc[people]
    approaches = {
        panel.BASELINE: baselines.guess_the_average(truth, columns),
        LABELS[simulate.PERSONAS]: llm_rows[LABELS[simulate.PERSONAS]],
        BEHAVIOR: behavior,
        LABELS[simulate.TWIN]: llm_rows[LABELS[simulate.TWIN]],
    }
    result = panel.build_panel(panel.PanelInput(truth, retest, approaches), bootstrap=args.bootstrap,
                               n_jobs=baselines.BEHAVIOR_JOBS)

    scored_truth, scored_rows, mask = panel.scoring_frames(panel.PanelInput(truth, retest, approaches))
    stab = {name: metrics.stability(scored_rows[name], scored_truth, mask) for name in llm_rows}
    for name, values in stab.items():
        for key, value in values.items():
            result.loc[result["approach"] == name, key] = value

    per_sample_effects = [
        effects.effect_table({HUMAN: truth, **{name: s[i] for name, s in llm_rows.items()}, BEHAVIOR: behavior}, HUMAN)
        for i in range(min(len(s) for s in llm_rows.values()))
    ]
    effect_rows = pd.concat(per_sample_effects).groupby("experiment", sort=False).mean(numeric_only=True).reset_index()
    effect_rows["human_significant"] = effect_rows["human_significant"] > 0.5
    sims = [*llm_rows, BEHAVIOR]
    for name in sims:
        effect_rows[f"{name}_same_direction"] = np.sign(effect_rows[f"{name}_d"]) == np.sign(effect_rows["human_d"])
        effect_rows[f"{name}_size_ratio"] = effect_rows[f"{name}_d"] / effect_rows["human_d"]
    summary = effects.effect_summary(effect_rows, sims)

    print(json.dumps({**counts, "people": len(people), "samples": {k: len(v) for k, v in llm_rows.items()}}))
    print(report.write(PANEL_NAME, result, effect_rows, summary).read_text())


if __name__ == "__main__":
    main()
