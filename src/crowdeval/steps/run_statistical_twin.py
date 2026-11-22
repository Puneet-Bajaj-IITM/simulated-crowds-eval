"""Panel A and Panel B with the statistical twin added (exploratory): tables, effects, and the data behind the figures.

Run: uv run python -m crowdeval.steps.run_statistical_twin
"""

from pathlib import Path

import pandas as pd

from crowdeval import config
from crowdeval.core import effects, panel, report
from crowdeval.models import baselines
from crowdeval.steps.run_exploratory import statistical_twin_inputs
from crowdeval.steps.run_panel_a import HUMAN

OUT_DIR = config.RESULTS_DIR / "exploratory"
CSV_NAME = "statistical_twin_panel_{letter}.csv"
MARKDOWN_FILE = OUT_DIR / "statistical_twin_panels.md"
TITLE = "# Main panel metrics with the statistical twin added (exploratory)"


def output_path(letter: str) -> Path:
    return OUT_DIR / CSV_NAME.format(letter=letter.lower())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sections = [TITLE]
    for letter, inp in statistical_twin_inputs().items():
        result = panel.build_panel(inp, bootstrap=config.BOOTSTRAP_SAMPLES, n_jobs=baselines.BEHAVIOR_JOBS)
        result.to_csv(output_path(letter), index=False)
        section = f"## Panel {letter}\n\n{report.panel_markdown(result)}"
        simulated = {k: v for k, v in inp.approaches.items() if k != panel.BASELINE}
        if all(isinstance(v, pd.DataFrame) for v in simulated.values()):
            summary = effects.effect_summary(effects.effect_table({HUMAN: inp.truth, **simulated}, HUMAN), list(simulated))
            section += f"\n\n{summary.to_markdown(index=False, floatfmt='.2f')}"
        sections.append(section)
    MARKDOWN_FILE.write_text("\n\n".join(sections) + "\n")
    print(MARKDOWN_FILE.read_text())


if __name__ == "__main__":
    main()
