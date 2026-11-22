"""The paper's two figures: accuracy and between-person spread, Panel A and Panel B, one colour per method.

Reads results/exploratory/statistical_twin_panel_{a,b}.csv (python -m crowdeval.steps.run_statistical_twin).
Run: uv run python -m crowdeval.steps.figures
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from crowdeval import config
from crowdeval.steps.run_statistical_twin import output_path

OUT_DIR = config.RESULTS_DIR / "figures"
DPI = 200
PERCENT = 100
INK = "#000000"
GRID = "#eeeeee"
HUMAN_LINE = "#888888"
TITLE_SIZE = 18
PANEL_SIZE = 15
TICK_SIZE = 13
VALUE_SIZE = 12
BAR_HEIGHT = 0.55
BAR_ALPHA = 0.85
TITLE_ROOM = (0, 0, 1, 0.965)

MEDIAN, PERSONA, BEHAVIOUR, TWIN, STAT_TWIN, HUMAN = (
    "Guess the average", "Prompted personas", "Trained on behavior", "Digital twins", "Statistical twin", "Humans (retest)",
)
COLOURS = {MEDIAN: "#1f77b4", PERSONA: "#ff7f0e", BEHAVIOUR: "#1a9e8f",
           TWIN: "#d62728", STAT_TWIN: "#9467bd", HUMAN: "#a0522d"}
PANELS = (("A", "Released GPT-4.1-mini"), ("B", "Fresh Gemini 3.8 Flash"))


@dataclass(frozen=True)
class Chart:
    title: str
    metric: str
    rows: tuple[tuple[str, str], ...]
    size: tuple[float, float]
    xlim: tuple[float, float]
    xticks: tuple[float, ...]
    xlabel: str
    value_format: str
    label_gap: float
    bars: bool
    show_people: bool
    file_name: str


ACCURACY = Chart(
    "Held-out prediction: the typical answer is a strong baseline", "accuracy",
    ((MEDIAN, "Median (not personalised)"), (PERSONA, "Demographic persona"), (BEHAVIOUR, "Behaviour-trained model"),
     (TWIN, "Text digital twin"), (STAT_TWIN, "Statistical twin (50 neighbours)"), (HUMAN, "Human retest")),
    (10, 8), (65, 88), (65, 70, 75, 80, 85),
    "Task-averaged individual accuracy (%)  |  bars: 95% bootstrap CI", "{:.1f}", 0.6, False, True, "accuracy.png",
)
SPREAD = Chart(
    "Across methods, simulated responses are too similar", "spread_ratio",
    ((PERSONA, "Demographic persona"), (BEHAVIOUR, "Behaviour-trained model"),
     (TWIN, "Text digital twin"), (STAT_TWIN, "Statistical twin")),
    (10, 7.2), (0, 112), (0, 20, 40, 60, 80, 100),
    "Between-person response spread (% of human spread)  |  bars: 95% bootstrap CI", "{:.0f}%", 1.5, True, False,
    "spread.png",
)


def _rows(letter: str, chart: Chart) -> pd.DataFrame:
    table = pd.read_csv(output_path(letter))
    picked = []
    for key, label in chart.rows:
        match = table[table["approach"].str.startswith(key)]
        if len(match) != 1:
            raise ValueError(f"Panel {letter}: expected one row for {key!r}, found {len(match)}")
        picked.append(match.iloc[0].rename(label))
    return pd.DataFrame(picked)


def _panel(ax: plt.Axes, letter: str, model: str, chart: Chart) -> None:
    table = _rows(letter, chart)
    positions = list(range(len(chart.rows)))[::-1]
    for y, (key, label) in zip(positions, chart.rows):
        row = table.loc[label]
        point, lo, hi = (PERCENT * row[f"{chart.metric}{s}"] for s in ("", "_lo", "_hi"))
        error = [[point - lo], [hi - point]]
        if chart.bars:
            ax.barh(y, point, height=BAR_HEIGHT, color=COLOURS[key], alpha=BAR_ALPHA)
            ax.errorbar(point, y, xerr=error, fmt="none", ecolor=INK, elinewidth=1.8, capsize=5, capthick=1.8)
        else:
            ax.errorbar(point, y, xerr=error, fmt="o", color=COLOURS[key], markersize=9,
                        elinewidth=2.2, capsize=4, capthick=1.8)
        ax.text(hi + chart.label_gap, y, chart.value_format.format(point), va="center", color=INK, fontsize=VALUE_SIZE)
    people = f", n={int(table['people'].iloc[0]):,}" if chart.show_people else ""
    ax.set_title(f"{letter}: {model}{people}", loc="left", fontweight="bold", fontsize=PANEL_SIZE, pad=14)
    ax.set_yticks(positions, [label for _, label in chart.rows], fontsize=TICK_SIZE)
    ax.set_ylim(-0.6, len(chart.rows) - 0.4)
    ax.set_xlim(*chart.xlim)
    ax.set_xticks(list(chart.xticks))
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if chart.bars:
        ax.axvline(PERCENT, color=HUMAN_LINE, linestyle="--", linewidth=2)


def draw(chart: Chart) -> plt.Figure:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=chart.size, sharex=True)
    for ax, (letter, model) in zip(axes, PANELS):
        _panel(ax, letter, model, chart)
    axes[-1].set_xlabel(chart.xlabel, fontsize=TICK_SIZE)
    fig.suptitle(chart.title, fontweight="bold", fontsize=TITLE_SIZE)
    fig.tight_layout(h_pad=2.5, rect=TITLE_ROOM)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for chart in (ACCURACY, SPREAD):
        draw(chart).savefig(OUT_DIR / chart.file_name, dpi=DPI, facecolor="white", bbox_inches="tight")
        print("wrote", OUT_DIR / chart.file_name)


if __name__ == "__main__":
    main()
