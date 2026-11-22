"""Write result tables as CSV and Markdown."""

from pathlib import Path

import pandas as pd

from crowdeval import config

PERCENT_KEYS = ("accuracy", "accuracy_pct_ceiling", "gain_over_average")
COLUMNS = (
    ("accuracy", "Accuracy"),
    ("accuracy_pct_ceiling", "% of human ceiling"),
    ("gain_over_average", "Gain over guessing the average"),
    ("mean_error_sd", "Mean error (human SDs)"),
    ("spread_ratio", "Spread vs. humans"),
    ("js_divergence", "JS divergence"),
    ("between_sample_sd", "Drift between samples (human SDs)"),
    ("stability_ratio", "Drift ÷ error"),
)


def _cell(row: pd.Series, key: str) -> str:
    if key not in row or pd.isna(row[key]):
        return "—"
    scale, fmt = (100, "{:.1f}") if key in PERCENT_KEYS else (1, "{:.2f}")
    value = fmt.format(row[key] * scale)
    lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if lo is None or pd.isna(lo):
        return value
    return f"{value} ({fmt.format(lo * scale)}–{fmt.format(hi * scale)})"


def panel_markdown(panel: pd.DataFrame) -> str:
    shown = [(k, label) for k, label in COLUMNS if k in panel.columns and panel[k].notna().any()]
    header = "| Approach | " + " | ".join(label for _, label in shown) + " |"
    rule = "|---" * (len(shown) + 1) + "|"
    lines = [header, rule]
    for _, row in panel.iterrows():
        lines.append(f"| {row['approach']} | " + " | ".join(_cell(row, k) for k, _ in shown) + " |")
    return "\n".join(lines)


def write(name: str, panel: pd.DataFrame, effects: pd.DataFrame, effect_summary: pd.DataFrame) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.RESULTS_DIR / f"{name}.csv", index=False)
    effects.to_csv(config.RESULTS_DIR / f"{name}_effects.csv", index=False)
    effect_summary.to_csv(config.RESULTS_DIR / f"{name}_effect_summary.csv", index=False)
    md = config.RESULTS_DIR / f"{name}.md"
    md.write_text(
        f"# {name}\n\n{panel_markdown(panel)}\n\n"
        "Accuracy, % of ceiling and gain are percentages; brackets are bootstrap 95% intervals over people.\n\n"
        f"## Bias effects\n\n{effect_summary.to_markdown(index=False, floatfmt='.2f')}\n"
    )
    return md
