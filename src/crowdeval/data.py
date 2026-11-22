"""Download and load Twin-2K-500 files at a pinned revision."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from crowdeval import config


def fetch(repo_path: str) -> Path:
    """Download one dataset file (cached) and return its local path."""
    return Path(
        hf_hub_download(
            repo_id=config.HF_DATASET,
            repo_type="dataset",
            filename=repo_path,
            revision=config.HF_REVISION,
            local_dir=config.DATA_DIR,
        )
    )


def spec_path(spec: str, filename: str) -> str:
    return f"{config.SIM_ROOT}/{spec}/{config.FORMATTED_SUBDIR}/{filename}"


def read_formatted(path: Path) -> pd.DataFrame:
    """Read an authors' formatted CSV: skip the description row, uppercase columns, numeric values."""
    df = pd.read_csv(path, skiprows=[config.DESCRIPTION_ROW])
    df.columns = df.columns.str.upper()
    df = df.drop(columns=[c for c in df.columns if c == "WAVE"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[config.ID_COLUMN])
    df[config.ID_COLUMN] = df[config.ID_COLUMN].astype(int)
    return df.set_index(config.ID_COLUMN).sort_index()


@dataclass(frozen=True)
class SpecData:
    """Human wave 1–3 answers (target), wave-4 retest, and one simulator's answers."""

    name: str
    wave1_3: pd.DataFrame
    wave4: pd.DataFrame
    simulated: pd.DataFrame


def load_spec(spec: str) -> SpecData:
    frames = [read_formatted(fetch(spec_path(spec, f))) for f in config.FORMATTED_FILES]
    common = frames[0].index.intersection(frames[1].index).intersection(frames[2].index)
    wave1_3, wave4, simulated = (f.loc[common] for f in frames)
    return SpecData(name=spec, wave1_3=wave1_3, wave4=wave4, simulated=simulated)


def load_released_summary(spec: str) -> dict[str, pd.DataFrame] | None:
    """The authors' own accuracy summary for a run, or None if they never scored it."""
    try:
        path = fetch(f"{config.SIM_ROOT}/{spec}/{config.SUMMARY_FILE}")
    except EntryNotFoundError:
        return None
    return pd.read_excel(path, sheet_name=None)
