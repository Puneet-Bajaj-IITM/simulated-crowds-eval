"""Load the mega-study's released outcome values from the pinned results archive.

Each study folder in results.zip holds one subfolder per released run. Its
`meta analysis individual level.csv` has the authors' outcome variables for every
person, once as answered by the human and once as predicted by the model.
"""

import functools
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from crowdeval import config
from crowdeval.new_questions import mega_outcomes

RESULTS_ROOT = "results"
VALUES_FILE = "meta analysis individual level.csv"
AUTHORS_SUMMARY_FILE = "meta analysis.csv"
RAW_ANSWERS_FILE = "consolidated_original_answers_values.csv"
RAW_EXTRA_HEADER_ROWS = [1, 2]
MEMBER_DEPTH = 4

PERSON = "TWIN_ID"
VARIABLE = "variable_name"
VALUE = "value"
RESPONDENT = "respondent_type"
HUMAN = "human"
MODEL = "twin"

TWIN = "full_persona"
PERSONAS = "demographics_only"
PRIOR = "empty_persona"
TWIN_TEMPERATURE_ZERO = "temperature_zero"
RUN_DATE = r"_\d{4}-\d{2}-\d{2}$"
RUN_FOLDERS = {
    TWIN: re.compile(r"^full_persona_without_re(a)?soning" + RUN_DATE),
    PERSONAS: re.compile(r"^demographics_only" + RUN_DATE),
    PRIOR: re.compile(r"^empty_persona" + RUN_DATE),
    TWIN_TEMPERATURE_ZERO: re.compile(r"^temperature_zero" + RUN_DATE),
}


def archive_path() -> Path:
    return Path(
        hf_hub_download(
            repo_id=config.MEGA_DATASET,
            repo_type="dataset",
            filename=config.MEGA_ARCHIVE,
            revision=config.MEGA_REVISION,
            local_dir=config.MEGA_DATA_DIR,
        )
    )


@functools.cache
def _archive() -> zipfile.ZipFile:
    return zipfile.ZipFile(archive_path())


@functools.cache
def run_folders() -> dict[tuple[str, str], str]:
    """(study, run) -> released run folder, e.g. ("junk_fees", TWIN) -> "full_persona_without_reasoning_2025-07-04"."""
    found: dict[tuple[str, str], str] = {}
    for name in _archive().namelist():
        parts = name.split("/")
        if len(parts) != MEMBER_DEPTH or parts[0] != RESULTS_ROOT or parts[3] != VALUES_FILE:
            continue
        for run, pattern in RUN_FOLDERS.items():
            if pattern.match(parts[2]):
                if (parts[1], run) in found:
                    raise ValueError(f"two {run} runs for {parts[1]}")
                found[(parts[1], run)] = parts[2]
    return found


def read_released(study: str, run: str, filename: str, **read_options) -> pd.DataFrame:
    member = f"{RESULTS_ROOT}/{study}/{run_folders()[(study, run)]}/{filename}"
    with _archive().open(member) as handle:
        return pd.read_csv(handle, **read_options)


@functools.cache
def _long_values(study: str, run: str) -> pd.DataFrame:
    return read_released(study, run, VALUES_FILE)


def outcome_values(run: str, respondent: str, studies: list[str]) -> pd.DataFrame:
    """People × outcome keys: the authors' values (stored codes) for one released run."""
    frames = []
    for study in studies:
        long = _long_values(study, run)
        wide = long[long[RESPONDENT] == respondent].pivot(index=PERSON, columns=VARIABLE, values=VALUE)
        wide.columns = [mega_outcomes.key(study, v) for v in wide.columns]
        frames.append(wide)
    out = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce")
    out.index = out.index.astype(int)
    out.index.name = config.ID_COLUMN
    return out.sort_index()


def raw_answers(study: str) -> pd.DataFrame:
    """A study's raw human answers as text, indexed by person (used only for randomized conditions)."""
    raw = read_released(study, TWIN, RAW_ANSWERS_FILE, skiprows=RAW_EXTRA_HEADER_ROWS, dtype=str, low_memory=False)
    raw = raw.dropna(subset=[PERSON])
    raw.index = raw[PERSON].astype(float).astype(int)
    return raw


@dataclass(frozen=True)
class Released:
    """Human answers (the target) and each released run's predictions, recoded, on the included outcomes."""

    truth: pd.DataFrame
    models: dict[str, pd.DataFrame]


def load_released(outcomes: pd.DataFrame, runs: tuple[str, ...]) -> Released:
    studies = sorted(outcomes[mega_outcomes.STUDY].unique())
    columns = list(outcomes.index)
    humans = [outcome_values(run, HUMAN, studies).reindex(columns=columns) for run in runs]
    for run, other in zip(runs[1:], humans[1:]):
        if not other.reindex(humans[0].index).equals(humans[0]) or not other.index.equals(humans[0].index):
            raise ValueError(f"human answers in the {run} files differ from the {runs[0]} files")
    truth = humans[0].dropna(how="all")
    models = {run: outcome_values(run, MODEL, studies).reindex(index=truth.index, columns=columns) for run in runs}
    return Released(mega_outcomes.recode(truth), {run: mega_outcomes.recode(f) for run, f in models.items()})
