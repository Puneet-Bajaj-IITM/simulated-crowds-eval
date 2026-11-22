"""Prompts and reply parsing for re-running the authors' text-persona simulation.

Prompt text, system prompt and answer post-processing are the authors' own
(crowdeval.upstream, ported from tianyipeng-lab/Digital-Twin-Simulation f1eed51).
Two conditions share one wave-4 question block per person:
  "twin"      persona = the person's full wave 1-3 answers (the released persona text)
  "personas"  persona = only the Demographics block, the authors' "demographic" variant
"""

import functools
import json
import math
import re
from collections.abc import Mapping

import pandas as pd

from crowdeval import config, data
from crowdeval.upstream.answer_extraction import extract_numeric
from crowdeval.upstream.format_instructions import FORMAT_INSTRUCTIONS
from crowdeval.upstream.question_text import persona_text
from crowdeval.upstream.response_validation import impute_answers
from crowdeval.upstream.survey_prompt import SYSTEM_INSTRUCTION, combined_prompt, question_block

WAVE_SPLIT_FILES = tuple(f"wave_split/chunks/wave_persona_chunk_{i:03d}.parquet" for i in range(1, 8))
PID = "pid"
PERSONA_TEXT = "wave1_3_persona_text"
PERSONA_JSON = "wave1_3_persona_json"
WAVE4_QUESTIONS_WAVE1_3_ANSWERS = "wave4_Q_wave1_3_A"
WAVE4_QUESTIONS = "wave4_Q_wave4_A"
PID_PREFIX = "pid_"

TWIN = "twin"
PERSONAS = "personas"
CONDITIONS = (TWIN, PERSONAS)

PRIOR = "prior"
RECALL = "recall"
ID_ONLY = "id_only"
EXPLORATORY_CONDITIONS = (PRIOR, RECALL, ID_ONLY)
ALL_CONDITIONS = CONDITIONS + EXPLORATORY_CONDITIONS

FULL_PERSONA_FILES = tuple(f"full_persona/chunks/persona_chunk_{i:03d}.parquet" for i in range(1, 8))
FULL_PERSONA_TEXT = "full_persona_text"
SOURCE_PERSONA_TEXT = "persona_text"
NO_PERSONA_TEXT = "No information about this individual is available."
ID_ONLY_TEMPLATE = "This individual is participant {pid} in the Twin-2K-500 dataset. No other information is provided."

SYSTEM_PROMPT = SYSTEM_INSTRUCTION
RESPONSE_FORMAT_INSTRUCTIONS = FORMAT_INSTRUCTIONS

MAPPED_SOURCE = "catalog_csv_column"
MAPPED_TARGET = "formatted_column"

FENCE_PATTERN = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)(?:```|\Z)", re.DOTALL)
TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
BYTE_ORDER_MARK = "﻿"
DIGIT_SEPARATOR = "_"


def load_wave_split(columns: list[str] | None = None) -> pd.DataFrame:
    """All 2,058 people: pid, wave 1-3 persona (text, JSON) and wave-4 questions with both answer sets."""
    frames = [pd.read_parquet(data.fetch(path), columns=columns) for path in WAVE_SPLIT_FILES]
    return pd.concat(frames, ignore_index=True)


def load_full_personas() -> pd.DataFrame:
    """The `full_persona` text for every person. It contains their answers to the holdout questions (recall test only)."""
    frames = [pd.read_parquet(data.fetch(path), columns=[PID, SOURCE_PERSONA_TEXT]) for path in FULL_PERSONA_FILES]
    full = pd.concat(frames, ignore_index=True).rename(columns={SOURCE_PERSONA_TEXT: FULL_PERSONA_TEXT})
    full[PID] = full[PID].astype(int)
    return full


def _persona_for(row: Mapping, condition: str) -> str:
    if condition == TWIN:
        return row[PERSONA_TEXT]
    if condition == PERSONAS:
        return persona_text(json.loads(row[PERSONA_JSON]), demographic_only=True)
    if condition == PRIOR:
        return NO_PERSONA_TEXT
    if condition == RECALL:
        return row[FULL_PERSONA_TEXT]
    return ID_ONLY_TEMPLATE.format(pid=row[PID])


def build_prompt(row: Mapping, condition: str) -> str:
    """The user message for one person: persona (per condition), then their wave-4 questions with answers masked."""
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"condition must be one of {ALL_CONDITIONS}, got {condition!r}")
    return combined_prompt(_persona_for(row, condition), question_block(json.loads(row[WAVE4_QUESTIONS])))


def parse_response(pid: int | str, raw_text: str) -> dict[str, float]:
    """Formatted wave-4 columns (uppercase, as in scoring.COLUMN_RANGES) -> numeric answer.

    Questions the reply skips or answers invalidly are absent. An unparseable reply or
    an unknown pid gives {}.
    """
    reply = _decode_reply(raw_text)
    blocks = _wave4_blocks(pid)
    if reply is None or blocks is None:
        return {}
    columns = _formatted_columns()
    out: dict[str, float] = {}
    for question in impute_answers(blocks, reply):
        for key, value in extract_numeric(question).items():
            number = _as_number(value)
            if key in columns and number is not None:
                out[columns[key]] = number
    return out


@functools.cache
def _formatted_columns() -> dict[str, str]:
    """Qualtrics cell key (e.g. QID287_10) -> uppercase formatted column (e.g. FALSE CONS. SELF _8)."""
    entries = json.loads(data.fetch(config.HOLDOUT_MAPPING_FILE).read_text(encoding="utf-8"))
    return {e[MAPPED_SOURCE]: e[MAPPED_TARGET].upper() for e in entries}


@functools.cache
def _wave4_question_json() -> dict[int, str]:
    frame = load_wave_split(columns=[PID, WAVE4_QUESTIONS])
    return dict(zip(frame[PID].astype(int), frame[WAVE4_QUESTIONS]))


def _wave4_blocks(pid: int | str) -> list | None:
    try:
        key = int(str(pid).removeprefix(PID_PREFIX))
    except ValueError:
        return None
    raw = _wave4_question_json().get(key)
    return None if raw is None else json.loads(raw)


def _candidates(text: str) -> list[str]:
    found = [text]
    found.extend(m.group(1) for m in FENCE_PATTERN.finditer(text))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        found.append(text[start : end + 1])
    return found


def _decode_reply(raw_text: object) -> dict | None:
    """The reply's JSON object, tolerating code fences, surrounding prose and trailing commas."""
    if not isinstance(raw_text, str):
        return None
    text = raw_text.strip().lstrip(BYTE_ORDER_MARK)
    for candidate in _candidates(text):
        for attempt in (candidate, TRAILING_COMMA_PATTERN.sub(r"\1", candidate)):
            try:
                decoded = json.loads(attempt)
            except (ValueError, RecursionError):
                continue
            if isinstance(decoded, dict):
                return decoded
    return None


def _as_number(value: object) -> float | None:
    """Numeric value of a CSV cell, as data.read_formatted's to_numeric would read it."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or DIGIT_SEPARATOR in text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None
