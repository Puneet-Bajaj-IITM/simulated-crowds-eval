import copy
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from crowdeval import config, data
from crowdeval.llm import simulate
from crowdeval.upstream import survey_prompt
from crowdeval.upstream.question_text import format_question_text, persona_text
from crowdeval.upstream.response_validation import numbered_questions

UPSTREAM_SHA256 = {
    "SYSTEM_PROMPT": "4423af09f4cef23d119fe877981f4964c08cd179b3a7688962b0fd471b4e9f4b",
    "RESPONSE_FORMAT_INSTRUCTIONS": "8ad2f416afa4cea6beda0de2c1ec8fead992dfd4b822a52a86bfda9f81826521",
    "GENERAL_LLM_INSTRUCTION": "42b607dafb9a2dac4494214ccd122ab6c57307103a081c0da77640e7149487e5",
    "COMBINED_PROMPT_HEADER": "8138bbd31c4216eb7b0c30283e4b7cb40a5bfe05aab8fcf4e6eb97fb02b40328",
    "COMBINED_PROMPT_SEPARATOR": "0fe9a47d7e0870d7b025ac1740437533503e09707a6bd552c74c5cd586402516",
}
REPLY_TYPE_LABELS = {"Matrix": "Matrix", "MC": "Single Choice", "Slider": "Slider", "TE": "Text Entry"}
ROUND_TRIP_AGREEMENT = 0.99
ROUND_TRIP_COVERAGE = 0.99
PROMPT_SAMPLE_SIZE = 60
DEMOGRAPHICS_BLOCK = "Demographics"
MAX_PERSONAS_SHARE = 0.10
QUESTIONS_PER_PERSON = 63
PEOPLE = 2058
QUESTION_LABEL = re.compile(r"^Q\d+:$", re.MULTILINE)
SENTINEL = "LEAKCHECK-7731"
SENTINEL_POSITION = 9731
MASKED_ANSWER = "Answer: [Masked]"
ANSWER_PREFIX = "Answer:"


@pytest.fixture(scope="module")
def wave_split() -> pd.DataFrame:
    return simulate.load_wave_split()


@pytest.fixture(scope="module")
def sample(wave_split: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    return wave_split.iloc[rng.choice(len(wave_split), PROMPT_SAMPLE_SIZE, replace=False)]


def perfect_reply(blocks: list) -> str:
    """What a model that knew the person's real wave-4 answers would send back."""
    reply = {
        f"Q{n}": {"Question Type": REPLY_TYPE_LABELS[q["QuestionType"]], "Answers": q["Answers"]}
        for n, q in numbered_questions(blocks)
    }
    return json.dumps(reply, indent=4)


def split_prompt(prompt: str) -> tuple[str, str]:
    assert prompt.startswith(survey_prompt.COMBINED_PROMPT_HEADER)
    persona, questions = prompt[len(survey_prompt.COMBINED_PROMPT_HEADER) :].split(
        survey_prompt.COMBINED_PROMPT_SEPARATOR
    )
    return persona, questions


def scrub_answers(blocks: list) -> list:
    """Same questions, every answer value replaced by a sentinel of the same shape."""
    scrubbed = copy.deepcopy(blocks)
    for block in scrubbed:
        for question in block["Questions"]:
            answers = question.get("Answers")
            if not answers:
                continue
            for key, value in answers.items():
                filler = SENTINEL_POSITION if key == "SelectedByPosition" else SENTINEL
                answers[key] = [filler] * len(value) if isinstance(value, list) else filler
    return scrubbed


def test_prompt_constants_match_upstream():
    values = {
        "SYSTEM_PROMPT": simulate.SYSTEM_PROMPT,
        "RESPONSE_FORMAT_INSTRUCTIONS": simulate.RESPONSE_FORMAT_INSTRUCTIONS,
        "GENERAL_LLM_INSTRUCTION": survey_prompt.GENERAL_LLM_INSTRUCTION,
        "COMBINED_PROMPT_HEADER": survey_prompt.COMBINED_PROMPT_HEADER,
        "COMBINED_PROMPT_SEPARATOR": survey_prompt.COMBINED_PROMPT_SEPARATOR,
    }
    digests = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in values.items()}
    assert digests == UPSTREAM_SHA256


def test_wave_split_shape(wave_split: pd.DataFrame):
    assert wave_split[simulate.PID].is_unique
    assert len(wave_split) == PEOPLE
    counts = {sum(1 for _ in numbered_questions(json.loads(s))) for s in wave_split[simulate.WAVE4_QUESTIONS]}
    assert counts == {QUESTIONS_PER_PERSON}


def test_round_trip_matches_released_wave4(wave_split: pd.DataFrame):
    replies = {
        int(pid): simulate.parse_response(pid, f"```json\n{perfect_reply(json.loads(blocks))}\n```")
        for pid, blocks in zip(wave_split[simulate.PID], wave_split[simulate.WAVE4_QUESTIONS])
    }
    ours = pd.DataFrame.from_dict(replies, orient="index")
    released = data.load_spec(config.SPEC_TWIN_PAPER).wave4
    people = ours.index.intersection(released.index)
    columns = sorted(set(ours.columns) & set(released.columns))
    assert len(people) >= 50 and len(columns) == len(simulate._formatted_columns())
    ours, released = ours.loc[people, columns], released.loc[people, columns]

    both = ours.notna() & released.notna()
    agree = (ours == released) & both
    mismatches = (both & ~agree).sum()
    if mismatches.any():
        print("per-column mismatches:\n", mismatches[mismatches > 0].to_string())
    agreement = agree.to_numpy().sum() / both.to_numpy().sum()
    coverage = both.to_numpy().sum() / released.notna().to_numpy().sum()
    print(f"round trip: {len(people)} people, agreement {agreement:.4%}, coverage {coverage:.4%}")
    assert agreement >= ROUND_TRIP_AGREEMENT
    assert coverage >= ROUND_TRIP_COVERAGE


def test_parse_response_tolerates_noise(wave_split: pd.DataFrame):
    pid, blocks = wave_split[simulate.PID].iloc[0], json.loads(wave_split[simulate.WAVE4_QUESTIONS].iloc[0])
    clean = perfect_reply(blocks)
    expected = simulate.parse_response(pid, clean)
    assert len(expected) >= QUESTIONS_PER_PERSON
    noisy = [
        f"```json\n{clean}\n```",
        f"```\n{clean}\n```",
        f"Here are my answers:\n{clean}\nHope this helps.",
        clean.replace("\n    }", ",\n    }"),
        f"```json\n{clean}",
    ]
    for reply in noisy:
        assert simulate.parse_response(pid, reply) == expected
    assert simulate.parse_response(str(pid), clean) == expected
    assert simulate.parse_response(f"pid_{pid}", clean) == expected


@pytest.mark.parametrize("reply", ["", "   ", "not json", "[1, 2, 3]", '{"Q1": ', None, 42, "{}"])
def test_parse_response_bad_reply_is_empty(wave_split: pd.DataFrame, reply: object):
    assert simulate.parse_response(wave_split[simulate.PID].iloc[0], reply) == {}


def test_parse_response_unknown_pid_is_empty(wave_split: pd.DataFrame):
    clean = perfect_reply(json.loads(wave_split[simulate.WAVE4_QUESTIONS].iloc[0]))
    assert simulate.parse_response(-1, clean) == {}
    assert simulate.parse_response("nobody", clean) == {}


def test_parse_response_drops_only_invalid_questions(wave_split: pd.DataFrame):
    pid, blocks = wave_split[simulate.PID].iloc[0], json.loads(wave_split[simulate.WAVE4_QUESTIONS].iloc[0])
    reply = json.loads(perfect_reply(blocks))
    reply["Q1"]["Answers"]["SelectedByPosition"] = 3
    reply["Q3"]["Question Type"] = "Ranking"
    del reply["Q24"]
    parsed = simulate.parse_response(pid, json.dumps(reply))
    full = simulate.parse_response(pid, perfect_reply(blocks))
    assert not any(c.startswith("FALSE CONS. SELF") for c in parsed)
    assert "1_Q295" not in parsed
    assert "Q157" not in parsed and "Q158" not in parsed
    assert {k: v for k, v in full.items() if k in parsed} == parsed
    assert all(isinstance(v, float) for v in parsed.values())


def test_twin_persona_is_released_persona_text(sample: pd.DataFrame):
    for _, row in sample.iterrows():
        persona, _ = split_prompt(simulate.build_prompt(row, simulate.TWIN))
        assert persona == row[simulate.PERSONA_TEXT]
        assert persona_text(json.loads(row[simulate.PERSONA_JSON])) == persona


def test_prompts_do_not_leak_wave4_answers(sample: pd.DataFrame):
    for _, row in sample.iterrows():
        scrubbed_row = row.copy()
        scrubbed_row[simulate.WAVE4_QUESTIONS] = json.dumps(scrub_answers(json.loads(row[simulate.WAVE4_QUESTIONS])))
        answered = [
            format_question_text(q, with_answers=True)
            for column in (simulate.WAVE4_QUESTIONS, simulate.WAVE4_QUESTIONS_WAVE1_3_ANSWERS)
            for _, q in numbered_questions(json.loads(row[column]))
        ]
        for condition in simulate.CONDITIONS:
            prompt = simulate.build_prompt(row, condition)
            assert prompt == simulate.build_prompt(scrubbed_row, condition)
            assert not any(rendered in prompt for rendered in answered)
            _, questions = split_prompt(prompt)
            answer_lines = [line for line in questions.splitlines() if line.startswith(ANSWER_PREFIX)]
            assert answer_lines and all(line.startswith(MASKED_ANSWER) for line in answer_lines)
            assert len(QUESTION_LABEL.findall(questions)) == QUESTIONS_PER_PERSON


def test_personas_prompt_is_demographics_only(sample: pd.DataFrame):
    for _, row in sample.iterrows():
        blocks = json.loads(row[simulate.PERSONA_JSON])
        demographics, others = blocks[0], blocks[1:]
        assert demographics["BlockName"] == DEMOGRAPHICS_BLOCK
        twin = simulate.build_prompt(row, simulate.TWIN)
        personas = simulate.build_prompt(row, simulate.PERSONAS)
        persona, questions = split_prompt(personas)

        assert persona == persona_text([demographics])
        assert row[simulate.PERSONA_TEXT].startswith(persona)
        assert questions == split_prompt(twin)[1]
        assert len(persona) < MAX_PERSONAS_SHARE * len(row[simulate.PERSONA_TEXT])
        assert len(personas) < len(twin) / 2
        other_answers = [format_question_text(q, with_answers=True) for b in others for q in b.get("Questions", [])]
        assert not any(rendered in personas for rendered in other_answers)


def test_build_prompt_rejects_unknown_condition(sample: pd.DataFrame):
    with pytest.raises(ValueError):
        simulate.build_prompt(sample.iloc[0], "summary")
