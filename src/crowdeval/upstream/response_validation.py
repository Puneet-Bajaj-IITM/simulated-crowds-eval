"""Validate a model's per-question answers and write them into the wave-4 question JSON.

Ported from tianyipeng-lab/Digital-Twin-Simulation, commit f1eed51,
text_simulation/postprocess_responses.py (Apache-2.0). See NOTICE.
The validators are unchanged. update_question_json_with_response is rewritten to work in
memory and to drop, rather than keep, questions whose reply is missing or invalid: the
original leaves the human's wave-4 answer in place for those questions in the file it writes.
"""

import copy
from collections.abc import Iterator

DESCRIPTIVE_TYPE = "DB"
QUESTION_KEY = "Q{number}"


def is_valid_number(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def is_in_range(value: object, low: float, high: float) -> bool:
    return is_valid_number(value) and low <= float(value) <= high


def validate_matrix_response(response: dict, question: dict) -> bool:
    if not isinstance(response, dict):
        return False
    if "SelectedByPosition" not in response or "SelectedText" not in response:
        return False
    if len(response["SelectedByPosition"]) != len(response["SelectedText"]):
        return False
    if not all(isinstance(p, (int, str)) for p in response["SelectedByPosition"]):
        return False
    return all(isinstance(t, str) for t in response["SelectedText"])


def validate_single_choice_response(response: dict, question: dict) -> bool:
    if not isinstance(response, dict):
        return False
    if "SelectedByPosition" not in response or "SelectedText" not in response:
        return False
    position = response["SelectedByPosition"]
    if not isinstance(position, (int, str)):
        return False
    if isinstance(position, str):
        try:
            int(position)
        except ValueError:
            return False
    return isinstance(response["SelectedText"], str)


def validate_slider_response(response: dict, question: dict) -> bool:
    if not isinstance(response, dict) or "Values" not in response:
        return False
    values = response["Values"]
    if not isinstance(values, list) or not all(is_valid_number(v) for v in values):
        return False
    constraints = question.get("NumericConstraints", {})
    if "MinValue" in constraints and "MaxValue" in constraints:
        return all(is_in_range(v, constraints["MinValue"], constraints["MaxValue"]) for v in values)
    return True


def validate_text_entry_response(response: dict, question: dict) -> bool:
    return isinstance(response, dict) and isinstance(response.get("Text"), str)


VALIDATORS = {
    "Matrix": validate_matrix_response,
    "Single Choice": validate_single_choice_response,
    "Slider": validate_slider_response,
    "Text Entry": validate_text_entry_response,
}


def validate_response(response: object, question: dict) -> bool:
    if not isinstance(response, dict):
        return False
    question_type = response.get("QuestionType") or response.get("Question Type")
    answers = response.get("Answers")
    if not question_type or not answers:
        return False
    validator = VALIDATORS.get(question_type)
    return bool(validator and validator(answers, question))


def numbered_questions(blocks: list) -> Iterator[tuple[int, dict]]:
    """(n, question) for every non-descriptive question, numbered as in the prompt."""
    count = 0
    for block in blocks:
        for question in block.get("Questions", []):
            if question.get("QuestionType") != DESCRIPTIVE_TYPE:
                count += 1
                yield count, question


def impute_answers(blocks: list, reply: dict) -> list[dict]:
    """Copies of the questions the reply answered validly, with Answers replaced by the reply's."""
    imputed = []
    for number, question in numbered_questions(blocks):
        retrieved = reply.get(QUESTION_KEY.format(number=number))
        try:
            valid = retrieved is not None and validate_response(retrieved, question)
        except (TypeError, ValueError, AttributeError):
            valid = False
        if valid:
            answered = copy.deepcopy(question)
            answered["Answers"] = retrieved["Answers"]
            imputed.append(answered)
    return imputed
