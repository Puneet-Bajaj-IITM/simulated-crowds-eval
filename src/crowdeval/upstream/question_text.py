"""Render survey questions (with or without answers) as the authors' plain text.

Ported from tianyipeng-lab/Digital-Twin-Simulation, commit f1eed51,
text_simulation/convert_persona_to_text.py (Apache-2.0). See NOTICE.
Output is byte-identical to the original for the question types in Twin-2K-500.
"""

import re

TAG_PATTERN = re.compile(r"<[^>]*>")
WHITESPACE_PATTERN = re.compile(r"\s+")
NBSP = "&nbsp;"

MASKED = "[Masked]"
NO_ANSWER = "[No Answer Provided]"
NO_STATEMENT = "[No Statement Needed]"
DESCRIPTIVE = "[Descriptive Information]\n\n"

SINGLE_SELECTORS = ("SAVR", "SAHR")
MULTI_SELECTORS = ("MAVR", "MAHR")
LINE_SELECTORS = ("SL", "ML")
FORM_SELECTOR = "FORM"


def strip_html(text: object) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = TAG_PATTERN.sub(" ", text)
    text = text.replace(NBSP, " ")
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _answers(question: dict) -> dict:
    return question.get("Answers") or {}


def _matrix(question: dict, with_answers: bool) -> str:
    columns = question.get("Columns", [])
    answers = _answers(question)
    out = ["Question Type: Matrix\n"]
    if columns:
        out.append("Options:\n")
        out.extend(f"  {i} = {strip_html(c)}\n" for i, c in enumerate(columns, 1))
        texts = answers.get("SelectedText", [])
        positions = answers.get("SelectedByPosition", [])
        for i, row in enumerate(question.get("Rows", [])):
            out.append(f"{i + 1}. {strip_html(row)}\n")
            detail = NO_ANSWER
            if i < len(positions) and i < len(texts):
                detail = f"{positions[i]} - {strip_html(texts[i])}"
            out.append(f"Answer: {detail if with_answers else MASKED}\n")
        out.append("\n")
    return "".join(out)


def _multiple_choice(question: dict, with_answers: bool) -> str:
    options = question.get("Options", [])
    answers = _answers(question)
    selector = question.get("Settings", {}).get("Selector")
    out = []
    if selector in MULTI_SELECTORS:
        out.append("Question Type: Multiple Choice\n")
    elif selector in SINGLE_SELECTORS:
        out.append("Question Type: Single Choice\n")
    if options:
        out.append("Options:\n")
        out.extend(f"  {i} - {strip_html(o)}\n" for i, o in enumerate(options, 1))
        positions = answers.get("SelectedByPosition", [])
        texts = answers.get("SelectedText", [])
        if not with_answers:
            out.append(f"Answer: {MASKED}\n")
        elif positions is None:
            out.append(f"Answer: {NO_ANSWER}\n")
        else:
            if selector in SINGLE_SELECTORS:
                positions, texts = [positions], [texts]
            for i, position in enumerate(positions):
                if i < len(texts):
                    out.append(f"Answer: {position} - {strip_html(texts[i])}\n")
    out.append("\n")
    return "".join(out)


def _text_entry(question: dict, with_answers: bool) -> str:
    answers = _answers(question)
    selector = question.get("Settings", {}).get("Selector")
    out = ["Question Type: Text Entry (Form)\n" if selector == FORM_SELECTOR else "Question Type: Text Entry\n"]
    if selector == FORM_SELECTOR:
        lookup = {}
        for item in answers.get("Text", []):
            if isinstance(item, dict):
                lookup.update(item)
        for label in question.get("Rows", []):
            value = strip_html(str(lookup.get(label, NO_ANSWER)))
            out.append(f"{strip_html(label)}: {value if with_answers else MASKED}\n")
    elif selector in LINE_SELECTORS:
        text = answers.get("Text")
        if not with_answers:
            out.append(f"Answer: {MASKED}\n")
        elif text is None:
            out.append(f"Answer: {NO_ANSWER}\n")
        else:
            out.append(f"Answer: {strip_html(str(text))}\n")
    out.append("\n")
    return "".join(out)


def _slider(question: dict, with_answers: bool) -> str:
    values = _answers(question).get("Values")
    out = ["Question Type: Slider\n"]
    if values:
        statements = question.get("Statements")
        for i, value in enumerate(values):
            statement = NO_STATEMENT if statements[i] == "" else statements[i]
            out.append(f"{i + 1}. {strip_html(statement)}\n")
            out.append(f"Answer: {strip_html(str(value)) if with_answers else MASKED}\n")
    else:
        out.append(f"Answer: {NO_ANSWER}\n")
    out.append("\n")
    return "".join(out)


def _descriptive(question: dict, with_answers: bool) -> str:
    return DESCRIPTIVE


RENDERERS = {
    "Matrix": _matrix,
    "MC": _multiple_choice,
    "TE": _text_entry,
    "Slider": _slider,
    "DB": _descriptive,
}


def format_question_text(question: dict, with_answers: bool = False) -> str:
    question_type = question.get("QuestionType")
    if question_type not in RENDERERS:
        raise ValueError(f"Unhandled question type: {question_type}")
    body = RENDERERS[question_type](question, with_answers)
    return strip_html(question.get("QuestionText", "")) + "\n" + body


def _collect(elements: list, lines: list[str], demographic_only: bool) -> None:
    for element in elements:
        kind = element.get("ElementType")
        if kind == "Block":
            if demographic_only and lines:
                continue
            lines.extend(format_question_text(q, with_answers=True) for q in element.get("Questions", []))
        elif kind == "Branch":
            _collect(element.get("Elements", []), lines, demographic_only)


def persona_text(elements: list, demographic_only: bool = False) -> str:
    """The "full" or "demographic" persona variant of convert_persona_to_text."""
    lines: list[str] = []
    _collect(elements, lines, demographic_only)
    return "".join(lines)
