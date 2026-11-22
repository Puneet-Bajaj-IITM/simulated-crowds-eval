"""Turn one answered question into the authors' numeric CSV cells, keyed by Qualtrics id.

Ported from tianyipeng-lab/Digital-Twin-Simulation, commit f1eed51,
evaluation/json2csv.py, AnswerExtractor in ExtractionMode.NUMERIC (Apache-2.0). See NOTICE.
Only the MC, Matrix, Slider and TE processors are ported: they are the only answerable
question types in wave 4. Keys look like "QID157", "QID287_10", "QID164_TEXT".
"""

SINGLE_SELECTORS = ("SAVR", "SAHR")
LINE_SELECTORS = ("SL", "ML")
ITEM_KEYS = ("Statements", "StatementsID", "Rows", "RowsID")
TEXT_SUFFIX = "_TEXT"


def _item_ids(question: dict) -> list[str]:
    if "Statements" in question or "StatementsID" in question:
        ids, texts = question.get("StatementsID", []), question.get("Statements", [])
    else:
        ids, texts = question.get("RowsID", []), question.get("Rows", [])
    if not ids and texts:
        ids = [str(i + 1) for i in range(len(texts))]
    return ids


def _multiple_choice(qid: str, question: dict, answers: dict) -> dict:
    if question.get("Settings", {}).get("Selector", "") not in SINGLE_SELECTORS:
        return {f"{qid}_{choice}": 1 for choice in answers.get("Selected", [])}
    position = answers.get("SelectedByPosition", "")
    if isinstance(position, (int, str)) and position:
        try:
            return {qid: int(position)}
        except ValueError:
            pass
    return {qid: answers.get("SelectedText", "")}


def _matrix(qid: str, question: dict, answers: dict) -> dict:
    out = {}
    selected = answers.get("SelectedByPosition", [])
    for i, item_id in enumerate(_item_ids(question)):
        key = f"{qid}_{item_id}"
        out[key] = None
        if selected and i < len(selected):
            try:
                out[key] = int(selected[i])
            except (ValueError, TypeError):
                texts = answers.get("SelectedText", [])
                if texts and i < len(texts):
                    out[key] = texts[i]
    return out


def _slider(qid: str, question: dict, answers: dict) -> dict:
    values = answers.get("Values", [])
    if len(values) > 1 and any(k in question for k in ITEM_KEYS):
        ids = _item_ids(question)
        return {f"{qid}_{ids[i]}": None if v is None else float(v) for i, v in enumerate(values) if i < len(ids)}
    if values:
        return {qid: None if values[0] is None else float(values[0])}
    return {}


def _text_entry(qid: str, question: dict, answers: dict) -> dict:
    selector = question.get("Settings", {}).get("Selector", "")
    if selector in LINE_SELECTORS and any(k in question for k in ITEM_KEYS):
        texts = answers.get("Text", [])
        if isinstance(texts, list):
            ids = _item_ids(question)
            return {f"{qid}_{ids[i]}": None if t is None else str(t) for i, t in enumerate(texts) if i < len(ids)}
        return {qid + TEXT_SUFFIX: str(texts) if texts else None}
    text = answers.get("Text", "")
    return {qid + TEXT_SUFFIX: str(text) if text else None}


PROCESSORS = {
    "MC": _multiple_choice,
    "Matrix": _matrix,
    "Slider": _slider,
    "TE": _text_entry,
}


def extract_numeric(question: dict) -> dict:
    """Numeric-mode cells for one question; {} when the original would log and skip it."""
    qid = question.get("QuestionID", "")
    answers = question.get("Answers", {})
    processor = PROCESSORS.get(question.get("QuestionType", ""))
    if not qid or not answers or processor is None:
        return {}
    try:
        return processor(qid, question, answers)
    except Exception:
        return {}
