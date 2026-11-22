"""Assemble the authors' persona + wave-4 question prompt and their system prompt.

Ported from tianyipeng-lab/Digital-Twin-Simulation, commit f1eed51 (Apache-2.0), see NOTICE:
- text_simulation/convert_question_json_to_text.py: GENERAL_LLM_INSTRUCTION, process_json_file
- text_simulation/create_text_simulation_input.py: COMBINED_PROMPT_HEADER / _SEPARATOR
- text_simulation/llm_helper.py: GEMINI_SYSTEM_INSTRUCTION, which LLMConfig falls back to.
  run_LLM_simulations.py reads system_instruction from an `llm_config` section that
  openai_config.yaml does not have, so this is the system prompt the OpenAI runs received
  (the YAML copy is the same text plus a trailing newline).
"""

from crowdeval.upstream.format_instructions import FORMAT_INSTRUCTIONS
from crowdeval.upstream.question_text import format_question_text

GENERAL_LLM_INSTRUCTION = (
    "Please answer the following questions as if you were taking this survey. "
    "The expected output is a JSON object and the format is provided in the end. \n---\n\n"
)

COMBINED_PROMPT_HEADER = "## Persona Profile (This individual's past survey responses):\n"
COMBINED_PROMPT_SEPARATOR = (
    "\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):\n"
)

SYSTEM_INSTRUCTION = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' as if you are the "
    "person described in the 'Persona Profile' (which consists of their past survey responses). \n"
    "Adhere to the persona by being consistent with their previous answers and stated characteristics. \n"
    "Follow all instructions provided for the new question carefully regarding the format of your answer."
)

BLOCK = "Block"
DESCRIPTIVE_TYPE = "DB"


def question_block(blocks: list) -> str:
    """Wave-4 questions with answers masked, numbered Q1..Qn, then the format instructions."""
    parts = [GENERAL_LLM_INSTRUCTION]
    count = 0
    for element in blocks:
        if element.get("ElementType") != BLOCK or not element.get("Questions"):
            continue
        for question in element["Questions"]:
            text = format_question_text(question, with_answers=False)
            if question.get("QuestionType") == DESCRIPTIVE_TYPE:
                parts.append(text)
            else:
                count += 1
                parts.append(f"Q{count}:\n{text}")
    parts.append(FORMAT_INSTRUCTIONS)
    return "".join(parts)


def combined_prompt(persona: str, questions: str) -> str:
    return f"{COMBINED_PROMPT_HEADER}{persona}{COMBINED_PROMPT_SEPARATOR}{questions}"
