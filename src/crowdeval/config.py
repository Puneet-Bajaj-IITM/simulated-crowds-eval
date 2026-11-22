from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

HF_DATASET = "LLM-Digital-Twin/Twin-2K-500"
HF_REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"

SIM_ROOT = "LLM_simulation_results/llm_simulations_all_specifications"
FORMATTED_SUBDIR = "csv_comparison/csv_formatted"
WAVE1_3_FILE = "responses_wave1_3_formatted.csv"
WAVE4_FILE = "responses_wave4_formatted.csv"
LLM_FILE = "responses_llm_imputed_formatted.csv"
FORMATTED_FILES = (WAVE1_3_FILE, WAVE4_FILE, LLM_FILE)

SPEC_DEMOGRAPHICS = "Demographics Only - GPT4.1-mini"
SPEC_TWIN_DEFAULT_TEMP = "Text Persona (Default Temperature) - GPT4.1-mini"
SPEC_TWIN_PAPER = "Text Persona - GPT4.1-mini"
RELEASED_SPECS = (SPEC_DEMOGRAPHICS, SPEC_TWIN_DEFAULT_TEMP, SPEC_TWIN_PAPER)

SUMMARY_FILE = "accuracy_evaluation/mad_accuracy_summary.xlsx"

HUMAN_DIR = "question_catalog_and_human_response_csv"
HUMAN_WAVE1_3_FILE = f"{HUMAN_DIR}/wave1_3_response.csv"
QUESTION_CATALOG_FILE = f"{HUMAN_DIR}/question_catalog.json"
HOLDOUT_MAPPING_FILE = "LLM_simulation_results/wave4_formatted_to_catalog_mapping.json"

MEGA_DATASET = "LLM-Digital-Twin/Twin-2K-500-Mega-Study"
MEGA_REVISION = "0401b715a341ac4b5f98b4424b4aecf9d29570d0"
MEGA_ARCHIVE = "results.zip"
MEGA_DATA_DIR = DATA_DIR / "mega_study"

ID_COLUMN = "TWIN_ID"
DESCRIPTION_ROW = 1

RANDOM_SEED = 20261121
CV_FOLDS = 5
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95
JSD_BINS = 10
