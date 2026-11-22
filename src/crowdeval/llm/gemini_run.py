"""Panel B model calls through the Gemini API (temperature 1.0, reasoning off, one seed per sample).

Each call sends the system instruction, temperature, seed and ThinkingConfig(thinking_budget=0).
Replies are appended to data/runs/replies_<name>.jsonl, one call per line.
Resumable: finished calls in the output file are skipped. Needs GEMINI_API_KEY.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from crowdeval import config
from crowdeval.llm import simulate

API_MODEL = "gemini-3.8-flash"
PLATFORM = "gemini-api"
API_KEY_ENV = "GEMINI_API_KEY"
INPUT_USD_PER_TOKEN = 0.75 / 1e6
OUTPUT_USD_PER_TOKEN = 3.75 / 1e6
WORKERS = 12
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 4.0
DEFAULT_CAP_USD = 50.0
RUNS_DIR = config.DATA_DIR / "runs"
TEMPERATURE = 1.0
REASONING = "none"


def sample_people(pids: np.ndarray, count: int) -> list[int]:
    rng = np.random.default_rng(config.RANDOM_SEED)
    return sorted(int(p) for p in rng.choice(np.sort(pids), size=count, replace=False))


def _request_config(sample: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=simulate.SYSTEM_PROMPT or None,
        temperature=TEMPERATURE,
        seed=sample,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def _call(client: genai.Client, prompt: str, sample: int) -> dict:
    started, error = time.time(), None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(model=API_MODEL, contents=prompt, config=_request_config(sample))
            usage = response.usage_metadata
            tokens_in = usage.prompt_token_count or 0
            tokens_out = usage.candidates_token_count or 0
            return {
                "reply": response.text,
                "error": None,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": tokens_in * INPUT_USD_PER_TOKEN + tokens_out * OUTPUT_USD_PER_TOKEN,
                "model_version": getattr(response, "model_version", None),
                "attempts": attempt + 1,
                "seconds": round(time.time() - started, 1),
            }
        except Exception as exc:  # noqa: BLE001 - retried with backoff, then logged
            error = repr(exc)
            time.sleep(BACKOFF_SECONDS * 2**attempt)
    return {"reply": None, "error": error, "input_tokens": None, "output_tokens": None, "cost_usd": 0.0,
            "attempts": MAX_ATTEMPTS, "seconds": round(time.time() - started, 1)}


def _done(path: Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    with path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return {(r["pid"], r["condition"], r["sample"]) for r in records if r.get("reply")}


def run(name: str, people: int, samples: int, cap_usd: float = DEFAULT_CAP_USD,
        conditions: tuple[str, ...] = simulate.CONDITIONS) -> dict:
    client = genai.Client(api_key=os.environ[API_KEY_ENV])
    personas = simulate.load_wave_split()
    if simulate.RECALL in conditions:
        personas = personas.merge(simulate.load_full_personas(), on=simulate.PID, how="left")
    personas = personas.set_index(simulate.PID, drop=False)
    chosen = sample_people(personas.index.to_numpy(), people)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"replies_{name}.jsonl"
    done = _done(out_path)
    jobs = [(pid, cond, s) for pid in chosen for cond in conditions for s in range(samples)
            if (pid, cond, s) not in done]
    prompts = {(pid, cond): simulate.build_prompt(personas.loc[pid].to_dict(), cond)
               for pid, cond, _ in jobs}
    lock, totals = threading.Lock(), {"spent_usd": 0.0, "calls": 0, "failed": 0, "skipped_at_cap": 0}

    def work(job: tuple[int, str, int]) -> None:
        pid, cond, sample = job
        with lock:
            if totals["spent_usd"] >= cap_usd:
                totals["skipped_at_cap"] += 1
                return
        result = _call(client, prompts[(pid, cond)], sample)
        record = {"pid": pid, "condition": cond, "sample": sample, "model": API_MODEL, "platform": PLATFORM,
                  "temperature": TEMPERATURE, "seed": sample, "reasoning": REASONING,
                  **result, "utc": datetime.now(timezone.utc).isoformat()}
        with lock:
            totals["spent_usd"] += record["cost_usd"]
            totals["calls"] += 1
            totals["failed"] += record["reply"] is None
            with out_path.open("a") as out:
                out.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, _ in enumerate(as_completed([pool.submit(work, job) for job in jobs]), start=1):
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} calls, ${totals['spent_usd']:.2f}", flush=True)
    return {"output": str(out_path), "pending_before": len(jobs), **totals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--people", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    parser.add_argument("--conditions", nargs="+", default=list(simulate.CONDITIONS), choices=simulate.ALL_CONDITIONS)
    args = parser.parse_args()
    print(run(args.name, args.people, args.samples, args.cap_usd, tuple(args.conditions)))


if __name__ == "__main__":
    main()
