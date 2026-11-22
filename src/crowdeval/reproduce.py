"""Reproduce every result and figure. Run: uv run python -m crowdeval.reproduce

Each step reads the published model replies or the downloaded dataset and writes to results/.
No model API is called. A step starts as soon as the steps it needs have finished, so
independent steps run side by side.
"""

import subprocess
import sys
import time
from dataclasses import dataclass

POLL_SECONDS = 1.0
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class Step:
    module: str
    needs: tuple[str, ...]
    description: str


STEPS = (
    Step("steps.run_panel_a", (), "Panel A: released GPT-4.1-mini outputs and the behaviour model, all 2,058 people"),
    Step("steps.run_generalization", (), "New questions: the authors' mega-study"),
    Step("steps.run_panel_b", ("steps.run_panel_a",), "Panel B: the published Gemini 3.8 Flash replies, 200 people"),
    Step("steps.run_statistical_twin", ("steps.run_panel_a",), "Statistical twin added to both panels"),
    Step("steps.run_exploratory", ("steps.run_statistical_twin",), "Exploratory: chance and prior corrections, rank agreement, memory"),
    Step("steps.run_individual", ("steps.run_statistical_twin",), "Personal price model on the 40 pricing decisions"),
    Step("steps.figures", ("steps.run_statistical_twin",), "Figures 1 and 2"),
)


def _start(step: Step) -> subprocess.Popen:
    print(f"start  {step.module}: {step.description}", flush=True)
    return subprocess.Popen([sys.executable, "-m", f"crowdeval.{step.module}"], stdout=subprocess.DEVNULL)


def main() -> None:
    began = time.time()
    pending, running, done = list(STEPS), {}, set()
    while pending or running:
        for step in [s for s in pending if set(s.needs) <= done]:
            running[step.module] = (_start(step), time.time())
            pending.remove(step)
        for module, (process, started) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[module]
            if code != 0:
                for other, _ in running.values():
                    other.terminate()
                raise SystemExit(f"crowdeval.{module} failed with exit code {code}")
            done.add(module)
            print(f"done   {module} in {time.time() - started:.0f}s", flush=True)
        time.sleep(POLL_SECONDS)
    print(f"\nAll results are in results/ ({(time.time() - began) / SECONDS_PER_MINUTE:.0f} min).")


if __name__ == "__main__":
    main()
