import pandas as pd

from crowdeval import config, data
from crowdeval.core import panel, scoring
from crowdeval.models import baselines

PEOPLE = 40
DRAWS = 6
JOBS = 3


def test_parallel_bootstrap_matches_a_single_process():
    spec = data.load_spec(config.SPEC_DEMOGRAPHICS)
    truth, retest = spec.wave1_3.iloc[:PEOPLE], spec.wave4.iloc[:PEOPLE]
    columns = [c for c in truth.columns if c in scoring.COLUMN_RANGES]
    approaches = {panel.BASELINE: baselines.guess_the_average(truth, columns),
                  "Personas": spec.simulated.loc[truth.index]}
    inp = panel.PanelInput(truth, retest, approaches)

    single = panel.build_panel(inp, bootstrap=DRAWS, n_jobs=1)
    parallel = panel.build_panel(inp, bootstrap=DRAWS, n_jobs=JOBS)

    pd.testing.assert_frame_equal(single, parallel)
