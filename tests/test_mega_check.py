"""Pipeline check on the real release: our scoring reproduces the authors' numbers."""

import pytest

from crowdeval.new_questions import mega_check, mega_conditions, mega_data, mega_effects, mega_outcomes

RUNS = (mega_data.TWIN, mega_data.PERSONAS, mega_data.PRIOR, mega_data.TWIN_TEMPERATURE_ZERO)
TWIN_LABEL = "twin"
TWO_LEVEL_OUTCOMES = 46


@pytest.fixture(scope="module")
def check():
    return mega_check.pipeline_check(RUNS).set_index("run")


def test_per_outcome_accuracy_matches_the_authors_release(check):
    assert (check["matching_authors"] == check["outcomes"]).all()
    assert check.loc[mega_data.TWIN, "outcomes"] == 161


def test_published_averages_are_reproduced(check):
    published = check.dropna(subset=["published"])
    assert set(published.index) == {mega_data.TWIN, mega_data.PRIOR, mega_data.TWIN_TEMPERATURE_ZERO}
    assert published["reproduces_published"].all()


def test_condition_effects_match_the_authors_release():
    released = mega_data.load_released(mega_outcomes.included(), (mega_data.TWIN,))
    cells = mega_conditions.conditions(released.truth, mega_data.raw_answers)
    mask = released.truth.notna() & released.models[mega_data.TWIN].notna()
    table = mega_effects.effect_table(released.truth, {TWIN_LABEL: released.models[mega_data.TWIN]}, cells, mask,
                                      mega_conditions.two_level_outcomes(cells))
    result = mega_check.effect_check(table, f"{TWIN_LABEL}_d").iloc[0]
    assert result["outcomes"] == TWO_LEVEL_OUTCOMES
    assert result["matching_authors"] == result["outcomes"]
