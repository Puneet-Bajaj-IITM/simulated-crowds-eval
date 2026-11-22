import numpy as np
import pandas as pd
import pytest

from crowdeval.new_questions import mega_conditions, mega_data, mega_outcomes

INCLUDED_OUTCOMES = 159
INCLUDED_STUDIES = 18
AUTHORS_OUTCOMES = 164
TWO_LEVEL_OUTCOMES = 46
PREREGISTERED_LEVELS = {"accuracy_nudges": 4, "idea_evaluation": 3, "privacy": 6, "hiring_algorithms": 2}
DONOR = "promiscuous_donors/rating_1_1"


def test_included_outcomes_match_the_fixed_set():
    table = mega_outcomes.included()
    assert len(mega_outcomes.authors_ranges()) == AUTHORS_OUTCOMES
    assert len(table) == INCLUDED_OUTCOMES
    assert table[mega_outcomes.STUDY].nunique() == INCLUDED_STUDIES
    assert mega_outcomes.EXCLUDED.isdisjoint(table.index)
    assert "idea_generation" not in set(table[mega_outcomes.STUDY])
    assert table[[mega_outcomes.LOW, mega_outcomes.HIGH]].notna().all().all()


def test_scale_fixes_follow_the_questionnaire():
    table = mega_outcomes.included()
    for outcome in ("infotainment/Profile1", "infotainment/Profile18", "digital_certification/DV1", DONOR):
        assert tuple(table.loc[outcome, [mega_outcomes.LOW, mega_outcomes.HIGH]]) == mega_outcomes.SEVEN_POINT
    authors = mega_outcomes.authors_ranges()
    assert tuple(authors.loc["infotainment/Profile1", [mega_outcomes.LOW, mega_outcomes.HIGH]]) == (1, 8)


def test_donor_codes_map_to_seven_scale_points_and_unknown_codes_fail():
    frame = pd.DataFrame({DONOR: [1.0, 4.0, 9.0, np.nan], "other/x": [2.0, 3.0, 9.0, 1.0]})
    recoded = mega_outcomes.recode(frame)
    assert recoded[DONOR].tolist()[:3] == [1, 2, 7] and np.isnan(recoded[DONOR].iloc[3])
    assert recoded["other/x"].equals(frame["other/x"])
    with pytest.raises(ValueError, match="unexpected answer codes"):
        mega_outcomes.recode(pd.DataFrame({DONOR: [2.0]}))


def test_defaults_condition_is_the_version_shown():
    answers = pd.DataFrame({"OPTOUT_Green": ["1", np.nan, " "], "OPTIN_Green": [np.nan, "2", "1"]})
    labels = mega_conditions.labels_for("default_eric/target_behavior_green", answers)
    assert list(labels) == [mega_conditions.OPT_OUT, mega_conditions.OPT_IN, mega_conditions.OPT_IN]


def test_hiring_jobs_two_and_four_take_the_other_evaluator():
    answers = pd.DataFrame({mega_conditions.ALGORITHM_FIRST: ["1", np.nan], mega_conditions.TEAM_FIRST: [np.nan, "1"]})
    job1 = mega_conditions.labels_for("hiring_algorithms/job1_item3", answers)
    job2 = mega_conditions.labels_for("hiring_algorithms/job2_item3", answers)
    assert list(job1) == [mega_conditions.ALGORITHM, mega_conditions.TEAM]
    assert list(job2) == [mega_conditions.TEAM, mega_conditions.ALGORITHM]
    assert not mega_conditions.has_condition("hiring_algorithms/Q6")


def test_an_answer_without_a_condition_is_an_error():
    truth = pd.DataFrame({"privacy/PPV": [3.0, 4.0]}, index=[1, 2])
    raw = pd.DataFrame({"Group": ["UA", np.nan]}, index=[1, 2])
    with pytest.raises(ValueError, match="without a condition"):
        mega_conditions.conditions(truth, lambda study: raw)


@pytest.fixture(scope="module")
def released():
    return mega_data.load_released(mega_outcomes.included(), (mega_data.TWIN, mega_data.PRIOR))


def test_every_answer_has_a_condition_and_levels_match_the_plan(released):
    cells = mega_conditions.conditions(released.truth, mega_data.raw_answers)
    assert len(mega_conditions.two_level_outcomes(cells)) == TWO_LEVEL_OUTCOMES
    levels = cells.apply(lambda c: c.dropna().nunique())
    for study, count in PREREGISTERED_LEVELS.items():
        assert levels[[c for c in cells.columns if mega_outcomes.study_of(c) == study]].max() == count


def test_released_people_are_twin_2k_500_participants(released):
    from crowdeval.new_questions import mega_baselines

    assert released.truth.index.isin(mega_baselines.wave1_3_features().index).all()
    for model in released.models.values():
        assert model.index.equals(released.truth.index)
