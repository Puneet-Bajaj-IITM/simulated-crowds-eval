import numpy as np
import pandas as pd
import pytest

from crowdeval.core import information, ranks

BINARY = "Q183"
FIVE_POINT = "Q171"


def frame(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.RangeIndex(len(next(iter(values.values())))))


def test_chance_accuracy_is_half_on_binary_items():
    truth = frame({BINARY: [1.0, 2.0, 1.0, 2.0]})
    mask = truth.notna()
    assert information.chance_accuracy(truth, mask) == pytest.approx(0.5)


def test_chance_accuracy_on_five_point_scale_matches_hand_calculation():
    truth = frame({FIVE_POINT: [1.0, 3.0]})
    expected = np.mean([1 - np.mean(np.abs(np.arange(1, 6) - y)) / 4 for y in (1, 3)])
    assert information.chance_accuracy(truth, truth.notna()) == pytest.approx(expected)


def test_coverage_and_harmonic_f1():
    assert information.coverage(0.8, 0.6, 1.0) == pytest.approx(0.5)
    assert information.harmonic_f1(0.5, 0.5) == pytest.approx(0.5)
    assert information.harmonic_f1(0.9, -0.1) == 0.0


def test_individuality_rewards_correct_departures_over_a_random_predictor():
    rng = np.random.default_rng(0)
    truth = frame({FIVE_POINT: rng.integers(1, 6, 400).astype(float)})
    prior = frame({FIVE_POINT: np.full(400, 3.0)})
    oracle = truth.copy()
    mask = truth.notna()
    scores = information.individuality(oracle, prior, truth, prior, mask, seed=1)
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1_corrected"] > 0.5
    noise = frame({FIVE_POINT: rng.integers(1, 6, 400).astype(float)})
    assert abs(information.individuality(noise, prior, truth, prior, mask, seed=1)["f1_corrected"]) < 0.15


def test_person_order_detects_perfect_and_reversed_ranking():
    truth = frame({FIVE_POINT: np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 4)})
    mask = truth.notna()
    assert ranks.person_order(truth, truth, mask) == pytest.approx(1.0)
    assert ranks.person_order(6 - truth, truth, mask) == pytest.approx(-1.0)
