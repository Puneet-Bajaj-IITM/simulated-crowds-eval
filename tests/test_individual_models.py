import numpy as np
import pandas as pd
import pytest

from crowdeval.models import baselines, individual_models, pricing

PEOPLE = 400
PRODUCTS = 40
MAX_PRICE = 60
FREE_SHARE = 0.1
PERSON_SPREAD = 2.5
MIN_PERSONAL_GAIN = 0.05


@pytest.fixture(scope="module")
def market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """People who differ a lot in how much they buy, at random prices for random products."""
    rng = np.random.default_rng(7)
    columns = list(pricing.PRICING_COLUMNS)
    index = pd.Index(np.arange(PEOPLE), name="TWIN_ID")
    codes = np.array([rng.permutation(PRODUCTS) for _ in range(PEOPLE)])
    prices = np.where(rng.random((PEOPLE, PRODUCTS)) < FREE_SHARE, 0.0, rng.uniform(1, MAX_PRICE, (PEOPLE, PRODUCTS)))
    product_effect = rng.normal(2.0, 0.5, PRODUCTS)
    person_effect = rng.normal(0.0, PERSON_SPREAD, PEOPLE)[:, None]
    logit = product_effect[codes] + person_effect - 0.8 * np.log1p(prices)
    truth = np.where(rng.random(logit.shape) < 1 / (1 + np.exp(-logit)), 1.0, 2.0)
    frame = lambda values: pd.DataFrame(values, index=index, columns=columns)
    return frame(truth), frame(codes.astype(float)), frame(prices)


def accuracy(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    return float((pred == truth).to_numpy().mean())


def test_predictions_are_answers_on_every_cell(market):
    truth, codes, prices = market
    for pred in individual_models.price_models(truth, codes, prices).values():
        assert set(np.unique(pred.to_numpy())) <= {individual_models.BUY, individual_models.SKIP}
        assert pred.notna().all().all()


def test_personal_decisions_beat_the_population_model_when_people_differ(market):
    truth, codes, prices = market
    preds = individual_models.price_models(truth, codes, prices)
    assert accuracy(preds["personal"], truth) > accuracy(preds["population"], truth) + MIN_PERSONAL_GAIN


def test_a_held_out_answer_never_changes_its_own_prediction(market):
    """Flip one column for a whole fold of people: none of those cells may change its own prediction."""
    truth, codes, prices = market
    people = baselines.person_folds(truth.index)[0][1]
    column = truth.columns[5]
    flipped = truth.copy()
    flipped.loc[people, column] = 3.0 - truth.loc[people, column]
    before = individual_models.price_models(truth, codes, prices)
    after = individual_models.price_models(flipped, codes, prices)
    for name in before:
        assert after[name].loc[people, column].equals(before[name].loc[people, column])
