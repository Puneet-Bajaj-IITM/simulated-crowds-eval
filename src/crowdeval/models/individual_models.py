"""Personal price model: a logit fitted to each person's own buy decisions.

A population logit (one effect per product, one slope on log(1 + price)) is fitted on the people in
the other folds and gives every offer a baseline log-odds. Each person then gets their own intercept
and slope shift on top of it, ridge-shrunk towards zero, fitted on four fifths of their decisions and
used to predict the remaining fifth. log(1 + price) stands in for log price so that the free offers
(about 9%) can be predicted too.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from crowdeval import config
from crowdeval.models import baselines, pricing

BUY = 1.0
SKIP = 2.0
PERSONAL_PENALTY = 1.0
POPULATION_C = 1e4
POPULATION_MAX_ITER = 1000
NEWTON_STEPS = 25
NEWTON_TOLERANCE = 1e-8


def _matrices(truth: pd.DataFrame, codes: pd.DataFrame, prices: pd.DataFrame) -> tuple[np.ndarray, ...]:
    columns = list(pricing.PRICING_COLUMNS)
    truth, codes, prices = (f.reindex(index=truth.index, columns=columns) for f in (truth, codes, prices))
    valid = (truth.notna() & codes.notna() & prices.notna()).to_numpy()
    buys = (truth.to_numpy() == BUY).astype(float)
    products = np.nan_to_num(codes.to_numpy(dtype=float), nan=0).astype(int)
    log_price = np.log1p(np.nan_to_num(prices.to_numpy(dtype=float), nan=0.0))
    return valid, buys, products, log_price


def fit_population(buys: np.ndarray, products: np.ndarray, log_price: np.ndarray, n_products: int) -> tuple[np.ndarray, float]:
    """Product effects and the common slope, from flat arrays of decisions."""
    design = np.column_stack([np.eye(n_products)[products], log_price])
    model = LogisticRegression(C=POPULATION_C, max_iter=POPULATION_MAX_ITER, fit_intercept=False)
    model.fit(design, buys)
    return model.coef_[0][:n_products], float(model.coef_[0][n_products])


def fit_personal(buys: np.ndarray, log_price: np.ndarray, offset: np.ndarray, penalty: float = PERSONAL_PENALTY) -> np.ndarray:
    """Ridge-penalized (intercept, slope shift) on top of a fixed offset, by Newton–Raphson."""
    design = np.column_stack([np.ones_like(log_price), log_price])
    beta = np.zeros(2)
    for _ in range(NEWTON_STEPS):
        p = expit(offset + design @ beta)
        gradient = design.T @ (buys - p) - penalty * beta
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < NEWTON_TOLERANCE:
            break
    return beta


def price_models(truth: pd.DataFrame, codes: pd.DataFrame, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Out-of-sample buy (1) / skip (2) predictions: "population" uses only other people, "personal" adds the person's other decisions."""
    valid, buys, products, log_price = _matrices(truth, codes, prices)
    n_products = int(products[valid].max()) + 1
    population, personal = np.full(buys.shape, np.nan), np.full(buys.shape, np.nan)
    decision_folds = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    position = pd.Series(np.arange(len(truth)), index=truth.index)
    for train_people, test_people in baselines.person_folds(truth.index):
        train = position[train_people].to_numpy()
        seen = valid[train]
        effects, slope = fit_population(buys[train][seen], products[train][seen], log_price[train][seen], n_products)
        for row in position[test_people].to_numpy():
            cells = np.flatnonzero(valid[row])
            offset = effects[products[row, cells]] + slope * log_price[row, cells]
            population[row, cells] = np.where(offset >= 0, BUY, SKIP)
            if len(cells) < config.CV_FOLDS:
                continue
            for fit, predict in decision_folds.split(cells):
                beta = fit_personal(buys[row, cells[fit]], log_price[row, cells[fit]], offset[fit])
                logit = offset[predict] + beta[0] + beta[1] * log_price[row, cells[predict]]
                personal[row, cells[predict]] = np.where(logit >= 0, BUY, SKIP)
    frame = lambda values: pd.DataFrame(values, index=truth.index, columns=list(pricing.PRICING_COLUMNS))
    return {"population": frame(population), "personal": frame(personal)}
