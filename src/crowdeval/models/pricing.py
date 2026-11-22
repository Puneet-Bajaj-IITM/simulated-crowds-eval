"""The pricing study: each person saw 40 products at random prices and answered buy (1) or not (2).

The product and price behind each column come from the wave-4 question text,
which is exactly what humans and simulators were shown.
"""

import json
import re

import pandas as pd

from crowdeval import config
from crowdeval.llm import simulate

PRICING_COLUMNS = tuple(f"{i}_Q295" for i in range(1, 41))
QUESTION_PREFIX = "QID9_"
COLUMN_SUFFIX = "_Q295"
OFFER_PATTERN = re.compile(r"in that category: (?P<product>.+?)\. The product is priced at: \$(?P<price>\d+(?:\.\d+)?)\.")


def offers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Product code and price for every person × pricing column, parsed from the wave-4 question text."""
    split = simulate.load_wave_split([simulate.PID, simulate.WAVE4_QUESTIONS])
    products: dict[int, dict[str, str]] = {}
    prices: dict[int, dict[str, float]] = {}
    for pid, blocks in zip(split[simulate.PID].astype(int), split[simulate.WAVE4_QUESTIONS]):
        products[pid], prices[pid] = {}, {}
        for block in json.loads(blocks):
            for question in block.get("Questions", []):
                qid = question["QuestionID"]
                match = OFFER_PATTERN.search(question.get("QuestionText", "")) if qid.startswith(QUESTION_PREFIX) else None
                if match:
                    column = qid.removeprefix(QUESTION_PREFIX) + COLUMN_SUFFIX
                    products[pid][column] = match["product"]
                    prices[pid][column] = float(match["price"])
    names = pd.DataFrame.from_dict(products, orient="index").reindex(columns=list(PRICING_COLUMNS)).sort_index()
    codes = pd.DataFrame(pd.factorize(names.to_numpy().ravel(), sort=True)[0].reshape(names.shape), names.index, names.columns)
    price_frame = pd.DataFrame.from_dict(prices, orient="index").reindex(index=names.index, columns=list(PRICING_COLUMNS))
    for frame in (codes, price_frame):
        frame.index.name = config.ID_COLUMN
    return codes.where(names.notna()), price_frame
