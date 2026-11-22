# Personal price model (exploratory)

Accuracy on the 40 pricing decisions only (buy or not), by the main accuracy rule, so it equals the share of decisions predicted correctly. Brackets are person-bootstrap 95% intervals. Coverage is the share of the gap from guessing the average to the human retest that an approach closes. The personal model sees 32 of the person's own 40 pricing answers when predicting the other 8, so it has information no other row has.

| Panel | Approach | Uses | Samples | Accuracy, % | Coverage |
|---|---|---|---|---|---|
| A | Guess the average | other people's answers | 1 | 56.9 (56.2 to 57.7) | 0.00 |
| A | Price model, population only | other people's pricing answers, product and price | 1 | 73.2 (72.8 to 73.7) | 0.60 |
| A | Personal price model | as above, plus the person's own other pricing answers (32 of 40) | 1 | 77.0 (76.6 to 77.4) | 0.74 |
| A | Trained on behavior | the person's wave 1–3 answers (holdout items excluded) | 1 | 54.0 (53.6 to 54.4) | -0.11 |
| A | Statistical twin (nearest neighbours) | the person's wave 1–3 answers (holdout items excluded) | 1 | 57.5 (56.9 to 58.1) | 0.02 |
| A | Prompted personas (GPT-4.1-mini) | demographics | 1 | 58.5 (57.9 to 59.1) | 0.06 |
| A | Digital twins (GPT-4.1-mini) | wave 1–3 persona text (holdout items excluded) | 1 | 66.6 (66.1 to 67.0) | 0.36 |
| A | Humans (retest) | the same person answering again | 1 | 83.9 (83.4 to 84.3) | 1.00 |
| B | Guess the average | other people's answers | 1 | 56.4 (54.1 to 58.6) | 0.00 |
| B | Price model, population only | other people's pricing answers, product and price | 1 | 72.4 (70.9 to 74.0) | 0.61 |
| B | Personal price model | as above, plus the person's own other pricing answers (32 of 40) | 1 | 76.4 (75.0 to 77.8) | 0.76 |
| B | Trained on behavior | the person's wave 1–3 answers (holdout items excluded) | 1 | 54.5 (53.3 to 55.9) | -0.07 |
| B | Statistical twin (nearest neighbours) | the person's wave 1–3 answers (holdout items excluded) | 1 | 56.8 (54.9 to 58.8) | 0.02 |
| B | Digital twins (gemini-3.8-flash) | wave 1–3 persona text (holdout items excluded) | 3 | 71.4 (69.9 to 72.9) | 0.57 |
| B | Prompted personas (gemini-3.8-flash) | demographics | 3 | 71.2 (69.9 to 72.7) | 0.56 |
| B | Humans (retest) | the same person answering again | 1 | 82.7 (81.3 to 84.1) | 1.00 |
