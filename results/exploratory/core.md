# Exploratory core analyses

Percent columns are percentages. Coverage and F1 scales: 0 = reference point, 100 = ceiling. Panel A has no GPT-4.1-mini prior run, so its coverage uses the data prior (the median).

## Panel A: released GPT-4.1-mini outputs

people: 2058, chance: 0.592, ceiling: 0.817, median: 0.755

| Approach | Accuracy | Chance-corrected | Information coverage (vs own prior) | Person-order ρ | Item-order ρ | Departure precision | Departure recall | Individuality F1 (chance-corrected) | Memory R | Generalization G | Memory–generalization F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Guess the average | 75.5 | 72.4 | 0.0 | -0.04 | 0.91 | 0.0 | 0.0 | 0.0 | — | — | — |
| Prompted personas (GPT-4.1-mini) | 68.0 | 39.3 | -119.9 | 0.04 | 0.47 | 41.7 | 50.9 | 8.9 | — | — | — |
| Trained on behavior | 75.1 | 70.7 | -6.1 | 0.16 | 0.90 | 51.5 | 39.6 | 15.5 | 42.1 | -6.1 | 0.0 |
| Digital twins (GPT-4.1-mini) | 71.2 | 53.5 | -68.7 | 0.06 | 0.71 | 45.3 | 39.4 | 14.7 | — | — | — |
| Statistical twin (nearest neighbours) | 75.7 | 73.3 | 3.3 | 0.09 | 0.88 | 53.8 | 20.9 | 9.6 | — | — | — |

## Panel B: fresh gemini-3.8-flash runs

people: 200, chance: 0.594, ceiling: 0.812, median: 0.740, llm_prior: 0.731

| Approach | Accuracy | Chance-corrected | Information coverage (vs own prior) | Person-order ρ | Item-order ρ | Departure precision | Departure recall | Individuality F1 (chance-corrected) | Memory R | Generalization G | Memory–generalization F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Guess the average | 74.0 | 67.2 | 0.0 | -0.15 | 0.83 | 0.0 | 0.0 | 0.0 | — | — | — |
| LLM prior (gemini-3.8-flash) | 73.1 | 63.1 | -12.6 | 0.04 | 0.59 | 50.3 | 42.2 | 14.6 | — | — | — |
| ID only (gemini-3.8-flash) | 73.3 | 64.0 | 2.4 | 0.02 | 0.64 | 52.4 | 8.7 | 2.1 | — | — | — |
| Digital twins (gemini-3.8-flash) | 73.8 | 66.3 | 8.7 | 0.15 | 0.62 | 55.2 | 19.9 | 11.3 | 100.0 | 8.7 | 16.0 |
| Prompted personas (gemini-3.8-flash) | 73.0 | 62.7 | -1.1 | 0.11 | 0.57 | 51.8 | 18.5 | 9.6 | — | — | — |
| Trained on behavior | 74.5 | 69.4 | 6.6 | 0.18 | 0.78 | 52.1 | 41.9 | 17.1 | 43.4 | 6.6 | 11.4 |
| Statistical twin (nearest neighbours) | 74.7 | 70.4 | 9.7 | 0.10 | 0.79 | 53.2 | 23.5 | 10.3 | — | — | — |

**Training-memory check:** ID-only minus LLM-prior accuracy = 0.20 points (95% CI 0.02 to 0.37).
