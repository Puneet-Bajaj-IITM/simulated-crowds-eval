# panel_a

| Approach | Accuracy | % of human ceiling | Gain over guessing the average | Mean error (human SDs) | Spread vs. humans | JS divergence |
|---|---|---|---|---|---|---|
| Guess the average | 75.5 (75.2–75.8) | 92.4 (92.0–92.8) | 0.0 (0.0–0.0) | 0.35 (0.35–0.36) | 0.01 (0.01–0.01) | 0.41 (0.40–0.41) |
| Prompted personas (GPT-4.1-mini) | 68.0 (67.7–68.3) | 83.2 (82.8–83.7) | -119.9 (-131.4–-109.3) | 0.71 (0.70–0.72) | 0.38 (0.37–0.39) | 0.32 (0.31–0.32) |
| Trained on behavior | 75.1 (74.8–75.4) | 91.9 (91.5–92.3) | -6.1 (-9.1–-3.1) | 0.18 (0.17–0.19) | 0.34 (0.34–0.34) | 0.22 (0.22–0.22) |
| Digital twins (GPT-4.1-mini) | 71.2 (70.9–71.5) | 87.1 (86.7–87.6) | -68.7 (-75.8–-61.9) | 0.52 (0.51–0.53) | 0.52 (0.51–0.53) | 0.21 (0.21–0.22) |
| Humans (retest) | 81.7 (81.4–82.0) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | — | — | — |

Accuracy, % of ceiling and gain are percentages; brackets are bootstrap 95% intervals over people.

## Bias effects

| approach                         |   human_effects |   same_direction |   median_size_ratio |   median_raw_ratio_exploratory |
|:---------------------------------|----------------:|-----------------:|--------------------:|-------------------------------:|
| Prompted personas (GPT-4.1-mini) |              12 |               10 |                1.01 |                           0.26 |
| Trained on behavior              |              12 |               12 |                4.12 |                           1.11 |
| Digital twins (GPT-4.1-mini)     |              12 |                9 |                0.75 |                           0.39 |
