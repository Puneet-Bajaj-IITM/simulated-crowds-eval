# panel_b

| Approach | Accuracy | % of human ceiling | Gain over guessing the average | Mean error (human SDs) | Spread vs. humans | JS divergence | Drift between samples (human SDs) | Drift ÷ error |
|---|---|---|---|---|---|---|---|---|
| Guess the average | 74.0 (73.1–75.0) | 91.2 (89.8–92.6) | 0.0 (0.0–0.0) | 0.37 (0.35–0.40) | 0.04 (0.04–0.05) | 0.39 (0.38–0.41) | — | — |
| Prompted personas (gemini-3.8-flash) | 73.0 (72.0–74.1) | 90.0 (88.6–91.5) | -13.9 (-23.7–-4.9) | 0.53 (0.51–0.57) | 0.33 (0.31–0.34) | 0.29 (0.28–0.30) | 0.02 | 0.06 |
| Trained on behavior | 74.5 (73.6–75.5) | 91.8 (90.5–93.2) | 6.6 (-1.9–15.6) | 0.24 (0.23–0.28) | 0.35 (0.34–0.36) | 0.23 (0.22–0.25) | — | — |
| Digital twins (gemini-3.8-flash) | 73.8 (72.8–74.9) | 91.0 (89.6–92.3) | -2.8 (-11.5–5.3) | 0.49 (0.46–0.52) | 0.40 (0.37–0.41) | 0.25 (0.24–0.27) | 0.02 | 0.08 |
| Humans (retest) | 81.2 (80.1–82.2) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | — | — | — | — | — |

Accuracy, % of ceiling and gain are percentages; brackets are bootstrap 95% intervals over people.

## Bias effects

| approach                             |   human_effects |   same_direction |   median_size_ratio |   median_raw_ratio_exploratory |
|:-------------------------------------|----------------:|-----------------:|--------------------:|-------------------------------:|
| Digital twins (gemini-3.8-flash)     |              11 |               11 |                5.32 |                           1.78 |
| Prompted personas (gemini-3.8-flash) |              11 |               11 |                5.22 |                           1.63 |
| Trained on behavior                  |              11 |               11 |                4.16 |                           1.24 |
