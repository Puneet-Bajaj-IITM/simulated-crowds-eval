# Main panel metrics with the statistical twin added (exploratory)

## Panel A

| Approach | Accuracy | % of human ceiling | Gain over guessing the average | Mean error (human SDs) | Spread vs. humans | JS divergence |
|---|---|---|---|---|---|---|
| Guess the average | 75.5 (75.2–75.8) | 92.4 (92.0–92.8) | 0.0 (0.0–0.0) | 0.35 (0.35–0.36) | 0.01 (0.01–0.01) | 0.41 (0.40–0.41) |
| Prompted personas (GPT-4.1-mini) | 68.0 (67.7–68.3) | 83.2 (82.8–83.7) | -119.9 (-131.4–-109.3) | 0.71 (0.70–0.72) | 0.38 (0.37–0.39) | 0.32 (0.31–0.32) |
| Trained on behavior | 75.1 (74.8–75.4) | 91.9 (91.5–92.3) | -6.1 (-9.1–-3.1) | 0.18 (0.17–0.19) | 0.34 (0.34–0.34) | 0.22 (0.22–0.22) |
| Digital twins (GPT-4.1-mini) | 71.2 (70.9–71.5) | 87.1 (86.7–87.6) | -68.7 (-75.8–-61.9) | 0.52 (0.51–0.53) | 0.52 (0.51–0.53) | 0.21 (0.21–0.22) |
| Statistical twin (nearest neighbours) | 75.7 (75.4–76.0) | 92.6 (92.3–93.0) | 3.3 (1.4–5.1) | 0.31 (0.30–0.32) | 0.29 (0.29–0.30) | 0.24 (0.24–0.25) |
| Humans (retest) | 81.7 (81.4–82.0) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | — | — | — |

| approach                              |   human_effects |   same_direction |   median_size_ratio |   median_raw_ratio_exploratory |
|:--------------------------------------|----------------:|-----------------:|--------------------:|-------------------------------:|
| Prompted personas (GPT-4.1-mini)      |              12 |               10 |                1.01 |                           0.26 |
| Trained on behavior                   |              12 |               12 |                4.12 |                           1.11 |
| Digital twins (GPT-4.1-mini)          |              12 |                9 |                0.75 |                           0.39 |
| Statistical twin (nearest neighbours) |              12 |               12 |                4.65 |                           1.09 |

## Panel B

| Approach | Accuracy | % of human ceiling | Gain over guessing the average | Mean error (human SDs) | Spread vs. humans | JS divergence |
|---|---|---|---|---|---|---|
| Guess the average | 74.0 (73.1–75.0) | 91.2 (89.8–92.6) | 0.0 (0.0–0.0) | 0.37 (0.35–0.40) | 0.04 (0.04–0.05) | 0.39 (0.38–0.41) |
| Digital twins (gemini-3.8-flash) | 73.8 (72.8–74.9) | 91.0 (89.6–92.3) | -2.8 (-11.5–5.3) | 0.49 (0.46–0.52) | 0.40 (0.37–0.41) | 0.25 (0.24–0.27) |
| Prompted personas (gemini-3.8-flash) | 73.0 (72.0–74.1) | 90.0 (88.6–91.5) | -13.9 (-23.7–-4.9) | 0.53 (0.51–0.57) | 0.33 (0.31–0.34) | 0.29 (0.28–0.30) |
| Trained on behavior | 74.5 (73.6–75.5) | 91.8 (90.5–93.2) | 6.6 (-1.9–15.6) | 0.24 (0.23–0.28) | 0.35 (0.34–0.36) | 0.23 (0.22–0.25) |
| Statistical twin (nearest neighbours) | 74.7 (73.8–75.8) | 92.1 (90.7–93.5) | 9.7 (1.9–17.4) | 0.35 (0.33–0.39) | 0.30 (0.27–0.31) | 0.26 (0.25–0.28) |
| Humans (retest) | 81.2 (80.1–82.2) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | — | — | — |
