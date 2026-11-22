# Generalization to new questions

**Exploratory.** The design (data, people, outcomes, approaches, metrics) was fixed before any score on these data was computed.

The questions are the dataset authors' follow-up mega-study (Peng et al., 2026, *Science Advances* 12(36), eaeh8260): new studies fielded on the Twin-2K-500 panel, none of which any participant had seen in waves 1–4. Source: Hugging Face `LLM-Digital-Twin/Twin-2K-500-Mega-Study` at revision `0401b715a341ac4b5f98b4424b4aecf9d29570d0`. The LLM rows are the authors' released GPT-4.1 predictions; no fresh model calls were made.

people: 1784, outcomes: 159, studies: 18, answers: 127429, bootstrap_draws: 2000, largest_feature_target_r: 0.616 (QID235_1 vs hiring_algorithms/Q6), chance accuracy: 61.7

## Accuracy and gain over the median

Percentages, except spread and mean error. Brackets: bootstrap 95% intervals.

| Approach | Accuracy | Gain over the median (points) | Gain (% of gap to perfect) | Chance-corrected | Information coverage (vs own prior) | Spread vs humans | Mean error (human SDs) |
|---|---|---|---|---|---|---|---|
| Guess the average | 76.0 (75.6 to 76.4) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) | 37.4 (36.4 to 38.3) | 0.0 (0.0 to 0.0) | 0.16 (0.16 to 0.16) | 0.25 (0.24 to 0.26) |
| LLM prior (GPT-4.1, empty persona) | 71.9 (71.5 to 72.2) | -4.1 (-4.5 to -3.7) | -17.2 (-18.9 to -15.4) | 26.6 (25.8 to 27.5) | -17.2 (-18.9 to -15.4) | 0.49 (0.48 to 0.49) | 0.43 (0.42 to 0.45) |
| Prompted personas (GPT-4.1) | 73.5 (73.1 to 73.9) | -2.5 (-2.8 to -2.2) | -10.3 (-11.8 to -9.0) | 30.9 (30.0 to 31.8) | 5.8 (4.5 to 7.1) | 0.59 (0.58 to 0.60) | 0.36 (0.35 to 0.38) |
| Trained on behavior | 76.3 (76.0 to 76.6) | 0.3 (-0.0 to 0.6) | 1.2 (-0.1 to 2.4) | 38.1 (37.3 to 38.9) | 1.2 (-0.1 to 2.4) | 0.49 (0.48 to 0.49) | 0.09 (0.08 to 0.11) |
| Statistical twin (nearest neighbours) | 76.4 (76.0 to 76.7) | 0.4 (0.3 to 0.5) | 1.7 (1.1 to 2.2) | 38.4 (37.5 to 39.3) | 1.7 (1.1 to 2.2) | 0.31 (0.30 to 0.32) | 0.21 (0.20 to 0.23) |
| Digital twins (GPT-4.1) | 73.9 (73.5 to 74.3) | -2.1 (-2.4 to -1.7) | -8.6 (-10.1 to -7.1) | 32.0 (31.0 to 32.9) | 7.3 (5.9 to 8.7) | 0.66 (0.65 to 0.67) | 0.38 (0.38 to 0.40) |
| Digital twins (GPT-4.1, temperature 0) | 73.9 (73.5 to 74.3) | -2.1 (-2.4 to -1.7) | -8.6 (-10.1 to -7.2) | 32.0 (31.1 to 32.8) | 7.3 (5.9 to 8.6) | 0.65 (0.64 to 0.67) | 0.38 (0.38 to 0.40) |

## Accuracy under the two weightings

| Approach | Each study counts equally | Each outcome counts equally |
|---|---|---|
| Guess the average | 76.0 (75.6 to 76.4) | 76.8 (76.5 to 77.1) |
| LLM prior (GPT-4.1, empty persona) | 71.9 (71.5 to 72.2) | 73.0 (72.7 to 73.3) |
| Prompted personas (GPT-4.1) | 73.5 (73.1 to 73.9) | 74.4 (74.1 to 74.7) |
| Trained on behavior | 76.3 (76.0 to 76.6) | 77.4 (77.1 to 77.7) |
| Statistical twin (nearest neighbours) | 76.4 (76.0 to 76.7) | 77.5 (77.2 to 77.8) |
| Digital twins (GPT-4.1) | 73.9 (73.5 to 74.3) | 74.5 (74.3 to 74.8) |
| Digital twins (GPT-4.1, temperature 0) | 73.9 (73.5 to 74.3) | 75.0 (74.7 to 75.3) |

## Accuracy by study (point estimates, %)

| study                     |   outcomes |   people |   Guess the average |   LLM prior (GPT-4.1, empty persona) |   Prompted personas (GPT-4.1) |   Trained on behavior |   Statistical twin (nearest neighbours) |   Digital twins (GPT-4.1) |   Digital twins (GPT-4.1, temperature 0) |
|:--------------------------|-----------:|---------:|--------------------:|-------------------------------------:|------------------------------:|----------------------:|----------------------------------------:|--------------------------:|-----------------------------------------:|
| accuracy_nudges           |       16.0 |   1003.0 |                72.4 |                                 67.4 |                          68.8 |                  73.4 |                                    73.3 |                      69.8 |                                     70.8 |
| affective_priming         |        4.0 |   1000.0 |                85.1 |                                 83.6 |                          84.0 |                  86.8 |                                    85.9 |                      85.0 |                                     84.8 |
| consumer_minimalism       |        4.0 |    200.0 |                68.3 |                                 61.2 |                          64.2 |                  70.8 |                                    69.0 |                      70.2 |                                     64.3 |
| context_effects           |        3.0 |    501.0 |                58.9 |                                 46.4 |                          59.0 |                  52.8 |                                    57.0 |                      57.9 |                                     57.8 |
| default_eric              |        2.0 |    600.0 |                60.5 |                                 57.7 |                          59.9 |                  59.0 |                                    60.6 |                      63.6 |                                     63.3 |
| digital_certification     |        1.0 |    600.0 |                80.0 |                                 79.4 |                          79.1 |                  80.4 |                                    80.2 |                      78.7 |                                     78.9 |
| hiring_algorithms         |       40.0 |    999.0 |                80.9 |                                 78.5 |                          78.5 |                  81.5 |                                    81.5 |                      78.4 |                                     78.7 |
| idea_evaluation           |       20.0 |   1174.0 |                74.1 |                                 74.0 |                          74.7 |                  74.6 |                                    75.1 |                      72.2 |                                     74.9 |
| infotainment              |       18.0 |    300.0 |                75.5 |                                 64.9 |                          71.1 |                  76.9 |                                    77.3 |                      74.1 |                                     74.7 |
| junk_fees                 |        3.0 |    400.0 |                81.8 |                                 69.2 |                          70.2 |                  83.0 |                                    82.1 |                      71.7 |                                     71.5 |
| obedient_twins            |        7.0 |   1001.0 |                81.2 |                                 79.3 |                          78.8 |                  82.2 |                                    81.7 |                      78.0 |                                     78.2 |
| preference_redistribution |        6.0 |   1163.0 |                72.6 |                                 71.4 |                          71.4 |                  76.2 |                                    74.2 |                      71.6 |                                     71.6 |
| privacy                   |        1.0 |   1200.0 |                80.1 |                                 74.9 |                          75.7 |                  79.2 |                                    79.4 |                      74.1 |                                     74.1 |
| promiscuous_donors        |       10.0 |    799.0 |                75.4 |                                 73.7 |                          75.5 |                  76.1 |                                    76.0 |                      75.8 |                                     76.6 |
| quantitative_intuition    |        5.0 |   1323.0 |                84.1 |                                 82.1 |                          82.6 |                  84.0 |                                    84.5 |                      82.6 |                                     82.6 |
| recommendation_algorithms |       10.0 |    435.0 |                73.7 |                                 65.5 |                          65.1 |                  73.1 |                                    73.9 |                      63.3 |                                     63.0 |
| story_beliefs             |        8.0 |    251.0 |                82.4 |                                 84.3 |                          84.3 |                  81.9 |                                    82.3 |                      84.4 |                                     84.5 |
| targeting_fairness        |        1.0 |    357.0 |                80.7 |                                 80.1 |                          80.3 |                  81.0 |                                    80.9 |                      79.4 |                                     80.4 |

## Condition effects

Over the two-level randomized outcomes whose human effect is significant (|d| > 1.96 SE).

| approach                               |   human_effects |   same_direction |   median_size_ratio |   median_raw_ratio_exploratory |
|:---------------------------------------|----------------:|-----------------:|--------------------:|-------------------------------:|
| LLM prior (GPT-4.1, empty persona)     |              32 |               22 |                0.65 |                           0.18 |
| Prompted personas (GPT-4.1)            |              32 |               22 |                0.46 |                           0.22 |
| Trained on behavior                    |              32 |               32 |                2.79 |                           1.01 |
| Statistical twin (nearest neighbours)  |              32 |               32 |                3.29 |                           0.86 |
| Digital twins (GPT-4.1)                |              32 |               29 |                0.34 |                           0.27 |
| Digital twins (GPT-4.1, temperature 0) |              32 |               27 |                0.33 |                           0.21 |

## Pipeline check

Our per-outcome accuracy against the authors' released `meta analysis.csv`, on their ranges and stored codes. The paper's averages drop the outcome not released for every run (the creativity rating), so the published figure is compared with the average over the outcomes all four runs share.

| run               |   outcomes |   matching_authors |   max_abs_difference |   mean_accuracy_all_outcomes |   shared_outcomes |   mean_accuracy_shared_outcomes |   published | reproduces_published   |
|:------------------|-----------:|-------------------:|---------------------:|-----------------------------:|------------------:|--------------------------------:|------------:|:-----------------------|
| empty_persona     |        160 |                160 |               0.0000 |                       0.7338 |               160 |                          0.7338 |      0.7340 | True                   |
| demographics_only |        160 |                160 |               0.0000 |                       0.7464 |               160 |                          0.7464 |    nan      |                        |
| full_persona      |        161 |                161 |               0.0000 |                       0.7491 |               160 |                          0.7483 |      0.7480 | True                   |
| temperature_zero  |        160 |                160 |               0.0000 |                       0.7523 |               160 |                          0.7523 |      0.7520 | True                   |

Condition effects: our human and twin Cohen's d equal the authors' released `effect size based on human` and `based on twin` (up to the order of levels) on 46 of 46 two-level outcomes, which checks the ported condition definitions.

## Notes

- New questions have no human retest, so the ceiling is 1 (exact reproduction). Gains and coverage are shares of the gap to perfect, not to a retest ceiling as in Panels A and B, and are not comparable with those panels' values.
- Accuracy is 1 − |prediction − answer| / range, averaged over people within an outcome, then over outcomes within a study, then over studies. "Each outcome counts equally" is the paper's weighting.
- Guess the average, the behavior model and the statistical twin are learned within each randomized condition cell from other cross-validation folds. So they need real answers to each new question from other people (the other four folds, about 80% of each study's sample); the LLM rows use none. Their inputs about the person are wave 1–3 answers only; a guard checked that no input names or copies a target (|r| < 0.95 throughout).
- Own prior: the LLM prior for personas and twins, the median for the rest. The LLM prior's own row is measured against the median.
- Five of the authors' 164 outcomes are excluded (no range, or not a closed-ended answer), and three scales are corrected to the questionnaire.
- Intervals are bootstrap 95% percentile intervals over people (2000 resamples; each person's answers move together; the same resamples for every approach). Effects are point estimates.
