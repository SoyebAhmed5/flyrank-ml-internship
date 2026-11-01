# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Soyeb Ahmed
- **Lane:** Refresh / Content Opportunity Scoring
- **Date:** 2026-11-01
- **Decision supported:** Which pseudonymized content items should be reviewed first for possible refresh?

## 0. Abstract

This study asks whether prior search and content signals can rank pages that are more likely to experience a decline in the next 30-day comparison window. I use the bundled 30,000-row anonymized FlyRank starter dataset, where the label is whether the most recent 30 days of impressions declined by more than 20% versus the preceding 30 days. I build a transparent baseline and compare logistic regression, a decision tree, and a random forest using only prior-window and static features, with pseudonymous client holdout validation. The random forest achieved ROC-AUC 0.870 and Precision@50 0.760, versus 0.788 and 0.320 for the baseline on the same holdout. The resulting score is a decision-support queue for human review, not a prediction of Google's algorithm or a causal claim about content refresh.

## 1. Problem framing

The unit of analysis is one content item. The output is a ranked refresh-review score plus reason codes and a suggested action. A reviewer can start with the highest-ranked items, inspect the page and editorial context, and decide whether a refresh, expansion, competitive review, or monitoring is appropriate. The model is useful because the queue must prioritize limited review time rather than merely classify every page.

## 2. Data safety and contract

The run uses `data/raw/content_refresh_anonymized.csv`: 30,000 rows × 44 raw columns. The dataset is an anonymized starter slice covering pseudonymous clients and aggregated search/engagement signals. No client names, domains, URLs, titles, raw queries, or credentials are used.

The label is `trend_direction == "down"`, corresponding to a decline greater than 20% in the latest 30-day impressions versus the previous 30-day window.

**Leakage control:** because the label is defined by the latest 30-day window, the model deliberately excludes latest-window measurements and 90-day aggregates that include that target period. It also excludes `trend_direction`, `trend_pct`, current freshness fields, pseudonymous IDs, and other post-period measurements. Features are based on the previous 30-day window and static metadata; `content_age_at_start` is current content age minus 30 days.

## 3. Baseline

The transparent baseline combines prior demand, prior traffic, content depth, and competition:

- 55% prior-impression percentile
- 25% prior-session percentile
- 15% inverse word-count percentile
- 5% competition percentile

On the client-holdout test set, its Precision@50 was **0.320** and ROC-AUC was **0.788**.

## 4. Method

Models compared:

| Logistic Regression | 0.816 | 0.695 | 0.840 | 0.560 | 0.616 |
| Decision Tree | 0.848 | 0.698 | 0.680 | 0.914 | 0.749 |
| Random Forest | 0.870 | 0.739 | 0.760 | 0.824 | 0.752 |
| Transparent baseline | 0.788 | 0.598 | 0.320 | — | — |

The random forest was selected using Precision@50 because the operational use case is a ranked review queue where the first 50 recommendations matter more than an arbitrary classification threshold.

Validation uses a client-holdout split: 27,675 training rows and 2,325 test rows, with a fixed random seed of 42. This prevents the same pseudonymous client from appearing in both train and test.

## 5. Results

| Model                | ROC-AUC | Average Precision | Precision@50 | Recall |    F1 |
| -------------------- | ------: | ----------------: | -----------: | -----: | ----: |
| Logistic Regression  |   0.816 |             0.695 |        0.840 |  0.560 | 0.616 |
| Decision Tree        |   0.848 |             0.698 |        0.680 |  0.914 | 0.749 |
| Random Forest        |   0.870 |             0.739 |        0.760 |  0.824 | 0.752 |
| Transparent baseline |   0.788 |             0.598 |        0.320 |      — |     — |

The random forest's Precision@50 is **2.38×** the baseline's Precision@50 on the same holdout. The positive-class base rate is 0.542, so Precision@50 should be interpreted together with the base rate rather than in isolation.

Top model features by random-forest importance include `log_impressions_prev_30d`, `prior_session_rate`, `content_age_at_start`, `log_sessions_prev_30d`, `word_count`, and `char_count`. These are associations used for ranking, not causal effects.

## 6. Limitations and honest framing

1. This run uses the bundled 30,000-row starter dataset, not the gated ~79M-row warehouse release.
2. The label is an observed historical decline, not a causal outcome of a content refresh.
3. Feature importance does not prove that changing a feature will cause improvement.
4. Pseudonymous client holdout improves separation across clients but is not the same as a future-calendar-time test.
5. The score should be reviewed by a human and combined with editorial context.
6. The dataset contains aggregated measurements; it does not contain the page text needed to judge content quality directly.
7. The score must not be described as predicting or reverse-engineering Google's ranking algorithm.

## 7. Ranked recommendations

The final queue blends 75% full-data random-forest probability with 25% the transparent baseline score. This produces a review priority, not a publication decision.

**Top actions in the ranked queue:**

- `refresh`: prioritize pages with strong prior demand/traffic and high model decline risk.
- `expand_and_refresh`: prioritize high-risk pages that also have low measured word count.
- `review_competitive_position`: prioritize high-competition pages with prior demand.
- `monitor`: keep lower-priority items under observation rather than automatically changing them.

The top 50 ranked items have an observed decline rate of **98.0%** in this retrospective evaluation. This is descriptive evidence about this dataset, not a promise of future performance.

## 8. Reproducibility

From the repository root:

```bash
pip install -r requirements.txt
python work/scripts/build_capstone.py
```

The notebook is `work/notebooks/capstone.ipynb`. Metrics are saved in `work/outputs/capstone_metrics.json`; the top-50 ranked recommendation table is saved in `work/outputs/top50_ranked_recommendations.csv`.

Random seed: **42**.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset. See [FlyRank](https://flyrank.ai).

## Public-safety statement

This paper contains only the anonymized starter data supplied with the internship repository. No private client names, domains, URLs, raw queries, credentials, or raw exports are included. Findings are framed as observed, measured, directional, and decision-support.
