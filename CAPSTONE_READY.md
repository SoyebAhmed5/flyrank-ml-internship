# FlyRank ML Capstone — Ready to Review

## Included
- `work/notebooks/capstone.ipynb` — completed notebook
- `work/capstone_report.md` — research-paper draft
- `docs/capstone/index.html` — GitHub Pages-ready paper
- `docs/capstone/flyrank_capstone_research_paper.pdf` — PDF
- `work/scripts/build_capstone.py` — reproducible build
- `work/outputs/capstone_metrics.json` — metrics receipt
- `work/outputs/top50_ranked_recommendations.csv` — top-50 queue
- `work/figures/` — paper charts

## Main result
Random Forest:
- ROC-AUC: 0.870
- Average Precision: 0.739
- Precision@50: 0.760
- Baseline Precision@50: 0.320
- Precision@50 lift: 2.38x

## Important scope note
This run uses the bundled 30,000-row anonymized starter dataset. It does not claim to use the gated full warehouse release. If your reviewer explicitly requires full-warehouse results, rerun the same leakage-safe methodology against the approved warehouse release before submitting.

## GitHub Pages
Push the repo, enable GitHub Pages from the `docs` folder, then put the actual live paper URL in `submission/paper_url.txt`. Do not submit a made-up URL.
