# Students’ Social Media Addiction (Quantitative Analysis Project)

This project was done and presented during a class to study the social media impact using data from a survey of 705 participants.

Final deliverables:
- Slide deck: `docs/QA_Project_Social_Media_Group7.pptx`
- Supporting writeups: `reports/`

Dataset is coming from kaggle:
https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships


## Repo structure
- `Data/`: input datasets (`media_addiction.csv`, `sunshine_hours_world.csv`)
- `scripts/`: reproducible analysis scripts
- `notebooks/`: exploratory notebooks (clustering + TikTok vs Instagram)
- `results/`: generated outputs (tables/figures used in the writeups)
- `docs/`: presentation + PDF handout

## Key analyses (matches the PPT)
- Platform comparison (Instagram vs TikTok): `notebooks/instagram_vs_tiktok.ipynb`
- Sunshine exposure vs addiction (ANOVA): uses `Data/sunshine_hours_world.csv`
- Academic impact (binary outcome, effect sizes + logistic): `scripts/academic_impact_analysis.py` + `reports/academic_impact_results_report.md`
- Clustering (4 profiles): `notebooks/clustering_and_hypothesis_testing.ipynb` + `reports/cluster_academic_impact_summary.md`

## Reproduce results
The repo includes a snapshot of outputs under `results/`. To regenerate:

```bash
# Stdlib-only (no third-party deps)
python3 scripts/academic_impact_analysis.py --out-dir results/academic_impact
python3 scripts/academic_impact_analysis.py --exclude-high-school --out-dir results/academic_impact_no_hs
python3 scripts/academic_impact_analysis.py --l2 1.0 --out-dir results/academic_impact_ridge
python3 scripts/academic_impact_analysis.py --l2 1.0 --exclude-high-school --out-dir results/academic_impact_ridge_no_hs
python3 scripts/generate_academic_impact_results_report.py

# Requires pandas/matplotlib/seaborn
python3 scripts/analyze_data.py
python3 scripts/generate_percentage_impact_figure.py
```

## Python dependencies (for notebooks + plotting scripts)
Install (optional) dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
