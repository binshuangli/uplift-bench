# UpliftBench — Community Leaderboard

This leaderboard tracks models evaluated under the **outer-test-isolated protocol** described
in the paper. All results use repeated stratified 3-fold CV (3 seeds, nested HP tuning),
reported as mean ± 95% CI across folds.

> ⚠️ **Read this first — the metric matters more than the model.** Each dataset is ranked by
> the metric that is *actually knowable*: **√PEHE** where treatment effects are known (IHDP,
> synthetic), **policy risk** (Jobs), and the **Qini** proxy only where no counterfactual
> exists (marketing RCTs). Two findings back this up (decomposed, per the paper): on the
> continuous benchmarks **Qini shows no detectable alignment with effect accuracy** (all-100
> IHDP mean Spearman ρ = +0.07 [−0.03, +0.16]) while AUUC is consistently more aligned
> (paired gap +0.49 [+0.40, +0.59]); on Jobs, **every ranking metric diverges from the
> identified policy objective**. The DR-Learner tops Qini on several IHDP splits while being
> the *worst* by √PEHE. **Do not use the marketing Qini tables below for model selection
> without validation** — see `results/findings_wp6.md` (produced by `make analyze`) and the paper.

**Want to add your model?** See [CONTRIBUTING.md](CONTRIBUTING.md) and
[Submitting a result](#submitting-a-result) below.

---

## Benchmark protocol

| Parameter | Value |
|-----------|-------|
| Outer folds | 3 |
| Seeds | 3 (9 fold evaluations per cell) |
| HP budget | 10 random-search configs |
| Inner folds | 2 |
| Base learner | LightGBM 4.6.0 (shared HP space) |
| Subsampling cap | 10,000 for Lenta / X5 / MegaFon |
| Primary metric | √PEHE (IHDP, synthetic); policy risk (Jobs); Qini (marketing, proxy only) |
| Calibration | ECE (10 quantile bins) |

---

## Leaderboard — Qini coefficient (mean across folds)

Numbers from the paper run (`git hash` stamped in each parquet). Higher = better.
`—` = model not applicable (binary-outcome models on continuous-outcome IHDP).
`TO` = timed out (> 3600s per job).

### Semi-synthetic: IHDP (10 splits, continuous outcome) — ranked by √PEHE (ground truth)

**√PEHE = RMSE of the estimated CATE. Lower = better.** This is the correct metric for IHDP
(individual effects are known). Bold = best per split.

Cells are the **arithmetic mean of fold-level √PEHE** (9 folds per cell), so a single
divergent fold inflates the affected cell.

| Model | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | **Avg** |
|-------|----|----|----|----|----|----|----|----|----|----|---------|
| T-Learner | 0.93 | 0.88 | 0.93 | **1.08** | **1.19** | 0.85 | **0.42** | 0.92 | **10.01** | **3.28** | **2.05** |
| S-Learner | 0.70 | **0.60** | **0.64** | 1.13 | 1.35 | 0.75 | 0.91 | **0.82** | 13.15 | 4.78 | 2.48 |
| X-Learner | 0.87 | 0.83 | 0.79 | 1.48 | 1.90 | 0.79 | **0.42** | 1.11 | 14.60 | 4.96 | 2.77 |
| CausalForest | **0.64** | 0.75 | 0.68 | 1.67 | 2.08 | **0.69** | 0.54 | 1.01 | 17.74 | 6.63 | 3.24 |
| R-Learner | 1.31 | 1.52 | 2.77 | 2.35 | 2.23 | 1.74 | 1.40 | 1.47 | 12.78 | 4.90 | 3.25 |
| DR-Learner | 1,777 | 19.6 | 1,429 | 28,512 | 1.91 | 16,088 | 24.8 | 1.00 | 22.9 | 8.20 | 4,788 |

*Binary-outcome models (ClassTrans, TwoModel, SoloModel, UpliftRF-\*) not applicable to IHDP's
continuous outcome.*

> **The DR-Learner is the cautionary tale.** By Qini it appears to *win* IHDP (Qini = 199.6 on
> s8 — the highest cell in the whole benchmark). By √PEHE it is the **worst** estimator: its
> AIPW nuisance estimation diverges on particular small folds (fold-level √PEHE up to
> 153,013 on s3), inflating the affected cell means. The high Qini is an
> outcome-scale artifact, not skill. The raw Qini values are in
> [results/master_summary.parquet](results/master_summary.parquet) but should **not** be used
> to rank IHDP models.

### Semi-synthetic: Jobs (LaLonde, 10 splits, binary outcome) — ranked by RCT-estimated policy risk

**RCT-estimated reference policy risk** $= 1 - \widehat{\mathbb{E}}[Y(\pi)]$, an IPW estimate
of the policy's value on the randomized experimental subset (Shalit et al., 2017); it is
*not* an oracle optimal-policy comparison, and Jobs provides no individual treatment effects.
**Lower = better.** Averaged across the 10 splits.

| Model | Policy risk ↓ | | Model | Policy risk ↓ |
|-------|---------------|---|-------|---------------|
| **DR-Learner** | **0.206** | | UpliftRF-KL | 0.238 |
| S-Learner | 0.215 | | ClassTrans | 0.240 |
| R-Learner | 0.221 | | UpliftRF-Chi | 0.240 |
| SoloModel | 0.228 | | X-Learner | 0.243 |
| CausalForest | 0.233 | | T-Learner | 0.244 |
| UpliftRF-ED | 0.234 | | TwoModel | 0.253 |

> **Metric disagreement again.** By Qini, **ClassTrans wins all 10 Jobs splits**; by the
> RCT-estimated policy risk it ranks **8th of 12** (DR-Learner / S-Learner lead). The proxy and
> the target rank the field differently here too (the spread is narrow, 0.21–0.25, so most
> differences are not significant). The full Qini-by-split table is in
> [results/master_summary.parquet](results/master_summary.parquet).

### Marketing RCT datasets — Qini only (no ground truth; proxy — validate before trusting)

| Model | Hillstrom | Lenta | X5 | MegaFon |
|-------|-----------|-------|----|---------|
| S-Learner | 47.4 | 4.6 | −3.0 | 40.3 |
| T-Learner | 36.1 | −2.8 | −2.7 | 38.6 |
| **X-Learner** | 46.3 | −0.7 | −2.1 | **40.4** |
| R-Learner | 45.8 | −2.1 | −6.7 | TO |
| DR-Learner | 44.0 | TO | −8.5 | TO |
| **ClassTrans** | 40.0 | **20.3** | −6.1 | 25.2 |
| TwoModel | 40.9 | 3.9 | −4.3 | 39.5 |
| **SoloModel** | **49.1** | 1.8 | **−0.3** | 36.4 |
| CausalForest | 15.8 | −1.8 | −7.5 | 29.7 |
| UpliftRF-KL | TO | 5.9 | −1.1 | TO |
| UpliftRF-ED | TO | −2.0 | −2.5 | TO |
| UpliftRF-Chi | TO | 0.4 | −1.9 | TO |

`TO` = timed out (>3600 s per cell; shown, not hidden). Bold = best per dataset. On MegaFon
the X- and S-Learner are effectively tied (40.4 vs 40.3). Verify any cell against
[results/master_summary.parquet](results/master_summary.parquet); `make check-results` does
this automatically.

### Criteo (supplement, added in v1.4.2) — Qini/AUUC/uplift@0.3 (no ground truth; proxy)

The canonical industry uplift RCT (Criteo Uplift v2.1, 13.9M rows; Diemert et al.). Run at
the **1M tier** with the same 10K-row evaluation cap and protocol as the other large RCTs
(`make repro` equivalent: `run_bench.py --tier criteo`; parquets in
[results_criteo/](results_criteo/)). Outcome = `visit` (binary), treated fraction 0.85.
These rows are a **repository supplement**: the paper reports no Criteo results (no
ground-truth effects, so neither F1 nor F2 is testable here).

| Model | Qini | AUUC | uplift@0.3 |
|-------|------|------|------------|
| **ClassTrans** | **+11.09** | **+0.0280** | **+0.0297** |
| CausalForest | +5.14 | +0.0107 | +0.0217 |
| TwoModel | +4.23 | +0.0214 | +0.0255 |
| T-Learner | +3.34 | +0.0179 | +0.0218 |
| R-Learner | +2.81 | +0.0152 | +0.0253 |
| SoloModel | +1.68 | +0.0092 | +0.0195 |
| UpliftRF-Chi | −0.10 | +0.0095 | +0.0138 |
| S-Learner | −0.52 | +0.0029 | +0.0189 |
| DR-Learner | −0.70 | +0.0127 | +0.0210 |
| UpliftRF-ED | −1.37 | +0.0100 | +0.0195 |
| UpliftRF-KL | −1.75 | +0.0056 | +0.0123 |
| X-Learner | −2.31 | +0.0090 | +0.0163 |

Two observations. (1) **The binary-regime metric agreement replicates out of sample**: the
cross-metric rank agreement on Criteo (Qini~AUUC ρ=+0.74, Qini~uplift@k +0.80,
AUUC~uplift@k +0.92 across the 12 estimators) sits inside the released binary-panel range
(marketing RCTs +0.61 to +0.96; Jobs +0.64 to +0.99) — consistent with F1 being specific to
the continuous-outcome regime. (2) ClassTransformation dominating Criteo matches published
results for this dataset — and it does so at treated fraction 0.85, so near-balanced
treatment is not necessary for its strong performance (see the class_transformation model
card).

### Synthetic (ground truth known) — ranked by √PEHE

**√PEHE, lower = better.** The Qini column is shown alongside to illustrate the disagreement:
by √PEHE the CausalForest and S-Learner win, but by Qini the TwoModel wins and S-Learner looks
near-worst.

| Model | √PEHE ↓ | Qini (proxy) |
|-------|---------|--------------|
| **CausalForest** | **0.09** | 0.48 |
| **S-Learner** | **0.09** | −0.20 |
| SoloModel | 0.09 | −0.35 |
| X-Learner | 0.23 | 1.52 |
| T-Learner | 0.25 | 1.98 |
| R-Learner | 0.29 | 0.02 |
| TwoModel | 0.41 | 2.70 |
| ClassTrans | 0.45 | 1.79 |
| DR-Learner | 721 | −2.55 |

The S-Learner is the sharpest illustration: **2nd-best by √PEHE, near-worst by Qini.**

---

## Calibration ECE leaderboard (lower = better)

`s_learner` and `solo_model` win on calibration in the majority of datasets even when
not top-ranked by Qini. See `results/tables/tab2_master_ece.tex` (produced by `make analyze`)
for the full table.

---

## Submitting a result

To add your model to the leaderboard, open a pull request with:

### 1. Your result parquets

Place output parquets in:
```
results/submissions/<your_method>/
```

Each file should follow the naming convention `<dataset>__<model>.parquet` and contain
the same columns as the master benchmark (see `results/master_raw.parquet` for the schema).

### 2. Your model card

Add `model_cards/<your_method>.md` using the [template](model_cards/TEMPLATE.md).

### 3. Your row in this table

Fill in the Qini values for every dataset where your method is applicable.  
Mark `—` where not applicable, `TO` where timed out.

### 4. Your entry in the submission log (bottom of this file)

---

## Submission log

| Date | Method | Submitter | PR | Notes |
|------|--------|-----------|-----|-------|
| 2026-06-16 | Baseline (WP5 paper run) | Li, Binshuang | — | Initial release |
