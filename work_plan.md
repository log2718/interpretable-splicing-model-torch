# Work Plan

## 1. Clean Up Performance Pipeline (immediate priority)

### Problem
Scripts in `performance/` were written incrementally — duplicated logic, hardcoded paths, can't reuse across configs.

### New structure
performance/
  lib/
    kl.py          — single KL divergence definition (scalar + vectorised)
    rnafold.py     — single RNAfold subprocess wrapper
    plots.py       — all plot functions (violin, heatmap, density histogram)
  enrich.py        — unified enrichment pipeline (replaces 3 old scripts)
  analyze.py       — unified analysis + plots (replaces 2 old scripts)
  compare.py       — cross-config comparison (new)

### Files to delete after migration
- add_data_features.py, add_loss_metrics.py, add_vienna_predictions.py
- model_performance_MFE.py, model_performance_extended.py

### Output per config
generated_files/{config_name}/
  summary.md, stats.csv
  violin_mfe.png, violin_mfe_log.png
  violin_freq_mfe.png, violin_ens_div.png
  heatmap_mfe_freq.png, density_psi_pred.png

### Plot types (10 total per config)
1. Line plot — mean KL per feature bin
2. Violin (linear KL) — KL distribution per bin
3. Violin (log KL + p60–p95 overlays) — log-scaled per bin
4. 1D histogram — KL / log-KL distribution
5. KL curve — theoretical KL(p,q) vs q
6. Boxplot + jitter — 3-panel (normal / log / zoomed)
7. Proportion line — P(KL > threshold) per bin
8. 3×3 heatmap — P(KL > 1) by MFE × feature tertile
9a. 2D KL heatmap — median/q90/mean KL per (feature×feature) cell
    [pcolormesh + LogNorm + YlOrRd — professor's visual style]
9b. 2D density histogram — joint feature density + Pearson r
    [pcolormesh + LogNorm + Blues — professor's exact function]
10. 10×10 violin grid — mini-violins per (feature×feature) cell

---

## 2. Set Up HPC (in progress separately)
- SLURM access needed for repeated training runs
- Goal: run each config 5–10× with different random seeds
- Test statistical significance of KL differences between configs

---

## 3. Longer Upstream Flanks
- Current: 40nt left + 30nt right
- Planned: 100L+30R, 200L+30R, 500L+30R
- Requires: longer biological sequence for left flank (from construct/genome)
- Changes: utils.py (flank constants → CLI args), prepare_dataset.py, train.py --input-length

---

## 4. New Input Features

### G-quadruplex (ViennaRNA --gquad)
- Add --gquad flag to RNAfold calls in prepare_dataset.py
- Add new input channel to model (or encode in existing struct channel)
- Files: utils.py, prepare_dataset.py, model.py

### Loop type identification
- Parse dot-bracket notation to label positions: hairpin / internal / bulge / multi-loop / external
- Encode as new multi-class input channel
- Files: utils.py, prepare_dataset.py, model.py

---

## 5. ContraFold RNA
- Replace ViennaRNA with ContraFold as structure predictor
- ContraFold outputs dot-bracket; adapt subprocess call in utils.py
- Add --folder contrafoldrna flag to prepare_dataset.py

---

## 6. Statistical Significance Testing
- After HPC setup: run each config N times (different seeds)
- Compare mean KL with bootstrap CI or paired t-test
- New script: performance/significance_test.py
