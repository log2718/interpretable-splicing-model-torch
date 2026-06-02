# Work Plan

## Status as of 2026-05-30

### Completed
- Performance pipeline fully refactored (see `performance/` layout below)
- `flank_40_30` retrained with **corrected** LEFT_FLANK (`CTGACTCTCTCTGCCTATGTCTTTCTCTGCCATCCAGGTT`)
- `flank_150_30` trained, enriched, analyzed (val_loss 0.1126)
- G-quadruplex features added end-to-end (see §4 below)
- Analysis report now has 3 sections (full / full-2D / restricted-20%-MFE)

---

## 1. Performance Pipeline (done)

```
performance/
  lib/
    kl.py          — KL divergence (scalar + vectorised)
    rnafold.py     — RNAfold wrappers: fold_mfe, fold_ensemble, fold_gquad
    plots.py       — all plot functions
  enrich.py        — unified enrichment: inference → KL → RNAfold → gquad
  analyze.py       — 3-section analysis report per config
  compare.py       — cross-config comparison
```

### enrich.py stages (all on by default; use flags to subset)
1. `inference`  → `predicted_PSI_{config}`
2. `kl`         → `kl_{config}`
3. `rnafold`    → `MFE_{config}`, `freq_MFE_{config}`, `ens_div_{config}`
4. `gquad`      → `gquad_present_{config}`, `MFE_delta_gquad_{config}` (only with `--gquad`)

### analyze.py output per config
- **Section 1**: mean KL + p90 KL vs 20 equal-freq bins, for MFE / freq_MFE / ens_div / MFE_delta_gquad
- **Section 2**: 10×10 mean-KL pcolormesh heatmaps for all feature pairs (linear scale, per-cell n=)
- **Section 3**: same as 1 & 2 restricted to bottom 20% MFE
- Plus: KL histogram, boxplot, violin, proportion P(KL>1), PSI density, gquad bar chart, stats.csv

---

## 2. Set Up HPC

- SLURM access needed for repeated training runs
- Goal: run each config 5–10× with different random seeds
- Test statistical significance of KL differences between configs

---

## 3. Longer Upstream Flanks

Custom flanks now pass through the full pipeline via `--left-flank` / `--right-flank` on
`prepare_dataset.py` and `enrich.py` — no need to edit `utils.py`.

| Config | Left flank | Status |
|---|---|---|
| `flank_40_30` | 40 nt (corrected) | done |
| `flank_150_30` | 150 nt | done |
| `flank_100_30` | 100 nt | planned — need biological sequence |
| `flank_200_30` | 200 nt | planned — need biological sequence |

Note: the 150nt flank ends in `...CATCCAGGTT` (2 extra nt past LEFT_FLANK). This is intentional
per the genomic sequence; LEFT_FLANK is embedded at positions 109–148 of the 150nt string.

---

## 4. G-quadruplex Features (done)

- `utils.py`: `RNAfold(..., gquad=True)` adds `-g` flag; `make_dataset_dict(..., gquad=True)` computes features
- `prepare_dataset.py`: `--gquad` flag produces `gquad_present` + `MFE_delta_gquad` columns in output CSV
- `performance/lib/rnafold.py`: `fold_gquad(seq, bin, temp)` for single-sequence use in enrich.py
- `performance/enrich.py`: `--gquad` flag adds Stage 4 (calls `fold_gquad` per exon)
- `performance/analyze.py`: `MFE_delta_gquad_{config}` auto-detected; `kl_by_gquad.png` bar chart added

### Using gquad as a model *input feature* (future)
Currently gquad is analysis-only (not fed to the model). To use as input:
- Add a new channel to the struct or a separate channel in `model.py`
- Encode `gquad_present` (per-position or scalar) in `prepare_dataset.py`

---

## 5. Loop Type Identification (planned)

- Parse dot-bracket notation to label each position: hairpin / internal / bulge / multi-loop / external
- Encode as new multi-class input channel
- Files: `utils.py`, `prepare_dataset.py`, `model.py`

---

## 6. ContraFold RNA (planned)

- Replace ViennaRNA with ContraFold as structure predictor
- ContraFold outputs dot-bracket; adapt subprocess call in `utils.py`
- Add `--folder contrafoldrna` flag to `prepare_dataset.py`

---

## 7. Statistical Significance Testing (blocked on HPC)

- After HPC setup: run each config N times (different seeds)
- Compare mean KL with bootstrap CI or paired t-test
- New script: `performance/significance_test.py`
