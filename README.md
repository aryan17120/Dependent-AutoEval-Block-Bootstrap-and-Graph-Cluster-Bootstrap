# Dependent AutoEval: Block Bootstrap and Graph-Cluster Bootstrap

Official implementation for the paper:

**AutoEval for Dependent Data via Block and Graph-Cluster Bootstrap**

*Anonymous submission — JMLR 2026*

---

## The Problem

The AutoEval framework combines small human-labeled datasets with large synthetic-label datasets via prediction-powered inference (PPI++) to produce statistically valid performance estimates. It works well ; but only when observations are independent.

In practice, this assumption is routinely violated. Consider evaluating a model on sensor readings from an air quality station: consecutive measurements are correlated, so the i.i.d. variance estimator underestimates the true uncertainty. The resulting confidence interval is too narrow, and coverage collapses silently. The point estimate is unbiased , only the variance step is wrong.

The same failure occurs on graph-structured data. Evaluating a GCN on Citeseer nodes treats each node as independent, ignoring the community structure that links neighboring predictions. Standard PPI++ coverage drops to 0.884 , below the nominal 0.900 , because the variance estimate ignores intra-community correlation.

This failure mode affects any evaluation pipeline where data has temporal, spatial, or relational structure. No existing AutoEval method corrects for it.

**Dependent AutoEval fills this gap with:**

- A non-overlapping block bootstrap for time series that correctly estimates the long-run variance
- A graph-cluster bootstrap for networks that resamples at community granularity
- Theoretical guarantees connecting both procedures to the Lahiri (2003) bootstrap consistency framework
- Empirical validation across five datasets spanning synthetic and real-world dependence structures

---

## Key Results

All results use n=200, 250 trials, 90% nominal CI.

### Time Series

| Dataset | Cov (cls) | Cov (ppi) | Cov (blk) | ESS (ppi) | ESS (blk) | MSE ratio |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| AR(1) | 0.900 | 0.932 | **0.960** | 1.521 | 1.078 | 0.604 |
| MA(2) | 0.908 | 0.928 | **0.968** | 1.686 | 1.248 | 0.503 |
| Air Quality | 0.880 | 0.920 | **1.000** | 4.141 | 1.696 | 0.182 |

### Graphs

| Dataset | Cov (cls) | Cov (ppi) | Cov (blk) | ESS (ppi) | ESS (blk) | MSE ratio |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| Cora | 0.900 | 0.912 | **0.996** | 1.267 | 0.370 | 0.763 |
| Citeseer | 0.904 | 0.884 | **1.000** | 1.167 | 0.435 | 0.913 |

**Headline findings:**

- Citeseer PPI++ coverage 0.884: the starkest failure of standard AutoEval under dependence, falling below nominal without any warning.
- Air Quality ESS(ppi) 4.141, MSE ratio 0.182: 82% MSE reduction from using the synthetic annotator on a strongly correlated sensor signal.
- Air Quality classical coverage 0.880: dependence affects even the classical estimator, motivating correction at the variance level.
- Graph ESS(blk) 0.370–0.435: resampling at community granularity (K=7 communities for Cora, K=6 for Citeseer) correctly reflects the true information content of the graph. This is a principled cost, not a failure.

For full per-dataset breakdowns and figures, see `results/experiments.md`.

---

## What We Build

### The Core Problem: Long-Run Variance

For dependent data, the i.i.d. variance underestimates the true long-run variance:

$$\sigma^2_{\mathrm{LRV}} = \sigma^2_Z \left(1 + 2\sum_{k=1}^{\infty} \rho_k\right)$$

For an AR(1) process with autocorrelation ρ, the inflation factor is (1+ρ)/(1−ρ). At ρ=0.5 this is 3×, and at ρ=0.7 it reaches 5.7×, meaning standard PPI++ confidence intervals can be less than half as wide as they should be.

Both bootstrap procedures leave the PPI++ point estimate μ̂ and optimal λ* unchanged. Only the variance estimation step is corrected.

### Two Procedures

| Setting | Method | Block unit | Block size |
|---------|--------|------------|------------|
| Time series | Non-overlapping block bootstrap | Contiguous window | L = ⌊√T⌋ |
| Networks | Graph-cluster bootstrap | Community | Greedy modularity maximization |

**λ* is held fixed across all bootstrap replicates** — estimated once on the original data and reused. This is a deliberate design choice that enables the Lahiri (2003) Thm 3.1 consistency argument to apply directly.

**Community detection uses greedy modularity maximization** (`greedy_modularity_communities` in NetworkX), not Louvain. Communities smaller than 5 nodes are merged into the largest community before resampling.

### Estimator Summary

| Variant | Variance estimator | Valid under dependence? |
|---------|--------------------|------------------------|
| Classical | Sample variance / n | No |
| PPI++ (standard) | i.i.d. PPI++ variance | No |
| Block Bootstrap PPI++ | Bootstrap variance over resampled blocks | Yes |

---

## Installation

```bash
git clone https://github.com/[anonymous]/Dependent-AutoEval-Block-Bootstrap-and-Graph-Cluster-Bootstrap.git
cd Dependent-AutoEval-Block-Bootstrap-and-Graph-Cluster-Bootstrap
conda activate autoeval
pip install -e .
```

**Dependencies:** Python 3.10, NumPy 1.24, SciPy 1.10, scikit-learn 1.2, NetworkX 3.1, PyTorch ≥ 2.0, PyTorch Geometric 2.3, pandas 1.5, matplotlib 3.7.

---

## Data Setup

**Cora / Citeseer:** Downloaded automatically by PyTorch Geometric at first run. No manual steps needed.

**UCI Air Quality:** Manual download required.

1. Visit: https://archive.ics.uci.edu/dataset/360/air+quality
2. Download `AirQualityUCI.zip` and extract it.
3. Place `AirQualityUCI.xlsx` in `data/airquality/`.

Column used: `PT08.S1(CO)` as synthetic annotator for `CO(GT)` ground truth (Pearson r = 0.88 after standardisation).

---

## Experiment Scripts

Each script is self-contained: it loads data, runs all three estimators (classical, PPI++, block bootstrap), scores results, and saves figures.

| Script | Setting | Dataset | Key Finding |
|--------|---------|---------|-------------|
| `run_extension2a_ar1.py` | Synthetic TS | AR(1), T=2000 | Coverage 0.932 → 0.960; ESS 1.521 |
| `run_extension2a_ma2.py` | Synthetic TS | MA(2), T=2000 | Coverage 0.928 → 0.968; ESS 1.686 |
| `run_extension2b_graphs.py` | Real graphs | Cora + Citeseer | Citeseer 0.884 → 1.000 |
| `run_extension2_airquality.py` | Real TS | UCI Air Quality, T=7344 | Coverage 0.920 → 1.000; 82% MSE reduction |
| `compile_extension2_summary.py` | All | Cross-dataset | Summary figures |
| `generate_motivating_figure.py` | All | Citeseer + AQ | Motivating figure for paper |

### Running Experiments

```bash
conda activate autoeval

# Synthetic time series
python scripts/run_extension2a_ar1.py
python scripts/run_extension2a_ma2.py

# Real-world graphs (auto-downloads Cora + Citeseer)
python scripts/run_extension2b_graphs.py

# UCI Air Quality (requires AirQualityUCI.xlsx in data/airquality/)
python scripts/run_extension2_airquality.py

# Cross-dataset summary figures
python scripts/compile_extension2_summary.py
```

### Output Locations

| Script | Output directory |
|--------|-----------------|
| AR(1) | `results/extension2a/` |
| MA(2) | `results/extension2a/ma2/` |
| Graphs | `results/extension2b/` |
| Air Quality | `results/extension2_airquality/` |
| Summary | `results/extension2_summary/` |

Each directory contains a CSV of per-trial results and PNG figures. CSVs and figures are version-controlled so results can be inspected without re-running.

### Expected Runtimes

| Script | Runtime |
|--------|---------|
| `run_extension2a_ar1.py` | ~2–3 min |
| `run_extension2a_ma2.py` | ~2–3 min |
| `run_extension2b_graphs.py` | ~5–8 min |
| `run_extension2_airquality.py` | ~3–5 min |
| `compile_extension2_summary.py` | ~1 min |

All experiments run single-core CPU. Seed 42 throughout.

---

## Implementation Details

### Block Length Selection

Block length L = ⌊√T⌋ throughout. For T=2000 (synthetic): L=44, B=45 blocks. For T=7344 (Air Quality): L=85, B=86 blocks.

### Synthetic Time Series

AR(1)/MA(2) data: Yₜ = sin(Xₜ) + ηₜ, ηₜ ~ N(0, 0.25). Synthetic annotator is a weak LinearRegression (Pearson r ≈ 0.61).

### Graph Experiments

GCN: 2-layer, 64 hidden units, 200 epochs, Adam lr=0.01, weight decay 5e-4, standard 140-node training split. Synthetic annotator uses softmax confidence. Community detection: greedy modularity maximization, min community size 5. K=7 for Cora, K=6 for Citeseer.

### n-sweep for Graphs

n ∈ {50, 100, 150, 200, 250, 300}. Stops at 300 because the labeled fraction exceeds 9% of Citeseer beyond this point.

---

## Repository Structure

```
dependent_autoeval/
├── __init__.py
├── estimators.py          # ppi_estimate: PPI++ point estimate + lambda*
└── bootstrap.py           # block_bootstrap_ci, graph_cluster_bootstrap_ci
scripts/
├── run_extension2a_ar1.py
├── run_extension2a_ma2.py
├── run_extension2b_graphs.py
├── run_extension2_airquality.py
├── compile_extension2_summary.py
└── generate_motivating_figure.py
results/
└── experiments.md         # Full results table + reproduction notes
data/
├── graphs/README.md       # PyTorch Geometric auto-download
└── airquality/README.md   # UCI manual download instructions
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Companion Repository

The companion paper addresses covariate shift in AutoEval:

**Adaptive AutoEval: Learning Importance Weights from Unlabeled Data under Covariate Shift**
*Anonymous submission — NeurIPS 2026 Main Track* (Under Review)

---

## Citation

```bibtex
@article{dependent_autoeval_2026,
  title   = {{AutoEval} for Dependent Data via Block and Graph-Cluster Bootstrap},
  author  = {Anonymous},
  journal = {Journal of Machine Learning Research},
  year    = {2026}
}
```