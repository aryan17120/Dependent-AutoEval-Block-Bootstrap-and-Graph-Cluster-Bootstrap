# Dependent AutoEval: Block Bootstrap and Graph-Cluster Bootstrap

Official implementation for the paper:

> **AutoEval for Dependent Data via Block and Graph-Cluster Bootstrap**  
> Aryan Saini, Vinayak M, Sanjay Singh  
> *IEEE Transactions on Artificial Intelligence* (under review)

This repository extends the [AutoEval / PPI++ framework](https://github.com/Akoasm666/autoeval)
to non-i.i.d. settings. Standard AutoEval fails under temporal or graph-structured
dependence because the i.i.d. variance estimator underestimates the true long-run
variance, producing confidence intervals that are too narrow. We propose two
bootstrap procedures that correct this without changing the point estimate.

**Companion repository (covariate shift):**
[Adaptive-AutoEval-Learning-Importance-Weights-from-Unlabeled-Data-under-Covariate-Shift](https://github.com/aryan17120/Adaptive-AutoEval-Learning-Importance-Weights-from-Unlabeled-Data-under-Covariate-Shift)

---

## Method Overview

### Problem

For dependent data, the i.i.d. variance underestimates the true long-run variance:

$$\sigma^2_{\mathrm{LRV}} = \sigma^2_Z \left(1 + 2\sum_{k=1}^{\infty} \rho_k\right)$$

This causes PPI++ confidence intervals to under-cover the true parameter.

### Two Procedures

| Setting | Method | Block unit |
|---------|--------|------------|
| Time series | Non-overlapping block bootstrap | Length $L = \lfloor\sqrt{T}\rfloor$ |
| Networks | Graph-cluster bootstrap | Community (greedy modularity maximization) |

$\lambda^*$ is held fixed across bootstrap replicates (one-time estimation on original data),
following Lahiri (2003) Thm 3.1.

---

## Results Summary

| Dataset | Cov (ppi) | Cov (blk) | ESS (ppi) | ESS (blk) | MSE ratio |
|---------|-----------|-----------|-----------|-----------|-----------|
| AR(1) | 0.932 | **0.960** | 1.521 | 1.078 | 0.604 |
| MA(2) | 0.928 | **0.968** | 1.686 | 1.248 | 0.503 |
| Cora | 0.912 | **0.996** | 1.267 | 0.370 | 0.763 |
| Citeseer | 0.884 | **1.000** | 1.167 | 0.435 | 0.913 |
| Air Quality | 0.920 | **1.000** | 4.141 | 1.696 | 0.182 |

Nominal coverage: 90%. n=200, 250 trials.

ESS(blk) < 1 on graph datasets is expected: resampling at community granularity
(K=7 communities for n=200 labeled nodes) reflects the true information content
of the graph structure.

---

## Installation

```bash
git clone https://github.com/aryan17120/Dependent-AutoEval-Block-Bootstrap-and-Graph-Cluster-Bootstrap.git
cd Dependent-AutoEval-Block-Bootstrap-and-Graph-Cluster-Bootstrap
conda activate autoeval
pip install -e .
```

**Dependencies:** Python 3.10, NumPy 1.24, SciPy 1.10, scikit-learn 1.2,
NetworkX 3.1, PyTorch ≥ 2.0, PyTorch Geometric 2.3, pandas 1.5, matplotlib 3.7.

---

## Data Setup

**Cora / Citeseer:** Downloaded automatically by PyTorch Geometric at runtime.

**UCI Air Quality:** Download manually from
[UCI repository](https://archive.ics.uci.edu/dataset/360/air+quality),
extract, and place `AirQualityUCI.xlsx` in `data/airquality/`.

---

## Reproducing Experiments

```bash
conda activate autoeval

python scripts/run_extension2a_ar1.py        # AR(1) synthetic time series
python scripts/run_extension2a_ma2.py        # MA(2) synthetic time series
python scripts/run_extension2b_graphs.py     # Cora + Citeseer
python scripts/run_extension2_airquality.py  # UCI Air Quality
python scripts/compile_extension2_summary.py # Cross-dataset figures
```

Results are saved to `results/`. CSVs and figures are version-controlled
so reviewers can inspect outputs without re-running experiments.

---

## Repository Structure

```
dependent_autoeval/
├── __init__.py
├── estimators.py     # ppi_estimate (PPI++ point estimate + lambda*)
└── bootstrap.py      # block_bootstrap_ci, graph_cluster_bootstrap_ci
scripts/              # Experiment scripts (one per dataset)
results/
└── experiments.md    # Full results table and reproduction instructions
data/
├── graphs/README.md  # PyTorch Geometric auto-download
└── airquality/README.md
```

---

## Citation

```bibtex
@article{saini2025dependent,
  title     = {{AutoEval} for Dependent Data via Block and Graph-Cluster Bootstrap},
  author    = {Saini, Aryan and M, Vinayak and Singh, Sanjay},
  journal   = {IEEE Transactions on Artificial Intelligence},
  year      = {2025},
  note      = {Under review}
}
```

---
