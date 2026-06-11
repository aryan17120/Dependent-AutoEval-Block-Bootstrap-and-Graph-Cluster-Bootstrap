# Experiment Results

All results use n=200 labeled nodes/observations, 250 trials, 90% nominal CI (α=0.10).

## Summary Table

| Dataset      | Type        | Ann. r | Cov (cls) | Cov (ppi) | Cov (blk) | ESS (ppi) | ESS (blk) | MSE ratio |
|--------------|-------------|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| AR(1)        | Synthetic TS | 0.61  | 0.900     | 0.932     | 0.960     | 1.521     | 1.078     | 0.604     |
| MA(2)        | Synthetic TS | 0.61  | 0.908     | 0.928     | 0.968     | 1.686     | 1.248     | 0.503     |
| Cora         | Real Graph  | 0.47   | 0.900     | 0.912     | 0.996     | 1.267     | 0.370     | 0.763     |
| Citeseer     | Real Graph  | 0.38   | 0.904     | 0.884     | 1.000     | 1.167     | 0.435     | 0.913     |
| Air Quality  | Real TS     | 0.88   | 0.880     | 0.920     | 1.000     | 4.141     | 1.696     | 0.182     |

## Key Observations

- **Citeseer PPI++ coverage 0.884** (below nominal 0.900): starkest failure of standard AutoEval under dependence.
- **Air Quality classical coverage 0.880**: dependence affects even the classical estimator.
- **Air Quality ESS(ppi) 4.141, MSE ratio 0.182**: 82% reduction in MSE from using the synthetic annotator.
- **Graph ESS(blk) 0.370–0.435**: reflects the structural cost of resampling at community granularity (K/n = 7/200 = 0.035 for Cora). This is a principled information-theoretic cost, not a failure.

## Reproducing Results

```bash
conda activate autoeval

# Synthetic time series
python scripts/run_extension2a_ar1.py
python scripts/run_extension2a_ma2.py

# Real-world graphs
python scripts/run_extension2b_graphs.py

# UCI Air Quality
python scripts/run_extension2_airquality.py

# Cross-dataset summary figures
python scripts/compile_extension2_summary.py
```

Results are saved to `results/extension2a/`, `results/extension2b/`,
`results/extension2_airquality/`, and `results/extension2_summary/`.