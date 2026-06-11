# Graph Data

Cora and Citeseer are downloaded automatically by PyTorch Geometric at runtime.

The download is triggered by `run_extension2b_graphs.py` via:

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root="data/graphs", name="Cora")
```

Downloaded files are excluded from version control (see `.gitignore`).
If you are running offline, download the datasets manually from:
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html