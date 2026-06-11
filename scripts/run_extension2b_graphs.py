"""
Extension 2b: AutoEval for Non-I.I.D. Data — Graphs (Cora & Citeseer)
Boyeau et al. (2025) "AutoEval Done Right" — Extension Plan Section 3.4

Target result:
  Coverage ~0.90, ESS improvement ~25% on Cora and Citeseer
  node classification with GCN predictions as synthetic annotator.

Pipeline:
  1. Load Cora / Citeseer via torch_geometric
  2. Train 2-layer GCN → softmax probabilities as synthetic annotator
  3. Detect communities via greedy modularity (networkx)
  4. Run PPI++ with cluster-based graph bootstrap
  5. Compare: Classical vs Standard PPI++ vs Cluster Bootstrap PPI++

Metric: accuracy of the GCN model (estimated from labeled subset)
  phi(v)  = 1(GCN_pred(v) == true_label(v))   [ground truth correctness]
  E_hat(v)= GCN_softmax(v)[pred_class]          [synthetic expectation]

Outputs:
  results/extension2b/ext2b_{dataset}_results.csv
  results/extension2b/ext2b_combined.png
  results/extension2b/ext2b_coverage.png
  results/extension2b/ext2b_ess.png
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["svg.fonttype"] = "none"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_DIR    = "data/graphs"
OUT_DIR     = "results/extension2b"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

DATASETS    = ["Cora", "Citeseer"]
N_TRIALS    = 250
N_LIST      = [50, 100, 150, 200, 250, 300]
ALPHA       = 0.1
N_BOOTSTRAP = 500
GCN_EPOCHS  = 200
GCN_HIDDEN  = 64
GCN_LR      = 0.01
GCN_WD      = 5e-4
SEED        = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------
# STEP 1 — GCN model definition
# --------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_probs(self, x, edge_index):
        """Return softmax probabilities (not log)."""
        self.eval()
        with torch.no_grad():
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return F.softmax(x, dim=1)

# --------------------------------------------------
# STEP 2 — Train GCN and extract predictions
# --------------------------------------------------

def train_gcn(data, epochs=200, hidden=64, lr=0.01, wd=5e-4):
    """Train GCN and return softmax probabilities for all nodes."""
    device = torch.device("cpu")
    model  = GCN(data.num_features, hidden, data.y.max().item() + 1).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for epoch in range(epochs):
        optim.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optim.step()

    probs = model.get_probs(data.x, data.edge_index)  # (N, C)
    return probs.numpy()

# --------------------------------------------------
# STEP 3 — Graph clustering (community detection)
# --------------------------------------------------

def detect_communities(data, min_size=5):
    """
    Detect communities using greedy modularity maximization (networkx).
    Returns cluster_ids: array of shape (N,) with cluster index per node.
    Falls back to degree-based binning if networkx community fails.
    """
    G = to_networkx(data, to_undirected=True)

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G)
        cluster_ids = np.zeros(data.num_nodes, dtype=int)
        for c_idx, community in enumerate(communities):
            for node in community:
                cluster_ids[node] = c_idx
        # Merge tiny clusters into nearest large one
        unique, counts = np.unique(cluster_ids, return_counts=True)
        small = unique[counts < min_size]
        if len(small) > 0:
            # Reassign small clusters to cluster 0
            for s in small:
                cluster_ids[cluster_ids == s] = 0
        print(f"  Communities detected: {len(np.unique(cluster_ids))} "
              f"(min size {min_size})")
        return cluster_ids

    except Exception as e:
        print(f"  Community detection fallback: {e}")
        # Fallback: bin nodes by degree into K groups
        degrees = np.array([G.degree(n) for n in range(data.num_nodes)])
        K       = max(10, data.num_nodes // 50)
        cluster_ids = np.digitize(degrees,
                                  bins=np.percentile(degrees,
                                                     np.linspace(0, 100, K)))
        return cluster_ids - 1

# --------------------------------------------------
# STEP 4 — PPI++ estimators
# --------------------------------------------------

def ppi_estimate(phi_lab, syn_lab, syn_unl):
    """Standard PPI++ with i.i.d. variance."""
    n = len(phi_lab)
    N = len(syn_unl)
    if n < 3 or N < 3:
        return float(phi_lab.mean()), float(phi_lab.var()/n), 0.0
    cov_num   = float(np.cov(phi_lab, syn_lab)[0, 1])
    var_denom = (n/N)*float(np.var(syn_unl)) + float(np.var(syn_lab))
    lambd     = float(np.clip(cov_num/(var_denom+1e-12), 0.0, 1.0))
    mu_hat    = lambd*syn_unl.mean() + (phi_lab - lambd*syn_lab).mean()
    resid     = phi_lab - lambd*syn_lab
    var_hat   = float(np.var(resid)/n + lambd**2*np.var(syn_unl)*(n/N)/n)
    return float(mu_hat), var_hat, lambd


def cluster_bootstrap_ppi(phi_all, syn_all, cluster_ids,
                           n_lab, rng_trial, n_boot=500):
    """
    Cluster Bootstrap PPI++ for graphs (Extension Plan §3.4 steps 3-4).

    Resamples graph clusters (communities) with replacement.
    Each bootstrap resample maintains within-cluster node relationships.

    phi_all     : (N_nodes,) ground-truth correctness for all nodes
    syn_all     : (N_nodes,) synthetic expectations for all nodes
    cluster_ids : (N_nodes,) cluster index per node
    n_lab       : number of labeled nodes
    """
    N_nodes   = len(phi_all)
    clusters  = np.unique(cluster_ids)
    n_clusters = len(clusters)

    # Build cluster-level data structures
    cluster_phi = {c: phi_all[cluster_ids == c] for c in clusters}
    cluster_syn = {c: syn_all[cluster_ids == c] for c in clusters}

    # Original labeled/unlabeled split for point estimate
    idx_lab  = rng_trial.choice(N_nodes, size=n_lab, replace=False)
    idx_unl  = np.setdiff1d(np.arange(N_nodes), idx_lab)
    phi_lab  = phi_all[idx_lab]
    syn_lab  = syn_all[idx_lab]
    syn_unl  = syn_all[idx_unl]
    mu_hat, var_iid, lambd = ppi_estimate(phi_lab, syn_lab, syn_unl)

    if n_clusters < 3:
        return mu_hat, var_iid, phi_lab, syn_lab, syn_unl

    # Bootstrap: resample clusters with replacement
    boot_ests = []
    rng_b     = np.random.RandomState(77)

    for _ in range(n_boot):
        # Resample clusters
        chosen_c  = rng_b.choice(clusters, size=n_clusters, replace=True)
        phi_boot  = np.concatenate([cluster_phi[c] for c in chosen_c])
        syn_boot  = np.concatenate([cluster_syn[c] for c in chosen_c])
        N_b       = len(phi_boot)

        if N_b < n_lab + 3:
            continue

        # Random labeled/unlabeled split within bootstrap graph
        idx_b  = rng_b.choice(N_b, size=n_lab, replace=False)
        idx_ub = np.setdiff1d(np.arange(N_b), idx_b)

        try:
            mb, _, _ = ppi_estimate(
                phi_boot[idx_b], syn_boot[idx_b], syn_boot[idx_ub]
            )
            boot_ests.append(mb)
        except:
            continue

    if len(boot_ests) < 10:
        return mu_hat, var_iid, phi_lab, syn_lab, syn_unl

    return float(mu_hat), float(np.var(boot_ests)), phi_lab, syn_lab, syn_unl


def run_graph_experiment(phi_all, syn_all, mu_gt, cluster_ids,
                          n_list, n_trials, n_boot):
    """Run Monte Carlo for one dataset."""
    results = []

    for n in n_list:
        cov_cls, cov_ppi, cov_blk = [], [], []
        wid_cls, wid_ppi, wid_blk = [], [], []
        mse_cls, mse_ppi, mse_blk = [], [], []
        ess_ppi, ess_blk, lam_list = [], [], []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 1000 + n)

            mu_b, var_b, phi_lab, syn_lab, syn_unl = cluster_bootstrap_ppi(
                phi_all, syn_all, cluster_ids,
                n_lab=n, rng_trial=rng, n_boot=n_boot
            )
            mu_c  = float(phi_lab.mean())
            var_c = float(phi_lab.var() / n)
            mu_p, var_p, lam = ppi_estimate(phi_lab, syn_lab, syn_unl)

            z = norm.ppf(1 - ALPHA / 2)

            def cov_fn(mu, var):
                lo = mu - z*np.sqrt(max(var, 1e-12))
                hi = mu + z*np.sqrt(max(var, 1e-12))
                return float(mu_gt >= lo and mu_gt <= hi)

            def wid_fn(var):
                return float(2*z*np.sqrt(max(var, 1e-12)))

            cov_cls.append(cov_fn(mu_c, var_c))
            cov_ppi.append(cov_fn(mu_p, var_p))
            cov_blk.append(cov_fn(mu_b, var_b))
            wid_cls.append(wid_fn(var_c))
            wid_ppi.append(wid_fn(var_p))
            wid_blk.append(wid_fn(var_b))
            mse_cls.append((mu_c - mu_gt)**2)
            mse_ppi.append((mu_p - mu_gt)**2)
            mse_blk.append((mu_b - mu_gt)**2)
            ess_ppi.append(var_c / (var_p + 1e-12))
            ess_blk.append(var_c / (var_b + 1e-12))
            lam_list.append(lam)

        sm = lambda l: float(np.mean(l))
        row = dict(n=n,
                   cov_cls=sm(cov_cls), cov_ppi=sm(cov_ppi),
                   cov_blk=sm(cov_blk),
                   wid_cls=sm(wid_cls), wid_ppi=sm(wid_ppi),
                   wid_blk=sm(wid_blk),
                   mse_cls=sm(mse_cls), mse_ppi=sm(mse_ppi),
                   mse_blk=sm(mse_blk),
                   ess_ppi=sm(ess_ppi), ess_blk=sm(ess_blk),
                   lambda_mean=sm(lam_list))
        results.append(row)
        print(f"  n={n:4d} | Cov cls={row['cov_cls']:.3f} "
              f"ppi={row['cov_ppi']:.3f} blk={row['cov_blk']:.3f} | "
              f"ESS ppi={row['ess_ppi']:.3f} blk={row['ess_blk']:.3f}")

    return pd.DataFrame(results)

# --------------------------------------------------
# STEP 5 — Main loop over datasets
# --------------------------------------------------

all_results = {}

for dataset_name in DATASETS:
    print()
    print("=" * 72)
    print(f"Dataset: {dataset_name}")
    print("=" * 72)

    # Load dataset
    dataset = Planetoid(root=DATA_DIR, name=dataset_name)
    data    = dataset[0]
    N_nodes = data.num_nodes
    print(f"Nodes: {N_nodes} | Edges: {data.edge_index.shape[1]//2} | "
          f"Features: {data.num_features} | Classes: {dataset.num_classes}")

    # Train GCN
    print(f"Training GCN ({GCN_EPOCHS} epochs) ...")
    probs = train_gcn(data, GCN_EPOCHS, GCN_HIDDEN, GCN_LR, GCN_WD)

    # Derive phi and synthetic expectations
    labels      = data.y.numpy()                        # (N,) true labels
    pred_class  = probs.argmax(axis=1)                  # (N,) predicted class
    phi_all     = (pred_class == labels).astype(float)  # (N,) correctness
    syn_all     = probs[np.arange(N_nodes), pred_class] # (N,) top softmax score

    mu_gt = float(phi_all.mean())
    corr  = float(np.corrcoef(phi_all, syn_all)[0, 1])
    print(f"GCN accuracy (full): {mu_gt:.4f}")
    print(f"Annotator corr:      {corr:.4f}")

    # Community detection
    print("Detecting communities ...")
    cluster_ids = detect_communities(data)
    n_clusters  = len(np.unique(cluster_ids))

    # Run experiment
    print(f"Running Monte Carlo: {N_TRIALS} trials x {len(N_LIST)} n values ...")
    print(f"Clusters: {n_clusters} | Bootstrap: {N_BOOTSTRAP} resamples")
    print("-" * 72)

    df = run_graph_experiment(
        phi_all, syn_all, mu_gt, cluster_ids,
        N_LIST, N_TRIALS, N_BOOTSTRAP
    )

    csv_path = os.path.join(OUT_DIR, f"ext2b_{dataset_name.lower()}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    all_results[dataset_name] = df

# --------------------------------------------------
# STEP 6 — Figures
# --------------------------------------------------

try:
    CC, CP, CB = "#4DAF4A", "#E41A1C", "#377EB8"

    # --- Figure 1: Combined coverage for both datasets ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, dname in zip(axes, DATASETS):
        df = all_results[dname]
        ns = df["n"].values
        ax.plot(ns, df["cov_cls"], "o-",  color=CC,
                label="Classical", linewidth=2)
        ax.plot(ns, df["cov_ppi"], "s--", color=CP,
                label="PPI++ (standard)", linewidth=2)
        ax.plot(ns, df["cov_blk"], "o-",  color=CB,
                label="Cluster Bootstrap PPI++", linewidth=2)
        ax.axhline(1-ALPHA, color="black", ls=":", lw=1.5,
                   label="Target 90%")
        ax.set_xlabel("# labeled nodes")
        ax.set_ylabel("Coverage of 90% CIs")
        ax.set_title(f"{dname} — Coverage")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Extension 2b: Graph Bootstrap AutoEval — Cora & Citeseer\n"
        "GCN predictions as synthetic annotator | "
        "Cluster-based graph bootstrap",
        fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "ext2b_coverage.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {p1}")
    plt.close()

    # --- Figure 2: ESS for both datasets ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, dname in zip(axes, DATASETS):
        df = all_results[dname]
        ns = df["n"].values
        ax.plot(ns, df["ess_ppi"], "s--", color=CP,
                label="PPI++ (standard)", linewidth=2)
        ax.plot(ns, df["ess_blk"], "o-",  color=CB,
                label="Cluster Bootstrap PPI++", linewidth=2)
        ax.axhline(1.0, color=CC, ls="--", lw=1.5,
                   label="Classical (=1)")
        ax.set_xlabel("# labeled nodes")
        ax.set_ylabel("ESS")
        ax.set_title(f"{dname} — ESS")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Extension 2b: ESS — Graph Bootstrap AutoEval (Cora & Citeseer)",
        fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "ext2b_ess.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {p2}")
    plt.close()

    # --- Figure 3: Combined 3-panel for one dataset (Cora) ---
    df  = all_results["Cora"]
    ns  = df["n"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ns, df["cov_cls"], "o-",  color=CC, label="Classical")
    ax.plot(ns, df["cov_ppi"], "s--", color=CP, label="PPI++ (standard)")
    ax.plot(ns, df["cov_blk"], "o-",  color=CB,
            label="Cluster Bootstrap PPI++")
    ax.axhline(1-ALPHA, color="black", ls=":", lw=1.2, label="Target 90%")
    ax.set_xlabel("# labeled nodes"); ax.set_ylabel("Coverage")
    ax.set_title("(a) Coverage"); ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ns, df["mse_cls"], "o-",  color=CC, label="Classical")
    ax.plot(ns, df["mse_ppi"], "s--", color=CP, label="PPI++ (standard)")
    ax.plot(ns, df["mse_blk"], "o-",  color=CB,
            label="Cluster Bootstrap PPI++")
    ax.set_xlabel("# labeled nodes"); ax.set_ylabel("MSE")
    ax.set_title("(b) MSE")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ns, df["ess_ppi"], "s--", color=CP, label="PPI++ (standard)")
    ax.plot(ns, df["ess_blk"], "o-",  color=CB,
            label="Cluster Bootstrap PPI++")
    ax.axhline(1.0, color=CC, ls="--", lw=1.0, label="Classical (=1)")
    ax.set_xlabel("# labeled nodes"); ax.set_ylabel("ESS")
    ax.set_title("(c) ESS"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    fig.suptitle(
        "Extension 2b: Graph Bootstrap AutoEval — Cora\n"
        "GCN synthetic annotator | Greedy modularity communities",
        fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "ext2b_combined.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {p3}")
    plt.close()

except Exception as e:
    print(f"Plotting error: {e}")
    import traceback; traceback.print_exc()

# --------------------------------------------------
# STEP 7 — Verification summary
# --------------------------------------------------

print("\n" + "="*72)
print("VERIFICATION SUMMARY — Extension 2b")
print("="*72)

for dname in DATASETS:
    df = all_results[dname]
    print(f"\n{dname}:")
    print(f"{'n':>5} | {'cov_cls':>8} | {'cov_ppi':>8} | "
          f"{'cov_blk':>8} | {'ESS_ppi':>8} | {'ESS_blk':>8}")
    print("-"*60)
    for _, r in df.iterrows():
        print(f"{int(r.n):>5} | {r.cov_cls:>8.3f} | {r.cov_ppi:>8.3f} | "
              f"{r.cov_blk:>8.3f} | {r.ess_ppi:>8.3f} | {r.ess_blk:>8.3f}")

    # ESS improvement at n=200
    row200 = df[df["n"] == 200].iloc[0]
    ess_improvement = (row200["ess_ppi"] - 1.0) * 100
    print(f"\n  ESS improvement at n=200: {ess_improvement:.1f}%")
    print(f"  Coverage (blk) at n=200: {row200['cov_blk']:.3f} "
          f"(target: {1-ALPHA:.2f})")

print()
print("Target: coverage ~0.90, ESS improvement ~25%")
print("\nDone. Extension 2b complete.")