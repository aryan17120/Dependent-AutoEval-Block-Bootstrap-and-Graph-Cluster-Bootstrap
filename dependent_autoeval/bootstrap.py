"""
Bootstrap confidence intervals for AutoEval under dependent data.

Two procedures:
  - block_bootstrap_ci      : non-overlapping block bootstrap for time series
  - graph_cluster_bootstrap_ci : community-cluster bootstrap for networks
"""

import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from .estimators import ppi_estimate


# ---------------------------------------------------------------------------
# Block bootstrap for time series
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    phi_series,
    syn_series,
    n_labeled,
    alpha=0.10,
    n_boot=500,
    block_len=None,
    seed=99,
):
    """
    Non-overlapping block bootstrap CI for PPI++ on time-series data.

    Parameters
    ----------
    phi_series : np.ndarray, shape (T,)
        True scores for the full time series (labeled positions drawn from this).
    syn_series : np.ndarray, shape (T,)
        Synthetic annotator scores for the full time series.
    n_labeled : int
        Number of labeled observations to draw per replicate.
    alpha : float
        Nominal error rate; CI covers at 1-alpha.
    n_boot : int
        Number of bootstrap replicates.
    block_len : int or None
        Block length L. Defaults to floor(sqrt(T)).
    seed : int
        Seed for the bootstrap RNG (separate from trial-level seeds).

    Returns
    -------
    ci_low : float
    ci_high : float
    mu_hat : float
        Point estimate on the original (non-bootstrapped) data.
    ess : float
        Effective sample size ratio: Var_iid / Var_block.
    """
    T = len(phi_series)
    if block_len is None:
        block_len = int(np.floor(np.sqrt(T)))

    n_blocks = T // block_len
    rng = np.random.RandomState(seed)

    # Point estimate on original data
    lab_idx = rng.choice(T, size=n_labeled, replace=False)
    phi_lab = phi_series[lab_idx]
    syn_lab = syn_series[lab_idx]
    syn_unl = np.delete(syn_series, lab_idx)
    mu_hat, var_iid, lam_opt = ppi_estimate(phi_lab, syn_lab, syn_unl)

    # Block indices
    blocks = [
        np.arange(b * block_len, (b + 1) * block_len)
        for b in range(n_blocks)
    ]

    boot_mus = []
    for _ in range(n_boot):
        chosen = rng.choice(n_blocks, size=n_blocks, replace=True)
        boot_idx = np.concatenate([blocks[b] for b in chosen])
        boot_phi = phi_series[boot_idx]
        boot_syn = syn_series[boot_idx]

        T_boot = len(boot_idx)
        lab_idx_b = rng.choice(T_boot, size=n_labeled, replace=False)
        phi_lab_b = boot_phi[lab_idx_b]
        syn_lab_b = boot_syn[lab_idx_b]
        syn_unl_b = np.delete(boot_syn, lab_idx_b)

        mu_b, _, _ = ppi_estimate(phi_lab_b, syn_lab_b, syn_unl_b, lam=lam_opt)
        boot_mus.append(mu_b)

    boot_mus = np.array(boot_mus)
    ci_low = float(np.percentile(boot_mus, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_mus, 100 * (1 - alpha / 2)))

    var_block = float(np.var(boot_mus, ddof=1))
    ess = float(var_iid / var_block) if var_block > 1e-15 else np.nan

    return ci_low, ci_high, mu_hat, ess


# ---------------------------------------------------------------------------
# Graph-cluster bootstrap for networks
# ---------------------------------------------------------------------------

def graph_cluster_bootstrap_ci(
    G,
    phi_all,
    syn_all,
    n_labeled,
    alpha=0.10,
    n_boot=500,
    min_community_size=5,
    seed=99,
):
    """
    Community-cluster bootstrap CI for PPI++ on graph-structured data.

    Community detection uses greedy modularity maximization
    (networkx.algorithms.community.greedy_modularity_communities).

    Parameters
    ----------
    G : nx.Graph
        The full graph. Node indices must correspond to rows of phi_all/syn_all.
    phi_all : np.ndarray, shape (N,)
        True label scores for all N nodes.
    syn_all : np.ndarray, shape (N,)
        Synthetic annotator scores for all N nodes.
    n_labeled : int
        Number of labeled nodes to draw per replicate.
    alpha : float
        Nominal error rate.
    n_boot : int
        Number of bootstrap replicates.
    min_community_size : int
        Communities smaller than this are merged into the largest community.
    seed : int
        Seed for the bootstrap RNG.

    Returns
    -------
    ci_low : float
    ci_high : float
    mu_hat : float
    ess : float
    communities : list of lists
        Final community assignments after size-based merging.
    """
    N = len(phi_all)
    rng = np.random.RandomState(seed)

    # Community detection
    raw_communities = list(greedy_modularity_communities(G))
    communities = _merge_small_communities(raw_communities, min_community_size)
    K = len(communities)

    # Point estimate on original data
    lab_idx = rng.choice(N, size=n_labeled, replace=False)
    phi_lab = phi_all[lab_idx]
    syn_lab = syn_all[lab_idx]
    syn_unl = np.delete(syn_all, lab_idx)
    mu_hat, var_iid, lam_opt = ppi_estimate(phi_lab, syn_lab, syn_unl)

    boot_mus = []
    for _ in range(n_boot):
        chosen = rng.choice(K, size=K, replace=True)
        boot_nodes = np.concatenate([list(communities[c]) for c in chosen])
        boot_phi = phi_all[boot_nodes]
        boot_syn = syn_all[boot_nodes]

        N_boot = len(boot_nodes)
        lab_idx_b = rng.choice(N_boot, size=n_labeled, replace=False)
        phi_lab_b = boot_phi[lab_idx_b]
        syn_lab_b = boot_syn[lab_idx_b]
        syn_unl_b = np.delete(boot_syn, lab_idx_b)

        mu_b, _, _ = ppi_estimate(phi_lab_b, syn_lab_b, syn_unl_b, lam=lam_opt)
        boot_mus.append(mu_b)

    boot_mus = np.array(boot_mus)
    ci_low = float(np.percentile(boot_mus, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_mus, 100 * (1 - alpha / 2)))

    var_block = float(np.var(boot_mus, ddof=1))
    ess = float(var_iid / var_block) if var_block > 1e-15 else np.nan

    return ci_low, ci_high, mu_hat, ess, communities


def _merge_small_communities(communities, min_size):
    """Merge communities smaller than min_size into the largest community."""
    communities = [list(c) for c in communities]
    large = [c for c in communities if len(c) >= min_size]
    small = [c for c in communities if len(c) < min_size]

    if not large:
        # All communities are small; return as single group
        return [sum(communities, [])]

    largest_idx = int(np.argmax([len(c) for c in large]))
    for s in small:
        large[largest_idx].extend(s)

    return large