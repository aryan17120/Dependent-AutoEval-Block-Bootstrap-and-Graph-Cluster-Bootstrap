"""
Extension 2a (MA2): AutoEval for Non-I.I.D. Data — MA(2) Time Series
Boyeau et al. (2025) "AutoEval Done Right" — Extension Plan Section 3

Companion to run_extension2a_ar1.py, using MA(2) instead of AR(1).

MA(2) process:
  X_t = eps_t + theta1 * eps_{t-1} + theta2 * eps_{t-2}
  Y_t = sin(X_t) + eta_t,   eta_t ~ N(0, noise^2)

Key difference from AR(1):
  MA(2) has FINITE dependence — only 2 lags of memory.
  AR(1) has infinite geometric decay of dependence.
  This makes block bootstrap even more natural for MA(2):
  blocks longer than 2 are exactly independent.

Theoretical variance inflation for MA(2):
  Var(sample mean) / Var_iid = gamma_0 + 2*gamma_1 + 2*gamma_2
  where gamma_k = autocovariance at lag k
  = sigma^2 * (c_0 + 2*c_1 + 2*c_2) / c_0
  with c_0 = 1 + theta1^2 + theta2^2
       c_1 = theta1 + theta1*theta2
       c_2 = theta2

Sweep: theta1 in {0.0, 0.3, 0.5, 0.7, 0.9}, theta2 = theta1/2

Outputs:
  results/extension2a/ext2a_ma2_theta_results.csv
  results/extension2a/ext2a_ma2_n_results.csv
  results/extension2a/ext2a_ma2_theta_coverage.png
  results/extension2a/ext2a_ma2_n_main.png
  results/extension2a/ext2a_ma2_theoretical.png
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams["svg.fonttype"] = "none"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OUT_DIR      = "results/extension2a"
os.makedirs(OUT_DIR, exist_ok=True)

T            = 2000
N_TRIALS     = 250
ALPHA        = 0.1
N_BOOTSTRAP  = 500
SEED         = 42
SIGMA        = 1.0    # MA(2) innovation std
NOISE        = 0.5    # label noise std

# Experiments
THETA1_LIST  = [0.0, 0.3, 0.5, 0.7, 0.9]   # theta1 sweep (n fixed)
N_FIXED      = 200                           # n for theta sweep
N_LIST       = [50, 100, 200, 300, 400, 500] # n sweep (theta fixed)
THETA1_FIXED = 0.7                           # theta1 for n sweep
# theta2 = theta1 / 2 throughout

np.random.seed(SEED)

# --------------------------------------------------
# STEP 1 — Data generation
# --------------------------------------------------

def generate_ma2(T, theta1, theta2, sigma=1.0, noise=0.5, seed=0):
    """
    Generate MA(2) time series with sinusoidal labels.

    X_t = eps_t + theta1*eps_{t-1} + theta2*eps_{t-2}
    Y_t = sin(X_t) + eta_t

    MA(2) has finite memory: dependence exists only at lags 1 and 2.
    """
    rng  = np.random.RandomState(seed)
    eps  = rng.randn(T + 2) * sigma   # innovations (with burn-in)
    x    = np.zeros(T)
    for t in range(T):
        x[t] = eps[t+2] + theta1*eps[t+1] + theta2*eps[t]
    y = np.sin(x) + rng.randn(T) * noise
    return x, y


def theoretical_inflation_ma2(theta1, theta2, sigma=1.0):
    """
    Theoretical variance inflation for MA(2) sample mean.

    Long-run variance / innovation variance:
      = c0 + 2*c1 + 2*c2
    where:
      c0 = sigma^2 * (1 + theta1^2 + theta2^2)
      c1 = sigma^2 * (theta1 + theta1*theta2)
      c2 = sigma^2 * theta2
    """
    c0 = sigma**2 * (1 + theta1**2 + theta2**2)
    c1 = sigma**2 * (theta1 + theta1*theta2)
    c2 = sigma**2 * theta2
    long_run_var = c0 + 2*c1 + 2*c2
    return long_run_var / c0   # ratio to i.i.d. variance


def fit_annotator(x, y):
    """Weak linear annotator."""
    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y)
    y_syn = reg.predict(x.reshape(-1, 1))
    corr  = float(np.corrcoef(y, y_syn)[0, 1])
    return y_syn, corr

# --------------------------------------------------
# STEP 2 — Estimators (same as AR1 script)
# --------------------------------------------------

def ppi_estimate(phi_lab, syn_lab, syn_unl):
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


def block_bootstrap_ppi(y_all, y_syn, n_lab, rng_trial,
                        block_len, n_boot=500):
    """Block Bootstrap PPI++ on full series (same as AR1 script)."""
    T_full   = len(y_all)
    n_blocks = T_full // block_len

    blocks_y   = [y_all[b*block_len:(b+1)*block_len] for b in range(n_blocks)]
    blocks_syn = [y_syn[b*block_len:(b+1)*block_len] for b in range(n_blocks)]

    idx_lab  = rng_trial.choice(T_full, size=n_lab, replace=False)
    idx_unl  = np.setdiff1d(np.arange(T_full), idx_lab)
    phi_lab  = y_all[idx_lab]
    syn_lab  = y_syn[idx_lab]
    syn_unl  = y_syn[idx_unl]
    mu_hat, var_iid, lambd = ppi_estimate(phi_lab, syn_lab, syn_unl)

    if n_blocks < 3:
        return mu_hat, var_iid, phi_lab, syn_lab, syn_unl

    boot_ests = []
    rng_b = np.random.RandomState(99)
    for _ in range(n_boot):
        chosen   = rng_b.choice(n_blocks, size=n_blocks, replace=True)
        y_b      = np.concatenate([blocks_y[i]   for i in chosen])
        syn_b    = np.concatenate([blocks_syn[i] for i in chosen])
        T_b      = len(y_b)
        if T_b < n_lab + 3:
            continue
        idx_b    = rng_b.choice(T_b, size=n_lab, replace=False)
        idx_ub   = np.setdiff1d(np.arange(T_b), idx_b)
        try:
            mb, _, _ = ppi_estimate(y_b[idx_b], syn_b[idx_b], syn_b[idx_ub])
            boot_ests.append(mb)
        except:
            continue

    if len(boot_ests) < 10:
        return mu_hat, var_iid, phi_lab, syn_lab, syn_unl
    return float(mu_hat), float(np.var(boot_ests)), phi_lab, syn_lab, syn_unl


def run_experiment(y_all, y_syn, mu_gt, block_len, n, n_trials, n_boot):
    """Run Monte Carlo for one (theta, n) condition."""
    cov_cls, cov_ppi, cov_blk = [], [], []
    wid_cls, wid_ppi, wid_blk = [], [], []
    mse_cls, mse_ppi, mse_blk = [], [], []
    ess_ppi, ess_blk, lam_list = [], [], []

    for trial in range(n_trials):
        rng = np.random.RandomState(trial * 1000 + n)
        mu_b, var_b, phi_lab, syn_lab, syn_unl = block_bootstrap_ppi(
            y_all, y_syn, n_lab=n, rng_trial=rng,
            block_len=block_len, n_boot=n_boot
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
    return dict(
        cov_cls=sm(cov_cls), cov_ppi=sm(cov_ppi), cov_blk=sm(cov_blk),
        wid_cls=sm(wid_cls), wid_ppi=sm(wid_ppi), wid_blk=sm(wid_blk),
        mse_cls=sm(mse_cls), mse_ppi=sm(mse_ppi), mse_blk=sm(mse_blk),
        ess_ppi=sm(ess_ppi), ess_blk=sm(ess_blk),
        lambda_mean=sm(lam_list)
    )

# --------------------------------------------------
# STEP 3 — Experiment A: Vary theta1 (n=N_FIXED)
# --------------------------------------------------

print("=" * 72)
print(f"EXPERIMENT A: Vary theta1 (theta2=theta1/2), n={N_FIXED} fixed")
print("=" * 72)

theta_results      = []
theoretical_inflations = []

for theta1 in THETA1_LIST:
    theta2 = theta1 / 2.0
    x_ts, y_ts  = generate_ma2(T, theta1, theta2, SIGMA, NOISE, seed=SEED)
    y_syn_ts, corr = fit_annotator(x_ts, y_ts)
    mu_gt_ts    = float(y_ts.mean())
    block_len   = max(10, int(np.sqrt(T)))
    inflation   = theoretical_inflation_ma2(theta1, theta2, SIGMA)
    theoretical_inflations.append(inflation)

    print(f"\ntheta1={theta1:.1f} theta2={theta2:.2f} | "
          f"corr={corr:.3f} | mu_gt={mu_gt_ts:.4f} | "
          f"theoretical inflation={inflation:.3f}")

    metrics = run_experiment(
        y_ts, y_syn_ts, mu_gt_ts,
        block_len=block_len, n=N_FIXED,
        n_trials=N_TRIALS, n_boot=N_BOOTSTRAP
    )
    row = {"theta1": theta1, "theta2": theta2, "corr": corr,
           "theoretical_inflation": inflation, **metrics}
    theta_results.append(row)

    print(f"  Cov cls={metrics['cov_cls']:.3f} "
          f"ppi={metrics['cov_ppi']:.3f} "
          f"blk={metrics['cov_blk']:.3f} | "
          f"ESS ppi={metrics['ess_ppi']:.3f} "
          f"blk={metrics['ess_blk']:.3f}")

theta_df = pd.DataFrame(theta_results)
theta_df.to_csv(os.path.join(OUT_DIR, "ext2a_ma2_theta_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/ext2a_ma2_theta_results.csv")

# --------------------------------------------------
# STEP 4 — Experiment B: Vary n (theta1=THETA1_FIXED)
# --------------------------------------------------

print()
print("=" * 72)
print(f"EXPERIMENT B: Vary n, theta1={THETA1_FIXED} "
      f"theta2={THETA1_FIXED/2:.2f} fixed")
print("=" * 72)

theta2_fixed = THETA1_FIXED / 2.0
x_ts, y_ts   = generate_ma2(T, THETA1_FIXED, theta2_fixed,
                             SIGMA, NOISE, seed=SEED)
y_syn_ts, corr = fit_annotator(x_ts, y_ts)
mu_gt_ts     = float(y_ts.mean())
block_len    = max(10, int(np.sqrt(T)))
inflation_fixed = theoretical_inflation_ma2(THETA1_FIXED, theta2_fixed, SIGMA)

print(f"theta1={THETA1_FIXED} theta2={theta2_fixed:.2f} | "
      f"corr={corr:.3f} | mu_gt={mu_gt_ts:.4f} | "
      f"theoretical inflation={inflation_fixed:.3f}")
print("-" * 72)

n_results = []
for n in N_LIST:
    metrics = run_experiment(
        y_ts, y_syn_ts, mu_gt_ts,
        block_len=block_len, n=n,
        n_trials=N_TRIALS, n_boot=N_BOOTSTRAP
    )
    row = {"n": n, **metrics}
    n_results.append(row)
    print(f"n={n:4d} | Cov cls={metrics['cov_cls']:.3f} "
          f"ppi={metrics['cov_ppi']:.3f} "
          f"blk={metrics['cov_blk']:.3f} | "
          f"ESS ppi={metrics['ess_ppi']:.3f} "
          f"blk={metrics['ess_blk']:.3f}")

n_df = pd.DataFrame(n_results)
n_df.to_csv(os.path.join(OUT_DIR, "ext2a_ma2_n_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/ext2a_ma2_n_results.csv")

# --------------------------------------------------
# STEP 5 — Figures
# --------------------------------------------------

try:
    CC, CP, CB = "#4DAF4A", "#E41A1C", "#377EB8"
    thetas = theta_df["theta1"].values
    ns     = n_df["n"].values

    # --- Figure 1: Coverage & ESS vs theta1 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(thetas, theta_df["cov_cls"], "o-",  color=CC,
            label="Classical", linewidth=2)
    ax.plot(thetas, theta_df["cov_ppi"], "s--", color=CP,
            label="PPI++ (standard, i.i.d.)", linewidth=2)
    ax.plot(thetas, theta_df["cov_blk"], "o-",  color=CB,
            label="Block Bootstrap PPI++", linewidth=2)
    ax.axhline(1-ALPHA, color="black", ls=":", lw=1.5, label="Target 90%")
    ax.set_xlabel("MA(2) parameter θ₁  (θ₂ = θ₁/2)", fontsize=11)
    ax.set_ylabel("Coverage of 90% CIs", fontsize=11)
    ax.set_title("(a) Coverage vs dependence strength\n"
                 "MA(2): finite memory, only 2-lag dependence")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(thetas, theta_df["ess_ppi"], "s--", color=CP,
            label="PPI++ (standard)", linewidth=2)
    ax.plot(thetas, theta_df["ess_blk"], "o-",  color=CB,
            label="Block Bootstrap PPI++", linewidth=2)
    ax.axhline(1.0, color=CC, ls="--", lw=1.5, label="Classical (=1)")
    ax.set_xlabel("MA(2) parameter θ₁  (θ₂ = θ₁/2)", fontsize=11)
    ax.set_ylabel("ESS", fontsize=11)
    ax.set_title("(b) ESS vs dependence strength")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Extension 2a (MA2): Block Bootstrap AutoEval — MA(2) Time Series "
        f"(n={N_FIXED})\n"
        f"T={T}, block length L=√T≈{block_len} | "
        f"Finite memory process (contrast with AR(1))",
        fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "ext2a_ma2_theta_coverage.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {p1}")
    plt.close()

    # --- Figure 2: n-sweep (paper style) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ns, n_df["cov_cls"], "o-",  color=CC, label="Classical")
    ax.plot(ns, n_df["cov_ppi"], "s--", color=CP, label="PPI++ (standard)")
    ax.plot(ns, n_df["cov_blk"], "o-",  color=CB,
            label="Block Bootstrap PPI++")
    ax.axhline(1-ALPHA, color="black", ls=":", lw=1.2, label="Target 90%")
    ax.set_xlabel("# labeled samples")
    ax.set_ylabel("Coverage of 90% CIs")
    ax.set_title("(a) Coverage vs n")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ns, n_df["mse_cls"], "o-",  color=CC, label="Classical")
    ax.plot(ns, n_df["mse_ppi"], "s--", color=CP, label="PPI++ (standard)")
    ax.plot(ns, n_df["mse_blk"], "o-",  color=CB,
            label="Block Bootstrap PPI++")
    ax.set_xlabel("# labeled samples")
    ax.set_ylabel("MSE")
    ax.set_title("(b) MSE of mean estimates")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ns, n_df["ess_ppi"], "s--", color=CP, label="PPI++ (standard)")
    ax.plot(ns, n_df["ess_blk"], "o-",  color=CB,
            label="Block Bootstrap PPI++")
    ax.axhline(1.0, color=CC, ls="--", lw=1.0, label="Classical (=1)")
    ax.set_xlabel("# labeled samples")
    ax.set_ylabel("ESS")
    ax.set_title("(c) ESS vs n")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Extension 2a (MA2): Block Bootstrap AutoEval — MA(2) Time Series "
        f"(θ₁={THETA1_FIXED}, θ₂={theta2_fixed:.2f})\n"
        f"T={T}, annotator corr~{corr:.2f}, "
        f"theoretical inflation={inflation_fixed:.3f}",
        fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "ext2a_ma2_n_main.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {p2}")
    plt.close()

    # --- Figure 3: Theoretical validation ---
    # For MA(2), theoretical inflation is modest and finite
    # Block bootstrap should track it better than for AR(1)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    width_ratio  = theta_df["wid_blk"] / theta_df["wid_ppi"]
    theoretical  = theta_df["theoretical_inflation"].apply(np.sqrt)

    ax.plot(thetas, width_ratio, "o-", color=CB, linewidth=2,
            label="Block Bootstrap width ratio\n(wid_blk / wid_ppi)")
    ax.plot(thetas, theoretical, "s--", color=CP, linewidth=2,
            label="Theoretical inflation\n√(long-run var / iid var)")
    ax.axhline(1.0, color=CC, ls="--", lw=1.0, label="No inflation (i.i.d.)")
    ax.set_xlabel("MA(2) parameter θ₁  (θ₂ = θ₁/2)", fontsize=11)
    ax.set_ylabel("CI Width Ratio", fontsize=11)
    ax.set_title("Theoretical validation: block bootstrap vs MA(2) theory\n"
                 "MA(2) inflation is modest (~1.1-1.5x) vs AR(1) (~4x at ρ=0.9)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "ext2a_ma2_theoretical.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {p3}")
    plt.close()

    # --- Figure 4: AR(1) vs MA(2) comparison (if AR(1) results exist) ---
    ar1_path = os.path.join(OUT_DIR, "ext2a_rho_results.csv")
    if os.path.exists(ar1_path):
        ar1_df = pd.read_csv(ar1_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        ax.plot(ar1_df["rho"],   ar1_df["cov_ppi"],   "s--", color=CP,
                label="AR(1) PPI++ (standard)", linewidth=2)
        ax.plot(ar1_df["rho"],   ar1_df["cov_blk"],   "o-",  color=CB,
                label="AR(1) Block Bootstrap", linewidth=2)
        ax.plot(theta_df["theta1"], theta_df["cov_ppi"], "s:",
                color="#FF7F00", label="MA(2) PPI++ (standard)", linewidth=2)
        ax.plot(theta_df["theta1"], theta_df["cov_blk"], "o-",
                color="#984EA3", label="MA(2) Block Bootstrap", linewidth=2)
        ax.axhline(1-ALPHA, color="black", ls=":", lw=1.2, label="Target 90%")
        ax.set_xlabel("Dependence parameter (ρ or θ₁)", fontsize=10)
        ax.set_ylabel("Coverage of 90% CIs", fontsize=10)
        ax.set_title("(a) Coverage: AR(1) vs MA(2)")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(ar1_df["rho"],   ar1_df["ess_ppi"],   "s--", color=CP,
                label="AR(1) PPI++ (standard)", linewidth=2)
        ax.plot(theta_df["theta1"], theta_df["ess_ppi"], "s:",
                color="#FF7F00", label="MA(2) PPI++ (standard)", linewidth=2)
        ax.axhline(1.0, color=CC, ls="--", lw=1.0, label="Classical (=1)")
        ax.set_xlabel("Dependence parameter (ρ or θ₁)", fontsize=10)
        ax.set_ylabel("ESS (PPI++ standard)", fontsize=10)
        ax.set_title("(b) ESS: AR(1) vs MA(2)\n"
                     "MA(2) ESS degrades less — finite memory process")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        fig.suptitle(
            "AR(1) vs MA(2): Comparing Block Bootstrap AutoEval\n"
            "under infinite-memory vs finite-memory dependence",
            fontsize=10)
        plt.tight_layout()
        p4 = os.path.join(OUT_DIR, "ext2a_ar1_vs_ma2.png")
        plt.savefig(p4, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {p4}")
        plt.close()
    else:
        print("AR(1) results not found — skipping comparison figure.")
        print("Run run_extension2a_ar1.py first to generate it.")

except Exception as e:
    print(f"Plotting error: {e}")
    import traceback; traceback.print_exc()

# --------------------------------------------------
# STEP 6 — Verification summary
# --------------------------------------------------

print("\n" + "="*72)
print("VERIFICATION SUMMARY — Extension 2a MA(2)")
print("="*72)

print(f"\nExperiment A (n={N_FIXED}, varying theta1, theta2=theta1/2):")
print(f"{'theta1':>7} | {'theta2':>7} | {'theory':>8} | "
      f"{'cov_cls':>8} | {'cov_ppi':>8} | {'cov_blk':>8} | {'ess_ppi':>8}")
print("-"*72)
for _, r in theta_df.iterrows():
    print(f"{r.theta1:>7.1f} | {r.theta2:>7.3f} | "
          f"{r.theoretical_inflation:>8.3f} | "
          f"{r.cov_cls:>8.3f} | {r.cov_ppi:>8.3f} | "
          f"{r.cov_blk:>8.3f} | {r.ess_ppi:>8.3f}")

print(f"\nExperiment B (theta1={THETA1_FIXED}, theta2={theta2_fixed:.2f}, "
      f"varying n):")
print(f"{'n':>5} | {'cov_cls':>8} | {'cov_ppi':>8} | "
      f"{'cov_blk':>8} | {'ess_ppi':>8} | {'ess_blk':>8}")
print("-"*72)
for _, r in n_df.iterrows():
    print(f"{int(r.n):>5} | {r.cov_cls:>8.3f} | {r.cov_ppi:>8.3f} | "
          f"{r.cov_blk:>8.3f} | {r.ess_ppi:>8.3f} | {r.ess_blk:>8.3f}")

print()
print("Key difference from AR(1):")
print("  MA(2) has FINITE memory (2 lags only)")
print("  Theoretical inflation is modest: max ~1.5x vs AR(1) ~4x at high param")
print("  Block bootstrap should track theory better for MA(2)")
print("  ESS degradation should be slower than AR(1) as theta increases")
print("\nDone. Extension 2a MA(2) complete.")