import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

OUT_PATH = "results/motivating_figure.png"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Load Citeseer results (n-sweep) ---
citeseer = pd.read_csv(
    "results/extension2b/ext2b_citeseer_results.csv"
)

# --- Load Air Quality results (n-sweep) ---
aq = pd.read_csv(
    "results/extension2_airquality/ext2_aq_n_results.csv"
)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
fig.subplots_adjust(wspace=0.06)

TARGET   = 0.90
LINE_KW  = dict(marker="o", markersize=5, linewidth=1.8)
COLOR_CLS = "#4CAF50"
COLOR_PPI = "#F44336"

for ax, df, title, note in [
    (axes[0], citeseer,
     "Citeseer (Real Graph)",
     r"$r = 0.38$,\ $n \leq 300$"),
    (axes[1], aq,
     "Air Quality (Real TS)",
     r"$r = 0.88$,\ $T = 7{,}344$"),
]:
    ax.plot(df["n"], df["cov_cls"],
            color=COLOR_CLS, label="Classical", **LINE_KW)
    ax.plot(df["n"], df["cov_ppi"],
            color=COLOR_PPI, label="PPI++ (naïve)", **LINE_KW)
    ax.axhline(TARGET, color="black", linestyle="--",
               linewidth=1.2, label="Target 90\%")
    ax.set_title(f"{title}\n{note}", pad=5)
    ax.set_xlabel(r"\# labeled samples ($n$)")
    ax.set_ylim(0.70, 1.05)
    ax.set_xticks(df["n"].tolist())
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Coverage of 90\% CIs")
axes[0].legend(loc="lower right", framealpha=0.9)

fig.suptitle(
    "Standard PPI++ confidence intervals undercover under dependence\n"
    "Coverage does not recover as $n$ grows",
    fontsize=10, y=1.03
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")