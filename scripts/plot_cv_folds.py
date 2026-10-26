#!/usr/bin/env python3
"""5-fold CV plot: paired 1ep vs 2ep per fold (MCC + AUC), classical folds.

Lines connect same fold (paired design); right panel = mean+-sd + paired delta.
Ham folds overlay when available (currently f0 only).

Output: reports/supplementary_5050/cv/cv_folds__pneumo.png + cv_folds.json.

Usage:
  pixi run python scripts/plot_cv_folds.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# data_test MCC/AUC from WS train logs (mix high5050-base + pneumo folds, seed 42)
CLASSIC = {
    "1ep": [0.8212, 0.7216, 0.7465, 0.7691, 0.7076],
    "1ep_auc": [0.9539, 0.9216, 0.9513, 0.9574, 0.9340],
    "2ep": [0.8678, 0.7460, 0.7891, 0.8178, 0.7500],
    "2ep_auc": [0.9576, 0.9291, 0.9504, 0.9609, 0.9454],
}
HAM = {"1ep": [0.7927], "2ep": [0.8355]}  # f0 only; f1-f4 running on 5090
OUT = Path("reports/supplementary_5050/cv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m1, m2 = np.array(CLASSIC["1ep"]), np.array(CLASSIC["2ep"])
    a1, a2 = np.array(CLASSIC["1ep_auc"]), np.array(CLASSIC["2ep_auc"])
    d = m2 - m1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(5)
    ax1.plot(x, m1, "o-", color="#1f77b4", label="1ep")
    ax1.plot(x, m2, "s-", color="#ff7f0e", label="2ep")
    for i in range(5):
        ax1.plot([i, i], [m1[i], m2[i]], "k-", alpha=0.35, lw=1.5)
    # AUC paired panel data (same folds)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"f{k}" for k in range(5)])
    ax1.set_ylabel("data_test MCC@0.5")
    ax1.set_title(f"paired by fold: 2ep-1ep mean {d.mean():+.4f} (5/5 positive)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    for vals, col, lab, off in ((m1, "#1f77b4", "1ep", -0.12), (m2, "#ff7f0e", "2ep", 0.12)):
        ax2.bar(off, vals.mean(), 0.2, yerr=vals.std(ddof=1), capsize=4, color=col, label=lab)
    ax2.set_xlim(-0.35, 0.35)
    ax2.set_xticks([])
    ax2.set_ylabel("mean MCC +- sd")
    ax2.set_title(f"1ep {m1.mean():.3f}+-{m1.std(ddof=1):.3f} | 2ep {m2.mean():.3f}+-{m2.std(ddof=1):.3f}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("pneumo 5-fold CV (classical, n=2792 unique, seed 42)")
    fig.tight_layout()
    fig.savefig(OUT / "cv_folds__pneumo.png", dpi=120)
    plt.close(fig)
    # AUC paired panel
    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, a1, "o-", color="#1f77b4", label="1ep")
    ax.plot(x, a2, "s-", color="#ff7f0e", label="2ep")
    for i in range(5):
        ax.plot([i, i], [a1[i], a2[i]], "k-", alpha=0.35, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{k}" for k in range(5)])
    ax.set_ylabel("data_test AUC")
    ax.set_title(f"paired by fold: 2ep-1ep mean {(a2-a1).mean():+.4f} ({((a2-a1)>0).sum()}/5 positive)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig2.suptitle("pneumo 5-fold CV AUC (classical, seed 42)")
    fig2.tight_layout()
    fig2.savefig(OUT / "cv_folds__auc.png", dpi=120)
    plt.close(fig2)
    json.dump({"classic_mcc": CLASSIC, "ham_mcc_f0": HAM,
               "paired_delta_mean": round(float(d.mean()), 4),
               "paired_delta_all_positive": bool((d > 0).all()),
               "note": "ham f1-f4 pending on 5090"},
              open(OUT / "cv_folds.json", "w"), indent=1)
    print(f"1ep {m1.mean():.4f}+-{m1.std(ddof=1):.4f} | 2ep {m2.mean():.4f}+-{m2.std(ddof=1):.4f} | paired d {d.mean():+.4f}")


if __name__ == "__main__":
    main()
