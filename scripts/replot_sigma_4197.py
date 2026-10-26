#!/usr/bin/env python3
"""Replot sigma 4197 ROCs from saved probs: no titles, FPR as 1-specificity.

Reads reports/supplementary_5050/sigma/roc4197_{set}__{alias}.csv + sigma_roc_4197.json.
Overwrites roc_4197__{set}.png. CPU only.

Usage:
  pixi run python scripts/replot_sigma_4197.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

OUT = Path("reports/supplementary_5050/sigma")
STYLES = {"benchmark": ":", "1ep": "-", "2ep": "--", "replica": "-."}
ALIASES = ["benchmark", "1ep", "2ep", "replica"]


def main():
    for name in ("SigA-prop", "SigX-prop", "SigA-all", "SigX-all"):
        fig, ax = plt.subplots(figsize=(5, 4))
        for alias in ALIASES:
            d = pd.read_csv(OUT / f"roc4197_{name}__{alias}.csv")
            y, s = d[d.columns[0]].to_numpy(), d[d.columns[1]].to_numpy()
            fpr, tpr, _ = roc_curve(y, s)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, linestyle=STYLES[alias],
                    label=f"{alias} (AUC {a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="1 - specificity", ylabel="TPR")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"roc_4197__{name}.png", dpi=120)
        plt.close(fig)
        print(name, "OK (no title, 1-specificity)")


if __name__ == "__main__":
    main()
