#!/usr/bin/env python3
"""Figure-6 replica (paper style) with OUR epochs + bootstrap error bars.

Grouped bars: Accuracy / F1 / MCC / Sensitivity / Specificity (@0.5 fixed)
x series: 1ep, 2ep, 3ep, benchmark (mini-promoter). Paper-test (n=22310).
CIs: paired bootstrap B=2000 per metric (our style, unlike the original).

Output: reports/supplementary_5050/benchmark/fig6_epochs__paper-test.png + json.

Usage:
  pixi run python scripts/plot_fig6_epochs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef)

MODELS = [
    ("1ep", "reports/supplementary_5050/grid_1eprev/ep1,paper_annot.csv", "#1f77b4"),
    ("2ep", "finetuning_outputs/eval_matrix_probs/mini-223k-d39v-tigr4-high5050__paper-test.csv", "#ff7f0e"),
    ("3ep", "finetuning_outputs/eval_matrix_probs/mini-223k-d39v-tigr4-high5050-3ep__paper-test.csv", "#2ca02c"),
    ("replica", "reports/supplementary_5050/replica_verdict_probs/replica-code__paper-test.csv", "#9467bd"),
    ("benchmark*", "finetuning_outputs/eval_matrix_probs/promoter-baseline__paper-test.csv", "#d62728"),
]
METRICS = ["Accuracy", "F1", "MCC", "Sensitivity", "Specificity"]
OUT = Path("reports/supplementary_5050/benchmark")
B = 2000
SEED = 42


def metrics_of(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {"Accuracy": accuracy_score(y, p), "F1": f1_score(y, p),
            "MCC": matthews_corrcoef(y, p),
            "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    vals, errs, n = {}, {}, 0
    for alias, fp, _ in MODELS:
        d = pd.read_csv(fp)
        y, s = (d[d.columns[0]].to_numpy(), d[d.columns[1]].to_numpy())
        n = len(y)
        p = (s >= 0.5).astype(int)
        vals[alias] = metrics_of(y, p)
        boots = {m: [] for m in METRICS}
        for _ in range(B):
            idx = rng.randint(0, n, n)
            for m, v in metrics_of(y[idx], p[idx]).items():
                boots[m].append(v)
        errs[alias] = {m: float(np.std(boots[m], ddof=1)) for m in METRICS}
        print(alias, {m: round(vals[alias][m], 4) for m in METRICS})

    x = np.arange(len(METRICS))
    w = 0.15
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (alias, _, color) in enumerate(MODELS):
        v = [vals[alias][m] for m in METRICS]
        e = [1.96 * errs[alias][m] for m in METRICS]
        ax.bar(x + (i - 2) * w, v, w, yerr=e, capsize=2, color=color, label=alias)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score (@0.5 fixed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.text(0.5, 0.01, "*benchmark saw test rows in train (train_overlap, not comparable)",
             ha="center", fontsize=6, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "fig6_epochs__paper-test.png", dpi=120)
    plt.close(fig)
    json.dump({"n": n, "metrics": vals,
               "ci95_halfwidth": errs, "note": "error bars = 1.96*sd over B=2000 bootstrap"},
              open(OUT / "fig6_epochs__paper-test.json", "w"), indent=1)
    print("fig6_epochs__paper-test.png OK")


if __name__ == "__main__":
    main()
