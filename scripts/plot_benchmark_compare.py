#!/usr/bin/env python3
"""3ep vs benchmark-prokbert (mini-promoter): ROC overlays + AUC table.

Reads matrix probs (both models) + 5050-test probs, writes one ROC PNG per
dataset to reports/supplementary_5050/benchmark/ + auc_compare.json.

Usage:
  pixi run python scripts/plot_benchmark_compare.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

P = Path("finetuning_outputs/eval_matrix_probs")
H = Path("reports/supplementary_5050/h2h")
OUT = Path("reports/supplementary_5050/benchmark")
M2 = "mini-223k-d39v-tigr4-high5050"
NEW = "mini-223k-d39v-tigr4-high5050-3ep"
REP = "mini-223k-replica"
BASE = "promoter-baseline"
# 1ep sources: paper-test from grid_1eprev (same 22310 rows, verified), rest from
# grid_1epfull eval_single (same rows as matrix; tokenization may differ per script)
E1P = Path("reports/supplementary_5050/grid_1eprev/ep1,paper_annot.csv")
E1G = Path("reports/supplementary_5050/grid_1epfull/mix-1ep")


def probs(p):
    d = pd.read_csv(p)
    c0, c1 = d.columns[0], d.columns[1]
    return d[c0].to_numpy(), d[c1].to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = [(f"{NEW}__paper-test.csv", f"{M2}__paper-test.csv", f"{REP}__paper-test.csv", f"{BASE}__paper-test.csv", "paper-test"),
            (f"{NEW}__ecoli-sigma70.csv", f"{M2}__ecoli-sigma70.csv", f"{REP}__ecoli-sigma70.csv", f"{BASE}__ecoli-sigma70.csv", "ecoli-sigma70"),
            (f"{NEW}__multispecies.csv", f"{M2}__multispecies.csv", f"{REP}__multispecies.csv", f"{BASE}__multispecies.csv", "multispecies"),
            (f"{NEW}__d39v-tigr4-dedup.csv", f"{M2}__d39v-tigr4-dedup.csv", f"{REP}__d39v-tigr4-dedup.csv", f"{BASE}__d39v-tigr4-dedup.csv", "dedup-all")]
    table = {}
    e1map = {"paper-test": E1P, "ecoli-sigma70": E1G.with_name("mix-1ep__ecoli-sigma70.csv"),
             "multispecies": E1G.with_name("mix-1ep__multispecies.csv"),
             "dedup-all": E1G.with_name("mix-1ep__d39v-tigr4-dedup.csv")}
    for fn, f2, fr, fb, name in sets:
        yn, sn = probs(P / fn)
        y2, s2 = probs(P / f2)
        yr, sr = probs(P / fr)
        yb, sb = probs(P / fb)
        y1, s1 = probs(e1map[name])
        assert (yn == y2).all() and (yn == yr).all() and (yn == yb).all() and (yn == y1).all()
        fig, ax = plt.subplots(figsize=(5, 4))
        for y, s, lab, st in [(yn, sn, "3ep", "-"), (y2, s2, "2ep", "--"),
                              (y1, s1, "1ep", "-."),
                              (yr, sr, "replica", (0, (3, 1, 1, 1))),
                              (yb, sb, "benchmark", ":")]:
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.5, linestyle=st, label=f"{lab} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC 1ep/2ep/3ep/replica vs benchmark — {name}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"roc_benchmark__{name}.png", dpi=120)
        plt.close(fig)
        aucs = [auc(*roc_curve(y, s)[:2]) for y, s in
                [(yn, sn), (y2, s2), (y1, s1), (yr, sr), (yb, sb)]]
        a3, a2, a1, ar, ab = aucs
        table[name] = {"n": int(len(yn)), "auc_3ep": round(float(a3), 4), "auc_2ep": round(float(a2), 4),
                       "auc_1ep": round(float(a1), 4), "auc_replica": round(float(ar), 4),
                       "auc_base": round(float(ab), 4)}
        print(f"{name}: 3ep {a3:.4f} | 2ep {a2:.4f} | 1ep {a1:.4f} | replica {ar:.4f} | base {ab:.4f}")
    # 5050-test (ad-hoc probs)
    yn, sn = probs(H / "v3_on_5050.csv")
    y2, s2 = probs(H / "new_on_5050.csv")
    yr, sr = probs(H / "cl_replica.csv")
    yb, sb = probs(Path("/tmp/base_on_5050.csv") if Path("/tmp/base_on_5050.csv").exists()
                   else H / "base_on_5050.csv")
    y1, s1 = probs(Path("reports/supplementary_5050/grid_1eprev/1ep_test.csv"))
    assert (yn == y2).all() and (yn == yr).all() and (yn == yb).all() and (yn == y1).all()
    fig, ax = plt.subplots(figsize=(5, 4))
    for y, s, lab, st in [(yn, sn, "3ep", "-"), (y2, s2, "2ep", "--"),
                          (y1, s1, "1ep", "-."),
                          (yr, sr, "replica", (0, (3, 1, 1, 1))),
                          (yb, sb, "benchmark-prokbert", ":")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.5, linestyle=st, label=f"{lab} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC 1ep/2ep/3ep/replica vs benchmark — 5050-test")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "roc_benchmark__5050-test.png", dpi=120)
    plt.close(fig)
    aucs = [auc(*roc_curve(y, s)[:2]) for y, s in [(yn, sn), (y2, s2), (y1, s1), (yr, sr), (yb, sb)]]
    a3, a2, a1, ar, ab = aucs
    table["5050-test"] = {"n": int(len(yn)), "auc_3ep": round(float(a3), 4), "auc_2ep": round(float(a2), 4),
                          "auc_1ep": round(float(a1), 4), "auc_replica": round(float(ar), 4),
                          "auc_base": round(float(ab), 4)}
    print(f"5050-test: 3ep {a3:.4f} | 2ep {a2:.4f} | 1ep {a1:.4f} | replica {ar:.4f} | base {ab:.4f}")
    json.dump(table, open(OUT / "auc_compare.json", "w"), indent=1)
    print("auc_compare.json OK")


if __name__ == "__main__":
    main()
