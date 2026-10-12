#!/usr/bin/env python3
"""Paired bootstrap CIs for headline model comparisons (reproducible).

Reads per-cell probs CSVs, resamples row indices with replacement (same index
for both models = paired), B=2000, seed 42. Reports delta (a-b) + 95% CI + p
(one-sided: fraction of resamples with delta <= 0) for MCC@0.5 and ROC-AUC.
Writes reports/model_matrix/bootstrap_cis.json, consumed by build_report.py.

Usage:
  pixi run python scripts/paired_bootstrap.py
  pixi run python scripts/paired_bootstrap.py --probs-dir finetuning_outputs/eval_matrix_probs --out reports/model_matrix/bootstrap_cis.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, ".")
from scripts.metrics import load_probs_csv

B = "promoter-baseline"
PAIRS = [
    ("multispecies", "mini-223k-d39v-tigr4", B),
    ("ecoli-sigma70", "mini-223k-replica", B),
    ("ecoli-sigma70", "mini-223k-d39v-tigr4", B),
    ("paper-test", "mini-223k-d39v-tigr4-high5050", "mini-223k-d39v-tigr4"),
    ("multispecies", "mini-223k-d39v-tigr4-high5050", "mini-223k-d39v-tigr4"),
    ("ecoli-sigma70", "mini-223k-d39v-tigr4-high5050", "mini-223k-d39v-tigr4"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Paired bootstrap CIs (B=2000, seed 42)")
    p.add_argument("--probs-dir", type=str, default="finetuning_outputs/eval_matrix_probs")
    p.add_argument("--out", type=str, default="reports/model_matrix/bootstrap_cis.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = {"method": f"paired bootstrap, B={args.n_boot}, seed={args.seed}, threshold 0.5 for MCC",
           "comparisons": []}
    for col, a, b in PAIRS:
        ya, sa = load_probs_csv(Path(args.probs_dir) / f"{a}__{col}.csv")
        yb, sb = load_probs_csv(Path(args.probs_dir) / f"{b}__{col}.csv")
        assert (ya == yb).all(), f"labels differ for {col}"
        y = ya
        n = len(y)
        dm, da = [], []
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            dm.append(matthews_corrcoef(y[idx], (sa[idx] >= 0.5).astype(int))
                      - matthews_corrcoef(y[idx], (sb[idx] >= 0.5).astype(int)))
            da.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
        row = {"dataset": col, "a": a, "b": b, "n": n, "metrics": {}}
        for name, arr in (("mcc", np.array(dm)), ("auc", np.array(da))):
            arr = np.asarray(arr, dtype=np.float64)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            p = float((arr <= 0).mean())
            row["metrics"][name] = {"delta": float(arr.mean()), "ci95": [float(lo), float(hi)],
                                    "p": p, "sig95": bool(lo > 0)}
        out["comparisons"].append(row)
        m, u = row["metrics"]["mcc"], row["metrics"]["auc"]
        print(f"{col:14s} {a:22s} mcc {m['delta']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}] "
              f"p={m['p']:.4f} {'SIG' if m['sig95'] else 'n.s.'} | auc {u['delta']:+.4f} {'SIG' if u['sig95'] else 'n.s.'}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
