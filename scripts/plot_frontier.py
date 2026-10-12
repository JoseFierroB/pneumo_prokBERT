#!/usr/bin/env python3
"""Frontier plot: MCC@0.5 per model x dataset (one metric per file).

Matrix 5 + supplementary A1/full (7 series). Overlap cells hatched (never ranked).
The dedup panel uses the CLEAN dedup-test slice recomputed from saved scores
(dedup-all mixes train rows in); other panels use matrix cells as-is.
Honest defense figure: clean heights decide, hatched bars show memorization context.

Usage:
  pixi run python scripts/plot_frontier.py
  pixi run python scripts/plot_frontier.py --out-dir reports/model_matrix/plots
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.metrics import cell_metrics, load_probs_csv

ORDER = ["mini-223k-replica", "mini-d39v-tigr4", "mini-223k-d39v-tigr4",
         "promoter-d39v-tigr4", "promoter-baseline", "mini-223k-1to2", "mini-223kfull-1to2-dedup"]
COLS = ["paper-test", "ecoli-sigma70", "multispecies", "d39v-tigr4-dedup"]
PROBS_DIRS = ["finetuning_outputs/eval_matrix_probs", "reports/supplementary_1to2",
              "reports/supplementary_full"]


def parse_args():
    p = argparse.ArgumentParser(description="MCC frontier plot (clean decides, overlap hatched)")
    p.add_argument("--out-dir", type=str, default="reports/model_matrix/plots")
    return p.parse_args()


def dedup_test_cell(alias):
    """Recompute the dedup-TEST clean slice from saved per-cell scores (no new inference)."""
    import pandas as pd
    src = next(str(Path(d) / f"{alias}__d39v-tigr4-dedup.csv") for d in PROBS_DIRS
               if (Path(d) / f"{alias}__d39v-tigr4-dedup.csv").exists())
    y, s = load_probs_csv(src)
    df = pd.read_csv("datasets/d39v_tigr4_dedup_3135/all.csv")
    assert len(df) == len(y), f"order mismatch {alias}"
    sp = pd.read_csv("datasets/d39v_tigr4_dedup_3135/test.csv")
    idx = [i for i, x in enumerate(df["segment"].str.upper()) if x in set(sp["segment"].str.upper())]
    m = cell_metrics(y[idx], s[idx])
    m["leakage"] = "clean (test split)"
    return m


def main():
    args = parse_args()
    m = json.load(open("reports/model_matrix/metrics.json"))
    a = json.load(open("reports/supplementary_1to2/mini-223k-1to2.json"))["mini-223k-1to2"]
    f = json.load(open("reports/supplementary_full/mini-223kfull-1to2-dedup.json"))["mini-223kfull-1to2-dedup"]
    res = dict(m)
    res["mini-223k-1to2"] = a
    res["mini-223kfull-1to2-dedup"] = f
    clean_dedup = {alias: dedup_test_cell(alias) for alias in ORDER}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for col in COLS:
        vals, hatch, colors = [], [], []
        title = col
        for alias in ORDER:
            if col == "d39v-tigr4-dedup":
                d = clean_dedup[alias]
                title = "d39v-tigr4-dedup-test (clean slice)"
            elif alias not in res or col not in res[alias]:
                vals.append(0.0)
                hatch.append("")
                colors.append("gray")
                continue
            else:
                d = res[alias][col]
            vals.append(d["mcc_05"])
            ov = d.get("leakage") == "train_overlap"
            hatch.append("///" if ov else "")
            colors.append("tab:gray" if ov else "tab:blue")
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(ORDER))
        bars = ax.bar(x, vals, color=colors)
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, rotation=25, fontsize=8, ha="right")
        ax.set(title=f"MCC frontier — {title} (clean decides, hatched = train_overlap)",
               ylabel="MCC@0.5")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / f"frontier__{col}.png", dpi=120)
        plt.close(fig)
        print(f"frontier__{col}.png OK")


if __name__ == "__main__":
    main()
