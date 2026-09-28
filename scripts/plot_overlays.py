#!/usr/bin/env python3
"""ROC and PR overlay plots — one metric per file.

Reads per-cell CSV (y_true, y_score) + a metrics dict for legend values.
Used by scripts/eval_matrix.py and reusable for supplementary model sets.

Usage:
  pixi run python scripts/plot_overlays.py --probs-dir finetuning_outputs/eval_matrix_probs \
      --metrics reports/model_matrix/metrics.json --out-dir reports/model_matrix/plots \
      --models mini-223k-replica mini-223k-d39v-tigr4 promoter-baseline promoter-d39v-tigr4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.metrics import load_probs_csv


def parse_args():
    p = argparse.ArgumentParser(description="Separate ROC/PR overlay plots per dataset")
    p.add_argument("--probs-dir", type=str, default="finetuning_outputs/eval_matrix_probs")
    p.add_argument("--probs-dirs", nargs="*", default=None,
                   help="extra probs CSV dirs searched in order (first hit wins); defaults to [--probs-dir]")
    p.add_argument("--metrics", type=str, default="reports/model_matrix/metrics.json")
    p.add_argument("--out-dir", type=str, default="reports/model_matrix/plots")
    p.add_argument("--models", nargs="*", default=None, help="subset of aliases (default: all in metrics)")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of columns (default: all)")
    p.add_argument("--suffix", type=str, default="", help="filename suffix, e.g. _supplementary")
    return p.parse_args()


def _find_csv(dirs, alias, col):
    for d in dirs:
        p = Path(d) / f"{alias}__{col}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"no CSV for {alias}__{col} in {dirs}")


def plot_overlays(probs_dir, out_dir, columns, aliases, results, suffix="", probs_dirs=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    probs_dir, out_dir = Path(probs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in (probs_dirs or [probs_dir])]
    for col in columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        for alias in aliases:
            y, s = load_probs_csv(_find_csv(dirs, alias, col))
            fpr, tpr, _ = roc_curve(y, s)
            m = results[alias][col]
            ax.plot(fpr, tpr, lw=1.5, label=f"{alias} (AUC {m['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=f"ROC — {col}", xlabel="FPR", ylabel="TPR")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"roc__{col}{suffix}.png", dpi=120)
        plt.close(fig)
        print(f"roc {col} OK")
    for col in columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        for alias in aliases:
            y, s = load_probs_csv(_find_csv(dirs, alias, col))
            prec, rec, _ = precision_recall_curve(y, s)
            m = results[alias][col]
            ax.plot(rec, prec, lw=1.5, label=f"{alias} (AP {m['ap']:.3f})")
        ax.set(title=f"PR — {col}", xlabel="Recall", ylabel="Precision")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pr__{col}{suffix}.png", dpi=120)
        plt.close(fig)
        print(f"pr {col} OK")


def main():
    args = parse_args()
    metrics = json.load(open(args.metrics))
    aliases = args.models or list(metrics.keys())
    first = metrics[aliases[0]]
    columns = args.datasets or list(first.keys())
    results = {a: metrics[a] for a in aliases}
    dirs = args.probs_dirs or [args.probs_dir]
    plot_overlays(args.probs_dir, args.out_dir, columns, aliases, results, suffix=args.suffix, probs_dirs=dirs)


if __name__ == "__main__":
    main()
