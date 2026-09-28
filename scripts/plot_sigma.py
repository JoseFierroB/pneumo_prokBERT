#!/usr/bin/env python3
"""Sigma-stratified score/recall plots (diagnostic track, separate from the matrix).

Joins canonical probs CSV scores with sigma tags (d39v via all.csv segments,
tigr4-high via metadata Sequence_ID) and draws, per dataset:
  - score stripplot by sigma group, one panel per model (single-metric files)
  - recall@0.5 bars by group and model (one file)
Groups with n<5 are annotated, never hidden.

Usage:
  pixi run python scripts/plot_sigma.py --models mini-223k-replica mini-223k-d39v-tigr4 promoter-baseline promoter-d39v-tigr4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.metrics import load_probs_csv


def parse_args():
    p = argparse.ArgumentParser(description="Sigma-stratified score/recall plots")
    p.add_argument("--models", nargs="*", default=["mini-223k-replica", "mini-223k-d39v-tigr4",
                                                   "promoter-baseline", "promoter-d39v-tigr4"])
    p.add_argument("--probs-dirs", nargs="*", default=["finetuning_outputs/eval_matrix_probs",
                                                       "reports/supplementary"])
    p.add_argument("--tigr4-meta", type=str,
                   default="datasets/external/positives_high_sigma_assigned_metadata.tsv")
    p.add_argument("--out-dir", type=str, default="reports/supplementary/plots_sigma")
    return p.parse_args()


def find_csv(dirs, alias, col):
    for d in dirs:
        p = Path(d) / f"{alias}__{col}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"no CSV for {alias}__{col}")


SIGA = {"SigA", "SigA-LOWCONF", "SigA_Putative"}


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alldf = pd.read_csv("datasets/d39v_full_989/all.csv")
    alldf["tag"] = alldf["id"].str.split("_").str[-1]
    seg2tag = dict(zip(alldf["segment"].str.upper(), alldf["tag"]))
    meta = pd.read_csv(args.tigr4_meta, sep="\t")
    meta["Sigma_Factor"] = meta["Sigma_Factor"].fillna("unassigned")
    tsig = dict(zip(meta["Sequence_ID"].astype(str), meta["Sigma_Factor"]))

    from Bio import SeqIO
    tseqs, ttags = [], []
    for r in list(SeqIO.parse("datasets/reference/tigr4-high/positives_81bp.fasta", "fasta")):
        tseqs.append(str(r.seq).upper())
        ttags.append(tsig.get(str(r.id), "SigA"))
    for r in list(SeqIO.parse("datasets/reference/tigr4-high/negatives_81bp.fasta", "fasta")):
        tseqs.append(str(r.seq).upper())
        ttags.append("neg")
    dseqs, dtags = [], []
    for r in list(SeqIO.parse("datasets/reference/d39v/positives_81bp.fasta", "fasta")):
        s = str(r.seq).upper()
        dseqs.append(s)
        dtags.append(seg2tag.get(s, "neg?"))
    for r in list(SeqIO.parse("datasets/reference/d39v/negatives_81bp.fasta", "fasta")):
        dseqs.append(str(r.seq).upper())
        dtags.append("neg")

    groups = {"d39v": (dseqs, dtags), "tigr4-high": (tseqs, ttags)}
    for dname, (seqs, tags) in groups.items():
        # --- stripplot per model (one file each) ---
        for alias in args.models:
            y, s = load_probs_csv(find_csv(args.probs_dirs, alias, dname))
            assert len(s) == len(seqs), f"order mismatch {alias}/{dname}"
            # positional alignment holds: CSV built in same fasta order as tags above
            order, vals = [], []
            for g in sorted(set(tags)):
                idx = [i for i, t in enumerate(tags) if t == g]
                v = s[idx]
                order.append(f"{g} (n={len(idx)})")
                vals.append(v)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(vals, tick_labels=order, showfliers=False)
            ax.set(title=f"scores by sigma — {dname} — {alias}", ylabel="P(promoter)")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out / f"strip__{dname}__{alias}.png", dpi=120)
            plt.close(fig)
            print(f"strip {dname}/{alias} OK")
        # --- recall@0.5 bars grouped (one file per dataset) ---
        fig, ax = plt.subplots(figsize=(9, 4))
        gnames = sorted(set(tags))
        x = np.arange(len(gnames))
        w = 0.8 / max(1, len(args.models))
        for j, alias in enumerate(args.models):
            y, s = load_probs_csv(find_csv(args.probs_dirs, alias, dname))
            rec = []
            for g in gnames:
                idx = [i for i, t in enumerate(tags) if t == g]
                rec.append(float(((s[idx] >= 0.5).sum() / max(1, len(idx)))))
            ax.bar(x + j * w, rec, width=w, label=alias)
        ax.set_xticks(x + w * (len(args.models) - 1) / 2)
        ax.set_xticklabels([f"{g}\n(n={[t for t in tags].count(g)})" for g in gnames], fontsize=7)
        ax.set(title=f"recall@0.5 by sigma — {dname}", ylabel="recall")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / f"recall__{dname}.png", dpi=120)
        plt.close(fig)
        print(f"recall {dname} OK")


if __name__ == "__main__":
    main()
