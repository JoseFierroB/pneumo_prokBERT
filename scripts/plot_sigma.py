#!/usr/bin/env python3
"""Sigma-stratified score/recall plots (diagnostic track, separate from the matrix).

Joins canonical probs CSV scores with sigma tags (d39v via all.csv segments,
tigr4-high via metadata Sequence_ID) and draws, per dataset:
  - score stripplot by sigma group, one panel per model (single-metric files)
  - recall@0.5 bars by group and model (one file)
  - ROC per sigma group vs negatives, one file per model (groups with n<2 skipped)
Groups with n<5 are annotated, never hidden.

Usage:
  pixi run python scripts/plot_sigma.py --models mini-223k-replica mini-d39v-tigr4 mini-223k-d39v-tigr4 promoter-baseline promoter-d39v-tigr4
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
    p.add_argument("--models", nargs="*", default=["mini-223k-replica", "mini-d39v-tigr4", "mini-223k-d39v-tigr4",
                                                   "promoter-baseline", "promoter-d39v-tigr4", "mini-223k-1to2-dedup"])
    p.add_argument("--probs-dirs", nargs="*", default=["finetuning_outputs/eval_matrix_probs",
                                                       "reports/supplementary",
                                                       "reports/supplementary_1to2",
                                                       "reports/supplementary_full"])
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

# Only these sigma groups are plotted (SigA family merged + competence SigX/SigX).
# LOWCONF merges into SigA by design. Everything else (None, TFs, unassigned,
# negatives) is not considered — except "neg", kept internally as the negative
# class for ROC computation (never plotted).
RELEVANT = {"SigA", "SigX"}


def _merge_siga(tags):
    return ["SigA" if t in SIGA else t for t in tags]


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
    # SOURCE label "ComX" mapped to our SigX (same competence factor).
    meta["Sigma_Factor"] = meta["Sigma_Factor"].replace({"ComX": "SigX"}).fillna("unassigned")
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

    groups = {"d39v": (dseqs, _merge_siga(dtags)), "tigr4-high": (tseqs, _merge_siga(ttags))}
    # --- joint pool: d39v eval+test positives + tigr4-high positives, one set ---
    # Clean-only per model: rows seen by THAT model's trains are excluded (counts vary).
    # Measured: HF train contains zero pneumo rows, so train_80-based models are clean
    # everywhere pneumo; dedup/1to2/sigx trains overlap heavily (199/292 d39v rows).
    _dd = {s.upper() for s in pd.read_csv("datasets/d39v_tigr4_dedup_3135/train.csv")["segment"]}
    _pp = {s.upper() for s in pd.read_csv("datasets/d39v_tigr4_1to2_plus_dedup/train.csv")["segment"]}
    SEEN = {"mini-223k-replica": set(), "promoter-baseline": set(),
            "mini-d39v-tigr4": _dd, "mini-223k-d39v-tigr4": _dd, "promoter-d39v-tigr4": _dd,
            "mini-223k-1to2-dedup": _pp}
    jpos = pd.concat([pd.read_csv("datasets/d39v_full_989/eval.csv"),
                      pd.read_csv("datasets/d39v_full_989/test.csv")], ignore_index=True)
    jpos = jpos[jpos.y == 1].copy()
    jtags = _merge_siga(jpos["id"].str.split("_").str[-1].tolist())
    jseqs = [str(s).upper() for s in jpos["segment"]]
    jseqs += tseqs
    jtags += _merge_siga(ttags)
    print(f"joint pool: {len(jseqs)} (d39v {len(jpos)} + tigr4 {len(tseqs)})")
    # union-clean: rows seen by ANY plotted model's trains are excluded everywhere,
    # so all models face identical rows (counts in titles/labels)
    union_seen = set().union(*[SEEN.get(a, _dd | _pp) for a in args.models]) if args.models else set()
    for dname, (seqs, tags) in groups.items():
        keep = [i for i in range(len(seqs)) if seqs[i] not in union_seen]
        tags_u = [tags[i] for i in keep]
        shown = [t for t in sorted(set(tags_u)) if t in RELEVANT]
        neg_idx = [i for i, t in enumerate(tags_u) if t == "neg"]
        print(f"{dname}: {len(keep)}/{len(seqs)} clean rows")
        # --- stripplot per model (one file each; clean-only rows) ---
        for alias in args.models:
            y, s = load_probs_csv(find_csv(args.probs_dirs, alias, dname))
            assert len(s) == len(seqs), f"order mismatch {alias}/{dname}"
            # positional alignment holds: CSV built in same fasta order as tags above
            s_u = s[keep]
            if not shown:
                print(f"strip {dname}/{alias} SKIPPED (no clean rows)")
                continue
            order, vals = [], []
            for g in shown:
                idx = [i for i, t in enumerate(tags_u) if t == g]
                v = s_u[idx]
                order.append(f"{g} (n={len(idx)})")
                vals.append(v)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(vals, tick_labels=order, showfliers=False)
            ax.set(title=f"scores by sigma, {dname} clean-only — {alias}", ylabel="P(promoter)")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out / f"strip__{dname}__{alias}.png", dpi=120)
            plt.close(fig)
            print(f"strip {dname}/{alias} OK")
        # --- recall@0.5 bars grouped (one file per dataset) ---
        fig, ax = plt.subplots(figsize=(9, 4))
        gnames = shown
        x = np.arange(len(gnames))
        w = 0.8 / max(1, len(args.models))
        for j, alias in enumerate(args.models):
            y, s = load_probs_csv(find_csv(args.probs_dirs, alias, dname))
            s_u = s[keep]
            rec = []
            for g in gnames:
                idx = [i for i, t in enumerate(tags_u) if t == g]
                rec.append(float(((s_u[idx] >= 0.5).sum() / max(1, len(idx)))))
            ax.bar(x + j * w, rec, width=w, label=alias)
        ax.set_xticks(x + w * (len(args.models) - 1) / 2)
        ax.set_xticklabels([f"{g}\n(n={[t for t in tags_u].count(g)})" for g in gnames], fontsize=7)
        ax.set(title=f"recall@0.5 by sigma, {dname} clean-only", ylabel="recall")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / f"recall__{dname}.png", dpi=120)
        plt.close(fig)
        print(f"recall {dname} OK")
    # --- joint strip per model (d39v eval+test + tigr4-high, one set) ---
    for alias in args.models:
        _, sd = load_probs_csv(find_csv(args.probs_dirs, alias, "d39v"))
        _, st = load_probs_csv(find_csv(args.probs_dirs, alias, "tigr4-high"))
        # order-free lookup: d39v CSVs follow d39v FASTA order, tigr4 CSVs the tigr4 order
        dd = {}
        for r, v in zip(list(SeqIO.parse("datasets/reference/d39v/positives_81bp.fasta", "fasta")) +
                        list(SeqIO.parse("datasets/reference/d39v/negatives_81bp.fasta", "fasta")), sd):
            dd[str(r.seq).upper()] = float(v)
        for r, v in zip(list(SeqIO.parse("datasets/reference/tigr4-high/positives_81bp.fasta", "fasta")) +
                        list(SeqIO.parse("datasets/reference/tigr4-high/negatives_81bp.fasta", "fasta")), st):
            dd[str(r.seq).upper()] = float(v)
        order, vals = [], []
        for g in sorted(set(jtags)):
            if g not in RELEVANT:
                continue
            rows = [(x, t) for x, t in zip(jseqs, jtags) if t == g and x not in union_seen]
            v = np.array([dd[s] for s, _ in rows])
            order.append(f"{g} (n={len(v)})")
            vals.append(v)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(vals, tick_labels=order, showfliers=False)
        ax.set(title=f"scores by sigma, joint d39v+tigr4 (clean-only) — {alias}", ylabel="P(promoter)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / f"strip__joint__{alias}.png", dpi=120)
        plt.close(fig)
        print(f"strip joint/{alias} OK")
        # --- ROC per relevant sigma group vs negatives (one file per model) ---
        from sklearn.metrics import roc_curve, auc
        for alias in args.models:
            y, s = load_probs_csv(find_csv(args.probs_dirs, alias, dname))
            s_u = s[keep]
            fig, ax = plt.subplots(figsize=(6, 5))
            for g in shown:
                idx = [i for i, t in enumerate(tags_u) if t == g]
                if len(idx) < 2 or not neg_idx:
                    continue
                yy = np.array([1] * len(idx) + [0] * len(neg_idx))
                ss = np.concatenate([s_u[idx], s_u[neg_idx]])
                fpr, tpr, _ = roc_curve(yy, ss)
                ax.plot(fpr, tpr, lw=1.5, label=f"{g} (n={len(idx)}, AUC {auc(fpr, tpr):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set(title=f"ROC by sigma vs neg, {dname} clean-only — {alias}", xlabel="FPR", ylabel="TPR")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"roc__{dname}__{alias}.png", dpi=120)
            plt.close(fig)
            print(f"roc {dname}/{alias} OK")


if __name__ == "__main__":
    main()
