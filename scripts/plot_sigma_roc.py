#!/usr/bin/env python3
"""Per-sigma promoter-detection ROC overlays (3ep/2ep/replica).

Positives grouped by sigma (TIGR4 via metadata join, D39V via id-suffix tags);
negatives = shared 5050-test negatives. One ROC PNG per sigma set.

Output: reports/supplementary_5050/sigma/roc_sigma__<set>.png + sigma_roc.json

Usage:
  pixi run python scripts/plot_sigma_roc.py [--batch-size 64] [--cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from prokbert_patches import apply_tokenizer_patches
apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import AutoModelForSequenceClassification

MODELS = {
    "3ep": "finetuning_outputs/mini_223k_d39v_tigr4_high5050_3ep/best",
    "2ep": "finetuning_outputs/mini_223k_d39v_tigr4_high5050_2ep/best",
    "replica": "finetuning_outputs/mini_223k_replica_2ep/best",
    "benchmark": "neuralbioinfo/prokbert-mini-promoter",
}
OUT = Path("reports/supplementary_5050/sigma")


def parse_args():
    p = argparse.ArgumentParser(description="Per-sigma ROC overlays")
    p.add_argument("--batch-size", type=int, default=64)
    add_device_args(p)
    return p.parse_args()


def scores_for(model, tok, segs, bs, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(segs), bs):
            enc = tok(segs[i:i + bs], add_special_tokens=True, padding=True,
                      truncation=False, return_tensors="pt")
            kw = {k: v.to(device) for k, v in enc.items()}
            out.extend(torch.softmax(model(**kw).logits, -1)[:, 1].cpu().numpy().tolist())
    return np.array(out)


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv("datasets/external/positives_high_sigma_assigned_metadata.tsv", sep="\t")
    msig = {str(k): ("SigX" if v == "ComX" else v)
            for k, v in zip(meta["Sequence_ID"].astype(str), meta["Sigma_Factor"])}
    mseq = {str(r.id): str(r.seq).upper()
            for r in SeqIO.parse("datasets/reference/tigr4-high/positives_81bp.fasta", "fasta")}
    tigr = {"TIGR4-SigA": [mseq[i] for i in msig if msig[i] == "SigA" and i in mseq],
            "TIGR4-SigX": [mseq[i] for i in msig if msig[i] == "SigX" and i in mseq]}
    dd = pd.read_csv("datasets/d39v_tigr4_dedup_3135/test.csv")
    dpos = dd[dd.y == 1]
    tag = dpos["id"].str.split("_").str[-1]
    dsiga = dpos[tag.isin(["SigA", "SigA-LOWCONF"])]["segment"].str.upper().tolist()
    dsigx = dpos[tag == "SigX"]["segment"].str.upper().tolist()
    neg = pd.read_csv("datasets/d39v_tigr4_high5050/test.csv")
    neg_segs = neg[neg.y == 0]["segment"].str.upper().tolist()
    sets = {"SigA": tigr["TIGR4-SigA"] + dsiga,
            "SigX": tigr["TIGR4-SigX"] + dsigx}
    print({k: len(v) for k, v in sets.items()}, "neg:", len(neg_segs))
    tok = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    table = {}
    for name, segs in sets.items():
        if not segs:
            continue
        y = np.array([1] * len(segs) + [0] * len(neg_segs))
        fig, ax = plt.subplots(figsize=(5, 4))
        row = {"n_pos": len(segs), "n_neg": len(neg_segs)}
        for alias, src, st in [("3ep", MODELS["3ep"], "-"), ("2ep", MODELS["2ep"], "--"),
                               ("replica", MODELS["replica"], "-."),
                               ("benchmark", MODELS["benchmark"], ":")]:
            model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True)
            model.to(device).eval()
            s = np.concatenate([scores_for(model, tok, segs, args.batch_size, device),
                                scores_for(model, tok, neg_segs, args.batch_size, device)])
            del model
            fpr, tpr, _ = roc_curve(y, s)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, linestyle=st, label=f"{alias} (AUC {a:.3f})")
            row[alias] = round(float(a), 4)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="FPR", ylabel="TPR", title=f"Promoter ROC — {name} vs shared negs")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"roc_sigma__{name}.png", dpi=120)
        plt.close(fig)
        table[name] = row
        print(name, row)
    json.dump(table, open(OUT / "sigma_roc.json", "w"), indent=1)
    print("sigma_roc.json OK")


if __name__ == "__main__":
    main()
