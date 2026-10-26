#!/usr/bin/env python3
"""Per-sigma ROC overlays on the 4197 benchmark set (SigA/SigX x prop/all negs).

Positives: SigA = SigA+SigA-LOWCONF (no distinction); SigX = SigX+ComX (merged,
precedent: promoter-tools sigma_stratify.py, plot_sigma_roc.py:65). None/other
TFs excluded. Negatives: the 2798 of the set (prop = subsample to n_pos, seed 42).

Models: benchmark (mini-promoter HF), mix-1ep, mix-2ep, replica-code-2ep.
Leakage: 1ep/2ep trained on high5050-train which overlaps this set -> per-set
overlap counts reported + clean-subset (unseen) AUC alongside full AUC.
Replica/benchmark never saw pneumo -> clean.

Output: reports/supplementary_5050/sigma/roc_4197__{SigA,SigX}-{prop,all}.png
  + sigma_roc_4197.json + probs CSVs.

Usage (WS-5090):
  pixi run python scripts/plot_sigma_roc_4197.py --gpu 0
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
from sklearn.utils import resample

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from scripts.metrics import save_probs_csv
from prokbert_patches import apply_tokenizer_patches
apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import AutoModelForSequenceClassification

MODELS = [
    ("benchmark", "neuralbioinfo/prokbert-mini-promoter", ":"),
    ("1ep", "finetuning_outputs/mini_223k_d39v_tigr4_high5050_1ep/best", "-"),
    ("2ep", "finetuning_outputs/mini_223k_d39v_tigr4_high5050_2ep/best", "--"),
    ("replica", "finetuning_outputs/replica_code_2ep/best", "-."),
]
MIX_TRAIN = "datasets/d39v_tigr4_high5050/train.csv"
DATA = "datasets/d39v_tigr4_high_1to2/all.csv"
OUT = Path("reports/supplementary_5050/sigma")
SEED = 42
N_BOOT = 2000


def parse_args():
    p = argparse.ArgumentParser(description="Per-sigma ROC overlays on 4197 set")
    p.add_argument("--batch-size", type=int, default=64)
    add_device_args(p)
    return p.parse_args()


def family(tag):
    if tag in ("SigA", "SigA-LOWCONF"):
        return "SigA"
    if tag in ("SigX", "ComX"):
        return "SigX"
    return "other"


def scores_for(model, tok, segs, bs, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(segs), bs):
            enc = tok(segs[i:i + bs], add_special_tokens=True, padding=True,
                      truncation=False, return_tensors="pt")
            kw = {k: v.to(device) for k, v in enc.items()}
            out.extend(torch.softmax(model(**kw).logits, -1)[:, 1].cpu().numpy().tolist())
    return np.array(out, dtype=np.float64)


def auc_ci(y, s, seed=SEED, n_boot=N_BOOT):
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], s[idx])
        vals.append(auc(fpr, tpr))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device}")
    OUT.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(DATA)
    d["tag"] = d["id"].str.extract(r"_([A-Za-z0-9\-]+)$")[0]
    d["fam"] = d["tag"].map(family)
    d["segment"] = d["segment"].str.upper()
    seen = set(pd.read_csv(MIX_TRAIN)["segment"].str.upper())
    d["in_mixtrain"] = d["segment"].isin(seen)
    print("pos-by-fam:", d[d.y == 1].fam.value_counts().to_dict())
    print("overlap with high5050-train:",
          d[d.fam.isin(["SigA", "SigX"])].groupby("fam")["in_mixtrain"].agg(["sum", "count"]).to_dict())

    negs = d[d.y == 0]
    neg_segs_all = negs["segment"].tolist()
    rng = np.random.RandomState(SEED)
    # TIGR4-ComX -> SigX (precedent: plot_sigma_roc.py:64-70). NOTE (verified
    # 2026-10-16): all 6 are EXACT duplicates of D39V rows already in the 4197,
    # so SigX rows = 27 but unique sequences = 21. Both numbers recorded.
    meta = pd.read_csv("datasets/external/positives_high_sigma_assigned_metadata.tsv", sep="\t")
    comx_ids = [str(i) for i in meta.loc[meta["Sigma_Factor"] == "ComX", "Sequence_ID"]]
    mseq = {str(r.id): str(r.seq).upper()
            for r in SeqIO.parse("datasets/reference/tigr4-high/positives_81bp.fasta", "fasta")}
    missing = [i for i in comx_ids if i not in mseq]
    assert not missing, f"ComX ids missing in fasta: {missing}"
    tigr_sigx = [mseq[i] for i in comx_ids]
    have = set(d["segment"])
    n_dup = sum(s in have for s in tigr_sigx)
    sigx_segs = [s for s in d[(d.y == 1) & (d.fam == "SigX")]["segment"].tolist()] + tigr_sigx
    assert len(sigx_segs) == 27, f"SigX rows {len(sigx_segs)} != 27"
    print(f"SigX rows=27 unique={len(set(sigx_segs))} ({n_dup} exact dupes of 4197 rows)")
    sets = {}
    for sig in ("SigA", "SigX"):
        if sig == "SigX":
            pos_segs = sigx_segs
            pos_keep = np.array([s in seen for s in pos_segs])
        else:
            pos = d[(d.y == 1) & (d.fam == sig)]
            pos_segs = pos["segment"].tolist()
            pos_keep = pos["in_mixtrain"].to_numpy()
        sub = negs.sample(n=len(pos_segs), random_state=SEED)
        sets[f"{sig}-prop"] = (pos_segs, sub["segment"].tolist(), pos_keep,
                               np.zeros(len(sub), dtype=bool))
        sets[f"{sig}-all"] = (pos_segs, neg_segs_all, pos_keep,
                              negs["in_mixtrain"].to_numpy())

    tok = ProkBERTTokenizer(tokenization_params={"kmer": 6, "shift": 1}, operation_space="sequence")
    table = {}
    for name, (pos_segs, neg_segs, pos_overlap, neg_overlap) in sets.items():
        y = np.array([1] * len(pos_segs) + [0] * len(neg_segs))
        overlap = np.concatenate([pos_overlap, neg_overlap])
        fig, ax = plt.subplots(figsize=(5, 4))
        row = {"n_pos": len(pos_segs), "n_neg": len(neg_segs),
               "n_overlap_mixtrain": int(overlap.sum())}
        if name.startswith("SigX"):
            row["n_pos_unique"] = len(set(pos_segs))
            row["composition"] = "21 D39V-SigX + 6 TIGR4-ComX (exact dupes of 4197 rows)"
        for alias, src, st in MODELS:
            model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True)
            model.to(device).eval()
            s = np.concatenate([scores_for(model, tok, pos_segs, args.batch_size, device),
                                scores_for(model, tok, neg_segs, args.batch_size, device)])
            del model
            torch.cuda.empty_cache()
            fpr, tpr, _ = roc_curve(y, s)
            a = auc(fpr, tpr)
            lo, hi = auc_ci(y, s)
            keep = ~overlap
            ac = float("nan")
            if keep.sum() > 10 and len(np.unique(y[keep])) == 2:
                fprc, tprc, _ = roc_curve(y[keep], s[keep])
                ac = auc(fprc, tprc)
            ax.plot(fpr, tpr, lw=1.5, linestyle=st, label=f"{alias} (AUC {a:.3f})")
            row[alias] = {"auc": round(float(a), 4), "ci95": [round(lo, 4), round(hi, 4)],
                          "auc_clean": round(ac, 4), "n_clean": int(keep.sum())}
            save_probs_csv(OUT / f"roc4197_{name}__{alias}.csv", y, s)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="FPR", ylabel="TPR", title=f"Promoter ROC — {name} (4197 set)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"roc_4197__{name}.png", dpi=120)
        plt.close(fig)
        table[name] = row
        print(name, json.dumps(row)[:220], flush=True)
    json.dump(table, open(OUT / "sigma_roc_4197.json", "w"), indent=1)
    print("sigma_roc_4197.json OK")


if __name__ == "__main__":
    main()
