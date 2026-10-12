#!/usr/bin/env python3
"""Generalization gap per model: ROC/AUC on train vs saved test scores.

Rule: before any other comparison, check train-vs-test ROC/AUC. Train side runs
inference (CSV splits and/or recomputed HF train_80); test side reuses saved
per-cell probs CSVs (never recomputed). One overlay PNG per model (same metric).

Usage:
  pixi run python scripts/generalization_gap.py --model finetuning_outputs/mini_d39v_tigr4_2ep/best \\
      --alias mini-d39v-tigr4 --train-csv datasets/d39v_tigr4_dedup_3135/train.csv \\
      --test-probs finetuning_outputs/eval_matrix_probs/mini-d39v-tigr4__d39v-tigr4-dedup.csv \\
      --test-name dedup-all --out-dir reports/model_matrix/plots --cpu
  pixi run python scripts/generalization_gap.py --model <ckpt> --alias mix \\
      --hf-train80 --train-csv datasets/d39v_tigr4_dedup_3135/train.csv --test-probs ... --gpu 1
  (or --test-csv f.csv to run test inference when no saved probs exist)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from scripts.metrics import cell_metrics, load_probs_csv
from prokbert_patches import apply_tokenizer_patches
apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser(description="Train-vs-test ROC/AUC generalization gap")
    p.add_argument("--model", required=True, help="checkpoint dir with model.safetensors")
    p.add_argument("--alias", required=True, help="short name (plot title + JSON row)")
    p.add_argument("--train-csv", action="append", default=[],
                   help="train CSV(s) with segment,y (repeatable)")
    p.add_argument("--hf-train80", action="store_true",
                   help="add recomputed HF train_80 (seed 42, len+sha asserted where known)")
    p.add_argument("--test-probs", type=str, default=None,
                   help="saved per-cell probs CSV (y_true,y_score); alternative to --test-csv")
    p.add_argument("--test-csv", action="append", default=[],
                   help="test CSV(s) with segment,y (runs inference; use when no saved probs exist)")
    p.add_argument("--dedup-unseen", type=str, default=None,
                   help="dedup dir: adds train.csv segments unseen in --train-csv (full-train reconstruction)")
    p.add_argument("--test-name", type=str, default="test")
    p.add_argument("--out-dir", type=str, default="reports/model_matrix/plots")
    p.add_argument("--batch-size", type=int, default=512)
    add_device_args(p)
    return p.parse_args()


def load_hf_train80():
    from datasets import load_dataset
    try:
        ds = load_dataset("neuralbioinfo/bacterial_promoters")
    except Exception:
        loc = Path("data/bacterial_promoters/train.parquet")
        assert loc.exists(), "no HF cache and no local parquet"
        df = pd.read_parquet(loc)
        from datasets import Dataset
        ds = {"train": Dataset.from_pandas(df)}
    tr = ds["train"]
    assert len(tr) == 223092, f"unexpected HF train size {len(tr)}"
    return tr.train_test_split(test_size=0.2, seed=42)["train"]


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device}")
    frames = []
    for f in args.train_csv:
        df = pd.read_csv(f)[["segment", "y"]]
        frames.append(df)
        print(f"train part {f}: {len(df)}")
    if args.hf_train80:
        h = load_hf_train80()
        frames.append(pd.DataFrame({"segment": h["segment"], "y": h["y"]}))
        print(f"train part hf-train80: {len(h)}")
    assert frames, "no train data (give --train-csv and/or --hf-train80)"
    tr = pd.concat(frames, ignore_index=True)
    n_extra = 0
    if args.dedup_unseen:
        seen = {str(s).upper() for s in tr["segment"]}
        dd = pd.read_csv(f"{args.dedup_unseen}/train.csv")[["segment", "y"]]
        extra = dd[~dd["segment"].str.upper().isin(seen)]
        n_extra = len(extra)
        print(f"dedup-unseen extra rows: {n_extra}")
        tr = pd.concat([tr, extra], ignore_index=True)
    ytr = tr["y"].to_numpy()
    tok = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True)
    model.to(device).eval()
    segs = [str(s).upper().replace("U", "T") for s in tr["segment"]]
    scores = []
    with torch.no_grad():
        for i in range(0, len(segs), args.batch_size):
            enc = tok(segs[i:i + args.batch_size], add_special_tokens=True, padding=True,
                      truncation=False, return_tensors="pt")
            kw = {k: v.to(device) for k, v in enc.items()}
            scores.extend(torch.softmax(model(**kw).logits, -1)[:, 1].cpu().numpy().tolist())
    str_ = np.array(scores, dtype=np.float64)
    mtr = cell_metrics(ytr, str_)
    if args.test_probs:
        yte, ste = load_probs_csv(args.test_probs)
    else:
        assert args.test_csv, "give --test-probs or --test-csv"
        te = pd.concat([pd.read_csv(f)[["segment", "y"]] for f in args.test_csv], ignore_index=True)
        yte = te["y"].to_numpy()
        segs_te = [str(s).upper().replace("U", "T") for s in te["segment"]]
        ste = []
        with torch.no_grad():
            for i in range(0, len(segs_te), args.batch_size):
                enc = tok(segs_te[i:i + args.batch_size], add_special_tokens=True, padding=True,
                          truncation=False, return_tensors="pt")
                kw = {k: v.to(device) for k, v in enc.items()}
                ste.extend(torch.softmax(model(**kw).logits, -1)[:, 1].cpu().numpy().tolist())
        ste = np.array(ste, dtype=np.float64)
    mte = cell_metrics(yte, ste)
    row = {"alias": args.alias, "test": args.test_name, "n_train": len(ytr),
           "train": {"auc": mtr["auc"], "mcc": mtr["mcc_05"], "acc": mtr["acc_05"]},
           "test_n": int(len(yte)),
           "test": {"auc": mte["auc"], "mcc": mte["mcc_05"], "acc": mte["acc_05"]},
           "gap_auc": float(mtr["auc"] - mte["auc"]), "gap_mcc": float(mtr["mcc_05"] - mte["mcc_05"])}
    print(json.dumps(row, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for y, s, lab in [(ytr, str_, f"train (AUC {mtr['auc']:.3f})"),
                      (yte, ste, f"{args.test_name} (AUC {mte['auc']:.3f})")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.5, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=f"ROC train vs test — {args.alias}", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"roc_gap__{args.alias}.png", dpi=120)
    plt.close(fig)
    print(f"roc_gap__{args.alias}.png OK (gap AUC {row['gap_auc']:+.4f}, MCC {row['gap_mcc']:+.4f})")


if __name__ == "__main__":
    main()
