#!/usr/bin/env python3
"""
Canonical evaluation matrix: 7 model aliases x 7 dataset columns.

Reads models.json + datasets_eval.json. For each cell computes N, balance,
CM@0.5, ACC/balACC/MCC/F1/ROC-AUC/PR-AUC, TPR/TNR/PPV/NPV/FPR, pred-pos-rate,
Youden threshold + metrics, and the leakage flag from the manifest.

One tokenizer call per dataset (pre-tokenized, no per-batch Pool storms).
Per-cell (y_true, y_score) saved as .npz to --probs_dir (ignored); tracked
outputs go to --reports_dir: metrics.json, summary_metrics.csv, ROC/PR plots.

Run from the repo root, e.g. on the WS 5090:
  pixi run python scripts/eval_matrix.py --canon_root /home/fierro/promoter-tools/data/benchmark/canonical_15_datasets
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve)

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

HF_DATASET = "neuralbioinfo/bacterial_promoters"
EXPECTED_TEST_SHA16 = "49a61e636397556b"


def parse_args():
    p = argparse.ArgumentParser(description="Canonical 7x7 evaluation matrix")
    p.add_argument("--models", nargs="*", default=None, help="subset of aliases (default: all in models.json)")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of evaluable columns (default: all except appendix)")
    p.add_argument("--canon_root", type=str,
                   default="../promoter-tools/data/benchmark/canonical_15_datasets",
                   help="Canonical FASTA root (default: sibling promoter-tools repo, relative to CWD=repo root)")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--reports_dir", type=str, default="reports/model_matrix")
    p.add_argument("--probs_dir", type=str, default="finetuning_outputs/eval_matrix_probs")
    add_device_args(p)
    return p.parse_args()


def sha_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def cell_metrics(y, scores, thr=0.5):
    pred = (scores >= thr).astype(int)
    fpr, tpr, ths = roc_curve(y, scores)
    idx = int(np.argmax(tpr - fpr))
    ythr = float(ths[idx])
    pred_y = (scores >= ythr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel().tolist()
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "cm_05": [[tn, fp], [fn, tp]],
        "acc_05": float(accuracy_score(y, pred)),
        "balacc_05": float(balanced_accuracy_score(y, pred)),
        "mcc_05": float(matthews_corrcoef(y, pred)),
        "f1_05": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "tpr_05": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tnr_05": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ppv_05": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "pred_pos_rate_05": float(pred.mean()),
        "youden_thr": ythr,
        "acc_y": float(accuracy_score(y, pred_y)),
        "mcc_y": float(matthews_corrcoef(y, pred_y)),
    }


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} (auto-detect; override with --gpu N or --cpu)")
    models = json.load(open("models.json"))
    manifest = json.load(open("datasets_eval.json"))
    aliases = args.models or [k for k in models if models[k]["status"] == "exists"]
    skipped = [k for k in (args.models or []) if models.get(k, {}).get("status") != "exists"]
    if skipped:
        print(f"skip (not trained yet): {skipped}")
    aliases = [k for k in aliases if models.get(k, {}).get("status") == "exists"]
    eval_only = [c for c in manifest["columns"] if manifest["columns"][c].get("kind") != "selection diagnostic (appendix, not ranked)"]
    columns = args.datasets or eval_only
    unknown = [c for c in columns if c not in manifest["columns"] or manifest["columns"][c].get("kind") == "selection diagnostic (appendix, not ranked)"]
    if unknown:
        raise SystemExit(f"not evaluable columns: {unknown}")
    print(f"aliases: {aliases}\ncolumns: {columns}")

    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')

    # --- build datasets once (segment lists + labels) ---
    data = {}
    if any(c.startswith("paper-") for c in columns):
        ds = load_dataset(HF_DATASET)
        tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
        test_10 = tmp["test"].train_test_split(test_size=0.5, seed=42)["test"]
        assert sha_ids(test_10["segment_id"])[:16] == EXPECTED_TEST_SHA16, "in-dist test IDs differ from manifest"
        data["paper-test"] = ([s.upper().replace("U", "T") for s in test_10["segment"]],
                              np.array(test_10["y"]))
        print("paper-test OK", len(data["paper-test"][0]))
    import pandas as pd
    if "ecoli-sigma70" in columns or "multispecies" in columns:
        ds = load_dataset(HF_DATASET)
        for col, split in [("ecoli-sigma70", "test_sigma70"), ("multispecies", "test_multispecies")]:
            if col in columns:
                d = ds[split]
                data[col] = ([s.upper().replace("U", "T") for s in d["segment"]], np.array(d["y"]))
                print(col, "OK", len(data[col][0]))
    from Bio import SeqIO
    for col, sub, pf, nf in [("d39v", "1_d39v", "positives_81bp.fasta", "negatives_81bp.fasta"),
                             ("tigr4-high", "2_tigr4_high", "positives_81bp.fasta", "negatives_81bp.fasta"),
                             ("d39v-tigr4", "4_d39v_tigr4_high", "positives_81bp.fasta", "negatives_81bp.fasta")]:
        if col in columns:
            pos = list(SeqIO.parse(f"{args.canon_root}/{sub}/{pf}", "fasta"))
            neg = list(SeqIO.parse(f"{args.canon_root}/{sub}/{nf}", "fasta"))
            seqs = [str(r.seq).upper() for r in pos + neg]
            data[col] = (seqs, np.array([1] * len(pos) + [0] * len(neg)))
            print(col, "OK", len(seqs))
    if "d39v-tigr4-dedup" in columns:
        df = pd.read_csv("datasets/d39v_tigr4_dedup_3135/all.csv")
        data["d39v-tigr4-dedup"] = (df["segment"].str.upper().tolist(), df["y"].to_numpy())
        print("d39v-tigr4-dedup OK", len(df))

    # --- pre-tokenize once per dataset ---
    tok_data = {}
    for col, (seqs, y) in data.items():
        enc = tokenizer(seqs, add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
        tok_data[col] = (enc, y)
        print(f"tokenized {col}: {enc['input_ids'].shape}")

    probs_dir = Path(args.probs_dir)
    probs_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for alias in aliases:
        entry = models[alias]
        src = entry["init"]["hf_id"] if alias == "hfpromoter-baseline" else entry["artifact"]
        print(f"\n=== {alias} <- {src} ===")
        model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True).to(device).eval()
        results[alias] = {}
        for col in columns:
            enc, y = tok_data[col]
            scores = []
            with torch.no_grad():
                for i in range(0, len(y), args.batch_size):
                    b = {k: v[i:i + args.batch_size].to(device) for k, v in enc.items()}
                    logits = model(**b).logits
                    scores.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy().tolist())
            scores = np.array(scores, dtype=np.float64)
            m = cell_metrics(y, scores)
            m["leakage"] = manifest["leakage"][alias][col]
            results[alias][col] = m
            np.savez_compressed(probs_dir / f"{alias}__{col}.npz", y_true=y, y_score=scores.astype(np.float32))
            print(f"  {col:22s} ACC {m['acc_05']:.4f} MCC {m['mcc_05']:.4f} AUC {m['auc']:.4f} AP {m['ap']:.4f} [{m['leakage']}]")
        del model
        torch.cuda.empty_cache()

    rep = Path(args.reports_dir)
    rep.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(rep / "metrics.json", "w"), indent=2)
    import csv
    with open(rep / "summary_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "n", "n_pos", "n_neg", "acc_05", "balacc_05", "mcc_05", "f1_05",
                    "auc", "ap", "tpr_05", "tnr_05", "ppv_05", "youden_thr", "acc_y", "mcc_y", "leakage",
                    "TN", "FP", "FN", "TP"])
        for alias in aliases:
            for col in columns:
                m = results[alias][col]
                tn, fp, fn, tp = m["cm_05"][0][0], m["cm_05"][0][1], m["cm_05"][1][0], m["cm_05"][1][1]
                w.writerow([alias, col, m["n"], m["n_pos"], m["n_neg"], f"{m['acc_05']:.4f}", f"{m['balacc_05']:.4f}",
                            f"{m['mcc_05']:.4f}", f"{m['f1_05']:.4f}", f"{m['auc']:.4f}", f"{m['ap']:.4f}",
                            f"{m['tpr_05']:.4f}", f"{m['tnr_05']:.4f}", f"{m['ppv_05']:.4f}",
                            f"{m['youden_thr']:.4f}", f"{m['acc_y']:.4f}", f"{m['mcc_y']:.4f}", m["leakage"],
                            tn, fp, fn, tp])
    print(f"\nsaved {rep/'metrics.json'} + summary_metrics.csv + {probs_dir}/*.npz")

    # --- ROC/PR overlays per dataset ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve
        for col in columns:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            for alias in aliases:
                z = np.load(probs_dir / f"{alias}__{col}.npz")
                y, s = z["y_true"], z["y_score"]
                fpr, tpr, _ = roc_curve(y, s)
                prec, rec, _ = precision_recall_curve(y, s)
                m = results[alias][col]
                ax[0].plot(fpr, tpr, lw=1.5, label=f"{alias} (AUC {m['auc']:.3f})")
                ax[1].plot(rec, prec, lw=1.5, label=f"{alias} (AP {m['ap']:.3f})")
            ax[0].plot([0, 1], [0, 1], "k--", lw=1)
            ax[0].set(title=f"ROC — {col}", xlabel="FPR", ylabel="TPR")
            ax[1].set(title=f"PR — {col}", xlabel="Recall", ylabel="Precision")
            for a in ax:
                a.legend(fontsize=7)
                a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(rep / f"roc_pr__{col}.png", dpi=120)
            plt.close(fig)
            print(f"plot {col} OK")
    except ImportError:
        print("matplotlib not available — plots skipped")


if __name__ == "__main__":
    main()
