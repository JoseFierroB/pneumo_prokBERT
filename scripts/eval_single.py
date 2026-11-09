#!/usr/bin/env python3
"""Evaluate one supplementary model (not in models.json) on the 4 columns.

Reuses builders/inference from scripts.eval_matrix. Writes metrics + CSV
under a supplementary dir (never touches the canonical matrix). Also writes
the characteristic model plots: ROC overlay, PR overlay, confusion matrices.

Usage:
  pixi run python scripts/eval_single.py --model finetuning_outputs/promoter_d39v_tigr4_2ep/best \
      --alias promoter-d39v-tigr4 --out-dir reports/supplementary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, ".")
from scripts.common import add_device_args, normalize, resolve_device
from scripts.eval_matrix import build_columns, cell_metrics, infer_model, pretokenize
from scripts.metrics import save_probs_csv
from prokbert.prokbert_tokenizer import ProkBERTTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Supplementary single-model eval (canonical matrix untouched)")
    p.add_argument("--model", required=True, help="HF model dir with model.safetensors")
    p.add_argument("--alias", required=True, help="short name for outputs")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of columns")
    p.add_argument("--out-dir", type=str, default="reports/supplementary")
    p.add_argument("--reference-root", type=str, default="datasets/reference")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-overlap", nargs="*", default=[],
                   help="columns to force-flag train_overlap (model trained on those rows)")
    p.add_argument("--leakage-base", type=str, default="promoter-baseline",
                   help="registry alias whose leakage flags seed the rest")
    p.add_argument("--hf-split-dir", type=str, default=None,
                   help="versioned HF split dir (manifest+ids): overrides paper-test "
                        "with its test part (use with matching train; set --leakage-base "
                        "and --train-overlap explicitly)")
    add_device_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device}")
    manifest = json.load(open("datasets_eval.json"))
    base_leak = manifest["leakage"].get(args.leakage_base, {})
    eval_only = [c for c in manifest["columns"]
                 if manifest["columns"][c].get("kind") != "selection diagnostic (appendix, not ranked)"]
    columns = args.datasets or eval_only
    args_ns = argparse.Namespace(reference_root=args.reference_root)
    tok = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    data = build_columns(args_ns, columns)
    if args.hf_split_dir and "paper-test" in data:
        import sys as _sys
        _sys.path.insert(0, ".")
        from scripts.hf_local import split_frames
        frames, spec = split_frames(args.hf_split_dir)
        te = frames["test"]
        data["paper-test"] = ([normalize(s) for s in te["segment"]],
                              te["y"].to_numpy())
        print(f"paper-test overridden by {args.hf_split_dir} test (spec {spec}, n={len(te)}) — "
              f"set --leakage-base/--train-overlap explicitly")
    tok_data = pretokenize(tok, data)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
    probed = infer_model(model, tok_data, columns, args.batch_size, device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    probed_scores = {}
    for col in columns:
        scores, y = probed[col]
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        probed_scores[col] = (y, scores)
        m = cell_metrics(y, scores)
        m["leakage"] = "train_overlap" if col in args.train_overlap else base_leak.get(col, "train_overlap")
        res[col] = m
        print(f"  {col:22s} ACC {m['acc_05']:.4f} MCC {m['mcc_05']:.4f} AUC {m['auc']:.4f} [{m['leakage']}]", flush=True)
        save_probs_csv(out / f"{args.alias}__{col}.csv", y, scores.astype(np.float32))
    json.dump({args.alias: res}, open(out / f"{args.alias}.json", "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    print(f"saved {out}/{args.alias}.json + .csv per column")
    plot_model_report(probed_scores, res, out, args.alias)


def plot_model_report(probed_scores, res, out_dir, alias):
    """Characteristic model plots: ROC overlay, PR overlay, confusion matrices."""
    from pathlib import Path
    out_dir = Path(out_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_curve, roc_curve)
    cols = list(probed_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    for col in cols:
        y, s = probed_scores[col]
        try:
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.5, label=f"{col} (AUC {res[col]['auc']:.3f})")
        except ValueError:
            print(f"  {col}: single class, ROC skipped")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=f"ROC — {alias}", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{alias}.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for col in cols:
        y, s = probed_scores[col]
        try:
            ap = average_precision_score(y, s)
            pr, rc, _ = precision_recall_curve(y, s)
            ax.plot(rc, pr, lw=1.5, label=f"{col} (AP {ap:.3f})")
        except ValueError:
            print(f"  {col}: single class, PR skipped")
    ax.set(title=f"Precision-Recall — {alias}", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_{alias}.png", dpi=120)
    plt.close(fig)
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), squeeze=False)
    for axi, col in zip(axes[0], cols):
        y, s = probed_scores[col]
        cm = confusion_matrix(y, (s >= 0.5).astype(int), labels=[0, 1])
        im = axi.imshow(cm, cmap="Blues")
        axi.set_xticks([0, 1])
        axi.set_yticks([0, 1])
        axi.set(title=f"{col}\nacc {res[col]['acc_05']:.3f} mcc {res[col]['mcc_05']:.3f}",
                xlabel="pred", ylabel="true")
        for i in range(2):
            for j in range(2):
                axi.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=axi, fraction=0.046)
    fig.suptitle(f"CM @0.5 — {alias}")
    fig.tight_layout()
    fig.savefig(out_dir / f"cm_{alias}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved roc_{alias}.png pr_{alias}.png cm_{alias}.png")


if __name__ == "__main__":
    main()
