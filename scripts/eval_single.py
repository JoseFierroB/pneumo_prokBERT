#!/usr/bin/env python3
"""Evaluate one supplementary model (not in models.json) on the 4 columns.

Reuses builders/inference from scripts.eval_matrix. Writes metrics + CSV
under a supplementary dir (never touches the canonical matrix).

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
from scripts.common import add_device_args, resolve_device
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
    tok_data = pretokenize(tok, data)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
    probed = infer_model(model, tok_data, columns, args.batch_size, device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for col in columns:
        scores, y = probed[col]
        m = cell_metrics(y, scores)
        m["leakage"] = "train_overlap" if col in args.train_overlap else base_leak.get(col, "train_overlap")
        res[col] = m
        print(f"  {col:22s} ACC {m['acc_05']:.4f} MCC {m['mcc_05']:.4f} AUC {m['auc']:.4f} [{m['leakage']}]", flush=True)
        save_probs_csv(out / f"{args.alias}__{col}.csv", y, scores.astype(np.float32))
    json.dump({args.alias: res}, open(out / f"{args.alias}.json", "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    print(f"saved {out}/{args.alias}.json + .csv per column")


if __name__ == "__main__":
    main()
