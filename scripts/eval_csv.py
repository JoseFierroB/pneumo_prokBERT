#!/usr/bin/env python3
"""Single-CSV inference: model + CSV(segment,y) -> probs CSV(y_true,y_score).

For ad-hoc head-to-head on non-matrix splits (leakage handled by caller
via clean slices). Reuses tokenizer/model pattern of eval_matrix.py.

Usage:
  pixi run python scripts/eval_csv.py --model <ckpt> --csv f.csv --out probs.csv [--batch-size 512] [--gpu 0|--cpu]
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from prokbert_patches import apply_tokenizer_patches
apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser(description="Single-CSV inference to probs CSV")
    p.add_argument("--model", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    add_device_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    df = pd.read_csv(args.csv)
    assert {"segment", "y"} <= set(df.columns)
    segs = [str(s).upper().replace("U", "T") for s in df["segment"]]
    tok = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True)
    model.to(device).eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(segs), args.batch_size):
            enc = tok(segs[i:i + args.batch_size], add_special_tokens=True, padding=True,
                      truncation=False, return_tensors="pt")
            kw = {k: v.to(device) for k, v in enc.items()}
            scores.extend(torch.softmax(model(**kw).logits, -1)[:, 1].cpu().numpy().tolist())
    pd.DataFrame({"y_true": df["y"].to_numpy(), "y_score": np.array(scores)}).to_csv(args.out, index=False)
    print(f"{args.out}: n={len(df)}")


if __name__ == "__main__":
    main()
