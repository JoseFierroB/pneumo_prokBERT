#!/usr/bin/env python3
"""Source ablation: positives + negatives from ONE source (CDS or IGR) at 1:2.

Within each high5050 split, bootstrap that split's SOURCE negatives to 2x its
positives (seed 42, high_1to2 method; copies get '#r<k>' suffix). No sequence
crosses splits. Columns segment,y,id,src.

Output: datasets/d39v_tigr4_{cds,igr}only/{train,eval,test,all}.csv + MANIFEST.json

Usage:
  pixi run python scripts/build_source_ablation.py --source CDS
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "datasets/d39v_tigr4_high5050"


def parse_args():
    p = argparse.ArgumentParser(description="Single-source 1:2 ablation")
    p.add_argument("--source", required=True, choices=["CDS", "IGR"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha16(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dst = Path(f"datasets/d39v_tigr4_{args.source.lower()}only")
    dst.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ("train", "eval", "test"):
        df = pd.read_csv(f"{SRC}/{split}.csv")
        pos = df[df.y == 1].reset_index(drop=True)
        neg = df[(df.y == 0) & (df.src == args.source)].reset_index(drop=True)
        need = 2 * len(pos)
        idx = rng.integers(0, len(neg), size=need)
        copies = neg.iloc[idx].reset_index(drop=True).copy()
        seen, ids = Counter(), []
        for i in copies["id"]:
            k = seen[i]
            ids.append(i if k == 0 else f"{i}#r{k}")
            seen[i] += 1
        copies["id"] = ids
        frames[split] = pd.concat([pos, copies], ignore_index=True)
        print(f"{split}: pos={len(pos)} neg={len(copies)} (unique {len(neg)}->{copies['segment'].nunique()})")
    for split, df in frames.items():
        df[["segment", "y", "id", "src"]].to_csv(dst / f"{split}.csv", index=False)
    pd.concat(frames.values(), ignore_index=True)[["segment", "y", "id", "src"]].to_csv(dst / "all.csv", index=False)
    sets = {s: set(frames[s]["segment"]) for s in frames}
    assert not ((sets["train"] & sets["eval"]) | (sets["train"] & sets["test"]) | (sets["eval"] & sets["test"]))
    man = {"source": f"{SRC} splits, {args.source}-only negatives bootstrapped 1:2 within split",
           "seed": args.seed, "method": "per-split bootstrap (high_1to2 recipe)",
           "counts": {s: {"n": len(df), "pos": int((df.y == 1).sum())} for s, df in frames.items()},
           "files_sha16": {f: sha16(dst / f) for f in ("train.csv", "eval.csv", "test.csv", "all.csv")}}
    (dst / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print("MANIFEST OK")


if __name__ == "__main__":
    main()
