#!/usr/bin/env python3
"""Build unified D39V+TIGR4-High set at 1:2 (+:-) from dedup splits.

Keeps positives and their split assignment; bootstraps negatives WITHIN each
split (seed 42) to exactly 2x positives, so no sequence crosses splits.
Copies carry id suffix '#r<k>' for traceability.

Output: datasets/d39v_tigr4_high_1to2/{train,eval,test,all}.csv + MANIFEST.json

Usage:
  pixi run python scripts/build_unified_1to2.py [--seed 42]
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "datasets/d39v_tigr4_dedup_3135"
DST = "datasets/d39v_tigr4_high_1to2"


def parse_args():
    p = argparse.ArgumentParser(description="Build unified 1:2 set from dedup splits")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(DST)
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ("train", "eval", "test"):
        df = pd.read_csv(f"{SRC}/{split}.csv")
        pos = df[df.y == 1].reset_index(drop=True)
        neg = df[df.y == 0].reset_index(drop=True)
        need = 2 * len(pos)
        idx = rng.integers(0, len(neg), size=need)
        copies = neg.iloc[idx].reset_index(drop=True).copy()
        # suffix only resampled duplicates; keep first occurrence id intact
        seen, ids = Counter(), []
        for i in copies["id"]:
            k = seen[i]
            ids.append(i if k == 0 else f"{i}#r{k}")
            seen[i] += 1
        copies["id"] = ids
        frames[split] = pd.concat([pos, copies], ignore_index=True)
        print(f"{split}: pos={len(pos)} neg={len(copies)} (unique {neg['segment'].nunique()}->{copies['segment'].nunique()})")
    for split, df in frames.items():
        df.to_csv(out / f"{split}.csv", index=False)
    alldf = pd.concat(frames.values(), ignore_index=True)
    alldf.to_csv(out / "all.csv", index=False)
    # cross-split overlap check (must be empty)
    sets = {s: set(frames[s]["segment"]) for s in frames}
    overlap = (sets["train"] & sets["eval"]) | (sets["train"] & sets["test"]) | (sets["eval"] & sets["test"])
    manifest = {"method": "per-split bootstrap of dedup negatives to 1:2 (pos:neg), positives untouched",
                "seed": args.seed, "source": SRC,
                "counts": {s: {"n": len(frames[s]), "pos": int((frames[s].y == 1).sum()),
                               "neg": int((frames[s].y == 0).sum())} for s in frames},
                "cross_split_overlap": len(overlap),
                "files_sha16": {f: hashlib.sha256(open(out / f, "rb").read()).hexdigest()[:16]
                                for f in ("train.csv", "eval.csv", "test.csv", "all.csv")}}
    json.dump(manifest, open(out / "MANIFEST.json", "w"), indent=2)
    assert len(overlap) == 0, f"cross-split overlap: {len(overlap)}"
    for s in frames:
        assert (frames[s].y == 1).sum() * 2 == (frames[s].y == 0).sum(), f"ratio off in {s}"
    print(f"OK {DST}: " + ", ".join(f"{s}={len(frames[s])}" for s in frames) + " ratio 1:2 exacto")


if __name__ == "__main__":
    main()
