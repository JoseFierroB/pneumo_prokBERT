#!/usr/bin/env python3
"""5-fold stratified split of high5050 train (pneumo side only; 223k fixed).

Per fold k: datasets/kfold_5050/fold{k}/{train,eval,test}.csv + MANIFEST.json.
train = 4/5 of high5050-train (stratified y, seed 42+k); eval/test = high5050
copies (selection/test locked, identical all folds).

Usage:
  pixi run python scripts/build_kfold.py [--n-splits 5] [--seed 42]
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

SRC = "datasets/d39v_tigr4_high5050"
DST = "datasets/kfold_5050"


def parse_args():
    p = argparse.ArgumentParser(description="5-fold split of 5050 train")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha16(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main():
    args = parse_args()
    tr = pd.read_csv(f"{SRC}/train.csv")
    ev = pd.read_csv(f"{SRC}/eval.csv")
    te = pd.read_csv(f"{SRC}/test.csv")
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for k, (_, te_idx) in enumerate(skf.split(tr, tr["y"])):
        hold = tr.iloc[te_idx].reset_index(drop=True)
        fit = tr.drop(tr.index[te_idx]).reset_index(drop=True)
        assert len(set(fit["segment"]) & set(hold["segment"])) == 0
        out = Path(DST) / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        fit[["segment", "y", "id", "src"]].to_csv(out / "train.csv", index=False)
        ev[["segment", "y", "id", "src"]].to_csv(out / "eval.csv", index=False)
        te[["segment", "y", "id", "src"]].to_csv(out / "test.csv", index=False)
        man = {"fold": k, "seed": args.seed, "method": "StratifiedKFold on high5050-train (eval/test locked copies)",
               "n_fit": len(fit), "n_hold": len(hold),
               "files_sha16": {f: sha16(out / f) for f in ("train.csv", "eval.csv", "test.csv")}}
        (out / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"fold{k}: fit={len(fit)} hold={len(hold)}")
    print("KFOLD OK")


if __name__ == "__main__":
    main()
