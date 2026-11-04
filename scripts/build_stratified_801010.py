#!/usr/bin/env python3
"""Stratified pneumo splits by (label, sigma-family, strain) with quotas.

Unlike build_kfold.py (y-only), this guarantees every minority stratum
(SigX, TIGR4-SigX) is present in train AND eval AND test:
  - positives: stratify key = (y=1, family, strain); family merges SigX+ComX;
    IDs missing from metadata form their own UNK stratum (kept, reported).
  - negatives: stratify key = (y=0, src, strain); strain from ID
    (NEG_SPV_*=D39V, TIGR4_*=TIGR4, NEG_IGR_{STRAIN}_*=that strain).
Proportional 70/15/15, seed 42, no resampling. Minimum check: every stratum
with n>=3 appears in all three splits (asserted; fails loudly otherwise).

Usage:
  pixi run python scripts/build_stratified_5050.py [--seed 42] [--out datasets/high5050_stratified]
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "datasets/d39v_tigr4_high5050"
META = "/home/fierro/promoter-tools/data/benchmark/datasets/2_d39v_tigr4_high_sigma/positives_81bp_metadata.tsv"


def sha16(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def fam_of(v):
    return "SigX" if v in ("SigX", "ComX") else (v if pd.notna(v) else "None")


def neg_strain(i):
    if i.startswith("NEG_IGR_"):
        return i.split("_")[2]
    if "TIGR4" in i:
        return "TIGR4"
    if "D39V" in i or i.startswith("NEG_SPV"):
        return "D39V"
    return "UNK"


def main():
    ap = argparse.ArgumentParser(description="Stratified (y, sigma, strain) pneumo splits.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="datasets/high5050_stratified")
    args = ap.parse_args()

    pool = pd.concat([pd.read_csv(f"{SRC}/{s}.csv", dtype=str)
                      for s in ("train", "eval", "test")],
                     ignore_index=True).drop_duplicates("segment").reset_index(drop=True)
    meta = pd.read_csv(META, sep="\t", dtype=str).set_index("Sequence_ID")

    def stratum(r):
        if r["y"] == "1":
            i = r["id"]
            if i in meta.index:
                m = meta.loc[i]
                ch = m["Chromosome"]
                st = "D39V" if ch == "D39V" else "TIGR4"
                return ("pos", fam_of(m["Sigma_Factor"]), st)
            return ("pos", "UNK", "UNK")
        return ("neg", r["src"], neg_strain(r["id"]))

    pool["__st__"] = [stratum(r) for _, r in pool.iterrows()]
    print("strata:", pool["__st__"].value_counts().to_dict())
    pool_all = pool.copy()
    unk_mask = pool["__st__"].astype(str) == "('pos', 'UNK', 'UNK')"
    pool_unk, pool = pool[unk_mask], pool[~unk_mask].reset_index(drop=True)
    print("UNK whole-train:", len(pool_unk))

    tr, rest = train_test_split(pool, test_size=0.20, random_state=args.seed,
                                stratify=pool["__st__"])
    tr = pd.concat([tr, pool_unk], ignore_index=True)
    ev, te = train_test_split(rest, test_size=0.50, random_state=args.seed,
                              stratify=rest["__st__"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in (("train", tr), ("eval", ev), ("test", te)):
        df = df.drop(columns=["__st__"]).reset_index(drop=True)
        df.to_csv(out / f"{name}.csv", index=False)
        c = df.merge(pool_all[["segment", "__st__"]].drop_duplicates(), left_on="segment",
                     right_on="segment", how="left")["__st__"].value_counts().to_dict()
        counts[name] = {str(k): int(v) for k, v in c.items()}
        print(name, len(df), "pos:", int((df.y == "1").sum()))

    # minimum-quota audit: strata with n>=3 overall must appear in all splits
    totals = pool["__st__"].value_counts()
    problems = []
    for st, n in totals.items():
        if str(st) == "('pos', 'UNK', 'UNK')":
            continue
        if n >= 3:
            for name in ("train", "eval", "test"):
                if counts[name].get(str(st), 0) == 0:
                    problems.append((str(st), name))
    if problems:
        raise SystemExit(f"ERROR: quota missed {problems}")
    man = {"seed": args.seed, "method": "stratified (y, sigma-family, strain), 80/10/10, quotas>=3 (UNK n=5 whole-train)",
           "counts": counts,
           "files_sha16": {f: sha16(out / f) for f in ("train.csv", "eval.csv", "test.csv")}}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print("STRATIFIED OK ->", args.out)


if __name__ == "__main__":
    main()

