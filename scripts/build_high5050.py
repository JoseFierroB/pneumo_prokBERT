#!/usr/bin/env python3
"""Build D39V+TIGR4-High 5050 set: 1399 positives + 2798 negatives (1:2),
negatives 50% CDS + 50% IGR, from sibling promoter-tools source.

Split 70/15/15 stratified by y (seed 42). No resampling: negatives already
fixed at 1:2 per design. Extra `src` column (pos/CDS/IGR) for audit;
loaders select segment,y (subset-safe).

Output: datasets/d39v_tigr4_high5050/{train,eval,test,all}.csv + MANIFEST.json

Usage:
  pixi run python scripts/build_high5050.py [--seed 42]
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "/home/fierro/Desktop/promoter-tools/data/benchmark/datasets/d39v_tigr4_high_dedup_1_to2_5050_cds_igrs"
DST = "datasets/d39v_tigr4_high5050"


def parse_args():
    p = argparse.ArgumentParser(description="Build High 5050 CDS/IGR set")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def readfa(p):
    ids, ss, cur, cid = [], [], [], None
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith(">"):
            if cur:
                ss.append("".join(cur).upper())
                ids.append(cid)
                cur = []
            cid = ln[1:]
        else:
            cur.append(ln)
    if cur:
        ss.append("".join(cur).upper())
        ids.append(cid)
    return pd.DataFrame({"id": ids, "segment": ss})


def sha16(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main():
    args = parse_args()
    pos = readfa(f"{SRC}/positives_81bp.fasta")
    pos["y"], pos["src"] = 1, "pos"
    neg = readfa(f"{SRC}/negatives_81bp.fasta")
    meta = pd.read_csv(f"{SRC}/negatives_81bp_metadata.tsv", sep="\t")
    srcmap = dict(zip(meta["Sequence_ID"], meta["Source"]))
    neg["y"] = 0
    neg["src"] = neg["id"].map(srcmap)
    assert neg["src"].isna().sum() == 0, "negatives without Source"
    assert len(pos) == 1399 and len(neg) == 2798, (len(pos), len(neg))
    assert (neg["src"] == "CDS").sum() == 1399 and (neg["src"] == "IGR").sum() == 1399
    assert all(len(s) == 81 for s in list(pos["segment"]) + list(neg["segment"]))
    assert len(set(pos["segment"]) & set(neg["segment"])) == 0
    # dedup vs 223k
    hf = set(pd.read_parquet("data/bacterial_promoters/train.parquet")["segment"].str.upper().str.replace("U", "T"))
    all4197 = pd.concat([pos, neg], ignore_index=True)
    assert len(set(all4197["segment"]) & hf) == 0, "overlap vs 223k"
    # 70/15/15 stratified
    tr, tmp = train_test_split(all4197, test_size=0.3, random_state=args.seed, stratify=all4197["y"])
    ev, te = train_test_split(tmp, test_size=0.5, random_state=args.seed, stratify=tmp["y"])
    frames = {"train": tr, "eval": ev, "test": te}
    for s, df in frames.items():
        print(f"{s}: n={len(df)} pos={(df.y == 1).sum()} neg={(df.y == 0).sum()} "
              f"CDS={((df.y == 0) & (df.src == 'CDS')).sum()} IGR={((df.y == 0) & (df.src == 'IGR')).sum()}")
    out = Path(DST)
    out.mkdir(parents=True, exist_ok=True)
    for s, df in frames.items():
        df[["segment", "y", "id", "src"]].to_csv(out / f"{s}.csv", index=False)
    all4197[["segment", "y", "id", "src"]].to_csv(out / "all.csv", index=False)
    sets = {s: set(frames[s]["segment"]) for s in frames}
    assert not ((sets["train"] & sets["eval"]) | (sets["train"] & sets["test"]) | (sets["eval"] & sets["test"])), "cross-split overlap"
    man = {"source": SRC, "seed": args.seed,
           "method": "70/15/15 stratified by y (no resampling; negatives fixed 1:2, 50% CDS + 50% IGR)",
           "counts": {s: {"n": len(df), "pos": int((df.y == 1).sum()), "neg": int((df.y == 0).sum()),
                          "neg_cds": int(((df.y == 0) & (df.src == "CDS")).sum()),
                          "neg_igr": int(((df.y == 0) & (df.src == "IGR")).sum())} for s, df in frames.items()},
           "overlap_223k": 0,
           "files_sha16": {f: sha16(out / f) for f in ("train.csv", "eval.csv", "test.csv", "all.csv")},
           "src_sha16": {f: sha16(f"{SRC}/{f}") for f in ("positives_81bp.fasta", "negatives_81bp.fasta",
                                                          "positives_81bp_metadata.tsv", "negatives_81bp_metadata.tsv")}}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print("MANIFEST OK")


if __name__ == "__main__":
    main()
