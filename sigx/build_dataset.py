#!/usr/bin/env python3
"""Build local binary SigA-vs-SigX set from D39V tags (no WS needed).

Positives y=1: SigX (21). Negatives y=0: SigA + SigA-LOWCONF (390).
Splits 70/15/15 stratified, seed 42 -> train 287 (272/15) / eval 62 (59/3) / test 62 (59/3).
Mirrors the legacy WS 411 recipe; TIGR4's 6 SigX reserved for later (not included).

Output: datasets/d39v_binary_siga_sigx/{train,eval,test,all}.csv + MANIFEST.json

Usage:
  pixi run python scripts/build_sigx_binary.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DST = "datasets/d39v_binary_siga_sigx"
SIGA = {"SigA", "SigA-LOWCONF"}


def main():
    alldf = pd.read_csv("datasets/d39v_full_989/all.csv")
    alldf["tag"] = alldf["id"].str.split("_").str[-1]
    pos = alldf[alldf["tag"] == "SigX"].copy()
    neg = alldf[alldf["tag"].isin(SIGA)].copy()
    assert len(pos) == 21 and len(neg) == 390, f"unexpected counts {len(pos)}/{len(neg)}"
    df = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    df = df[["segment", "y", "id", "tag", "sigma", "strain", "split"]]
    tr, tmp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["y"])
    ev, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp["y"])
    out = Path(DST)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", tr), ("eval", ev), ("test", te), ("all", df)]:
        d.to_csv(out / f"{name}.csv", index=False)
    manifest = {"source": "datasets/d39v_full_989/all.csv tags (SigX=21 vs SigA+SigA-LOWCONF=390)",
                "seed": 42, "method": "70/15/15 stratified; tag column kept for audit",
                "counts": {s: {"n": len(d), "pos": int((d.y == 1).sum())} for s, d in
                           [("train", tr), ("eval", ev), ("test", te), ("all", df)]},
                "files_sha16": {f: hashlib.sha256(open(out / f, "rb").read()).hexdigest()[:16]
                                for f in ("train.csv", "eval.csv", "test.csv", "all.csv")}}
    json.dump(manifest, open(out / "MANIFEST.json", "w"), indent=2)
    print(f"OK {DST}: " + ", ".join(f"{s}={len(d)}({int((d.y==1).sum())}+)" for s, d in
                                    [("train", tr), ("eval", ev), ("test", te)]))


if __name__ == "__main__":
    main()
