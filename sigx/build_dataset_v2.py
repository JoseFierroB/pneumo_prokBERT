#!/usr/bin/env python3
"""Build SigX v2: D39V SigX (21) + TIGR4 SigX (6) positives; SigA-family negatives
subsampled to 1:2 (54) — no duplicates. Splits 70/15/15 seed 42 stratified by y*strain.

Output: datasets/d39v_tigr4_sigx_1to2/{train,eval,test,all}.csv + MANIFEST.json
(includes overlap audit vs 1to2-train/dedup-train/sigx-v1 splits for honest flags).

Usage:
  pixi run python sigx/build_dataset_v2.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DST = "datasets/d39v_tigr4_sigx_1to2"
SEED = 42


def main():
    # positives: D39V SigX from d39v_full_989 tags + TIGR4 SigX from v2 metadata
    d39v = pd.read_csv("datasets/d39v_full_989/all.csv")
    dtag = d39v["id"].str.split("_").str[-1]
    dpos = d39v[dtag == "SigX"].copy()
    dpos["strain"] = "D39V"
    meta = pd.read_csv("datasets/external/positives_high_sigma_assigned_metadata.tsv", sep="\t")
    from Bio import SeqIO
    mseq = {str(r.id): str(r.seq).upper()
            for r in SeqIO.parse("datasets/reference/tigr4-high/positives_81bp.fasta", "fasta")}
    # SOURCE label is "ComX" (frozen vendor TSV); our label is SigX (same factor).
    tids = [str(i) for i in meta[meta["Sigma_Factor"] == "ComX"]["Sequence_ID"]]
    tpos = pd.DataFrame({"segment": [mseq[i] for i in tids], "id": tids,
                         "sigma": "SigX", "strain": "TIGR4"})
    pos = pd.concat([dpos[["segment", "id", "sigma", "strain"]], tpos], ignore_index=True)
    pos["tag"] = pos["id"].str.split("_").str[-1]
    assert len(pos) == 27, f"unexpected positives {len(pos)}"
    # negatives: SigA-family both strains, subsampled to 1:2 (54), no replacement
    dneg = d39v[dtag.isin(["SigA", "SigA-LOWCONF"])].copy()
    dneg["strain"] = "D39V"
    tsa = [str(i) for i in meta[meta["Sigma_Factor"] == "SigA"]["Sequence_ID"]]
    tneg = pd.DataFrame({"segment": [mseq[i] for i in tsa], "id": tsa,
                         "sigma": "SigA", "strain": "TIGR4"})
    negpool = pd.concat([dneg[["segment", "id", "sigma", "strain"]], tneg], ignore_index=True)
    neg = negpool.sample(n=2 * len(pos), random_state=SEED)
    df = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    df["split"] = ""
    # stratified splits by y*strain
    tr, tmp = train_test_split(df, test_size=0.3, random_state=SEED,
                               stratify=df["y"].astype(str) + df["strain"])
    ev, te = train_test_split(tmp, test_size=0.5, random_state=SEED,
                              stratify=tmp["y"].astype(str) + tmp["strain"])
    out = Path(DST)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", tr), ("eval", ev), ("test", te), ("all", df)]:
        d.to_csv(out / f"{name}.csv", index=False)
    # overlap audit vs other trains (honest flags)
    aud = {}
    for other, fp in [("1to2-train", "datasets/d39v_tigr4_high_1to2/train.csv"),
                      ("dedup-train", "datasets/d39v_tigr4_dedup_3135/train.csv"),
                      ("sigxv1-train", "datasets/d39v_binary_siga_sigx/train.csv")]:
        o = {s.upper() for s in pd.read_csv(fp)["segment"]}
        aud[other] = {s: int(sum(1 for x in (te if s == "test" else ev if s == "eval" else tr)["segment"] if x.upper() in o))
                      for s in ("train", "eval", "test")}
    manifest = {"source": "d39v_full_989 SigX (21) + TIGR4 SigX v2 (6); negatives subsampled 1:2 (54, no replacement)",
                "seed": SEED, "method": "70/15/15 stratified by y*strain; tag column kept",
                "counts": {s: {"n": len(d), "pos": int((d.y == 1).sum())} for s, d in
                           [("train", tr), ("eval", ev), ("test", te), ("all", df)]},
                "overlap_audit": aud,
                "files_sha16": {f: hashlib.sha256(open(out / f, "rb").read()).hexdigest()[:16]
                                for f in ("train.csv", "eval.csv", "test.csv", "all.csv")}}
    json.dump(manifest, open(out / "MANIFEST.json", "w"), indent=2)
    print(f"OK {DST}: " + ", ".join(f"{s}={len(d)}({int((d.y==1).sum())}+)" for s, d in
                                    [("train", tr), ("eval", ev), ("test", te)]))
    print("overlap audit:", json.dumps(aud))


if __name__ == "__main__":
    main()
