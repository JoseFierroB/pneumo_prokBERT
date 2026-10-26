#!/usr/bin/env python3
"""Build versioned HF splits from the local snapshot (offline, deterministic).

- 80/10/10: reproduces the canonical fixed split (same two-step code + seed 42 as
  train_mix/paper_replica) via datasets load; asserts the EXPECTED SHAs. Identical
  to history by construction.
- 70/15/15, 60/20/20: single stratified split (sklearn, seed 42) on snapshot keys
  y|species-group (positives, rares<25 -> OTHER) / y|prom_class-subtype (negatives).

Output per spec: datasets/hf_splits/<spec>/ids_{train,eval,test}.csv (segment_id,y)
+ manifest.json (method, seed, counts, y-rates, SHA16 of ids, source sha256).

Usage:
  pixi run python scripts/build_hf_splits.py
  pixi run python scripts/build_hf_splits.py --spec 70/15/15
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")
from scripts.hf_local import SOURCE_SHA256, check_snapshot, sha16

OUT_ROOT = Path("datasets/hf_splits")
EXPECTED = {"val_sha": "716c3c3320d22c23", "test_sha": "49a61e636397556b"}


def parse_ratio(spec):
    parts = [float(x) for x in spec.split("/")]
    assert len(parts) == 3 and abs(sum(parts) - 100) < 1e-9, f"bad ratio {spec}"
    return parts[0] / 100, parts[1] / 100, parts[2] / 100


def write_spec(name, tr_idx, ev_idx, te_idx, df, method, extra=None):
    dd = OUT_ROOT / name
    dd.mkdir(parents=True, exist_ok=True)
    ids = {}
    for k, idx in (("train", sorted(tr_idx)), ("eval", sorted(ev_idx)), ("test", sorted(te_idx))):
        sub = df.iloc[idx][["segment_id", "y"]]
        sub.to_csv(dd / f"ids_{k}.csv", index=False)
        ids[k] = sub
    _a = set(ids["train"]["segment_id"]); _b = set(ids["eval"]["segment_id"]); _c = set(ids["test"]["segment_id"])
    assert not (_a & _b) and not (_a & _c) and not (_b & _c), "id overlap"
    man = {
        "spec": name, "seed": 42, "method": method,
        "source": "data/bacterial_promoters/train.parquet", "source_sha256": check_snapshot(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": {k: {"n": len(v), "pos_rate": round(float(v["y"].mean()), 4),
                       "sha16": sha16(v["segment_id"])} for k, v in ids.items()},
    }
    if extra:
        man.update(extra)
    json.dump(man, open(dd / "manifest.json", "w"), indent=1)
    print(f"{name}: " + " ".join(f"{k}={len(v)}" for k, v in ids.items()) +
          " sha " + "/".join(man["splits"][k]["sha16"][:8] for k in ("train", "eval", "test")))
    return man


def build_fixed_80():
    """Canonical split: identical code path as train_mix (HF order, seed 42)."""
    from datasets import load_dataset
    ds = load_dataset("neuralbioinfo/bacterial_promoters")
    assert len(ds["train"]) == 223092
    tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
    tr, temp = tmp["train"], tmp["test"]
    vt = temp.train_test_split(test_size=0.5, seed=42)
    va, te = vt["train"], vt["test"]
    assert sha16(va["segment_id"]) == EXPECTED["val_sha"], "val SHA drift"
    assert sha16(te["segment_id"]) == EXPECTED["test_sha"], "test SHA drift"
    df = pd.DataFrame({"segment_id": [str(i) for i in tr["segment_id"]]})
    full = pd.concat([
        pd.DataFrame({"segment_id": [str(i) for i in tr["segment_id"]], "y": tr["y"]}),
        pd.DataFrame({"segment_id": [str(i) for i in va["segment_id"]], "y": va["y"]}),
        pd.DataFrame({"segment_id": [str(i) for i in te["segment_id"]], "y": te["y"]}),
    ], ignore_index=True)
    # positional blocks (train block, eval block, test block as built)
    n_tr, n_va = len(tr), len(va)
    write_spec("80/10/10", range(n_tr), range(n_tr, n_tr + n_va),
               range(n_tr + n_va, len(full)), full,
               "canonical fixed split (HF order, seed 42, SHA-asserted)",
               {"stratified": False, "canonical": True})
    return full


def build_strat(spec):
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet("data/bacterial_promoters/train.parquet",
                         columns=["segment_id", "y", "ppd_original_SpeciesName", "prom_class"])
    assert len(df) == 223092
    df["segment_id"] = df["segment_id"].astype(str)
    df["sp"] = df["ppd_original_SpeciesName"].fillna("UNKNOWN")
    counts = df["sp"].value_counts()
    rare = set(counts[counts < 25].index)
    df["spg"] = df["sp"].where(~df["sp"].isin(rare), "OTHER")
    df["key2"] = df["y"].astype(str) + "|" + np.where(df["y"] == 1, df["spg"], df["prom_class"])
    kc = df["key2"].value_counts()
    assert kc.min() >= 5, f"strat key too small: {kc.tail(3).to_dict()}"
    ftr, fva, _ = parse_ratio(spec)
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=1 - ftr, random_state=42, stratify=df["key2"])
    i_rep, i_sel = train_test_split(i_tmp, test_size=fva / (1 - ftr), random_state=42,
                                    stratify=df.iloc[i_tmp]["key2"])
    write_spec(spec, i_tr, i_sel, i_rep, df.reset_index(drop=True),
               "single stratified split (sklearn, seed 42)",
               {"stratified": True, "key": "y|species-group(pos, rares<25->OTHER)|prom_class(neg)",
                "n_keys": int(df['key2'].nunique()), "min_key_n": int(kc.min())})


def parse_args():
    p = argparse.ArgumentParser(description="Build versioned HF splits offline")
    p.add_argument("--spec", default="80/10/10,70/15/15,60/20/20",
                   help="comma-separated specs to (re)build")
    return p.parse_args()


def main():
    args = parse_args()
    for spec in [s.strip() for s in args.spec.split(",") if s.strip()]:
        if spec == "80/10/10":
            build_fixed_80()
        else:
            build_strat(spec)


if __name__ == "__main__":
    main()
