#!/usr/bin/env python3
"""Local HF snapshot access + versioned split manifests (offline-first).

Snapshot: data/bacterial_promoters/train.parquet (sha256 pinned in SOURCE.md).
Splits: datasets/hf_splits/<spec>/ with ids_{train,eval,test}.csv (segment_id,y)
+ manifest.json (method, seed, per-split SHA16, source sha256).

Usage:
  from scripts.hf_local import resolve_split  # -> {"train","eval","test"} HF Datasets
  resolve_split("70/15/15")            # datasets/hf_splits/70/15/15
  resolve_split("/abs/path/to/split") # explicit dir
  split_frames("70/15/15")             # -> ({train,eval,test} DataFrames, spec)
"""
import hashlib
import json
from pathlib import Path

PARQUET = Path("data/bacterial_promoters/train.parquet")
SPLITS_ROOT = Path("datasets/hf_splits")

SOURCE_SHA256 = "3c537705abefb2ad878e89e5e052b503329ded9b31f3121fedca083ee1c8e0fb"


def sha16(ids):
    return hashlib.sha256(("\n".join(str(i) for i in ids) + "\n").encode()).hexdigest()[:16]


def check_snapshot():
    import hashlib as _h
    h = _h.sha256(PARQUET.read_bytes()).hexdigest()
    assert h == SOURCE_SHA256, f"snapshot changed: {h} != {SOURCE_SHA256} (see SOURCE.md)"
    return h


def load_snapshot(columns=None):
    """Full snapshot DataFrame indexed by segment_id (offline, SHA-verified)."""
    import pandas as pd
    check_snapshot()
    return pd.read_parquet(PARQUET, columns=columns)


def _manifest(sdir):
    man = json.load(open(Path(sdir) / "manifest.json"))
    assert man["source_sha256"] == SOURCE_SHA256, f"manifest built from other snapshot: {sdir}"
    return man


def _segments():
    snap = load_snapshot(columns=["segment_id", "segment"])
    return dict(zip(snap["segment_id"].astype(str), snap["segment"]))


def resolve_split(spec_or_dir):
    """Return {train,eval,test} as HF Datasets with segment_id,segment,y (segments joined)."""
    import pandas as pd
    from datasets import Dataset
    sdir = Path(spec_or_dir)
    sdir = sdir if sdir.is_dir() else (SPLITS_ROOT / str(spec_or_dir))
    man = _manifest(sdir)
    check_snapshot()
    ids = {}
    for k in ("train", "eval", "test"):
        f = sdir / f"ids_{k}.csv"
        ids[k] = pd.read_csv(f, dtype={"segment_id": str})
        assert sha16(ids[k]["segment_id"]) == man["splits"][k]["sha16"], f"SHA mismatch {f}"
    seg = _segments()
    out = {}
    for k in ("train", "eval", "test"):
        df = ids[k].copy()
        df["segment"] = df["segment_id"].map(seg)
        assert df["segment"].notna().all(), f"segments missing for {k}"
        out[k] = Dataset.from_pandas(df[["segment_id", "segment", "y"]], preserve_index=False)
    _a = set(out["train"]["segment"])
    _b = set(out["eval"]["segment"])
    _c = set(out["test"]["segment"])
    assert not (_a & _b) and not (_a & _c) and not (_b & _c), "split overlap by segment"
    print(f"HF local split OK ({sdir.name}): train {len(out['train'])} eval {len(out['eval'])} "
          f"test {len(out['test'])} (spec {man['spec']}, sha {man['splits']['test']['sha16']})")
    return out


def split_frames(split_dir):
    """{train,eval,test} as DataFrames(segment,y) from a manifest dir (for eval/gap scripts)."""
    import pandas as pd
    d = Path(split_dir)
    man = _manifest(d)
    check_snapshot()
    seg = _segments()
    out = {}
    for k in ("train", "eval", "test"):
        ids = pd.read_csv(d / f"ids_{k}.csv", dtype={"segment_id": str})
        assert sha16(ids["segment_id"]) == man["splits"][k]["sha16"], f"SHA mismatch {k}"
        df = pd.DataFrame({"segment": ids["segment_id"].map(seg), "y": ids["y"]})
        assert df["segment"].notna().all()
        out[k] = df
    return out, man["spec"]
