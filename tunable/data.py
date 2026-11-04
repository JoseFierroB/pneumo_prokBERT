#!/usr/bin/env python3
"""Single data-loading API: HF species split + pneumo stratified dir.

Wraps scripts.train_mix223k loaders and scripts.hf_local manifests so every
consumer gets identical, SHA-verified splits. Also builds *-segments.csv
join files (segment,y) needed by gap evals.
"""
import hashlib
from pathlib import Path

import sys
sys.path.insert(0, ".")


def load_all(args):
    """Mirror the data stage of scripts/train_mix223k.main (parallel + full-train).

    Returns dict(train_80, val_10, test_10, ds, pneu_raw, pneu, train_ds,
    val_ds, locked_sets, manifest, data_name, extra).
    """
    import hashlib as _h
    from pathlib import Path as _P
    from scripts import train_mix223k as T

    data_name = _P(args.data_dir).name
    if args.hf_split_spec.startswith("local:"):
        from scripts.hf_local import resolve_split
        from datasets import load_dataset
        spec = args.hf_split_spec.split("local:", 1)[1]
        ds = load_dataset(T.HF_DATASET)
        local = resolve_split(spec)
        train_80, val_10, test_10 = local["train"], local["eval"], local["test"]
    else:
        ds, train_80, val_10, test_10 = T.load_hf_split(args.seed, args.hf_split_spec)
    hf_subset_info = None
    joint_fold = None
    referee_10 = test_10
    if args.hf_subset:
        assert args.hf_split_spec in ("80/10/10", "local:80/10/10"), \
            "--hf-subset needs the SHA-asserted 80/10/10 base (fixed or local:)"
    if args.hf_fold_dir:
        assert not args.full_train and not args.hf_subset, "--hf-fold-dir is parallel-mode, no subset"
        joint_fold = args.hf_fold_dir
        train_80, val_10, test_10 = T.load_hf_fold(args.hf_fold_dir)
    if args.hf_subset:
        assert not args.full_train, "--hf-subset is parallel-mode only"
        train_80, hf_subset_info = T.apply_hf_subset(train_80, args.hf_subset, args.seed)
    pneu_raw, pneu = T.load_data_splits(args, data_name)
    extra = {}
    if args.full_train:
        train_ds, val_ds, n_extra, n_excluded = T.compose_full(args, ds, pneu)
        manifest = {
            "mode": "full-train", "hf_dataset": T.HF_DATASET, "hf_full_train": 223092,
            "data_dir": args.data_dir, "data_name": data_name,
            "data_splits": {k: {"n": len(pneu_raw[k]), "sha256": _h.sha256(
                open(f"{args.data_dir}/{k}.csv", "rb").read()).hexdigest()} for k in ("train", "eval", "test")},
            "dedup_dir": args.dedup_dir, "dedup_unseen_extra": n_extra, "dedup_excluded_dup": n_excluded,
            "select": "sigma70 MCC (external, clean)",
            "combined": {"train": len(train_ds), "val": len(val_ds)},
        }
        locked_sets = [("val_sigma70", val_ds), ("data_test", pneu["test"]),
                       ("multispecies", ds["test_multispecies"])]
    else:
        train_ds, val_ds = T.compose_parallel(args, train_80, val_10, test_10, pneu)
        manifest = T.build_manifest(args, data_name, train_80, val_10, test_10, pneu_raw, train_ds, val_ds,
                                  hf_subset_info)
        locked_sets = [("val_combined", val_ds), ("test_10", test_10), ("data_test", pneu["test"]),
                       ("sigma70", ds["test_sigma70"]), ("multispecies", ds["test_multispecies"])]
        if joint_fold:
            manifest["hf_fold_dir"] = joint_fold
            locked_sets = [("val_combined", val_ds), ("fold_test", test_10), ("data_test", pneu["test"]),
                           ("test_10", referee_10),
                           ("sigma70", ds["test_sigma70"]), ("multispecies", ds["test_multispecies"])]
    return {"train_80": train_80, "val_10": val_10, "test_10": test_10, "ds": ds,
            "pneu_raw": pneu_raw, "pneu": pneu, "train_ds": train_ds, "val_ds": val_ds,
            "locked_sets": locked_sets, "manifest": manifest, "data_name": data_name, "extra": extra}


def ensure_segments_csv(split_dir, which="test", out_name=None):
    """Build <split_dir>/{which}_segments.csv (segment,y) from ids + snapshot.

    SHA-verified via the split manifest (same assertions as split_frames).
    Returns the CSV path.
    """
    import pandas as pd
    from scripts.hf_local import split_frames
    frames, spec = split_frames(split_dir)
    if which not in frames:
        raise SystemExit(f"split part {which} not in {split_dir} (have {sorted(frames)})")
    out = str(Path(split_dir) / (out_name or f"{which}_segments.csv"))
    frames[which][["segment", "y"]].to_csv(out, index=False)
    print(f"segments csv: {out} rows={len(frames[which])} spec={spec}")
    return out


def sha16_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]
