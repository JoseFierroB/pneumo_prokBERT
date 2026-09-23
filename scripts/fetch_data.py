#!/usr/bin/env python3
"""Single entry point for data readiness (works online and offline).

- Ensures data/bacterial_promoters/{train,test_sigma70,test_multispecies}.parquet
  (downloads from HF unless --no-download), verifying row counts vs datasets_eval.json.
- Validates each --data-dir (train.csv/eval.csv/test.csv with segment,y; no split overlap).
- Checks benchmark pneumo FASTAs under --canon_root.
- Prints a readiness report; exits nonzero on any failure. Idempotent.

Usage:
  pixi run python scripts/fetch_data.py
  pixi run python scripts/fetch_data.py --data-dir datasets/d39v_full_989 --data-dir datasets/d39v_tigr4_dedup_3135
  pixi run python scripts/fetch_data.py --no-download --canon_root /path/to/canonical_15_datasets
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")


def parse_args():
    p = argparse.ArgumentParser(description="Data readiness: HF parquet + --data-dir validation + benchmark FASTAs")
    p.add_argument("--data-dir", action="append", default=[],
                   help="Dataset dir to validate (repeatable): train.csv/eval.csv/test.csv with segment,y")
    p.add_argument("--canon_root", type=str,
                   default="../promoter-tools/data/benchmark/canonical_15_datasets",
                   help="Benchmark FASTA root (default: sibling promoter-tools repo, relative to CWD=repo root)")
    p.add_argument("--no-download", action="store_true",
                   help="never hit the network; fail if local parquet files are missing")
    p.add_argument("--bact_dir", type=str, default="data/bacterial_promoters")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.load(open("datasets_eval.json"))
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")
        ok = ok and cond

    # --- bacterial_promoters parquet ---
    import pandas as pd
    bdir = Path(args.bact_dir)
    for col, fname in [("paper-test", None), ("ecoli-sigma70", "test_sigma70.parquet"),
                       ("multispecies", "test_multispecies.parquet")]:
        if fname is None:
            continue
        p = bdir / fname
        if not p.exists() and not args.no_download:
            print(f"downloading {fname} from HF...")
            from datasets import load_dataset
            split = fname.replace(".parquet", "")
            load_dataset("neuralbioinfo/bacterial_promoters", split=split).to_pandas().to_parquet(p)
        if not p.exists():
            check(f"bacterial_promoters/{fname}", False, "(missing and --no-download)")
            continue
        n = len(pd.read_parquet(p))
        want = manifest["columns"][col]["n"]
        check(f"bacterial_promoters/{fname}", n == want, f"n={n} want={want}")
    tp = bdir / "train.parquet"
    if not tp.exists() and not args.no_download:
        print("downloading train.parquet from HF...")
        from datasets import load_dataset
        load_dataset("neuralbioinfo/bacterial_promoters", split="train").to_pandas().to_parquet(tp)
    if tp.exists():
        import pandas as pd
        check("bacterial_promoters/train.parquet", len(pd.read_parquet(tp)) == 223092,
              f"n={len(pd.read_parquet(tp))}")
    else:
        check("bacterial_promoters/train.parquet", False, "(missing and --no-download)")

    # --- --data-dir validation ---
    import pandas as pd
    for d in args.data_dir:
        dp = Path(d)
        name = dp.name
        files_ok = all((dp / f).exists() for f in ("train.csv", "eval.csv", "test.csv"))
        check(f"{name}: files", files_ok)
        if not files_ok:
            continue
        dfs = {k: pd.read_csv(dp / f"{k}.csv") for k in ("train", "eval", "test")}
        cols_ok = all({"segment", "y"} <= set(df.columns) for df in dfs.values())
        check(f"{name}: columns segment,y", cols_ok)
        sets = {k: set(df["segment"]) for k, df in dfs.items()}
        overlap = (sets["train"] & sets["eval"]) | (sets["train"] & sets["test"]) | (sets["eval"] & sets["test"])
        check(f"{name}: no split overlap", not overlap, f"overlap={len(overlap)}")
        check(f"{name}: counts", True,
              f"train={len(dfs['train'])} eval={len(dfs['eval'])} test={len(dfs['test'])}")

    # --- benchmark FASTAs ---
    from Bio import SeqIO
    for col, sub, pf, nf, want in [("d39v", "1_d39v", "positives_81bp.fasta", "negatives_81bp.fasta", (989, 1000)),
                                   ("tigr4-high", "2_tigr4_high", "positives_81bp.fasta", "negatives_81bp.fasta", (738, 738)),
                                   ("d39v-tigr4", "4_d39v_tigr4_high", "positives_81bp.fasta", "negatives_81bp.fasta", (1727, 1738))]:
        pp, np_ = Path(args.canon_root) / sub / pf, Path(args.canon_root) / sub / nf
        if not (pp.exists() and np_.exists()):
            check(f"benchmark {col}", False, f"(missing under {args.canon_root})")
            continue
        np_, nn = sum(1 for _ in SeqIO.parse(str(pp), "fasta")), sum(1 for _ in SeqIO.parse(str(np_), "fasta"))
        check(f"benchmark {col}", (np_, nn) == want, f"pos={np_} neg={nn}")

    print("READY" if ok else "NOT READY")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
