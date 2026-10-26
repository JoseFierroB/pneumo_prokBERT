#!/usr/bin/env python3
"""
Verdict for the exact-replica battery (paper-223k only).

Compares, on LOCKED sets (test_10 = paper-test, sigma70, multispecies):
  1. replica_code vs replica_paper (MCC@0.5 + AUC per column)
  2. winner vs published ProkBERT numbers (E.coli MCC 0.74, multi MCC 0.62 — reference)
  3. winner vs deployed mini-promoter baseline probs (if present, same rows)
  4. paired bootstrap B=2000 seed 42 for code-vs-paper per column (+ winner-vs-baseline)

Models load with the OFFICIAL local class (never AutoModel -> Ext).
Fires only when both replica best/ dirs exist; otherwise exits non-zero.

Usage:
  pixi run python scripts/verdict_replica.py --gpu 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from scripts.eval_matrix import build_columns, infer_model, pretokenize
from scripts.metrics import cell_metrics, load_probs_csv, save_probs_csv
from scripts.train_mix223k import EXPECTED

from prokbert_patches import apply_tokenizer_patches

apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.models.megatronbert import ProkBertForSequenceClassification

PUBLISHED = {"ecoli-sigma70": {"mcc": 0.74}, "multispecies": {"mcc": 0.62}}
COLS = ["paper-test", "ecoli-sigma70", "multispecies", "d39v-tigr4-dedup"]
# NOTE: best-by-val-AUC selection is OUR choice (standard early stopping);
# neither the paper nor upstream code specifies checkpoint selection.


def parse_args():
    p = argparse.ArgumentParser(description="Replica battery verdict (locked sets)")
    p.add_argument("--code-dir", default="finetuning_outputs/replica_code_2ep/best")
    p.add_argument("--paper-dir", default="finetuning_outputs/replica_paper_2ep/best")
    p.add_argument("--baseline-probs-dir", default="finetuning_outputs/eval_matrix_probs")
    p.add_argument("--baseline-alias", default="promoter-baseline")
    p.add_argument("--mix-aliases", nargs="*", default=["mini-223k-d39v-tigr4-high5050",
                                                        "mini-223k-d39v-tigr4-high5050-3ep"],
                   help="our mix models for paired comparison (same rows; dedup is "
                        "train_overlap for THEM, clean for replicas)")
    p.add_argument("--out", default="reports/supplementary_5050/replica_verdict.json")
    p.add_argument("--datasets", nargs="*", default=COLS)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    add_device_args(p)
    return p.parse_args()


def load_official(model_dir, device):
    assert Path(model_dir, "config.json").exists(), f"no config.json in {model_dir}"
    assert Path(model_dir, "model.safetensors").exists(), f"no model.safetensors in {model_dir}"
    m = ProkBertForSequenceClassification.from_pretrained(model_dir)
    assert m.__class__.__name__ == "ProkBertForSequenceClassification", m.__class__.__name__
    return m.to(device).eval()


def paired_delta(y, sa, sb, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    dm, da = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        dm.append(matthews_corrcoef(y[idx], (sa[idx] >= 0.5).astype(int))
                  - matthews_corrcoef(y[idx], (sb[idx] >= 0.5).astype(int)))
        da.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    out = {}
    for name, arr in (("mcc", np.array(dm)), ("auc", np.array(da))):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[name] = {"delta": float(arr.mean()), "ci95": [float(lo), float(hi)],
                     "sig95": bool(lo > 0)}
    return out


def main():
    args = parse_args()
    for d in (args.code_dir, args.paper_dir):
        if not Path(d, "model.safetensors").exists():
            print(f"WAIT: {d} not ready (replicas still training?)")
            raise SystemExit(2)
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device}")

    tok = ProkBERTTokenizer(tokenization_params={"kmer": 6, "shift": 1}, operation_space="sequence")
    data = build_columns(argparse.Namespace(reference_root="datasets/reference"), args.datasets)
    tok_data = pretokenize(tok, data)

    res, probs = {}, {}
    for alias, mdir in (("replica-code", args.code_dir), ("replica-paper", args.paper_dir)):
        model = load_official(mdir, device)
        out = infer_model(model, tok_data, args.datasets, args.batch_size, device)
        res[alias] = {}
        probs[alias] = {}
        for col in args.datasets:
            scores, y = out[col]
            m = cell_metrics(y, scores)
            m["leakage"] = "clean"  # replicas saw HF train_80 only; test_10/sigma70/multi unseen
            res[alias][col] = m
            probs[alias][col] = (y, scores)
            print(f"  {alias:14s} {col:14s} MCC {m['mcc_05']:.4f} AUC {m['auc']:.4f} [clean]", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n=== code vs paper (delta = code - paper) ===")
    boot = {}
    for col in args.datasets:
        y = probs["replica-code"][col][0]
        b = paired_delta(y, probs["replica-code"][col][1], probs["replica-paper"][col][1],
                         args.n_boot, args.seed)
        boot[f"code-vs-paper@{col}"] = b
        for k in ("mcc", "auc"):
            d = b[k]
            print(f"  {col:14s} d{k.upper()} {d['delta']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                  f"{'SIG' if d['sig95'] else 'n.s.'}")

    print("\n=== vs published (ProkBERT paper: E.coli MCC 0.74, multi MCC 0.62) ===")
    for col, ref in PUBLISHED.items():
        if col in args.datasets:
            for alias in ("replica-code", "replica-paper"):
                got = res[alias][col]["mcc_05"]
                print(f"  {alias:14s} {col:14s} MCC {got:.4f} vs published {ref['mcc']:.2f} "
                      f"(delta {got - ref['mcc']:+.4f})")

    print("\n=== vs our mix models (same rows, paired) ===")
    print("  leakage: dedup = train_overlap for MIX, clean for replicas;")
    print("  paper-test/sigma70/multi = clean for both.")
    for mix in args.mix_aliases:
        ok = all((Path(args.baseline_probs_dir) / f"{mix}__{col}.csv").exists()
                 for col in args.datasets)
        if not ok:
            print(f"  {mix}: probs missing — skipped")
            continue
        for col in args.datasets:
            ym, sm = load_probs_csv(Path(args.baseline_probs_dir) / f"{mix}__{col}.csv")
            y = probs["replica-code"][col][0]
            if not (ym == y).all():
                print(f"  {mix} {col}: row mismatch — skipped")
                continue
            mm = matthews_corrcoef(y, (sm >= 0.5).astype(int))
            for alias in ("replica-code", "replica-paper"):
                b = paired_delta(y, probs[alias][col][1], sm, args.n_boot, args.seed)
                boot[f"{alias}-vs-{mix}@{col}"] = b
                d = b["auc"]
                print(f"  {alias:14s} vs {mix:34s} {col:14s} mixMCC {mm:.4f} "
                      f"dAUC {d['delta']:+.4f} [{'SIG' if d['sig95'] else 'n.s.'}]")

    print("\n=== vs deployed mini-promoter (same rows) ===")
    base_ok = True
    for col in args.datasets:
        fp = Path(args.baseline_probs_dir) / f"{args.baseline_alias}__{col}.csv"
        if not fp.exists():
            print(f"  {col:14s} baseline probs missing ({fp}) — skipped")
            base_ok = False
            continue
        yb, sb = load_probs_csv(fp)
        y = probs["replica-code"][col][0]
        if not (yb == y).all():
            print(f"  {col:14s} row mismatch vs baseline — skipped")
            base_ok = False
            continue
        for alias in ("replica-code", "replica-paper"):
            b = paired_delta(y, probs[alias][col][1], sb, args.n_boot, args.seed)
            boot[f"{alias}-vs-baseline@{col}"] = b
            d = b["auc"]
            print(f"  {alias:14s} {col:14s} dAUC {d['delta']:+.4f} "
                  f"{'SIG' if d['sig95'] else 'n.s.'} (baseline MCC {matthews_corrcoef(y, (sb >= 0.5).astype(int)):.4f})")

    out = {"models": res, "bootstrap": boot,
           "baseline_compared": base_ok, "test_sha16": EXPECTED["test_sha"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    pd = Path(args.out).parent / "replica_verdict_probs"
    pd.mkdir(exist_ok=True)
    for alias in probs:
        for col in args.datasets:
            save_probs_csv(pd / f"{alias}__{col}.csv", *probs[alias][col])
    print(f"\nwrote {args.out} + probs in {pd}/")


if __name__ == "__main__":
    main()
