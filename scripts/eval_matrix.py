#!/usr/bin/env python3
"""
Evaluation matrix: 5 model aliases x 4 dataset columns.

Reads models.json + datasets_eval.json. For each cell computes N, balance,
CM@0.5, ACC/balACC/MCC/F1/ROC-AUC/PR-AUC, TPR/TNR/PPV/NPV/FPR, pred-pos-rate,
Youden threshold + metrics, and the leakage flag from the manifest.

One tokenizer call per dataset (pre-tokenized, no per-batch Pool storms).
Per-cell (y_true, y_score) saved as CSV to --probs-dir (ignored); tracked
outputs go to --reports-dir: metrics.json, summary_metrics.csv, ROC/PR plots.

Run from the repo root, e.g. on the WS 5090:
  pixi run python scripts/eval_matrix.py --reference-root datasets/reference

Usage:
  pixi run python scripts/eval_matrix.py --help                      # all flags
  pixi run python scripts/eval_matrix.py --models mini-d39v-tigr4          # one model, all columns
  pixi run python scripts/eval_matrix.py --models mini-d39v-tigr4 --datasets d39v tigr4-high --cpu
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from scripts.metrics import cell_metrics
from scripts.metrics import save_probs_csv
from scripts.reference_sets import CANONICAL_SETS, load_reference_sets
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

HF_DATASET = "neuralbioinfo/bacterial_promoters"
EXPECTED_TEST_SHA16 = "49a61e636397556b"


def parse_args():
    p = argparse.ArgumentParser(description="5x4 evaluation matrix")
    p.add_argument("--models", nargs="*", default=None, help="subset of aliases (default: all in models.json)")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of evaluable columns (default: all except appendix)")
    p.add_argument("--reference-root", type=str, default="datasets/reference",
                   help="Benchmark FASTA root with d39v/, tigr4-high/, d39v-tigr4/ subdirs (in-repo default)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--reports-dir", type=str, default="reports/model_matrix")
    p.add_argument("--probs-dir", type=str, default="finetuning_outputs/eval_matrix_probs")
    add_device_args(p)
    return p.parse_args()


def sha_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def build_columns(args, columns):
    """Build {col: (segments, labels)} once. Canonical FASTA sets come from
    scripts.reference_sets.load_reference_sets (shared with energy/ probes)."""
    if "paper-val" in columns:
        raise SystemExit("paper-val is selection-only (appendix): never evaluated, use val_10 from training logs")
    import pandas as pd
    data = {}
    if any(c.startswith("paper-") for c in columns):
        ds = load_dataset(HF_DATASET)
        tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
        test_10 = tmp["test"].train_test_split(test_size=0.5, seed=42)["test"]
        assert sha_ids(test_10["segment_id"])[:16] == EXPECTED_TEST_SHA16, "in-dist test IDs differ from manifest"
        data["paper-test"] = ([s.upper().replace("U", "T") for s in test_10["segment"]],
                              np.array(test_10["y"]))
        print("paper-test OK", len(data["paper-test"][0]))
    if "ecoli-sigma70" in columns or "multispecies" in columns:
        ds = load_dataset(HF_DATASET)
        for col, split in [("ecoli-sigma70", "test_sigma70"), ("multispecies", "test_multispecies")]:
            if col in columns:
                d = ds[split]
                data[col] = ([s.upper().replace("U", "T") for s in d["segment"]], np.array(d["y"]))
                print(col, "OK", len(data[col][0]))
    data.update(load_reference_sets(args.reference_root,
                                        [c for c in CANONICAL_SETS if c in columns], verbose=True))
    if "d39v-tigr4-dedup" in columns:
        df = pd.read_csv("datasets/d39v_tigr4_dedup_3135/all.csv")
        data["d39v-tigr4-dedup"] = (df["segment"].str.upper().tolist(), df["y"].to_numpy())
        print("d39v-tigr4-dedup OK", len(df))
    return data


def pretokenize(tokenizer, data):
    """One tokenizer call per dataset (avoids per-batch Pool storms)."""
    tok_data = {}
    for col, (seqs, y) in data.items():
        enc = tokenizer(seqs, add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
        tok_data[col] = (enc, y)
        print(f"tokenized {col}: {enc['input_ids'].shape}")
    return tok_data


def infer_model(model, tok_data, columns, batch_size, device):
    """Batched forward over pre-tokenized data -> {col: scores}."""
    out = {}
    with torch.no_grad():
        for col in columns:
            enc, y = tok_data[col]
            scores = []
            for i in range(0, len(y), batch_size):
                b = {k: v[i:i + batch_size].to(device) for k, v in enc.items()}
                logits = model(**b).logits
                scores.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy().tolist())
            out[col] = (np.array(scores, dtype=np.float64), y)
    return out


def resolve_plan(args, models, manifest):
    """Resolve (aliases, columns) from flags + registries; appendix columns never rank."""
    aliases = args.models or [k for k in models if models[k]["status"] == "exists"]
    skipped = [k for k in (args.models or []) if models.get(k, {}).get("status") != "exists"]
    if skipped:
        print(f"skip (not trained yet): {skipped}")
    aliases = [k for k in aliases if models.get(k, {}).get("status") == "exists"]
    if not aliases:
        raise SystemExit("no evaluable models (not in models.json with status exists; "
                         "use scripts/eval_single.py for supplementary models)")
    eval_only = [c for c in manifest["columns"] if manifest["columns"][c].get("kind") != "selection diagnostic (appendix, not ranked)"]
    columns = args.datasets or eval_only
    unknown = [c for c in columns if c not in manifest["columns"]]
    if unknown:
        raise SystemExit(f"not evaluable columns: {unknown}")
    appendix = [c for c in columns if manifest["columns"][c].get("kind") == "selection diagnostic (appendix, not ranked)"]
    if appendix:
        print(f"appendix (diagnostic, not ranked): {appendix}")
    print(f"aliases: {aliases}\ncolumns: {columns}")
    return aliases, columns


def evaluate_alias(alias, entry, tok_data, columns, manifest, probs_dir, batch_size, device):
    """Run one model over all columns: per-cell metrics + leakage flag + CSV probs."""
    src = entry["init"]["hf_id"] if alias == "promoter-baseline" else entry["artifact"]
    print(f"\n=== {alias} <- {src} ===")
    model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True).to(device).eval()
    cells = {}
    probed = infer_model(model, tok_data, columns, batch_size, device)
    for col in columns:
        scores, y = probed[col]
        m = cell_metrics(y, scores)
        m["leakage"] = manifest["leakage"][alias][col]
        cells[col] = m
        save_probs_csv(probs_dir / f"{alias}__{col}.csv", y, scores.astype(np.float32))
        print(f"  {col:22s} ACC {m['acc_05']:.4f} MCC {m['mcc_05']:.4f} AUC {m['auc']:.4f} AP {m['ap']:.4f} [{m['leakage']}]")
    del model
    torch.cuda.empty_cache()
    return cells


def write_reports(rep, probs_dir, aliases, columns, results):
    """Write metrics.json + summary_metrics.csv + ROC/PR overlays."""
    rep.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(rep / "metrics.json", "w"), indent=2)
    with open(rep / "summary_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "n", "n_pos", "n_neg", "acc_05", "balacc_05", "mcc_05", "f1_05",
                    "auc", "ap", "tpr_05", "tnr_05", "ppv_05", "youden_thr", "acc_y", "mcc_y", "leakage",
                    "TN", "FP", "FN", "TP"])
        for alias in aliases:
            for col in columns:
                m = results[alias][col]
                tn, fp, fn, tp = m["cm_05"][0][0], m["cm_05"][0][1], m["cm_05"][1][0], m["cm_05"][1][1]
                w.writerow([alias, col, m["n"], m["n_pos"], m["n_neg"], f"{m['acc_05']:.4f}", f"{m['balacc_05']:.4f}",
                            f"{m['mcc_05']:.4f}", f"{m['f1_05']:.4f}", f"{m['auc']:.4f}", f"{m['ap']:.4f}",
                            f"{m['tpr_05']:.4f}", f"{m['tnr_05']:.4f}", f"{m['ppv_05']:.4f}",
                            f"{m['youden_thr']:.4f}", f"{m['acc_y']:.4f}", f"{m['mcc_y']:.4f}", m["leakage"],
                            tn, fp, fn, tp])
    print(f"\nsaved {rep/'metrics.json'} + summary_metrics.csv + {probs_dir}/*.csv")

    # --- ROC and PR overlays: one metric per file, under plots/ ---
    try:
        from scripts.plot_overlays import plot_overlays
        plot_overlays(probs_dir, rep / "plots", columns, aliases, results)
    except ImportError:
        print("matplotlib not available — plots skipped")


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} (auto-detect; override with --gpu N or --cpu)")
    models = json.load(open("models.json"))
    manifest = json.load(open("datasets_eval.json"))
    aliases, columns = resolve_plan(args, models, manifest)

    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    data = build_columns(args, columns)
    tok_data = pretokenize(tokenizer, data)

    probs_dir = Path(args.probs_dir)
    probs_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for alias in aliases:
        results[alias] = evaluate_alias(alias, models[alias], tok_data, columns, manifest,
                                        probs_dir, args.batch_size, device)

    write_reports(Path(args.reports_dir), probs_dir, aliases, columns, results)


if __name__ == "__main__":
    main()
