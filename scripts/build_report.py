#!/usr/bin/env python3
"""Regenerate report documents from reports/model_matrix/metrics.json.

Default (no selectors) reproduces EVALUATION.md/json for the full matrix.
With --models/--datasets, renders only the requested slice (EVALUATION.* then hold
a subset; the full matrix always remains in reports/model_matrix/metrics.json —
re-run without selectors to restore the full documents).

Usage:
  pixi run python scripts/build_report.py
  pixi run python scripts/build_report.py --models base-223k-d39v hfpromoter-baseline --datasets ecoli-sigma70 tigr4-high
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")

COLS = ["paper-test", "ecoli-sigma70", "multispecies", "d39v",
        "tigr4-high", "d39v-tigr4", "d39v-tigr4-dedup"]

NOTES = """## Notes
- HF baseline is zero-shot `neuralbioinfo/prokbert-mini-promoter` (no fine-tuning here); it was upstream-fine-tuned on the full 223k train, so its `paper-test`/`paper-val` cells are `train_overlap`, not clean.
- `paper223k-replica` reproduces the upstream 223k setup (80/10/10 seed 42, paper hyperparams, fp32); matches the released HF model within ±0.006.
- `base-223k-d39v` / `base-223k-d39v-tigr4` use the parallel protocol (fit HF train_80 + pneumo train, combined val, locked tests); supersede the legacy full-train mix (see `legacy/`).
- `base-d39v` / `base-d39v-tigr4` train from the true base with `--warmup_ratio 0.1` (paper warmup 500 caps LR on 22–36 steps).
- `hfpromoter-d39v` uses the pooler architecture inherited from HF mini-promoter; other models use masked-attention pooling (documented, not unified).
- Youden = max(TPR-FPR) on the evaluated set (optimistic, for benchmark comparison).
- Per-model cards: `reports/model_matrix/<alias>/model_card.md`; ROC/PR overlays: `reports/model_matrix/plots/roc_pr__<dataset>.png`; registry: `MODELS.md`, `models.json`; datasets: `datasets_eval.json`."""


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate EVALUATION.md/json + model cards from metrics.json")
    p.add_argument("--models", nargs="*", default=None, help="subset of aliases (default: all, registry order)")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of columns (default: all, manifest order)")
    p.add_argument("--reports_dir", type=str, default="reports/model_matrix")
    return p.parse_args()


def main():
    args = parse_args()
    models = json.load(open("models.json"))
    metrics = json.load(open(f"{args.reports_dir}/metrics.json"))
    order = [k for k in models if k in metrics]
    aliases = args.models or order
    columns = args.datasets or COLS
    unknown_m = [a for a in aliases if a not in metrics]
    unknown_d = [c for c in columns if c not in COLS]
    if unknown_m or unknown_d:
        raise SystemExit(f"unknown models={unknown_m} datasets={unknown_d}")
    subset = not (set(aliases) == set(order) and list(columns) == COLS)

    lines = [f"# Evaluation Report — {len(aliases)} Models × {len(columns)} Datasets (5090, k6s1, @0.5 / Youden)", "",
             "Matrix from `reports/model_matrix/metrics.json` (evaluator `scripts/eval_matrix.py`, one tokenizer call per dataset).",
             "Leakage flags: `clean` = clean of fine-tuning labels in this project; `train_overlap` / `selection_overlap` otherwise.",
             "`paper-val` is the selection set (appendix): used for best-checkpoint selection, never for ranking.", "",
             "| Model | Dataset | N | ACC@0.5 | MCC@0.5 | AUC | AP | ACC@Youden | MCC@Youden | Thr | Leakage |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for alias in aliases:
        for c in columns:
            m = metrics[alias][c]
            lines.append(
                f"| {alias} | {c} | {m['n']} | {m['acc_05']:.4f} | {m['mcc_05']:.4f} | "
                f"{m['auc']:.4f} | {m['ap']:.4f} | {m['acc_y']:.4f} | {m['mcc_y']:.4f} | "
                f"{m['youden_thr']:.3f} | {m['leakage']} |")
    cis_path = os.path.join(args.reports_dir, "bootstrap_cis.json")
    if os.path.exists(cis_path):
        cis = json.load(open(cis_path))
        lines += ["", f"## Significance (paired bootstrap, B=2000, threshold 0.5)",
                  "| Dataset | Comparison (a − b) | Metric | Δ | 95% CI | p | 95% sig |",
                  "|---|---|---|---|---|---|---|"]
        for row in cis["comparisons"]:
            for metric, v in row["metrics"].items():
                lo, hi = v["ci95"]
                lines.append(f"| {row['dataset']} | {row['a']} − {row['b']} | {metric} | "
                             f"{v['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {v['p']:.4f} | "
                             f"{'yes' if v['sig95'] else 'no'} |")
    lines += ["", NOTES, "", "## Model Paths"]
    for alias in aliases:
        entry = models[alias]
        if alias == "hfpromoter-baseline":
            lines.append(f"- `{alias}`: `{entry['init']['hf_id']}` (pinned rev `{entry['init']['rev'][:7]}`)")
        else:
            lines.append(f"- `{alias}`: `{entry['artifact']}`")
    open("EVALUATION.md", "w").write("\n".join(lines) + "\n")
    json.dump({a: {c: metrics[a][c] for c in columns} for a in aliases}, open("EVALUATION.json", "w"), indent=2)

    for alias in aliases:
        entry = models[alias]
        d = os.path.join(args.reports_dir, alias)
        os.makedirs(d, exist_ok=True)
        cl = [f"# {alias}", "",
              f"- **Init:** `{entry['init']['hf_id']}` @ `{entry['init']['rev'][:12]}`",
              f"- **Status:** {entry['status']}",
              f"- **Train data:** {entry['train_data']}",
              f"- **Script:** `{entry['script']}`",
              f"- **Artifact:** `{entry['artifact']}`",
              f"- **sha256:** `{entry.get('sha256') or 'pending'}`"]
        if entry.get("note"):
            cl.append(f"- **Note:** {entry['note']}")
        if entry.get("legacy"):
            cl.append(f"- **Legacy superseded:** `{entry['legacy']['artifact']}` (sha `{entry['legacy']['sha256'][:12]}…`), "
                      f"recipe: {entry['legacy']['recipe']}")
        ri_path = entry.get("run_info")
        if ri_path and os.path.exists(ri_path):
            ri = json.load(open(ri_path))
            cl += ["", "## run_info.json (summary)",
                   f"- timestamp: {ri.get('timestamp_utc')}", f"- commit: {ri.get('model_commit_hash')}",
                   f"- warmup_steps={ri.get('warmup_steps')} warmup_ratio={ri.get('warmup_ratio')} fp16={ri.get('fp16')}"]
        cl += ["", "## Results matrix (@0.5 / Youden)",
               "| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for c in columns:
            m = metrics[alias][c]
            cl.append(f"| {c} | {m['n']} | {m['acc_05']:.4f} | {m['mcc_05']:.4f} | {m['auc']:.4f} | "
                      f"{m['ap']:.4f} | {m['acc_y']:.4f} | {m['mcc_y']:.4f} | {m['youden_thr']:.3f} | {m['leakage']} |")
        open(os.path.join(d, "model_card.md"), "w").write("\n".join(cl) + "\n")

    # summary_metrics.csv (selected cells)
    import csv
    with open(os.path.join(args.reports_dir, "summary_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "n", "n_pos", "n_neg", "acc_05", "balacc_05", "mcc_05", "f1_05",
                    "auc", "ap", "tpr_05", "tnr_05", "ppv_05", "youden_thr", "acc_y", "mcc_y", "leakage",
                    "TN", "FP", "FN", "TP"])
        for alias in aliases:
            for c in columns:
                m = metrics[alias][c]
                tn, fp, fn, tp = m["cm_05"][0][0], m["cm_05"][0][1], m["cm_05"][1][0], m["cm_05"][1][1]
                w.writerow([alias, c, m["n"], m["n_pos"], m["n_neg"], f"{m['acc_05']:.4f}", f"{m['balacc_05']:.4f}",
                            f"{m['mcc_05']:.4f}", f"{m['f1_05']:.4f}", f"{m['auc']:.4f}", f"{m['ap']:.4f}",
                            f"{m['tpr_05']:.4f}", f"{m['tnr_05']:.4f}", f"{m['ppv_05']:.4f}",
                            f"{m['youden_thr']:.4f}", f"{m['acc_y']:.4f}", f"{m['mcc_y']:.4f}", m["leakage"],
                            tn, fp, fn, tp])
    if subset:
        print(f"SUBSET written ({len(aliases)} models x {len(columns)} datasets) — full matrix stays in {args.reports_dir}/metrics.json; re-run without selectors to restore the full documents.")
    else:
        print("documents regenerated (full selection).")


if __name__ == "__main__":
    main()
