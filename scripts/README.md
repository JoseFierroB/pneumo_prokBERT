# scripts/ — what each script does

Run everything from the repo root (`pixi run python scripts/<name>.py ...`).
All GPU scripts auto-detect the device (`--gpu N` / `--cpu` via `common.py`).

## Training (the only two entry points)
- `train_mix223k.py` — mix HF-223k + pneumo. Flags: `--data-dir`, `--num-train-epochs`
  (float allowed), `--dropout`, `--hf-subset`, `--hf-split-spec` (`80/10/10` fixed/SHA,
  ratios, or `local:<spec>` offline), `--hf-fold-dir` (joint CV), `--full-train`,
  `--train-seed`, `--es-patience` + `--select-metric`, `--freeze-backbone` /
  `--probe-checkpoint` (control experiments), `--dry-run`.
- `train_replica.py` — paper-223k-only exact-replica harness (`--recipe code|paper`,
  `--full-train`, `--model-loader`, `--warmup-ratio`, `--metric-for-best`).

## Evaluation (same pipeline or nothing counts)
- `eval_single.py` — one model × matrix columns → metrics JSON + per-column probs CSVs
  (+`--hf-split-dir` override for paper-test, `--leakage-base/--train-overlap` flags).
- `eval_csv.py` — one model × one CSV → probs CSV (ad-hoc splits).
- `eval_matrix.py` — canonical 5×4 matrix (builders + batched inference).
- `generalization_gap.py` — train-vs-test ROC/AUC gap (+`--hf-split-dir/--hf-which`).
- `metrics.py` — shared cells: `cell_metrics` (fixed thr + Youden), `trainer_compute_metrics`.
- `paired_bootstrap.py` — paired B=2000 CIs for head-to-head claims.
- `verdict_replica.py` — locked code-vs-paper verdict + baseline comparison.

## Data builders (deterministic, SHA-manifested)
- `build_hf_splits.py` — versioned HF splits (`datasets/hf_splits/<spec>/`).
- `build_high5050.py`, `build_unified_1to2.py`, `build_kfold.py`, `build_source_ablation.py`.
- `hf_local.py` — offline snapshot access + manifest verification (library, not CLI).
- `fetch_data.py` — data readiness check (online/offline).
- `common.py` — `--gpu/--cpu` flags + device auto-detect (import, don't copy).

## Plots (CPU-only, read saved probs/CSVs, never retrain)
- `plot_fig6_epochs.py`, `plot_benchmark_compare.py`, `plot_sigma_roc_4197.py`,
  `replot_sigma_4197.py`, `plot_cv_folds.py`, `parse_curves_1ep.py`,
  `plot_gap_curve.py`, `plot_frontier.py`, `plot_overlays.py`, `plot_training.py`,
  `plot_defense.py`, `plot_screening.py` — each writes into its `reports/` folder.
- `reference_sets.py` — canonical FASTA loader shared by matrix + probes.

## Retired 2026-10-26 (archived out of repo)
`plot_sigma.py`, `audit_sigma.py`, `eval_sigma.py` — superseded by
`plot_sigma_roc_4197.py` + `eval_single --datasets`. Recover from
`~/pneumo_prokBERT_archive/2026-10-26/scripts/`.

Verification ladder (no test suite, by design): `py_compile` → `--help` →
`--dry-run` → subset run vs saved metrics → spot-check re-inference to `/tmp` and diff.
