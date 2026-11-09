# scripts/ — what each script does

Run everything from the repo root (`pixi run python scripts/<name>.py ...`).
All GPU scripts auto-detect the device (`--gpu N` / `--cpu` via `common.py`).

## Training (the only two entry points)
- `train_mix223k.py` — mix HF-223k + pneumo. Flags: `--data-dir`, `--num-train-epochs`
  (float allowed), `--dropout`, `--hf-subset`, `--hf-split-spec` (`80/10/10` fixed/SHA,
  ratios, or `local:<spec>` offline), `--hf-fold-dir` (joint CV), `--full-train`,
  `--train-seed`, `--select-metric`, `--save-steps/--save-total-limit`,
  `--freeze-backbone` / `--probe-checkpoint` (control experiments), `--dry-run`.
- `train_replica.py` — paper-223k-only exact-replica harness (`--recipe code|paper`,
  `--full-train`, `--model-loader`, `--warmup-ratio`, `--metric-for-best`).

## Evaluation (same pipeline or nothing counts)
- `eval_single.py` — one model × matrix columns → metrics JSON + per-column probs CSVs
  + ROC/PR/confusion-matrix plots
  (+`--hf-split-dir` override for paper-test, `--leakage-base/--train-overlap` flags).
- `generalization_gap.py` — train-vs-test ROC/AUC gap (+`--hf-split-dir/--hf-which`).
- `metrics.py` — shared cells: `cell_metrics` (fixed thr + Youden), `trainer_compute_metrics`..

## Data builders (deterministic, SHA-manifested)
- `build_hf_splits.py` — versioned HF splits (`datasets/hf_splits/<spec>/`).
- `build_stratified_801010.py`, `build_stratified_5050.py`,
  `build_species_stratified.py` — current species/pneumo splits + manifests.
- `build_kfold.py` — K-fold split with real holdout per fold.
- `hf_local.py` — offline snapshot access + manifest verification (library, not CLI).
- `common.py` — `normalize()` (canonical segment norm) + `--gpu/--cpu` flags
  + device auto-detect (import, don't copy).

## Plots (CPU-only, read saved probs/CSVs, never retrain)
- `plot_overlays.py` — overlay plots used by `eval_matrix` (kept as its dependency).
- `reference_sets.py` — canonical FASTA loader shared by matrix + probes.

