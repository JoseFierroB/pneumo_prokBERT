# Modular fine-tuning pipeline: using the code and configuring hyperparameters (smoke-validated)

Modules under `tunable/` re-implement the `scripts/train_mix223k.py` stages as
importable modules (config/data/model/train/evaluate/cli). New code only;
existing scripts are imported, never modified. Respects repo conventions:
`pixi run`, argparse, `datasets/<name>` + MANIFEST, `finetuning_outputs/<run>`
+ `run_info.json`, `reports/<topic>`, `models.json` registry, `prokbert/`
untouched.

## Acceptance gate (passed 2026-11-04)

    pixi run python tunable/smoke_replica.py --output-dir finetuning_outputs/tunable_smoke

Preset `replica-1ep` on the new data regime (species801010 + pneumo 80/10/10)
reproduces `ofat_lr4e4` bit-identically:
data_test MCC 0.6483169198887292 / AUC 0.9124489795918367, gap AUC +0.0096.
Artifact: `finetuning_outputs/tunable_smoke/best` (see `models.json`
`tunable-smoke-replica-1ep`).

## Layout

- `config.py`: paper-replica defaults + argparse groups (data/model/optim/run);
  `--preset replica|replica-1ep|ofat_center`, every flag overrides the preset.
- `data.py`: single split API (`species801010` via `hf_local` + pneumo
  `datasets/<name>`) with SHA/overlap asserts + `ensure_segments_csv()`.
- `model.py` / `train.py` / `evaluate.py`: builders, orchestration, combined
  loss overlays + `generalization_gap` wrapper.
- `cli.py`: `pixi run python -m tunable.cli ...` (see root README for commands).

## Command reference

    # replica-1ep single run
    pixi run python -m tunable.cli --preset replica-1ep \
        --data-dir datasets/high5050_stratified_801010 \
        --hf-split-spec local:species801010 --seed 42 \
        --output-dir finetuning_outputs/my_run

    # with gap vs test-set + loss overlay
    pixi run python -m tunable.cli --preset replica-1ep \
        --data-dir datasets/high5050_stratified_801010 \
        --hf-split-spec local:species801010 --output-dir finetuning_outputs/my_run \
        --gap --gap-test-csv datasets/hf_splits/species801010/test_segments.csv \
        --gap-test-name test-set --gap-out-dir reports/gaps/tunable

    # acceptance smoke (asserts bit-identical metrics, exits nonzero on mismatch)
    pixi run python tunable/smoke_replica.py --output-dir finetuning_outputs/tunable_smoke
