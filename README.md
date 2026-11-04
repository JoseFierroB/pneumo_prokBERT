# pneumo_prokBERT

Fine-tuning of ProkBERT (`neuralbioinfo/prokbert-mini`) for bacterial promoter
classification, focused on *Streptococcus pneumoniae* (D39V, TIGR4). 81 bp windows
centered on the TSS `[-60, +20]`; LCA tokenization `k=6, s=1`.

## Status

- **Primary model:** HF-223k + pneumo high5050 mix, **3ep** (paper 0.643/0.904, multi 0.642/0.902). Details and CIs in the local lab log `reports/supplementary_5050/battery.md` (not versioned).
- **Gap-AUC floor:** 1ep (+0.025 [.021,.030]); see `reports/supplementary_5050/1ep_dossier.md`.
- Model registry: `MODELS.md` (+ `reports/supplementary_5050/ws_models_manifest.json` for the 92 WS checkpoints).

## Quickstart

```bash
pixi install
pixi run python scripts/fetch_data.py --data-dir datasets/d39v_tigr4_high5050  # check data
# Mix training 1ep (5090: export CUDA_VISIBLE_DEVICES=0)
pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_high5050 --num-train-epochs 2 --gpu 0
# Exact paper-223k replica (no pneumo)
pixi run python scripts/train_replica.py --recipe code --gpu 0
# Evaluate a checkpoint (same pipeline, always)
pixi run python scripts/eval_single.py --model <ckpt>/best --alias <name> --out-dir reports/supplementary_5050/grid_<exp> --gpu 0
```

Every train writes `run_info.json` + `best/` into its own folder. Weights live in
`finetuning_outputs/` (git-ignored); the WS holds the 92 newer checkpoints.

## Data

See `datasets/MANIFEST.md`. Summary: `d39v_tigr4_high5050/` (primary pneumo data, 4197),
`d39v_tigr4_high_1to2/`, `d39v_tigr4_dedup_3135/`, `pt_benchmark/` (clean balanced),
`pneumo_5fold/` + `hf_splits/<80|70|60>` folds (offline, SHA-pinned). Local HF snapshot in
`data/bacterial_promoters/` (`SOURCE.md`); `test_sigma70`/`multispecies` load from the Hub.

## Map

```
scripts/            train_mix223k.py (mix) · train_replica.py (replica) · eval_*.py ·
                    generalization_gap.py · paired_bootstrap.py · builders · plots (CPU)
                    see scripts/README.md
datasets/           MANIFEST.md — what is canonical and where it comes from
reports/supplementary_5050/   1ep_dossier.md · gaps.json · gap_cis.json · benchmark/ · sigma/ · cv/ · curves_1ep/ · grid_*/ (local lab log battery.md not versioned)
prokbert/           vendored upstream, untouched (patches only in prokbert_patches.py)
MODELS.md           model registry (alias → init, data, command, hash)
EVALUATION.md       canonical 8-model × 4-dataset matrix (frozen Oct-12)
```

Root-level `finetuning.py` / `paper_replica.py` are legacy (pre-`scripts/`); don't use for new runs.
`energy/` is a separate track (fusion), not part of the main protocol.

## License and citation

## Using the code and configuring hyperparameters

The training pipeline is factored into importable modules under `tunable/`
(config/data/model/train/evaluate/cli). New code only; existing `scripts/*`
are imported, never modified. Defaults follow the paper replica (Frontiers
2.3.2); every hyperparameter is an argparse flag (groups: data/model/optim/run),
and `--preset replica|replica-1ep|ofat_center` starts from a known point that
any flag can override.

    # 1. replica smoke on the new data regime (acceptance gate)
    pixi run python tunable/smoke_replica.py --output-dir finetuning_outputs/tunable_smoke

    # 2. single run via the modular CLI (any flag overrides the preset)
    pixi run python -m tunable.cli --preset replica-1ep \
        --data-dir datasets/high5050_stratified_801010 \
        --hf-split-spec local:species801010 --seed 42 \
        --output-dir finetuning_outputs/my_run

    # 3. same + train-vs-test-set gap and loss overlay
    pixi run python -m tunable.cli --preset replica-1ep \
        --data-dir datasets/high5050_stratified_801010 \
        --hf-split-spec local:species801010 --output-dir finetuning_outputs/my_run \
        --gap --gap-test-csv datasets/hf_splits/species801010/test_segments.csv \
        --gap-test-name test-set --gap-out-dir reports/gaps/tunable

    # 4. hyperparameter sweep example (dropout variants, 1ep, new regime)
    for DO in 0.0 0.05 0.1 0.15 0.2 0.3; do
      pixi run python -m tunable.cli --preset replica-1ep \
        --data-dir datasets/high5050_stratified_801010 \
        --hf-split-spec local:species801010 --dropout $DO \
        --output-dir finetuning_outputs/tunable_do$DO
    done

    # 5. per-layer dropout + freeze/head variants use the same flags as scripts/train_mix223k.py
    #    (--dropout-head/--dropout-hidden/--dropout-attn, --freeze-backbone, ...)

Data builders (run from repo root):

    # pneumo stratified 80/10/10 (ratio 1:2, seed 42)
    pixi run python scripts/build_stratified_801010.py --out datasets/high5050_stratified_801010

    # test-segments join for gap evals (SHA-verified against the split manifest)
    pixi run python -c "from tunable.data import ensure_segments_csv; ensure_segments_csv('datasets/hf_splits/species801010', 'test')"

Eval and plots:

    pixi run python scripts/plot_training.py --runs my_run --out-dir reports/tunable/loss_curves
    pixi run python scripts/generalization_gap.py --model finetuning_outputs/my_run/best ...

Docs: reports/tunable/README.md (acceptance gate record), models.json
(tunable-smoke-replica-1ep entry).

## License and citation

Own code and docs: MIT (see `LICENSE`). `prokbert/` is upstream under its terms.

```bibtex
@article{ProkBERT2024,
  author  = {Ligeti et al.},
  journal = {Frontiers in Microbiology},
  title   = {ProkBERT family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14}
}
```
