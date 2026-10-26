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

## Honest-eval rules

- Same pipeline or it doesn't count (tokenizer and normalization fixed per script).
- Select on validation, report on locked test, once.
- Comparisons with paired bootstrap B=2000; method claims require multi-seed
  (seed lottery is ±0.02 MCC).
- Primary gap: gap-AUC (threshold-free); MCC for deployment and paper comparability.

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
