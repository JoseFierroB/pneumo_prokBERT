# Model registry — 5 aliases (short names, stable technical paths)

> Weights live in `finetuning_outputs/*/` (ignored by git). This file + `models.json`
> are the single source of truth: alias → init, data, script, artifact, hashes.
> Local `model.safetensors` files are copies of the WS artifacts; WS hashes re-verified in Phase 1.

## Naming (explicit)
- `mini-*` = init `prokbert-mini` (general base); `promoter-*` = init `mini-promoter` (upstream 223k).
- `223k` = HF `train_80` (178473) unless suffixed `full` (= full 223092 train, no holdout).
- **`mix` = single fit on MIXED data** (`train_80` + pneumo-train, combined val, locked tests).
  `mini-mix-1to2-cont` = continuation FROM the mix model (not a new mix).
- `cont` = pneumo-only continuation of a previous model (narrows generality; see supplementaries).
- `dedup` = `d39v_tigr4_dedup_3135`; `1to2` = `d39v_tigr4_high_1to2` (1:2 bootstrap of dedup negatives).
- `replica` = faithful paper-protocol replica (80/10/10 seed 42 on HF data).

| Alias | Init (HF id @ rev) | Training data | Script / command | Artifact (`best/`) | `model.safetensors` sha256 |
|---|---|---|---|---|---|
| `mini-223k-replica` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | HF train 80/10/10 seed 42 (178473/22309/22310) | legacy `/tmp/paper_faithful_2ep.py` (mirrored by `paper_replica.py`) | `finetuning_outputs/mini_223k_replica_2ep/best` | `ffb2fd34…` |
| `mini-d39v-tigr4` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | `datasets/d39v_tigr4_dedup_3135` (2194/470/471) | `finetuning.py --model-name neuralbioinfo/prokbert-mini --dataset-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/mini_d39v_tigr4_2ep --warmup-ratio 0.1` | `finetuning_outputs/mini_d39v_tigr4_2ep/best` | `1f9050d6…` |
| `mini-223k-d39v-tigr4` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | HF `train_80` (178473) + dedup train (2194); select on combined val; tests locked | `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/mini_223k_d39v_tigr4_2ep` | `finetuning_outputs/mini_223k_d39v_tigr4_2ep/best` | `cc3575b3…` |
| `promoter-d39v-tigr4` | `neuralbioinfo/prokbert-mini-promoter` @ `8502f02` | `datasets/d39v_tigr4_dedup_3135` (2194/470/471) | `finetuning.py --model-name neuralbioinfo/prokbert-mini-promoter --dataset-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/promoter_d39v_tigr4_2ep --cpu` | `finetuning_outputs/promoter_d39v_tigr4_2ep/best` | `4c7b25b6…` |
| `promoter-baseline` | `neuralbioinfo/prokbert-mini-promoter` @ `8502f02` | — (no training, zero-shot) | `scripts/eval_matrix.py` (same evaluator as all models) | HF hub (pinned rev) | n/a |

Legacy / appendix (not ranked in the matrix):
- Retired 2026-09-28 (checkpoints deleted local+WS): `base-d39v`, `base-223k-d39v`, `hfpromoter-d39v` (d39v-only variants, superseded by the dedup family); `d39v_989_base_true_2ep` (warmup-500 ablation, evidence for the warmup bug); `mix_223k_plus_d39v_2ep` (legacy full-train mix, WS `legacy/`).
- Repo-root pilot checkpoint (`model.safetensors` + `config.json`/`vocab.txt`, commit `7515b6a`, SigA+SigX 3ep CPU): retired from git — it was the contaminated init of the old `d39v_989_paper_2ep` run. Provenance lives in git history; inits now always come from pinned HF revisions above.
- Contaminated `d39v_989_paper_2ep` (pilot init): **retired/deleted**.

Architecture note: `promoter-*` use the pooler implementation (21K `models.py`, `bert.pooler` present);
`base-*`/`paper*` use masked-attention pooling (94K `models.py`, no pooler). Documented, not unified.

Paper hyperparams (all runs): AdamW lr 0.0004, betas 0.95/0.98, eps 5e-05, wd 0.01,
2 epochs, batch 128, seed 42, k6s1, 81 bp, best by MCC. Large runs: warmup 500, fp32.
Small runs (22–36 steps): `--warmup-ratio 0.1`, fp16 auto (as in `base-d39v`).
