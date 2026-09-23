# Model registry — 7 aliases (short names, stable technical paths)

> Weights live in `finetuning_outputs/*/` (ignored by git). This file + `models.json`
> are the single source of truth: alias → init, data, script, artifact, hashes.
> Local `model.safetensors` files are copies of the WS artifacts; WS hashes re-verified in Phase 1.

| Alias | Init (HF id @ rev) | Training data | Script / command | Artifact (`best/`) | `model.safetensors` sha256 |
|---|---|---|---|---|---|
| `paper223k-replica` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | HF train 80/10/10 seed 42 (178473/22309/22310) | legacy `/tmp/paper_faithful_2ep.py` (mirrored by `paper_replica.py`) | `finetuning_outputs/paper_80_10_10_2ep/best` | `ffb2fd34…` |
| `base-d39v` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | `datasets/d39v_full_989` (1381/296/296) | `finetuning.py --model_name neuralbioinfo/prokbert-mini --warmup_ratio 0.1 --output_dir finetuning_outputs/d39v_989_base_true_wr01_2ep` | `finetuning_outputs/d39v_989_base_true_wr01_2ep/best` | `8d233810…` |
| `base-d39v-tigr4` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | `datasets/d39v_tigr4_dedup_3135` (2194/470/471) | `finetuning.py` same as `base-d39v`, `--dataset_dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/base_d39v_tigr4_2ep` | `finetuning_outputs/base_d39v_tigr4_2ep/best` | `1f9050d6…` |
| `base-223k-d39v` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | HF `train_80` (178473) + D39V train (1381); select on combined val; tests locked | `scripts/train_mix223k.py --data-dir datasets/d39v_full_989 --output_dir finetuning_outputs/base_223k_d39v_2ep` | `finetuning_outputs/base_223k_d39v_2ep/best` | `c5670a4e…` |
| `base-223k-d39v-tigr4` | `neuralbioinfo/prokbert-mini` @ `feb2520a` | HF `train_80` (178473) + dedup train (2194); select on combined val; tests locked | `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/base_223k_d39v_tigr4_2ep` | `finetuning_outputs/base_223k_d39v_tigr4_2ep/best` | `cc3575b3…` |
| `hfpromoter-d39v` | `neuralbioinfo/prokbert-mini-promoter` @ `8502f02` | `datasets/d39v_full_989` (1381/296/296) | legacy `/tmp/train_from_promoter.py --dataset d39v` (warmup 500, fp32) | `finetuning_outputs/d39v_from_promoter_2ep/best` | `9507ff76…` |
| `hfpromoter-baseline` | `neuralbioinfo/prokbert-mini-promoter` @ `8502f02` | — (no training, zero-shot) | `scripts/eval_matrix.py` (same evaluator as all models) | HF hub (pinned rev) | n/a |

Legacy / appendix (not ranked in the matrix):
- `mix_223k_plus_d39v_2ep` (sha `2f80d52f…`): full HF train + D39V-all, 90/10 internal val → moved to `legacy/` (WS), superseded by `base-223k-d39v`.
- Repo-root pilot checkpoint (`model.safetensors` + `config.json`/`vocab.txt`, commit `7515b6a`, SigA+SigX 3ep CPU): retired from git — it was the contaminated init of the old `d39v_989_paper_2ep` run. Provenance lives in git history; inits now always come from pinned HF revisions above.
- `d39v_989_base_true_2ep` (sha `4ae67723…`): warmup-500 ablation, evidence for the warmup bug.
- Contaminated `d39v_989_paper_2ep` (pilot init): **retired/deleted**.

Architecture note: `hfpromoter-*` use the pooler implementation (21K `models.py`, `bert.pooler` present);
`base-*`/`paper*` use masked-attention pooling (94K `models.py`, no pooler). Documented, not unified.

Paper hyperparams (all runs): AdamW lr 0.0004, betas 0.95/0.98, eps 5e-05, wd 0.01,
2 epochs, batch 128, seed 42, k6s1, 81 bp, best by MCC. Large runs: warmup 500, fp32.
Small runs (22–36 steps): `--warmup_ratio 0.1`, fp16 auto (as in `base-d39v`).
