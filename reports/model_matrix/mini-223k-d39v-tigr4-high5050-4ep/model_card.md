# mini-223k-d39v-tigr4-high5050-4ep

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** hf_train_80 (178473) + high5050 train (2937); select on combined val; tests locked
- **Script:** `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_high5050 --output_dir finetuning_outputs/mini_223k_d39v_tigr4_high5050_4ep --num-train-epochs 4`
- **Artifact:** `finetuning_outputs/mini_223k_d39v_tigr4_high5050_4ep/best`
- **sha256:** `ed26c9f8649f3c7d3d1a993d29f5191ddc852a5c0af92fe62143692469c7cda8`
- **Note:** epoch ablation (E6); same protocol/data as high5050 2ep, 4 epochs

## run_info.json (summary)
- timestamp: 2026-10-12T13:57:21+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=500 warmup_ratio=None fp16=None

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.7965 | 0.5967 | 0.8807 | 0.8952 | 0.7968 | 0.5987 | 0.519 | clean |
| ecoli-sigma70 | 1864 | 0.8541 | 0.7078 | 0.9283 | 0.9268 | 0.8632 | 0.7248 | 0.589 | clean |
| multispecies | 22582 | 0.7945 | 0.5924 | 0.8778 | 0.8926 | 0.7952 | 0.5932 | 0.489 | clean |
| d39v-tigr4-dedup | 3135 | 0.8941 | 0.7875 | 0.9521 | 0.9629 | 0.9321 | 0.8638 | 0.865 | train_overlap |
