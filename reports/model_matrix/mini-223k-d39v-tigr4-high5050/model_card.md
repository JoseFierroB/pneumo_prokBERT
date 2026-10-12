# mini-223k-d39v-tigr4-high5050

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** hf_train_80 (178473) + high5050 train (2937); select on combined val; tests locked
- **Script:** `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_high5050 --output_dir finetuning_outputs/mini_223k_d39v_tigr4_high5050_2ep`
- **Artifact:** `finetuning_outputs/mini_223k_d39v_tigr4_high5050_2ep/best`
- **sha256:** `3afcd3cdc90e6cb909eb23d84a42057caa4858f02320f22a93724625a6607292`

## run_info.json (summary)
- timestamp: 2026-10-12T11:08:00+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=500 warmup_ratio=None fp16=None

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.8072 | 0.6146 | 0.8919 | 0.9059 | 0.8087 | 0.6199 | 0.564 | clean |
| ecoli-sigma70 | 1864 | 0.8766 | 0.7517 | 0.9392 | 0.9387 | 0.8782 | 0.7549 | 0.506 | clean |
| multispecies | 22582 | 0.8068 | 0.6138 | 0.8905 | 0.9040 | 0.8084 | 0.6174 | 0.521 | clean |
| d39v-tigr4-dedup | 3135 | 0.9088 | 0.8161 | 0.9584 | 0.9672 | 0.9372 | 0.8746 | 0.833 | train_overlap |
