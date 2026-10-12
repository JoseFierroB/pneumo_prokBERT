# mini-223k-d39v-tigr4-high5050-3ep

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** hf_train_80 (178473) + high5050 train (2937); select on combined val; tests locked
- **Script:** `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_high5050 --output_dir finetuning_outputs/mini_223k_d39v_tigr4_high5050_3ep --num-train-epochs 3`
- **Artifact:** `finetuning_outputs/mini_223k_d39v_tigr4_high5050_3ep/best`
- **sha256:** `a0a257c4e067ee9698770d730d367da33d2015a76674e0e2c41a3b6c23e28d59`
- **Note:** epoch ablation (E6); same protocol/data as high5050 2ep, 3 epochs

## run_info.json (summary)
- timestamp: 2026-10-12T13:00:28+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=500 warmup_ratio=None fp16=None

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.8212 | 0.6431 | 0.9043 | 0.9169 | 0.8234 | 0.6492 | 0.563 | clean |
| ecoli-sigma70 | 1864 | 0.8745 | 0.7489 | 0.9458 | 0.9457 | 0.8830 | 0.7651 | 0.716 | clean |
| multispecies | 22582 | 0.8209 | 0.6424 | 0.9018 | 0.9135 | 0.8233 | 0.6482 | 0.540 | clean |
| d39v-tigr4-dedup | 3135 | 0.9311 | 0.8605 | 0.9684 | 0.9742 | 0.9445 | 0.8898 | 0.889 | train_overlap |
