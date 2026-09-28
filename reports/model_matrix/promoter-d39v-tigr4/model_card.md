# promoter-d39v-tigr4

- **Init:** `neuralbioinfo/prokbert-mini-promoter` @ `8502f02d1019`
- **Status:** exists
- **Train data:** datasets/d39v_tigr4_dedup_3135 (2194/470/471)
- **Script:** `finetuning.py --model-name neuralbioinfo/prokbert-mini-promoter --dataset-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/promoter_d39v_tigr4_2ep --cpu`
- **Artifact:** `finetuning_outputs/promoter_d39v_tigr4_2ep/best`
- **sha256:** `4c7b25b6b8143c645cff25ec723b22478970a90636d7e1143eaa00459fad69dd`
- **Note:** pooler architecture (21K models.py); trained on CPU, warmup 500

## run_info.json (summary)
- timestamp: 2026-09-28T07:14:22+00:00
- commit: 8502f02d1019017fb44244274664e17e0e30ce1e
- warmup_steps=500 warmup_ratio=0.0 fp16=False

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.8889 | 0.7928 | 0.9769 | 0.9798 | 0.9238 | 0.8476 | 0.206 | train_overlap |
| ecoli-sigma70 | 1864 | 0.8589 | 0.7217 | 0.9357 | 0.9335 | 0.8632 | 0.7253 | 0.170 | clean |
| multispecies | 22582 | 0.7910 | 0.6022 | 0.8807 | 0.8905 | 0.8012 | 0.6054 | 0.267 | clean |
| d39v-tigr4-dedup | 3135 | 0.9301 | 0.8595 | 0.9539 | 0.9628 | 0.9337 | 0.8676 | 0.664 | train_overlap |
