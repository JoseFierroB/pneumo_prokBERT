# base-d39v

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** datasets/d39v_full_989 (1381/296/296)
- **Script:** `finetuning.py --model_name neuralbioinfo/prokbert-mini --warmup_ratio 0.1`
- **Artifact:** `finetuning_outputs/d39v_989_base_true_wr01_2ep/best`
- **sha256:** `8d233810b517a2fa99f610c073a8a80999cc235b67d0a35f1e8a98762448bb4e`

## run_info.json (summary)
- timestamp: 2026-09-21T07:42:17+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=0 warmup_ratio=0.1 fp16=True

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.5320 | 0.0768 | 0.5608 | 0.5611 | 0.5531 | 0.1064 | 0.046 | clean |
| ecoli-sigma70 | 1864 | 0.4866 | -0.0248 | 0.4904 | 0.4865 | 0.5536 | 0.0955 | 0.969 | clean |
| multispecies | 22582 | 0.5371 | 0.0847 | 0.5673 | 0.5669 | 0.5562 | 0.1127 | 0.049 | clean |
| d39v | 1989 | 0.9306 | 0.8663 | 0.9706 | 0.9749 | 0.9412 | 0.8839 | 0.071 | train_overlap |
| tigr4-high | 1476 | 0.8977 | 0.8071 | 0.9316 | 0.9542 | 0.9092 | 0.8238 | 0.081 | clean |
| d39v-tigr4 | 3465 | 0.9166 | 0.8408 | 0.9539 | 0.9661 | 0.9270 | 0.8570 | 0.079 | train_overlap |
| d39v-tigr4-dedup | 3135 | 0.9136 | 0.8307 | 0.9455 | 0.9542 | 0.9225 | 0.8450 | 0.079 | train_overlap |
