# mini-223k-d39v-tigr4

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** hf_train_80 (178473) + dedup train (2194); select on combined val; tests locked
- **Script:** `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/mini_223k_d39v_tigr4_2ep`
- **Artifact:** `finetuning_outputs/mini_223k_d39v_tigr4_2ep/best`
- **sha256:** `cc3575b3858862ed3cdf6645d4a838c374d8fc7aff20a2c795de8f2dddeaa30a`

## run_info.json (summary)
- timestamp: 2026-09-22T15:49:25+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=500 warmup_ratio=None fp16=None

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.7959 | 0.5922 | 0.8813 | 0.8966 | 0.7981 | 0.6016 | 0.610 | clean |
| ecoli-sigma70 | 1864 | 0.8686 | 0.7364 | 0.9326 | 0.9282 | 0.8702 | 0.7393 | 0.524 | clean |
| multispecies | 22582 | 0.7909 | 0.5820 | 0.8782 | 0.8930 | 0.7954 | 0.5935 | 0.570 | clean |
| d39v-tigr4-dedup | 3135 | 0.8957 | 0.7924 | 0.9580 | 0.9661 | 0.9330 | 0.8679 | 0.956 | train_overlap |
