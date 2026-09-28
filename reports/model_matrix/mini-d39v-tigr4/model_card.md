# mini-d39v-tigr4

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** datasets/d39v_tigr4_dedup_3135 (2194/470/471)
- **Script:** `finetuning.py --model_name neuralbioinfo/prokbert-mini --dataset_dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/mini_d39v_tigr4_2ep --warmup_ratio 0.1`
- **Artifact:** `finetuning_outputs/mini_d39v_tigr4_2ep/best`
- **sha256:** `1f9050d665bf2d23c36e9895f28534f404956fba20241e8d78ddb9f22c97790a`

## run_info.json (summary)
- timestamp: 2026-09-22T15:37:51+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=0 warmup_ratio=0.1 fp16=True

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.5470 | 0.0976 | 0.5701 | 0.5650 | 0.5566 | 0.1135 | 0.135 | clean |
| ecoli-sigma70 | 1864 | 0.4533 | -0.0740 | 0.4577 | 0.4545 | 0.5445 | 0.0650 | 0.980 | clean |
| multispecies | 22582 | 0.5498 | 0.1017 | 0.5709 | 0.5649 | 0.5547 | 0.1095 | 0.170 | clean |
| d39v-tigr4-dedup | 3135 | 0.9282 | 0.8578 | 0.9582 | 0.9650 | 0.9368 | 0.8735 | 0.233 | train_overlap |
