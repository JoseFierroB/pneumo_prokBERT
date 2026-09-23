# base-223k-d39v

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** hf_train_80 (178473) + d39v train (1381); select on combined val; tests locked
- **Script:** `scripts/train_mix223k.py --data-dir datasets/d39v_full_989 --output_dir finetuning_outputs/base_223k_d39v_2ep`
- **Artifact:** `finetuning_outputs/base_223k_d39v_2ep/best`
- **sha256:** `c5670a4e690f8236fd6a7419abffea42cb629425643363e94ae79663f4afd3ac`
- **Legacy superseded:** `finetuning_outputs/mix_223k_plus_d39v_2ep/best` (sha `2f80d52f0b29…`), recipe: full hf train + d39v-all, 90/10 internal val

## run_info.json (summary)
- timestamp: 2026-09-22T15:41:09+00:00
- commit: feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc
- warmup_steps=500 warmup_ratio=None fp16=None

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.8080 | 0.6211 | 0.8917 | 0.9056 | 0.8085 | 0.6215 | 0.490 | clean |
| ecoli-sigma70 | 1864 | 0.8686 | 0.7355 | 0.9371 | 0.9339 | 0.8723 | 0.7431 | 0.526 | clean |
| multispecies | 22582 | 0.8082 | 0.6213 | 0.8905 | 0.9031 | 0.8098 | 0.6229 | 0.472 | clean |
| d39v | 1989 | 0.9442 | 0.8885 | 0.9817 | 0.9867 | 0.9593 | 0.9195 | 0.834 | train_overlap |
| tigr4-high | 1476 | 0.8753 | 0.7507 | 0.9276 | 0.9507 | 0.8997 | 0.8037 | 0.793 | clean |
| d39v-tigr4 | 3465 | 0.9149 | 0.8297 | 0.9596 | 0.9721 | 0.9333 | 0.8688 | 0.817 | train_overlap |
| d39v-tigr4-dedup | 3135 | 0.9085 | 0.8150 | 0.9529 | 0.9626 | 0.9289 | 0.8573 | 0.817 | train_overlap |
