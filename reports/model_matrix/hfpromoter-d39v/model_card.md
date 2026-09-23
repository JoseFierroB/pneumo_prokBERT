# hfpromoter-d39v

- **Init:** `neuralbioinfo/prokbert-mini-promoter` @ `8502f02d1019`
- **Status:** exists
- **Train data:** datasets/d39v_full_989 (1381/296/296)
- **Script:** `legacy:/tmp/train_from_promoter.py --dataset d39v (warmup 500, fp32)`
- **Artifact:** `finetuning_outputs/d39v_from_promoter_2ep/best`
- **sha256:** `9507ff764fef971804aff51cbfbb48e4f55dcacafb9efdfbe1286398c93f38f8`
- **Note:** pooler architecture (21K models.py); effective LR capped by warmup on 22 steps

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.9269 | 0.8553 | 0.9796 | 0.9822 | 0.9285 | 0.8576 | 0.444 | train_overlap |
| ecoli-sigma70 | 1864 | 0.8578 | 0.7153 | 0.9378 | 0.9340 | 0.8680 | 0.7352 | 0.780 | clean |
| multispecies | 22582 | 0.8030 | 0.6077 | 0.8830 | 0.8901 | 0.8039 | 0.6109 | 0.545 | clean |
| d39v | 1989 | 0.9326 | 0.8656 | 0.9759 | 0.9820 | 0.9553 | 0.9115 | 0.955 | train_overlap |
| tigr4-high | 1476 | 0.8686 | 0.7371 | 0.9239 | 0.9458 | 0.9045 | 0.8157 | 0.968 | clean |
| d39v-tigr4 | 3465 | 0.9053 | 0.8108 | 0.9540 | 0.9669 | 0.9333 | 0.8697 | 0.966 | train_overlap |
| d39v-tigr4-dedup | 3135 | 0.8982 | 0.7949 | 0.9465 | 0.9558 | 0.9292 | 0.8587 | 0.966 | train_overlap |
