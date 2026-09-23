# Evaluation Report — 7 Models × 7 Datasets (5090, k6s1, @0.5 / Youden)

Matrix from `reports/model_matrix/metrics.json` (evaluator `scripts/eval_matrix.py`, one tokenizer call per dataset).
Leakage flags: `clean` = clean of fine-tuning labels in this project; `train_overlap` / `selection_overlap` otherwise.
`paper-val` is the selection set (appendix): used for best-checkpoint selection, never for ranking.

| Model | Dataset | N | ACC@0.5 | MCC@0.5 | AUC | AP | ACC@Youden | MCC@Youden | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|---|
| paper223k-replica | paper-test | 22310 | 0.7961 | 0.5941 | 0.8800 | 0.8951 | 0.7968 | 0.5959 | 0.510 | clean |
| paper223k-replica | ecoli-sigma70 | 1864 | 0.8492 | 0.7052 | 0.9318 | 0.9266 | 0.8600 | 0.7192 | 0.658 | clean |
| paper223k-replica | multispecies | 22582 | 0.7934 | 0.5882 | 0.8788 | 0.8942 | 0.7939 | 0.5906 | 0.529 | clean |
| paper223k-replica | d39v | 1989 | 0.9080 | 0.8192 | 0.9761 | 0.9832 | 0.9563 | 0.9137 | 0.937 | clean |
| paper223k-replica | tigr4-high | 1476 | 0.8604 | 0.7216 | 0.9289 | 0.9533 | 0.9085 | 0.8291 | 0.977 | clean |
| paper223k-replica | d39v-tigr4 | 3465 | 0.8877 | 0.7774 | 0.9563 | 0.9708 | 0.9356 | 0.8749 | 0.935 | clean |
| paper223k-replica | d39v-tigr4-dedup | 3135 | 0.8785 | 0.7589 | 0.9494 | 0.9611 | 0.9321 | 0.8652 | 0.935 | clean |
| base-d39v | paper-test | 22310 | 0.5320 | 0.0768 | 0.5608 | 0.5611 | 0.5531 | 0.1064 | 0.046 | clean |
| base-d39v | ecoli-sigma70 | 1864 | 0.4866 | -0.0248 | 0.4904 | 0.4865 | 0.5536 | 0.0955 | 0.969 | clean |
| base-d39v | multispecies | 22582 | 0.5371 | 0.0847 | 0.5673 | 0.5669 | 0.5562 | 0.1127 | 0.049 | clean |
| base-d39v | d39v | 1989 | 0.9306 | 0.8663 | 0.9706 | 0.9749 | 0.9412 | 0.8839 | 0.071 | train_overlap |
| base-d39v | tigr4-high | 1476 | 0.8977 | 0.8071 | 0.9316 | 0.9542 | 0.9092 | 0.8238 | 0.081 | clean |
| base-d39v | d39v-tigr4 | 3465 | 0.9166 | 0.8408 | 0.9539 | 0.9661 | 0.9270 | 0.8570 | 0.079 | train_overlap |
| base-d39v | d39v-tigr4-dedup | 3135 | 0.9136 | 0.8307 | 0.9455 | 0.9542 | 0.9225 | 0.8450 | 0.079 | train_overlap |
| base-d39v-tigr4 | paper-test | 22310 | 0.5470 | 0.0976 | 0.5701 | 0.5650 | 0.5566 | 0.1135 | 0.135 | clean |
| base-d39v-tigr4 | ecoli-sigma70 | 1864 | 0.4533 | -0.0740 | 0.4577 | 0.4545 | 0.5445 | 0.0650 | 0.980 | clean |
| base-d39v-tigr4 | multispecies | 22582 | 0.5498 | 0.1017 | 0.5709 | 0.5649 | 0.5547 | 0.1095 | 0.170 | clean |
| base-d39v-tigr4 | d39v | 1989 | 0.9437 | 0.8901 | 0.9766 | 0.9804 | 0.9548 | 0.9104 | 0.233 | train_overlap |
| base-d39v-tigr4 | tigr4-high | 1476 | 0.9126 | 0.8324 | 0.9483 | 0.9651 | 0.9207 | 0.8452 | 0.201 | train_overlap |
| base-d39v-tigr4 | d39v-tigr4 | 3465 | 0.9304 | 0.8653 | 0.9640 | 0.9737 | 0.9400 | 0.8820 | 0.233 | train_overlap |
| base-d39v-tigr4 | d39v-tigr4-dedup | 3135 | 0.9282 | 0.8578 | 0.9582 | 0.9650 | 0.9368 | 0.8735 | 0.233 | train_overlap |
| base-223k-d39v | paper-test | 22310 | 0.8080 | 0.6211 | 0.8917 | 0.9056 | 0.8085 | 0.6215 | 0.490 | clean |
| base-223k-d39v | ecoli-sigma70 | 1864 | 0.8686 | 0.7355 | 0.9371 | 0.9339 | 0.8723 | 0.7431 | 0.526 | clean |
| base-223k-d39v | multispecies | 22582 | 0.8082 | 0.6213 | 0.8905 | 0.9031 | 0.8098 | 0.6229 | 0.472 | clean |
| base-223k-d39v | d39v | 1989 | 0.9442 | 0.8885 | 0.9817 | 0.9867 | 0.9593 | 0.9195 | 0.834 | train_overlap |
| base-223k-d39v | tigr4-high | 1476 | 0.8753 | 0.7507 | 0.9276 | 0.9507 | 0.8997 | 0.8037 | 0.793 | clean |
| base-223k-d39v | d39v-tigr4 | 3465 | 0.9149 | 0.8297 | 0.9596 | 0.9721 | 0.9333 | 0.8688 | 0.817 | train_overlap |
| base-223k-d39v | d39v-tigr4-dedup | 3135 | 0.9085 | 0.8150 | 0.9529 | 0.9626 | 0.9289 | 0.8573 | 0.817 | train_overlap |
| base-223k-d39v-tigr4 | paper-test | 22310 | 0.7959 | 0.5922 | 0.8813 | 0.8966 | 0.7981 | 0.6016 | 0.610 | clean |
| base-223k-d39v-tigr4 | ecoli-sigma70 | 1864 | 0.8686 | 0.7364 | 0.9326 | 0.9282 | 0.8702 | 0.7393 | 0.524 | clean |
| base-223k-d39v-tigr4 | multispecies | 22582 | 0.7909 | 0.5820 | 0.8782 | 0.8930 | 0.7954 | 0.5935 | 0.570 | clean |
| base-223k-d39v-tigr4 | d39v | 1989 | 0.9231 | 0.8488 | 0.9808 | 0.9857 | 0.9573 | 0.9161 | 0.962 | train_overlap |
| base-223k-d39v-tigr4 | tigr4-high | 1476 | 0.8760 | 0.7523 | 0.9396 | 0.9589 | 0.9079 | 0.8260 | 0.956 | train_overlap |
| base-223k-d39v-tigr4 | d39v-tigr4 | 3465 | 0.9030 | 0.8074 | 0.9637 | 0.9746 | 0.9362 | 0.8769 | 0.956 | train_overlap |
| base-223k-d39v-tigr4 | d39v-tigr4-dedup | 3135 | 0.8957 | 0.7924 | 0.9580 | 0.9661 | 0.9330 | 0.8679 | 0.956 | train_overlap |
| hfpromoter-d39v | paper-test | 22310 | 0.9269 | 0.8553 | 0.9796 | 0.9822 | 0.9285 | 0.8576 | 0.444 | train_overlap |
| hfpromoter-d39v | ecoli-sigma70 | 1864 | 0.8578 | 0.7153 | 0.9378 | 0.9340 | 0.8680 | 0.7352 | 0.780 | clean |
| hfpromoter-d39v | multispecies | 22582 | 0.8030 | 0.6077 | 0.8830 | 0.8901 | 0.8039 | 0.6109 | 0.545 | clean |
| hfpromoter-d39v | d39v | 1989 | 0.9326 | 0.8656 | 0.9759 | 0.9820 | 0.9553 | 0.9115 | 0.955 | train_overlap |
| hfpromoter-d39v | tigr4-high | 1476 | 0.8686 | 0.7371 | 0.9239 | 0.9458 | 0.9045 | 0.8157 | 0.968 | clean |
| hfpromoter-d39v | d39v-tigr4 | 3465 | 0.9053 | 0.8108 | 0.9540 | 0.9669 | 0.9333 | 0.8697 | 0.966 | train_overlap |
| hfpromoter-d39v | d39v-tigr4-dedup | 3135 | 0.8982 | 0.7949 | 0.9465 | 0.9558 | 0.9292 | 0.8587 | 0.966 | train_overlap |
| hfpromoter-baseline | paper-test | 22310 | 0.9239 | 0.8480 | 0.9791 | 0.9817 | 0.9266 | 0.8535 | 0.605 | train_overlap |
| hfpromoter-baseline | ecoli-sigma70 | 1864 | 0.8509 | 0.7078 | 0.9373 | 0.9324 | 0.8659 | 0.7301 | 0.870 | clean |
| hfpromoter-baseline | multispecies | 22582 | 0.7950 | 0.5908 | 0.8810 | 0.8867 | 0.8021 | 0.6080 | 0.761 | clean |
| hfpromoter-baseline | d39v | 1989 | 0.8859 | 0.7787 | 0.9700 | 0.9780 | 0.9487 | 0.8980 | 0.974 | clean |
| hfpromoter-baseline | tigr4-high | 1476 | 0.8306 | 0.6654 | 0.9209 | 0.9416 | 0.8963 | 0.7991 | 0.984 | clean |
| hfpromoter-baseline | d39v-tigr4 | 3465 | 0.8623 | 0.7303 | 0.9493 | 0.9628 | 0.9255 | 0.8531 | 0.974 | clean |
| hfpromoter-baseline | d39v-tigr4-dedup | 3135 | 0.8510 | 0.7102 | 0.9417 | 0.9507 | 0.9206 | 0.8402 | 0.974 | clean |

## Significance (paired bootstrap, B=2000, threshold 0.5)
| Dataset | Comparison (a − b) | Metric | Δ | 95% CI | p | 95% sig |
|---|---|---|---|---|---|---|
| multispecies | base-223k-d39v − hfpromoter-baseline | mcc | +0.0305 | [+0.0208, +0.0406] | 0.0000 | yes |
| multispecies | base-223k-d39v − hfpromoter-baseline | auc | +0.0094 | [+0.0062, +0.0127] | 0.0000 | yes |
| multispecies | base-223k-d39v-tigr4 − hfpromoter-baseline | mcc | -0.0087 | [-0.0192, +0.0013] | 0.0930 | no |
| multispecies | base-223k-d39v-tigr4 − hfpromoter-baseline | auc | -0.0029 | [-0.0063, +0.0007] | 0.1170 | no |
| ecoli-sigma70 | base-223k-d39v − hfpromoter-baseline | mcc | +0.0276 | [-0.0019, +0.0591] | 0.0700 | no |
| ecoli-sigma70 | base-223k-d39v − hfpromoter-baseline | auc | -0.0002 | [-0.0085, +0.0080] | 0.9160 | no |
| ecoli-sigma70 | base-223k-d39v-tigr4 − hfpromoter-baseline | mcc | +0.0286 | [-0.0023, +0.0585] | 0.0750 | no |
| ecoli-sigma70 | base-223k-d39v-tigr4 − hfpromoter-baseline | auc | -0.0047 | [-0.0136, +0.0040] | 0.3020 | no |
| ecoli-sigma70 | paper223k-replica − hfpromoter-baseline | mcc | -0.0026 | [-0.0326, +0.0282] | 0.8850 | no |
| ecoli-sigma70 | paper223k-replica − hfpromoter-baseline | auc | -0.0055 | [-0.0141, +0.0033] | 0.2360 | no |
| tigr4-high | base-d39v − hfpromoter-baseline | mcc | +0.1417 | [+0.1035, +0.1788] | 0.0000 | yes |
| tigr4-high | base-d39v − hfpromoter-baseline | auc | +0.0107 | [-0.0038, +0.0249] | 0.1340 | no |
| tigr4-high | base-223k-d39v − hfpromoter-baseline | mcc | +0.0854 | [+0.0552, +0.1150] | 0.0000 | yes |
| tigr4-high | base-223k-d39v − hfpromoter-baseline | auc | +0.0067 | [-0.0006, +0.0143] | 0.0690 | no |

## Notes
- HF baseline is zero-shot `neuralbioinfo/prokbert-mini-promoter` (no fine-tuning here); it was upstream-fine-tuned on the full 223k train, so its `paper-test`/`paper-val` cells are `train_overlap`, not clean.
- `paper223k-replica` reproduces the upstream 223k setup (80/10/10 seed 42, paper hyperparams, fp32); matches the released HF model within ±0.006.
- `base-223k-d39v` / `base-223k-d39v-tigr4` use the parallel protocol (fit HF train_80 + pneumo train, combined val, locked tests); supersede the legacy full-train mix (see `legacy/`).
- `base-d39v` / `base-d39v-tigr4` train from the true base with `--warmup_ratio 0.1` (paper warmup 500 caps LR on 22–36 steps).
- `hfpromoter-d39v` uses the pooler architecture inherited from HF mini-promoter; other models use masked-attention pooling (documented, not unified).
- Youden = max(TPR-FPR) on the evaluated set (optimistic, for benchmark comparison).
- Per-model cards: `reports/model_matrix/<alias>/model_card.md`; ROC/PR overlays: `reports/model_matrix/plots/roc_pr__<dataset>.png`; registry: `MODELS.md`, `models.json`; datasets: `datasets_eval.json`.

## Model Paths
- `paper223k-replica`: `finetuning_outputs/paper_80_10_10_2ep/best`
- `base-d39v`: `finetuning_outputs/d39v_989_base_true_wr01_2ep/best`
- `base-d39v-tigr4`: `finetuning_outputs/base_d39v_tigr4_2ep/best`
- `base-223k-d39v`: `finetuning_outputs/base_223k_d39v_2ep/best`
- `base-223k-d39v-tigr4`: `finetuning_outputs/base_223k_d39v_tigr4_2ep/best`
- `hfpromoter-d39v`: `finetuning_outputs/d39v_from_promoter_2ep/best`
- `hfpromoter-baseline`: `neuralbioinfo/prokbert-mini-promoter` (pinned rev `8502f02`)
