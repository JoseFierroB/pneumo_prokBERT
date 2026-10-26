# Evaluation Report — 8 Models × 4 Datasets (5090, k6s1, @0.5 / Youden)

> Frozen Oct-12 snapshot. Per-model cards and `model_matrix/plots/` overlays were
> archived out of repo on 2026-10-26 (see `~/pneumo_prokBERT_archive/2026-10-26/`);
> the table below is intact. Current figures live in `reports/supplementary_5050/`.

Matrix from `reports/model_matrix/metrics.json` (evaluator `scripts/eval_matrix.py`, one tokenizer call per dataset).
Leakage flags: `clean` = clean of fine-tuning labels in this project; `train_overlap` / `selection_overlap` otherwise.
`paper-val` is the selection set (appendix): used for best-checkpoint selection, never for ranking.

| Model | Dataset | N | ACC@0.5 | MCC@0.5 | AUC | AP | ACC@Youden | MCC@Youden | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|---|
| mini-223k-replica | paper-test | 22310 | 0.7961 | 0.5941 | 0.8800 | 0.8951 | 0.7968 | 0.5959 | 0.510 | clean |
| mini-223k-replica | ecoli-sigma70 | 1864 | 0.8492 | 0.7052 | 0.9318 | 0.9266 | 0.8600 | 0.7192 | 0.658 | clean |
| mini-223k-replica | multispecies | 22582 | 0.7934 | 0.5882 | 0.8788 | 0.8942 | 0.7939 | 0.5906 | 0.529 | clean |
| mini-223k-replica | d39v-tigr4-dedup | 3135 | 0.8785 | 0.7589 | 0.9494 | 0.9611 | 0.9321 | 0.8652 | 0.935 | clean |
| mini-d39v-tigr4 | paper-test | 22310 | 0.5470 | 0.0976 | 0.5701 | 0.5650 | 0.5566 | 0.1135 | 0.135 | clean |
| mini-d39v-tigr4 | ecoli-sigma70 | 1864 | 0.4533 | -0.0740 | 0.4577 | 0.4545 | 0.5445 | 0.0650 | 0.980 | clean |
| mini-d39v-tigr4 | multispecies | 22582 | 0.5498 | 0.1017 | 0.5709 | 0.5649 | 0.5547 | 0.1095 | 0.170 | clean |
| mini-d39v-tigr4 | d39v-tigr4-dedup | 3135 | 0.9282 | 0.8578 | 0.9582 | 0.9650 | 0.9368 | 0.8735 | 0.233 | train_overlap |
| mini-223k-d39v-tigr4 | paper-test | 22310 | 0.7959 | 0.5922 | 0.8813 | 0.8966 | 0.7981 | 0.6016 | 0.610 | clean |
| mini-223k-d39v-tigr4 | ecoli-sigma70 | 1864 | 0.8686 | 0.7364 | 0.9326 | 0.9282 | 0.8702 | 0.7393 | 0.524 | clean |
| mini-223k-d39v-tigr4 | multispecies | 22582 | 0.7909 | 0.5820 | 0.8782 | 0.8930 | 0.7954 | 0.5935 | 0.570 | clean |
| mini-223k-d39v-tigr4 | d39v-tigr4-dedup | 3135 | 0.8957 | 0.7924 | 0.9580 | 0.9661 | 0.9330 | 0.8679 | 0.956 | train_overlap |
| mini-223k-d39v-tigr4-high5050 | paper-test | 22310 | 0.8072 | 0.6146 | 0.8919 | 0.9059 | 0.8087 | 0.6199 | 0.564 | clean |
| mini-223k-d39v-tigr4-high5050 | ecoli-sigma70 | 1864 | 0.8766 | 0.7517 | 0.9392 | 0.9387 | 0.8782 | 0.7549 | 0.506 | clean |
| mini-223k-d39v-tigr4-high5050 | multispecies | 22582 | 0.8068 | 0.6138 | 0.8905 | 0.9040 | 0.8084 | 0.6174 | 0.521 | clean |
| mini-223k-d39v-tigr4-high5050 | d39v-tigr4-dedup | 3135 | 0.9088 | 0.8161 | 0.9584 | 0.9672 | 0.9372 | 0.8746 | 0.833 | train_overlap |
| mini-223k-d39v-tigr4-high5050-3ep | paper-test | 22310 | 0.8212 | 0.6431 | 0.9043 | 0.9169 | 0.8234 | 0.6492 | 0.563 | clean |
| mini-223k-d39v-tigr4-high5050-3ep | ecoli-sigma70 | 1864 | 0.8745 | 0.7489 | 0.9458 | 0.9457 | 0.8830 | 0.7651 | 0.716 | clean |
| mini-223k-d39v-tigr4-high5050-3ep | multispecies | 22582 | 0.8209 | 0.6424 | 0.9018 | 0.9135 | 0.8233 | 0.6482 | 0.540 | clean |
| mini-223k-d39v-tigr4-high5050-3ep | d39v-tigr4-dedup | 3135 | 0.9311 | 0.8605 | 0.9684 | 0.9742 | 0.9445 | 0.8898 | 0.889 | train_overlap |
| mini-223k-d39v-tigr4-high5050-4ep | paper-test | 22310 | 0.7965 | 0.5967 | 0.8807 | 0.8952 | 0.7968 | 0.5987 | 0.519 | clean |
| mini-223k-d39v-tigr4-high5050-4ep | ecoli-sigma70 | 1864 | 0.8541 | 0.7078 | 0.9283 | 0.9268 | 0.8632 | 0.7248 | 0.589 | clean |
| mini-223k-d39v-tigr4-high5050-4ep | multispecies | 22582 | 0.7945 | 0.5924 | 0.8778 | 0.8926 | 0.7952 | 0.5932 | 0.489 | clean |
| mini-223k-d39v-tigr4-high5050-4ep | d39v-tigr4-dedup | 3135 | 0.8941 | 0.7875 | 0.9521 | 0.9629 | 0.9321 | 0.8638 | 0.865 | train_overlap |
| promoter-d39v-tigr4 | paper-test | 22310 | 0.8889 | 0.7928 | 0.9769 | 0.9798 | 0.9238 | 0.8476 | 0.206 | train_overlap |
| promoter-d39v-tigr4 | ecoli-sigma70 | 1864 | 0.8589 | 0.7217 | 0.9357 | 0.9335 | 0.8632 | 0.7253 | 0.170 | clean |
| promoter-d39v-tigr4 | multispecies | 22582 | 0.7910 | 0.6022 | 0.8807 | 0.8905 | 0.8012 | 0.6054 | 0.267 | clean |
| promoter-d39v-tigr4 | d39v-tigr4-dedup | 3135 | 0.9301 | 0.8595 | 0.9539 | 0.9628 | 0.9337 | 0.8676 | 0.664 | train_overlap |
| promoter-baseline | paper-test | 22310 | 0.9239 | 0.8480 | 0.9791 | 0.9817 | 0.9266 | 0.8535 | 0.605 | train_overlap |
| promoter-baseline | ecoli-sigma70 | 1864 | 0.8509 | 0.7078 | 0.9373 | 0.9324 | 0.8659 | 0.7301 | 0.870 | clean |
| promoter-baseline | multispecies | 22582 | 0.7950 | 0.5908 | 0.8810 | 0.8867 | 0.8021 | 0.6080 | 0.761 | clean |
| promoter-baseline | d39v-tigr4-dedup | 3135 | 0.8510 | 0.7102 | 0.9417 | 0.9507 | 0.9206 | 0.8402 | 0.974 | clean |

## Significance (paired bootstrap, B=2000, threshold 0.5)
| Dataset | Comparison (a − b) | Metric | Δ | 95% CI | p | 95% sig |
|---|---|---|---|---|---|---|
| multispecies | mini-223k-d39v-tigr4 − promoter-baseline | mcc | -0.0087 | [-0.0190, +0.0017] | 0.9540 | no |
| multispecies | mini-223k-d39v-tigr4 − promoter-baseline | auc | -0.0029 | [-0.0062, +0.0005] | 0.9515 | no |
| ecoli-sigma70 | mini-223k-replica − promoter-baseline | mcc | -0.0020 | [-0.0340, +0.0309] | 0.5540 | no |
| ecoli-sigma70 | mini-223k-replica − promoter-baseline | auc | -0.0054 | [-0.0142, +0.0034] | 0.8830 | no |
| ecoli-sigma70 | mini-223k-d39v-tigr4 − promoter-baseline | mcc | +0.0287 | [-0.0011, +0.0608] | 0.0290 | no |
| ecoli-sigma70 | mini-223k-d39v-tigr4 − promoter-baseline | auc | -0.0047 | [-0.0135, +0.0043] | 0.8445 | no |
| paper-test | mini-223k-d39v-tigr4-high5050 − mini-223k-d39v-tigr4 | mcc | +0.0224 | [+0.0133, +0.0314] | 0.0000 | yes |
| paper-test | mini-223k-d39v-tigr4-high5050 − mini-223k-d39v-tigr4 | auc | +0.0106 | [+0.0081, +0.0131] | 0.0000 | yes |
| multispecies | mini-223k-d39v-tigr4-high5050 − mini-223k-d39v-tigr4 | mcc | +0.0316 | [+0.0227, +0.0401] | 0.0000 | yes |
| multispecies | mini-223k-d39v-tigr4-high5050 − mini-223k-d39v-tigr4 | auc | +0.0123 | [+0.0098, +0.0148] | 0.0000 | yes |
| ecoli-sigma70 | mini-223k-d39v-tigr4-high5050 − mini-223k-d39v-tigr4 | mcc | +0.0152 | [-0.0100, +0.0399] | 0.1160 | no |
| ecoli-sigma70 | mini-223k-d39v-tigr4-high5050 − mini-223k-d39v-tigr4 | auc | +0.0066 | [+0.0001, +0.0126] | 0.0225 | yes |
| paper-test | mini-223k-d39v-tigr4-high5050-3ep − mini-223k-d39v-tigr4-high5050 | mcc | +0.0284 | [+0.0203, +0.0365] | 0.0000 | yes |
| paper-test | mini-223k-d39v-tigr4-high5050-3ep − mini-223k-d39v-tigr4-high5050 | auc | +0.0124 | [+0.0102, +0.0145] | 0.0000 | yes |
| multispecies | mini-223k-d39v-tigr4-high5050-3ep − mini-223k-d39v-tigr4-high5050 | mcc | +0.0286 | [+0.0204, +0.0367] | 0.0000 | yes |
| multispecies | mini-223k-d39v-tigr4-high5050-3ep − mini-223k-d39v-tigr4-high5050 | auc | +0.0113 | [+0.0091, +0.0135] | 0.0000 | yes |

## Notes
- HF baseline is zero-shot `neuralbioinfo/prokbert-mini-promoter` (no fine-tuning here); it was upstream-fine-tuned on the full 223k train, so its `paper-test`/`paper-val` cells are `train_overlap`, not clean.
- `mini-223k-replica` reproduces the upstream 223k setup (80/10/10 seed 42, paper hyperparams, fp32); matches the released HF model within ±0.006.
- `mini-223k-d39v-tigr4` uses the parallel protocol (fit HF train_80 + dedup train, combined val, locked tests).
- `mini-d39v-tigr4` trains from the true base with `--warmup-ratio 0.1` (paper warmup 500 caps LR on 22–36 steps).
- `promoter-d39v-tigr4` uses the pooler architecture inherited from HF mini-promoter; other models use masked-attention pooling (documented, not unified).
- Youden = max(TPR-FPR) on the evaluated set (optimistic, for benchmark comparison).
- Per-model cards: `reports/model_matrix/<alias>/model_card.md`; ROC/PR overlays: `reports/model_matrix/plots/roc__<dataset>.png` + `pr__<dataset>.png` (one metric per file); registry: `MODELS.md`, `models.json`; datasets: `datasets_eval.json`.

## Model Paths
- `mini-223k-replica`: `finetuning_outputs/mini_223k_replica_2ep/best`
- `mini-d39v-tigr4`: `finetuning_outputs/mini_d39v_tigr4_2ep/best`
- `mini-223k-d39v-tigr4`: `finetuning_outputs/mini_223k_d39v_tigr4_2ep/best`
- `mini-223k-d39v-tigr4-high5050`: `finetuning_outputs/mini_223k_d39v_tigr4_high5050_2ep/best`
- `mini-223k-d39v-tigr4-high5050-3ep`: `finetuning_outputs/mini_223k_d39v_tigr4_high5050_3ep/best`
- `mini-223k-d39v-tigr4-high5050-4ep`: `finetuning_outputs/mini_223k_d39v_tigr4_high5050_4ep/best`
- `promoter-d39v-tigr4`: `finetuning_outputs/promoter_d39v_tigr4_2ep/best`
- `promoter-baseline`: `neuralbioinfo/prokbert-mini-promoter` (pinned rev `8502f02`)
