# Data lineage: what each model fits, selects on, and tests

Rule: always select on `eval/val` (best MCC); `test` and externals locked, report only.
`train_overlap` = the test contains rows seen in train (diagnostic only, never the primary metric).

## Source splits (all 70/15/15 seed 42, stratified)
| split | train | eval | test |
|---|---|---|---|
| HF 223k (80/10/10 of 223092) | train_80 178473 | val_10 22309 | test_10 22310 |
| dedup_3135 (1399+/1736−) | 2194 (979+/1215−) | 470 | 471 |
| d39v_tigr4_high_1to2 (1399+/2798−, within-split bootstrap) | 2937 | 630 | 630 |
| d39v_full_989 (1973, tag-source appendix) | 1381 | 296 | 296 |

## Per model
| model | fit (train) | selection (val) | locked tests | notes |
|---|---|---|---|---|
| mini-223k-replica | train_80 | val_10 (MCC) | test_10, sigma70, multi | base mini, fp32, warmup 500 |
| mini-d39v-tigr4 | dedup-train | dedup-eval | dedup-test, sigma70, multi | same; canonical tigr4/d39v = overlap |
| mini-223k-d39v-tigr4 | train_80+dedup-train (180667) | combined 22779 | test_10, dedup-test, sigma70, multi | same |
| promoter-d39v-tigr4 | dedup-train | dedup-eval | dedup-test + rest | mini-promoter init, CPU, warmup 500 |
| promoter-baseline | — (upstream full-train) | — | sigma70, multi (paper-test/val = upstream overlap) | zero-shot here |

## Evaluation columns (4 canonical + internal splits)
Canonical: `paper-test` (clean only if the model never saw train_80), `ecoli-sigma70`, `multispecies`,
`d39v-tigr4-dedup` (see `datasets_eval.json` + per-cell flags).
Internal splits (dedup eval/test, val_10) evaluated separately for per-split curves.
