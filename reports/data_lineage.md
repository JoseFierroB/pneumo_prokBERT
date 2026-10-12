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

HF species: 77 total (69 in train_80); 4 species unseen in train
(`M. capsulatus Bath`, `N. gonorrhoeae`, `S. enterica CFSAN002069`, `Y. enterocolitica 8081`,
1 row each — negligible globally). y ≈ 0.50 in all splits. A sklearn
StratifiedShuffleSplit variant (32 buckets incl. RARE, chi2 ≈ 0.1) was tested and REJECTED:
re-splitting would invalidate all frozen metrics/shas for ~72 rows of rare species.

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

## Clean dedup slices (recomputed from saved scores, no new inference)
dedup-eval/test never overlap any train split (dedup is internally disjoint), so these
slices are clean for ALL models — including HF baseline (never saw dedup at all):

| model | dedup-eval MCC/AUC | dedup-test MCC/AUC |
|---|---|---|
| mini-223k-replica | 0.780 / 0.949 | 0.789 / 0.951 |
| mini-d39v-tigr4 | 0.839 / 0.939 | 0.848 / 0.951 |
| mini-223k-d39v-tigr4 | 0.788 / 0.950 | 0.758 / 0.950 |
| promoter-d39v-tigr4 | 0.846 / 0.949 | 0.867 / 0.947 |
| promoter-baseline (HF) | 0.718 / 0.940 | 0.681 / 0.942 |

## Clean ranking, all models (MCC; every cell below is clean)
| model | multispecies | ecoli-sigma70 |
|---|---|---|
| mini-223kfull-1to2-dedup | 0.636 | 0.759 |
| mini-223k-1to2 | 0.616 | 0.739 |
| promoter-d39v-tigr4 | 0.602 | 0.722 |
| promoter-baseline (HF) | 0.591 | 0.708 |
| mini-223k-replica | 0.588 | 0.705 |
| mini-223k-d39v-tigr4 | 0.582 | 0.736 |
| mini-d39v-tigr4 | 0.102 | -0.074 |
