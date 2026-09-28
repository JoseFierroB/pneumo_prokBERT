# Generalization errors (MCC @0.5 unless noted)

> Rows for `base-d39v`, `base-223k-d39v`, `hfpromoter-d39v` describe runs retired 2026-09-28 (checkpoints deleted); kept as historical record.

Sources: `run_info.json` (eval/test in-split), `trainer_state.json` best val, canonical matrix.
Legacy runs without `run_info.json` are marked; their internal numbers come from checkpoints/logs.

## In-split gaps (train → eval → test)
| model | eval (select) | test (locked) | gap | reading |
|---|---|---|---|---|
| mini-223k-replica | 0.634 (val_10, best@2650) | 0.594 (test_10) | −0.040 | selection effect visible, mild |
| base-d39v | 0.928 (eval296) | 0.821 (test296) | −0.107 | mild overfit or harder test |
| mini-d39v-tigr4 | 0.839 (eval470) | 0.848 (test471) | +0.009 | no overfit |
| base-223k-d39v | 0.624 (val comb.) | 0.621 (test_10) / 0.888 (d39v-test) | −0.003 | no overfit |
| mini-223k-d39v-tigr4 | 0.602 (val comb.) | 0.592 (test_10) / 0.758 (dedup-test) | −0.010 | no overfit |
| hfpromoter-d39v | legacy, no run_info | 0.906* (test296, run log) | n/a | gap not recoverable |

*Run-log value, not verifiable in-repo (no `run_info.json`, empty `trainer_state.json` history). Treat as indicative.
| ablation warmup500 | 0.568 | 0.505 | −0.063 | underfit (LR capped), not overfit |
| e1b_hardneg | 0.817 | 0.777 | −0.040 | mild |
| promoter-d39v-tigr4 | 0.846 | 0.867 | +0.021 | no overfit |

## In-dist → external gaps (distribution shift, not overfitting)
| model | in-dist test | sigma70 | multispecies | tigr4-high |
|---|---|---|---|---|
| mini-223k-replica | 0.594 | 0.705 | 0.588 | 0.722 |
| base-d39v | 0.821 | −0.025 | 0.085 | 0.807 |
| mini-d39v-tigr4 | 0.848 | −0.074 | 0.102 | 0.832† |
| base-223k-d39v | 0.621 | 0.736 | 0.621 | 0.751 |
| mini-223k-d39v-tigr4 | 0.592 | 0.736 | 0.582 | 0.752† |
| hfpromoter-d39v | 0.906 | 0.715 | 0.608 | 0.737 |
| promoter-baseline | — | 0.708 | 0.591 | 0.665 |

† train_overlap. Externals are clean except where flagged in the matrix.

## Diagnosis
1. No severe overfitting in any run (worst in-split gap −0.107).
2. The dominant error is **distributional**: pneumo-only models collapse outside pneumo
   (0.82→−0.03); mixes retain general + gain pneumo.
3. Replica externals > in-dist test is a distribution effect (E. coli sigma70 promoters
   are an easier, cleaner task), not leakage: leakage flags confirm test_10 was unseen.
4. Warmup-capped run underfits (low train *and* test) — distinct signature from overfit.
