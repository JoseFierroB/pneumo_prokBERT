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

## Overfitting verdict (measured, all trained models)
Evidence: 34 train-vs-test ROC overlays (`roc_gap__*.png`: 6 matrix + 28 ComX),
train/test loss + MCC curves (63 + 72 PNGs), post-hoc train AUCs, leakage flags.
- **Matrix, clean tests**: gaps AUC +0.03–0.05, MCC +0.06–0.12 (replica, mix, A1).
  Small, healthy, same direction — fitting, not memorizing.
- **Overlap cells**: gaps ≈ 0 (dedup-all, full-train paper-test). Expected: the model
  saw those rows. Flagged, never ranked as clean — memorization declared, not hidden.
- **Full-train**: train AUC 0.936 ≤ val 0.945 — zero overfit by the book.
- **ComX fresh runs** (S1/S2/S3/v2): gaps ~0 or negative (test easier than train at n=62).
  No overfit signal beyond small-n noise.
- **ComX committed inits** (P/A1/full): gaps +0.4–0.7 AUC — train memorized, test fails.
  Documented as failure mode (features too committed for 15 positives), not as results.
- **Curves**: train loss ↓ with eval loss ↓/flat and eval MCC plateaus in every kept run;
  no divergence anywhere. The one underfit (warmup-capped) shows low train *and* test.
- **Leakage/selection**: per-cell flags + locked holdouts + best-by-MCC on locked val only.
Conclusion: no hidden overfitting in any reported number; where memorization exists
(overlap cells, committed inits) it is measured, flagged, and excluded from headlines.

## Measured train-vs-test gaps (`scripts/generalization_gap.py`, ROC overlays `roc_gap__*.png`)
Rule: before any other comparison, check train-vs-test ROC/AUC. Train side runs inference;
test side reuses saved matrix scores.

| model | train | test | train AUC/MCC | test AUC/MCC | gap AUC/MCC |
|---|---|---|---|---|---|
| mini-223k-replica | train_80 | paper-test clean | 0.912/0.661 | 0.880/0.594 | +0.032/+0.067 |
| mini-d39v-tigr4 | dedup-train | dedup-all overlap | 0.964/0.864 | 0.958/0.858 | +0.005/+0.006 |
| mini-223k-d39v-tigr4 | train_80+dedup-train | paper-test clean | 0.911/0.659 | 0.881/0.592 | +0.030/+0.067 |
| promoter-d39v-tigr4 | dedup-train | dedup-all overlap | 0.957/0.861 | 0.954/0.860 | +0.003/+0.002 |
| mini-223k-1to2 | train_80+1to2-train | paper-test clean | 0.944/0.744 | 0.895/0.621 | +0.049/+0.124 |
| mini-223kfull-1to2-dedup | full+1to2+unseen | paper-test overlap | 0.896/0.627 | 0.895/0.621 | +0.001/+0.005 |
Reading: clean gaps are small and healthy (+0.03–0.05 AUC; A1 fits train hardest at
+0.124 MCC). Full-train gap ≈ 0 **confirms memorization** of paper-test (it trained on
it) — not generalization. Overlap-flagged cells (dedup-all) show near-zero gaps as expected.
