# Supplementary 1to2 (pneumo-only 1:2 continuations; NOT in the matrix)
## mini-mix-1to2-cont (pneumo-only continuation, WS 5090)
- **Recipe**: `finetuning.py --model-name finetuning_outputs/mini_223k_d39v_tigr4_2ep/best --dataset-dir datasets/d39v_tigr4_high_1to2 --output-dir finetuning_outputs/mini_mix_1to2_cont_2ep --warmup-ratio 0.1` (2ep, 46 steps, fp16).
- **run_info**: test MCC 0.8362 / AUC 0.9676 (1to2 test, clean).
- **4-column eval** (`eval_single.py`, leakage from `mini-223k-d39v-tigr4` + dedup override): JSON + 4 CSVs here.
- **Finding**: pneumo-only continuation narrows the model — dedup 0.79→0.90 MCC but paper-test 0.59→0.43, sigma70 0.74→0.41, multispecies 0.58→0.43 (all clean). General promoter features partially forgotten.

## promoter-dedup-1to2-cont (promoter init + 1to2, WS 5090, `warmup-ratio 0.1`)
- **Recipe**: `finetuning.py --model-name neuralbioinfo/prokbert-mini-promoter --dataset-dir datasets/d39v_tigr4_high_1to2 --output-dir finetuning_outputs/promoter_dedup_1to2_cont_2ep --warmup-ratio 0.1` (2ep, 46 steps).
- **run_info**: 1to2-eval MCC 0.903 / test MCC 0.929 (clean).
- **4-column eval**: dedup 0.931 (overlap) / sigma70 0.374 clean / multispecies 0.387 clean / paper-test 0.462 overlap.
- **Finding**: best pneumo specialist ever (dedup 0.93 > promoter 0.86) but worst general collapse (sigma70 0.72→0.37, multi 0.60→0.39 — below HF 0.71/0.59). Narrowing hits promoter-init harder than mini.
- **Why it collapses (deltas vs init)**: dedup run (warmup 500) backbone Δ 0.002 — the capped LR barely moved it, accidentally preserving upstream features. 1to2 run (warmup-ratio 0.1) backbone Δ 0.019 (9×), classifier 0.10 vs 0.05. Cause = effective LR, not the 1:2 data. Next: promoter+1to2 with capped warmup 500 as deliberate gentle adaptation.

## promoter-dedup-1to2-cont-w500 (gentle adaptation: warmup 500 cap, WS 5090)
- **Recipe**: same data, `--warmup-steps 500` → `finetuning_outputs/promoter_dedup_1to2_cont_w500_2ep`.
- **run_info**: 1to2-eval MCC 0.860 / test MCC 0.907 (clean).
- **4-column eval**: dedup 0.876 (overlap) / sigma70 0.661 clean / multispecies 0.571 clean / paper-test 0.723 overlap.
- **Deltas vs init**: backbone 0.0033 (≈ dedup run 0.0022) vs 0.0189 at full LR — capped warmup protects features as predicted.
- **Finding**: no collapse (sigma70 0.66 vs 0.37 at full LR; multi 0.57 vs 0.39) but below dedup-run generality (0.72/0.60) and HF (0.71/0.59) on multi. Gentle 1:2 gains pneumo (dedup 0.876 > 0.860) at a small general cost.

## mini-223k-1to2 (A1 replay with 1:2 balance, WS 5090)
- **Recipe**: `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_high_1to2 --output-dir finetuning_outputs/mini_223k_1to2_2ep` (train_80 + 1to2-train = 181410; combined val 22939; locked tests).
- **Locked**: val_combined 0.628 / test_10 0.621 / 1to2-test 0.819 / sigma70 0.739 / multispecies 0.616 (MCC).
- **4-column eval**: paper-test 0.621 / sigma70 0.739 / multispecies 0.616 / dedup 0.873 (MCC; dedup overlap).
- **Finding**: replay beats the dedup-mix on every clean cell (paper 0.59→0.62, multi 0.58→0.62) with no narrowing — the balanced pneumo data helps everywhere.
- **Per-split (A1, post-hoc, no new training)**:

| split | MCC | AUC | flag |
|---|---|---|---|
| 1to2-train | 0.910 | 0.982 | overlap (fit) |
| 1to2-eval | 0.825 | 0.952 | clean |
| 1to2-test | 0.819 | 0.956 | clean |
| dedup-train | 0.885 | 0.978 | overlap (1to2 built from dedup) |
| dedup-eval | 0.845 | 0.952 | clean |
| dedup-test | 0.845 | 0.954 | clean |
- **Why dedup is `train_overlap` for A1**: 1to2-train shares 1952 segments with dedup-all (all inside dedup-train, by within-split construction); dedup-eval/test have ZERO overlap. Clean slices recomputed from saved scores (no new inference): dedup-eval MCC 0.845 / AUC 0.952, dedup-test MCC 0.845 / AUC 0.954 — the 0.873 on dedup-all is real ability, not memorization.
- **Head-to-head A1 vs `mini-223k-d39v-tigr4`** (paired bootstrap B=2000):

| dataset | MCC Δ | AUC Δ |
|---|---|---|
| paper-test (clean) | +0.028 [+0.020,+0.038] SIG | +0.014 [+0.011,+0.016] SIG |
| multispecies (clean) | +0.034 [+0.025,+0.042] SIG | +0.014 [+0.011,+0.016] SIG |
| ecoli-sigma70 (clean) | +0.003 n.s. | +0.010 [+0.004,+0.016] SIG |

First significant clean-set wins of the project. 1:2 balance applies to pneumo data only;
HF train_80 keeps its natural distribution (no resampling upstream).

## A1 vs HF baseline (`promoter-baseline`, paired bootstrap B=2000)
| dataset | A1 MCC | HF MCC | Δ | flag HF |
|---|---|---|---|---|
| ecoli-sigma70 clean | 0.739 | 0.708 | +0.030 [+0.001,+0.061] SIG | clean |
| multispecies clean | 0.616 | 0.591 | +0.026 [+0.016,+0.035] SIG | clean |
| d39v-tigr4-dedup | 0.873 | 0.710 | +0.162 SIG | clean for HF, overlap for A1 |
| paper-test | 0.621 | 0.848 | — (HF `train_overlap` upstream; not a fair fight) | overlap |
A1 beats the HF promoter model on every clean column; it only loses paper-test, which HF memorized upstream.

## promoter-dedup-1to2-cont (local CPU, same recipe on 1to2 data)
- **Recipe**: `finetuning.py --model-name neuralbioinfo/prokbert-mini-promoter --dataset-dir datasets/d39v_tigr4_high_1to2 --warmup-ratio 0.1` → 1to2-test MCC 0.907 / AUC 0.962 (clean).
- **4-column eval**: paper-test 0.484 (overlap) / sigma70 0.382 (clean) / multispecies 0.393 (clean) / dedup 0.928 (overlap).
- **vs promoter-dedup** (sigma70 0.722, multi 0.602): the 1:2 ratio COLLAPSES general sets for promoter init while helping the specialty test (0.867→0.907). Ratio effect is init-dependent: mini base gained everywhere with 1:2 (A1), promoter init only gains in-domain.

## mini-223k-1to2-dedup (train_80 + 1to2-train + dedup-unseen 242, WS 5090)
- **Recipe**: `scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_1to2_plus_dedup --output-dir finetuning_outputs/mini_223k_1to2_dedup_2ep` (train_80 178473 + 3179 = 181652; combined val 22939; locked tests).
- **Locked**: val_combined 0.641 / test_10 0.626 / plus_dedup-test 0.772 / sigma70 0.729 / multispecies 0.625 (MCC).
- **4-column eval**: paper-test 0.626 / sigma70 0.729 / multispecies 0.625 / dedup 0.862 (MCC).
- **vs A1** (paired bootstrap): paper +0.006 n.s., sigma70 -0.010 n.s., multispecies +0.009 [+0.000,+0.017] SIG.
- **Finding**: the 242 extra dedup rows buy a tiny significant multispecies gain and nothing else — diminishing returns confirmed quantitatively. Dataset: `datasets/d39v_tigr4_1to2_plus_dedup/` (zero internal overlap, MANIFEST).
