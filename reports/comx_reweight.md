# ComX reweighting (simple protocol, clean-only, gradual)

Rule: always `clean`, never `train_overlap`. Internal splits have zero overlap (verified).
Note: 48/62 comx-test rows were seen by dedup-trained models → for THEM this test is
`train_overlap`; for a fresh model trained only below, it is `clean`.

## Dataset
`datasets/d39v_binary_siga_comx/` (built by `scripts/build_comx_binary.py`, no WS needed):
train 287 (272 SigA-fam / 15 SigX) / eval 62 (59/3) / test 62 (59/3).
SigX=21 vs SigA+SigA-LOWCONF=390 from `d39v_full_989` tags, 70/15/15 stratified seed 42
+ MANIFEST.json. TIGR4's 6 ComX reserved for later (v2 metadata).

## Stages (gradual, stop if a stage is conclusive)
- **S1**: comx-train only (unseen) × presets below → per-sample table on comx-test (clean).
- **S2**: train+test (349) → eval on comx-eval (still untouched).
- **S3**: + d39v_full_989 train rows → full-data model.

## Protocol
`finetuning.py --train_csv ... --eval_csv ... --test_csv ... --class-weight "w0,w1"`
presets: `1,1` (baseline) → `1,5` → `1,10` → `1,18.6` (inverse frequency 390/21).
Thresholds at 0.5 and eval-tuned Youden. Same seed 42, `warmup_ratio 0.1`, 2ep as everything else.

## How to read results (n=3 positives in test)
Aggregate MCC on 3 samples is noise (±33% recall per flip). Report instead:
1. **Per-sample table**: each of the 3 test ComX with P(ComX) per preset.
2. MCC/AUC with the explicit caveat.
3. If inconclusive, LOOCV over pooled SigX (21+5) as follow-up — not now.

## What NOT to do
No SMOTE on sequences (refuted with evidence); no oversampling with replacement
(duplicates 21 → overfit); no threshold tuning on test; no CIs from n=3.

## S1 results (comx-train 287, mini base, CPU 2ep; test clean, n=3)
| preset | test MCC | test AUC | P(3 ComX test) |
|---|---|---|---|
| 1,1 | 0.000 | 0.475 | 0.254, 0.281, 0.245 |
| 1,5 | 0.000 | 0.458 | 0.259, 0.294, 0.254 |
| 1,10 | 0.000 | 0.486 | 0.301, 0.353, 0.308 |
| 1,18.6 | -0.029 | 0.548 | 0.359, 0.496, 0.418 |
Weight shifts score mass monotonically, flips nothing @0.5. Runs: `finetuning_outputs/comx_w*_2ep`.

## S2 results (train+test 349, eval on comx-eval clean, n=3)
| preset | eval MCC | eval AUC | P(3 ComX eval) |
|---|---|---|---|
| 1,1 | 0.000 | 0.780 | 0.272, 0.270, 0.271 |
| 1,5 | 0.000 | 0.859 | 0.286, 0.290, 0.297 |
| 1,10 | 0.000 | 0.949 | 0.350, 0.364, 0.379 |
| 1,18.6 | 0.650 | 0.977 | 0.490, 0.513, 0.532 |
With 18 train positives the strong preset flips 2/3 @0.5. Dose-response consistent S1→S2;
n=3 keeps every claim fragile (±33% recall per flip). Runs: `finetuning_outputs/comx_s2_w*_2ep`
(test_csv = eval.csv by design; test metrics duplicate eval).
