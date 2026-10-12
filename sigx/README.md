# SigX (self-contained track, local CPU)

> Naming: competence factor is **SigX** (source label `ComX` in frozen vendor
> metadata, mapped at read). Paths use `sigx`; `runs/*/run_info.json` logs and
> `datasets/external/*.tsv` keep legacy bytes as historical record.

Layout: `build_dataset.py` (v1 builder) · `runs/` (all SigX checkpoints) · `reports/` (future tables/plots) · this README (protocol + results). Datasets live in `datasets/` (referenced, shared). Next: v2 dataset (27+54, 1:2, y×strain) — planned, not built.
Training curves (72 PNGs, 24 runs): `sigx/reports/` (6-step runs read as straight lines; numbers carry the signal, not the shapes).

# SigX reweighting (simple protocol, clean-only, gradual)

Rule: always `clean`, never `train_overlap`. Internal splits have zero overlap (verified).
Note: 48/62 sigx-test rows were seen by dedup-trained models → for THEM this test is
`train_overlap`; for a fresh model trained only below, it is `clean`.

## Dataset
`datasets/d39v_binary_siga_sigx/` (built by `sigx/build_dataset.py`, no WS needed):
train 287 (272 SigA-fam / 15 SigX) / eval 62 (59/3) / test 62 (59/3).
SigX=21 vs SigA+SigA-LOWCONF=390 from `d39v_full_989` tags, 70/15/15 stratified seed 42
+ MANIFEST.json. TIGR4's 6 SigX reserved for later (v2 metadata).

## Stages (gradual, stop if a stage is conclusive)
- **S1**: sigx-train only (unseen) × presets below → per-sample table on sigx-test (clean).
- **S2**: train+test (349) → eval on sigx-eval (still untouched).
- **S3**: + d39v_full_989 train rows → full-data model.
- **P/A1/full**: other inits under S1 setting (below).

## Protocol
`finetuning.py --train_csv ... --eval_csv ... --test_csv ... --class-weight "w0,w1"`
presets: `1,1` (baseline) → `1,5` → `1,10` → `1,18.6` (inverse frequency 390/21).
Thresholds at 0.5 and eval-tuned Youden. Same seed 42, `warmup_ratio 0.1`, 2ep as everything else.

## Verdict on reweighting (after S1/S2/S3/P/v2: no durable gain)
Weights shift scores monotonically but never bought a robust decision: S1 flipped nothing,
S2/S3 flips were n=3 noise, and on balanced v2 weights HURT (1,1: MCC 0.50; 1,10: -0.19).
What worked instead: more positives (15→31), real training length (40 steps), clean design.
Recommendation: plain CE (`1,1`) by default; weights only as diagnostic on imbalanced pilots.

## How to read results (n=3 positives in test)
Aggregate MCC on 3 samples is noise (±33% recall per flip). Report instead:
1. **Per-sample table**: each of the 3 test SigX with P(SigX) per preset.
2. MCC/AUC with the explicit caveat.
3. If inconclusive, LOOCV over pooled SigX (21+5) as follow-up — not now.

## S1 results (sigx-train 287, mini base, CPU 2ep; test clean, n=3)
| preset | test MCC | test AUC | P(3 SigX test) |
|---|---|---|---|
| 1,1 | 0.000 | 0.475 | 0.254, 0.281, 0.245 |
| 1,5 | 0.000 | 0.458 | 0.259, 0.294, 0.254 |
| 1,10 | 0.000 | 0.486 | 0.301, 0.353, 0.308 |
| 1,18.6 | -0.029 | 0.548 | 0.359, 0.496, 0.418 |
Weight shifts score mass monotonically, flips nothing @0.5. Runs: `sigx/runs/sigx_w*_2ep`.

## S2 results (train+test 349, eval on sigx-eval clean, n=3)
| preset | eval MCC | eval AUC | P(3 SigX eval) |
|---|---|---|---|
| 1,1 | 0.000 | 0.780 | 0.272, 0.270, 0.271 |
| 1,5 | 0.000 | 0.859 | 0.286, 0.290, 0.297 |
| 1,10 | 0.000 | 0.949 | 0.350, 0.364, 0.379 |
| 1,18.6 | 0.650 | 0.977 | 0.490, 0.513, 0.532 |
With 18 train positives the strong preset flips 2/3 @0.5. Dose-response consistent S1→S2;
n=3 keeps every claim fragile (±33% recall per flip). Runs: `sigx/runs/sigx_s2_w*_2ep`
(test_csv = eval.csv by design; test metrics duplicate eval).

## S3 results (sigx train+test + d39v_full-train SigX/SigA = 616 rows, 33 pos)
Trap found: `d39v_full_989` splits are NOT segment-disjoint (legacy flaw) — 36 sigx-eval rows
(2 positives) appear in d39v_full-train. Excluded those 36 from S3-train → 580 rows, 31 pos,
sigx-eval overlap 0 (clean honored).
| preset | eval MCC | eval AUC | P(3 SigX eval) |
|---|---|---|---|
| 1,1 | 0.000 | 0.836 | 0.081, 0.072, 0.081 |
| 1,5 | 0.000 | 0.977 | 0.158, 0.159, 0.198 |
| 1,10 | 0.810 | 0.972 | 0.413, 0.602, 0.627 |
| 1,18.6 | 0.632 | 0.977 | 0.505, 0.566, 0.620 |
With 31 positives the mid presets flip 2/3 @0.5. Same monotonic dose-response, stronger.
Runs: `sigx/runs/sigx_s3_w*_2ep` (train_csv `/tmp/s3_train.csv` rebuilt with full
`segment,y,id,tag,sigma,strain,split` schema; test_csv = eval.csv by design).

## P: promoter-init SigX (S1 setting, mini-promoter init) — FAILED
| preset | test MCC | test AUC | P(3 SigX test) |
|---|---|---|---|
| 1,1 | 0.000 | 0.147 | 0.004, 0.003, 0.003 |
| 1,5 | 0.000 | 0.079 | 0.011, 0.005, 0.003 |
| 1,10 | -0.435 | 0.085 | 0.270, 0.778, 0.040 |
| 1,18.6 | -0.315 | 0.096 | 0.163, 0.372, 0.175 |
The pooler head collapses on 15 positives (AUC < 0.15, worse than random; high weights
go erratic, not better). Mini base >> promoter init for few-shot SigX: promoter features
are destroyed, not reused. Runs: `sigx/runs/sigx_p_w*_2ep`. Weights chapter closed.

## A1/full inits (S1 setting, pneumo-aware inits) — WORSE than mini base
Overlap warning: A1/full inits saw 197/287 sigx-train, 42/62 eval (3/3 pos), 48/62 test
(2/3 pos) during THEIR training → all evals below are `train_overlap` (diagnostic only).
| init | preset | test MCC | test AUC | P(3 SigX test) |
|---|---|---|---|---|
| A1 | 1,1 | 0.000 | 0.277 | 0.012, 0.007, 0.006 |
| A1 | 1,5 | 0.000 | 0.311 | — |
| A1 | 1,10 | -0.059 | 0.328 | — |
| A1 | 1,18.6 | -0.059 | 0.373 | 0.121, 0.096, 0.068 |
| full | 1,1 | 0.000 | 0.209 | 0.016, 0.013, 0.012 |
| full | 1,5 | 0.000 | 0.186 | — |
| full | 1,10 | -0.029 | 0.186 | — |
| full | 1,18.6 | -0.041 | 0.277 | 0.065, 0.094, 0.115 |
Specialized inits collapse harder than plain mini base (S1: P 0.25–0.50, AUC 0.46–0.55):
committed features + 15 positives = worse few-shot. Plain mini base remains the best SigX
init. Runs: `sigx/runs/sigx_a1_w*_2ep`, `sigx_full_w*_2ep`.

## B2: where the weights move (relative L2 delta vs init, strongest preset each)
| run | backbone | pool/weighting | classifier | embeddings |
|---|---|---|---|---|
| S1-w18.6 (mini base) | 0.006 | 0.007 | 0.706 | 0.001 |
| S2-w18.6 (mini base) | 0.004 | 0.004 | 0.705 | 0.001 |
| S3-w10 (mini base) | 0.010 | 0.010 | 0.708 | 0.002 |
| S3-w18.6 (mini base) | 0.006 | 0.006 | 0.706 | 0.002 |
| P-w18.6 (promoter init) | 0.225 | 0.292 | 0.796 | 0.164 |
| A1-w18.6 (mix-1to2 init) | 0.108 | 0.121 | 0.796 | 0.091 |
Reading: mini-base runs adapt the **classifier only** (backbone ~frozen) — hence score
shifts without separation. Promoter/A1 inits **rewrite the backbone** (0.11–0.22) and
collapse: committed features + 15 positives = destruction, not adaptation.

## v2 results (27+54, mini base; test 13 rows: 3 D39V + 1 TIGR4 positives, clean)
Lesson first: 2ep × batch128 = 2 gradient steps → uniform P≈0.50 collapse (runs
`sigx_v2_w*` deleted). Micro-training (10ep × batch16 = 40 steps) learns:
| preset | test MCC | test AUC | P(3 D39V pos) | P(1 TIGR4 pos) |
|---|---|---|---|---|
| 1,1 | 0.501 | 0.861 | 0.98, 0.98, 0.98 | 0.013 |
| 1,2 | 0.501 | 0.861 | 0.98, 0.98, 0.98 | 0.025 |
| 1,5 | 0.278 | 0.778 | 0.98, 0.98, 0.03 | 0.027 |
| 1,10 | -0.192 | 0.583 | 0.21, 0.14, 0.10 | 0.061 |
Plain (1,1) wins; weights hurt once training is real. Key finding: the 3 D39V positives
score 0.98 while the TIGR4 positive scores ~0.02 — **no cross-strain transfer** with this
data. Runs: `sigx/runs/sigx_v2e10_w*_2ep`.

## Probing (retired: `--freeze-backbone` reverted, frozen embeddings lack the boundary)
| preset | test MCC | test AUC | P(4 pos) |
|---|---|---|---|
| 1,1 | -0.386 | 0.333 | 0.49, 0.49, 0.52, 0.50 |
| 1,18.6 | 0.000 | 0.361 | 0.51, 0.51, 0.54, 0.52 |
Frozen embeddings do NOT separate (all P≈0.50) — worse than full-finetune (MCC 0.50).
Lesson: the tiny backbone shift of full-finetune (Δ 0.01) was load-bearing after all;
frozen general features lack the SigX/SigA boundary. Runs: `sigx/runs/sigx_v2probe_*`.

## Generalization gaps (train-vs-test ROC, `scripts/generalization_gap.py`, 28 PNGs in `sigx/reports/`)
| stage | gap AUC range | gap MCC range | reading |
|---|---|---|---|
| S1 (test clean) | +0.16 → +0.43 (grows with weight) | 0 (0.73 at w18.6 = train memorized) | train fits, test flat |
| S2 (eval clean) | -0.19 → -0.04 | 0 → -0.14 | negative = test easier than train (n=62 noise) |
| S3 (eval clean) | -0.12 → -0.03 | 0 → -0.14 | same, no overfit signal beyond noise |
| P (test clean) | +0.17 → +0.76 | 0 → +0.47 | train memorized, test fails = classic overfit |
| A1/full (overlap) | +0.06 → +0.70 | 0 → +0.61 | memorization incl. overlap rows |
| v2e10 (test clean) | +0.14 → +0.41 | +0.50 → +1.11 | 55-row train memorized (MCC~0.9), test partial |

## Energy sample-weights (SantaLucia profiles as per-sample loss weights, NOT features)
Heuristic prior (documented): w = clip(1+|z|, 1, 3), z = standardized mean duplex-stability
over the Pribnow window (k-mers 40–55) vs train global. Energy carries class signal
(max per-position Δ 0.57 vs pooled std 0.37 near kmer 47). Mechanism: `WeightedTrainer`
unified loss (class × focal × sample weights, sum-normalized) + `--sample-weight-csv`
(id-matched, order-asserted). Weights: `sigx/energy_weights_v2.csv` (pos-mean 1.83 vs neg 1.70).
| setting (v2-test) | MCC | AUC | P(4 pos) |
|---|---|---|---|
| CE plain | 0.501 | 0.861 | 0.98 ×3, 0.03 |
| energy weights | 0.639 | 0.806 | 0.98 ×3, 0.01 |
| focal γ=1.0 | 0.822 | 0.889 | 0.81–0.92, 0.06 |
| energy × focal γ=1.0 | 0.822 | 0.889 | 0.84, 0.83, 0.83, 0.10 |
Energy beats plain; energy+focal ties focal alone (0.822); TIGR4 missed by all. Runs: `sigx/runs/sigx_v2ew_2ep`, `sigx/runs/sigx_v2_ewf10_2ep`.
Stacking check (logistic on [seqP, energy], fit eval → test): seq 0.889 = stacked 0.889,
energy alone 0.778 (TIGR4 ranked 6/13). Zero marginal value: energy signal is redundant
with sequence scores. Untried: concat at representation level, hard negatives by energy,
other energy tables.

## TIGR4 SigX probe (6 v2 sequences + 278 SigA background, all models)
| model | SigX P (6) | n≥0.5 | SigA-bg mean |
|---|---|---|---|
| S1-w18.6 (best S1) | 0.48, 0.41, 0.41, 0.41, 0.48, 0.37 | 0/6 | 0.397 |
| S2-w18.6 (best S2) | 0.50, 0.47, 0.48, 0.47, 0.49, 0.46 | 0/6 | 0.467 |
| S3-w18.6 (best S3) | 0.53, 0.45, 0.46, 0.45, 0.52, 0.42 | 2/6 | 0.438 |
| S3-w1_1 | 0.07 ×6 | 0/6 | 0.070 |
| replica/mix/promoter/HF (`P(promoter)` head) | 0.69–1.00 | 5–6/6 | 0.94–0.99 |
Reading: promoter-heads confirm all 6 are strong promoters (different task, expected).
SigX-heads show dose-response but do NOT separate from SigA background at decision
level (S2-w18.6: SigX ~0.48 vs bg 0.467) — 15–33 positives cannot buy separation.

## What NOT to do
No SMOTE on sequences (refuted with evidence); no oversampling with replacement
(duplicates 21 → overfit); no threshold tuning on test; no CIs from n=3.

## Threshold-moving (eval-Youden thr → test MCC; systematic, 32 runs)| runs | @0.5 | @you | verdict |
|---|---|---|---|
| S1 (v1-test) | 0.00 ×3, -0.03 | 0.08–0.25, -0.03 | helps a little (weak ranking) |
| S2 (v1-eval) | 0.00 ×3, 0.65 | 0.38–0.81 | helps a lot (good ranking, biased thr) |
| S3 (v1-eval) | 0.00, 0.00, 0.81, 0.63 | 0.36–0.81 | helps or ties, never hurts much |
| P / A1 / full (overlap) | ≤0 | ≤0 mostly | can't fix inverted ranking |
| v2e10 (v2-test) | 0.50, 0.50, 0.28, -0.19 | 0.43, 0.82, 0.64, 0.00 | helps (0.50→0.82 best case) |
| v2probe (v2-test) | -0.39, 0.00 | -0.39, -0.18 | no ranking to move along |
| v2focal (v2-test) | 0.82, 0.64 | 0.82, 0.82 | ties/leads at the top |
Rule: threshold-moving rides an existing ranking (AUC ≳ 0.8 → helps); it cannot create
one (AUC ≲ 0.5 → nothing). Preferred over reweighting: no retraining, eval-based (honest).

## LOOCV-21 (WS 5090, mini base w18.6; 15 D39V-train + 6 TIGR4 held out one by one)
Design: pool 21 (NOT 27 — the 6 sigx-eval/test positives stay out so selection stays
clean); negatives fixed 272; checkpoints deleted per run (table only).
Result: recall **21/21 = 1.000**, P band 0.505–0.569 (TIGR4: 0.505–0.542).
Verdict: TRIVIAL — mean eval MCC across runs is 0.06; w18.6 pushes everything over 0.5,
hits and background alike. Recall without specificity is not evidence. SigX stays negative.
Also fixed en route: `datasets/d39v_binary_siga_sigx/all.csv` split labels were stale
(row set identical, verified diff 0); rebuilt from split files + MANIFEST note.
