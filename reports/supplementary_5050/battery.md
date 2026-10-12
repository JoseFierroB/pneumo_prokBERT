# Battery E1–E8 — mini-223k-d39v-tigr4-high5050 (Lambda A10, one by one)

- E1 gaps: +0.096 (223k, train 0.710/test 0.615), +0.096 (pneumo, 0.812/0.716). Gray zone.
- E2 curves: no divergence; MCC-eval rises to ~0.63, no late decline; gradnorm normal.
- E3 calibration: T=1.47 thr=0.911 (fit eval); test MCC 0.716 -> 0.842. ECE base 0.12.
- E4 by source: CDS FPR 15% = IGR 16% (IGR FPs more confident 0.84 vs 0.71);
  D39V MCC 0.79 vs TIGR4 0.55 (FNR 25% TIGR4) — D39V-strong/TIGR4-weak.
- E5 drift vs init: 0.107 mean-rel (old mix 0.100) — identical; gap is data difficulty, not weights.
- E6 epochs (locked MCC): 1ep < 2ep < 3ep everywhere
  (test_10 0.593/0.615/0.643; data 0.660/0.716/0.737; multi 0.593/0.614/0.642).
- E7 train-seed 43: general +0.02 (paper 0.634, multi 0.634), pneumo -0.06 (data 0.659).
  Promotion margin seed-fragile on pneumo side, robust on general side.
- E8 dinuc-shuffled positives (true neg, n=630): meanP 0.19, median 0.07, 12.9% > 0.5.
  Motif learner, not composition. 6% > 0.8 residual tail.

Verdict: NOT overfit (underfit-leaning at 2ep; 3ep better everywhere).
Open problems: TIGR4 weakness, seed fragility (pneumo), ECE 0.12 (fixable post-hoc).
Follow-up proposed: 3ep as promoted config (needs matrix + h2h).

## Epoch config (E6b): 1<2<3>4, peak 3ep (matrix MCC)
- paper-test: 0.593 / 0.615 / 0.643 / 0.597
- multispecies: 0.593 / 0.614 / 0.642 / 0.592
- ecoli: 0.719 / 0.752 / 0.749 / 0.708
- 5050-test locked: 0.660 / 0.716 / 0.737 / 0.686 (clean-slice 3v2: tie,
  locked gap was contamination-driven)
- 3ep vs 2ep bootstrap: paper +0.028 SIG, multi +0.029 SIG, ecoli n.s.
- 4ep below 2ep everywhere: overfit cliff between 3 and 4.
Promoted config: 3ep (primary ★).

## E9: TIGR4 failures = non-canonical High-confidence TSS (3ep, 5050-test)
- Annotated sigma (SigA/SigX/SigX, n=82): FNR 0-1.4% both strains. Textbook promoters: perfect.
- Unannotated: D39V FNR 5% (4/82) vs TIGR4 FNR 33% (15/46, all Confidence=High).
- Misses lack -10 box: 5/20 (25%) vs hits 181/190 (95%).
Story: model learned canonical -10 promoters; TIGR4 non-canonical High TSS escape it.
Not a bug: training positives are canonical-dominated. Follow-up: SigX/non-canonical
augmentation (see sigx/ ceiling: n=27).

## E10: 3ep gaps (primary model)
- 223k: train 0.8097 / test 0.6431, gap +0.167 (crosses 0.15 line; 2ep was +0.096).
- pneumo: train 0.8679 / test 0.7369, gap +0.131 (gray-high; 2ep +0.096).
- best/ = checkpoint-3500/4254 (~2.5ep effective; val-selected, not forced 3ep).
Verdict: 3ep memorizes harder AND tests strictly better (paper/multi SIG).
At/past the gap boundary on 223k, but performance-justified; 4ep cliff is real
and adjacent. Promotion stands with this caveat.

## E11: 3ep calibration (fit 5050-eval, report 5050-test)
- T=1.675 thr=0.953 valMCC=0.851; test base 0.7369/ECE 0.105 -> +T+thr 0.8458.
- Same pattern as 2ep (T=1.47 thr=0.91 -> 0.842); higher T = more overconfidence.

## E12: FP anatomy (3ep, 5050-test, exact motifs)
- TP 43% exact TATAAT vs bg 6-7%; 57% TP without exact box (distributed).
- FP = background rate (7%) -> FPs NOT box-driven; other signal.
- FN 1/20 with TATAAT (concurs E9). TTGACA weak; TACGAA ~0 everywhere.

## E13: source ablation (2ep, locked + cross on 5050-test)
- cds-only: test_10 0.633, own-test 0.756, mixed-5050 0.674 (FPR IGR 27%).
- igr-only: test_10 0.611, own-test 0.719, mixed-5050 0.719 (FPR 14/17%).
- mix-50/50: 0.716 (FPR 15/16%). IGR negatives carry the signal; CDS-only
  does not generalize across sources. 50/50 kept (ties best, covers both).

## E14: 3ep vs specialist promoter-d39v-tigr4 (dedup-test)
- clean-185: spec 0.883 vs 3ep 0.789, d=+0.094 CI[+0.005,+0.182] SIG (tight).
- full-471: 0.867 vs 0.863 tie (3ep saw 286 rows in train).
- Specialist still king in-domain (clean); generalist ties only via contamination.

## Thresholds (3ep, 5050-test): operational curve
- thr 0.50: MCC 0.737, prec 0.76, rec 0.91 (canon 0.97 / noncanon 0.40), FPR 0.14
- thr 0.95: MCC 0.831, prec 0.90, rec 0.87 (canon 0.96 / noncanon 0.24), FPR 0.05
- Non-canonical recall caps at 0.44 (thr 0.3): thresholds cannot fix E9,
  they only trade precision/recall within the canonical world.
- Scale input: FPR 0.05-0.14 -> 50K-140K FP per 1M windows; gene-anchored
  (~4K/genome) -> ~200-560 candidates/genome needing downstream filter.

## E9b: FN anatomy WITHOUT annotation circularity (3ep, 210 pos, 20 misses)
Task is binary promoter/non-promoter; misses are plain FNs. Annotation-free traits:
- GC: hits 0.297 vs misses 0.387 (model is AT-loving; miss 3-mers: TCG/ACC/CGA/TGG/ACG/CGT).
- 5UTR length: hits 297 vs misses 582 (long leaders/internal TSS).
- TSS Score: hits 2838 vs misses 238 (12x weaker evidence, yet conf High).
- Strand: no bias. Strain: D39V hit-rate 0.967 vs TIGR4 0.746.
Clean story: AT-detector misses GC-rich, weak, long-UTR TSS, overrepresented in TIGR4.
No sigma labels needed; "no -10" analysis retired as circular.

## F3: position sensitivity (3ep, 151 D39V test positives, genome-anchored shifts)
- shift 0: recall 0.967 | ±5: 0.72/0.74 | ±10: 0.62/0.52 | ±20: 0.48/0.38 | ±40: 0.36/0.26
- Downstream (+) decays faster than upstream (-): upstream context rules.
- Scale rule: tiling step <=10bp (<=5bp safe) or gene-anchored candidates.
  (TSS at index 60 both strands; convention validated 151/151 vs D39V.fna.)

## F1: divergence robustness (3ep, 210 test positives, random substitutions)
- base 0.905 | 1%: 0.905 | 2%: 0.910 | 5%: 0.886 | 10%: 0.881. Flat.
- Clinical strain divergence (~1-5% core) does not degrade recall.

## Weights: per-layer drift + 2/3/4ep trajectory (rel. L2 vs init)
- Profile 3ep: L0=0.188 > L1=0.155 > L2=0.144 > L3=0.115 > L4=0.097 > L5=0.076,
  emb=0.146. Bottom moves most (k-mer-level retuning); top moves least.
  Head N/A: fresh classifier+weighting_layer (init had MLM head instead).
- Trajectory: 3ep > 2ep everywhere (more movement, better test);
  4ep < 3ep everywhere yet tests worse -> drift magnitude != performance;
  4ep wandered to a worse basin, not further along the same path.
- Near-duplicates (5050-test vs train): pos median Hamming 38/81 (min 1),
  neg min 12. Recall dist<10: 1.00 (13) vs dist>=10: 0.88: holds at distance.
  No memorization-via-near-dupe.
