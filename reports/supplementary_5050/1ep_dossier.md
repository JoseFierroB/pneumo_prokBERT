# Dossier 1ep — consolidated reference (2026-10-16)

One epoch of fine-tuning as: gap floor, stability probe, rare-sigma specialist,
and deployment floor. All numbers seed-42 mix high5050 unless noted; CIs where measured.

## 1. Gap (the solid claim)
- gap-AUC 223k **+0.0254 [0.0207, 0.0298]** (train80 n=178473 + test_10 n=22310,
  independent bootstrap B=2000 per side). Lowest of all configs; CI excludes 2ep.
- gap-AUC pneumo +0.0097. pAUC gaps (FPR<0.1/<0.2): 0.038/0.035 (1.5-1.8x total gap).
- gap-MCC +0.060 (thr artifact; collapses to ~+0.04 at val-thr).
- ROC: `gaps/roc_gap__gap1ep-223k.png`, `gaps/roc_gap__gap1ep-pneumo.png`.

## 2. Test performance (the floor)
- paper-test 0.593/0.8844 [0.8802,0.8887]; pneumo-test 0.660/0.9417;
  sigma70 0.719/0.935; multispecies 0.593/0.882.
- Lowest useful everywhere except SigX. 2ep beats it paired (HF 5-seed, CV 5/5, FN-regime).

## 3. Threshold structure
- argmax-MCC on pneumo-val: thr **0.94**, band [0.88,0.97] (bootstrap).
  test pneumo 0.660->0.820 (prec 0.900, rec 0.857).
- No transfer to paper (0.593->0.535). Per-domain operating points required.
- Per-strain: D39V 0.97 (0.92 MCC), TIGR4 0.92 (0.69 MCC).

## 4. Calibration
- 1ep does NOT need temperature scaling: T=1.403 fitted on val-NLL worsens ECE
  both sides (val 0.164->0.182, test 0.033->0.042). 1ep-test already well calibrated
  (0.033 < 2ep 0.047 pre-scaling). Less training = less overconfidence.

## 5. Error anatomy (MCC@0.5)
- paper-test CM: TN 8988 FP 2096 FN 2445 TP 8781 (TPR 0.782, FPR 0.189).
- pneumo-test CM: TN 332 FP 88 FN 20 TP 190 (TPR 0.905, FPR 0.210).
- Deployment problem = FPs (1 in 5 negatives), not FNs. thr 0.94 fixes FPs (prec 0.90) at rec 0.86.
- TIGR4 rec 0.78 vs D39V 0.97; FNs lack -10/sigma; head-species recall 0.62-0.70
  (exposure deficit, not signal deficit).
- Heatmap: `curves_1ep/cm__1ep.png`.

## 6. Stability
- Frozen harness 5-seed: test MCC 0.590+-0.0081 (PASS <0.01, barely), AUC 0.882+-0.004.
- Mix s42/s43: 0.593/0.603. CV pneumo 5-fold: 0.753+-0.045 (fold dominates).
- do005@1ep WITHDRAWN as claim (2-seed mean loses + head-recall losses).

## 7. SigX exception (the one regime 1ep dominates)
- 4197-set: SigX-prop 1.000, SigX-all 0.991 (n=21 D39V; n=27 with TIGR4-ComX dupes).
  Benchmark 0.914/0.932, 2ep 0.991/0.986. Less training = better rare-sigma generalization.
- Plots: `sigma/roc_4197__SigX-{prop,all}.png` (no titles, 1-specificity axis).

## 8. Training dynamics
- Curves in `curves_1ep/`: late-epoch peaks (max MCC@0.92ep, AUC@0.99ep), no early plateau.
- Dropout overlay: trajectories nearly identical (dropout moves endpoints, not dynamics).
- Schedule duel (paper recipe): ratio-0.1 > fixed-500 (dAUC +0.0035 SIG paper/multi).

## Verdict
1ep = gap floor + metrology anchor + SigX specialist. Not deployable as general model
(loses every paper-test regime incl. FN-critical vs 2ep/3ep). Its role in the gap story:
the "small gap, lower test" endpoint of the tradeoff curve, with CIs.
