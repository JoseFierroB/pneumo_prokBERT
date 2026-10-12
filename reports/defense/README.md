# Defense pack — evidence per requested item (all paths repo-relative)

## 1. Generalisation error train-vs-test (ROC/AUC before any comparison)
- Overlays: `reports/model_matrix/plots/roc_gap__*.png` (6 matrix) + `comx/reports/roc_gap__*.png` (28 ComX).
- Frontier (all models × datasets, overlap hatched): `reports/model_matrix/plots/frontier__<dataset>.png` (one file per dataset, full alias names; `scripts/plot_frontier.py`). The dedup panel uses the clean dedup-TEST slice recomputed from saved scores (dedup-all mixes train rows in).
- Verdict + table: `reports/generalization.md` (Measured train-vs-test gaps).
- Train AUCs: `run_info.json` locked_eval + `reports/supplementary_full/README.md` (full-train post-hoc 0.936).
- Reproduce: `scripts/generalization_gap.py` (train CSVs/HF-train80 vs saved test scores).
- Headlines: matrix clean gaps +0.03–0.05 (healthy); full-train gap ≈ 0 = memorization declared (overlap);
  ComX fresh ~0/noise; committed inits +0.4–0.7 = documented failure, not results.

## 2. Plots per sigma factor
- Strips + recall + ROC-vs-neg: `reports/supplementary/plots_sigma/` (32 PNGs, 6 models; d39v + tigr4-high sets + joint pool strips `strip__joint__*.png`).
- **Clean-only rule**: rows seen by ANY plotted model's trains are excluded everywhere (d39v 575/1989, tigr4 476/1476 clean); empty groups (e.g. joint SigX n=0) render explicitly as data limits, not hidden.
- Only relevant groups plotted (SigA family + SigX/ComX); rest excluded by design (`RELEVANT` in `plot_sigma.py`; neg kept internally as ROC reference).
- Tables: `reports/sigma_imbalance/siga_vs_sigx.{json,md,tsv}` (5 models; tigr4 = 278 SigA vs 6 ComX).
- Tables: `reports/sigma_imbalance/siga_vs_sigx.{json,md,tsv}` (5 models; tigr4 = 278 SigA vs 6 ComX).
- Code: `scripts/eval_sigma.py` (sets) + `scripts/plot_sigma.py` (strips; appendix cols on explicit request).
- Data: `datasets/reference/{d39v,tigr4-high}/*.fasta` + `datasets/d39v_full_989/all.csv` (tags) + `datasets/external/` (v2 metadata).
- Reading: ComX (n=6) and SigA (n=278) both at P≈1.0 — indistinguishable; `unassigned` also ≈1.0.

## 3. Twice as many negatives as positives (1:2)
- Unified: `datasets/d39v_tigr4_high_1to2/` (4197 = 1399+2798, MANIFEST + Counter-verified; `scripts/build_unified_1to2.py`).
- Plus-dedup: `datasets/d39v_tigr4_1to2_plus_dedup/` (1to2 + 242 dedup-unseen, zero internal overlap; MANIFEST).
- ComX v2: `datasets/d39v_tigr4_comx_1to2/` (27+54 subsampled, MANIFEST + overlap audit, 1 dup removed; `comx/build_dataset_v2.py`).
- Results: A1 paper-test 0.621 / multi 0.616 (SIG vs mix); split+dedup multi +0.009 SIG (diminishing returns);
  full-train sigma70 0.759 / multi 0.636 (SIG); v2e10 MCC 0.50, focal-γ1 0.82, threshold 0.82
  (`comx/README.md`, `reports/supplementary_1to2/`, `reports/supplementary_full/`).
- Canonical stack declared: README Data (HF-223k + dedup + 1:2, plain CE); naming: MODELS.md Naming.

## 4. Cross-validation
- LOOCV-21 executed (WS 5090, mini base w18.6, 15 D39V + 6 TIGR4 held out one by one,
  checkpoints deleted): recall 21/21 = 1.000, P band 0.505–0.569 — TRIVIAL
  (mean eval MCC 0.06; weight pushes everything over 0.5). Record: `comx/README.md` (LOOCV-21).
- Paired bootstrap B=2000 (reproducible): `scripts/paired_bootstrap.py` →
  `reports/model_matrix/bootstrap_cis.json` + `significance.tsv` (first clean SIG wins: A1/full).

## 5. Class imbalance, S. pneumoniae sigma factors
- Audit: `reports/sigma_imbalance/descriptives.{json,md}` (fragile 34, singletons 8,
  competence pool 27 = 21 D39V SigX + 6 TIGR4 ComX v2) + `sigma_{counts,gc,fragile}.tsv`.
- Tag source: `datasets/d39v_full_989/` (appendix) + `datasets/external/` (v2 metadata, 738 rows).

## 6. Energies (a la MLDSPP)
- Verdict: `energy/README.md` (E0/fusion parked: redundant, gate 0.0012).
- As sample weights (NOT features): `comx/energy_weights_v2.csv` + `--sample-weight-csv`
  (v2-test MCC 0.64; × focal-γ1 ties 0.82; stacking seq+energy = seq alone 0.889 → redundant).
