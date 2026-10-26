# Supplementary 5050 — index

Head-to-head + gap battery for `mini-223k-d39v-tigr4-high5050` (HF-train80 + pneumo
train) and replicas. Working log: `battery.md` (local-only, not versioned). Machine-readable: `gaps.json`
(MCC/AUC per config), `gap_cis.json` (gap-AUC with 95% CIs + pAUC gaps).

## Dossiers (read these first)
- `1ep_dossier.md` — 1-epoch reference: gap floor, calibration, error anatomy, SigX.
- `battery.md` — full experiment log (E1–E14 + replica/CV/Hamming/temperature/soup/verdicts).
- `replica_verdict.json` — code-vs-paper recipe verdict + full-train + selection-vs-last.
- `ws_models_manifest.json` — 92 WS checkpoints (sha16 + run_info).

## Figures (all dpi=120 PNGs, no titles on ROCs by convention)
- `benchmark/` — `fig6_epochs__paper-test` (Acc/F1/MCC/Sens/Spec × 1/2/3ep+replica+benchmark*, boot CIs), `roc_benchmark__{paper-test,ecoli-sigma70,multispecies,dedup-all,5050-test}` (1/2/3ep+replica+base).
- `sigma/` — `roc_4197__Sig{A,X}-{prop,all}` (benchmark/1ep/2ep/replica, 1-specificity axis) + `sigma_roc_4197.json` (AUCs, CIs, clean-subset AUCs).
- `gaps/` — 25 train-vs-test ROCs (base 1/3/4ep, dropout@1ep/@3ep, subsets, ep15/ep25, mix-ES, s44).
- `curves_1ep/` — per-arm train curves (loss + val MCC/AUC vs epoch) + `overlay__auc.png` + `cm__1ep.png`.
- `cv/` — `cv_table.csv` (+`_1ep`), paired fold plots, `cv_folds.json`.

## Grids (model × column probs CSVs + per-alias JSON)
`grid_{1eprev,1epfull,do1ep,dropout,subset,ls,ensemble,adopt,full,mix3ep,cells,ep1,soup}`.
Convention since 2026-10-26 cleanup: JSONs + PNGs + `*paper-test.csv` (+`,val.csv` where it exists) stay;
other per-column CSVs archived to `~/pneumo_prokBERT_archive/<date>/` (regenerable via
`eval_single.py` from WS checkpoints).

## Promotion history (preserved)
`mini-223k-d39v-tigr4-high5050` vs `mini-223k-d39v-tigr4`, identical protocol (2ep, seed 42):
fair slices (both-clean) 5050-test Δ+0.084 SIG, paper-test Δ+0.022 SIG, multispecies Δ+0.032 SIG.
Rule met → **PROMOTED to primary mix model**. Numbers: `h2h.json`.
Builder: `scripts/build_high5050.py`.
