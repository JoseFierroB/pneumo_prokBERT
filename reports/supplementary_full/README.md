# Supplementary full-train (HF-promoter analog: FULL 223k + pneumo; NOT in the matrix)

## mini-223kfull-1to2-dedup (WS 5090, `train_mix223k.py --full-train`)
- **Recipe**: full HF train (223092) + 1to2-train (2937) + dedup-train unseen-only rows (242; 1952 duplicates excluded) = 226271; select best by sigma70 MCC (external clean); paper hyperparams fp32 warmup 500.
- **Locked**: val_sigma70 + data_test + multispecies in `run_info.json`.
- **4-column eval**: paper-test 0.722 / sigma70 0.759 / multispecies 0.636 / dedup 0.849 (MCC; paper-test+dedup `train_overlap`).
- **vs A1** (paired bootstrap B=2000): multispecies +0.020 [+0.012,+0.029] SIG; sigma70 +0.021 n.s. (p=0.055).
- Best clean sigma70 (0.759) and multispecies (0.636) of all models; no in-dist holdout by design.
- **Per-split (post-hoc, no new training)**:

| split | MCC | AUC | flag |
|---|---|---|---|
| 1to2-train | 0.879 | 0.972 | overlap (fit) |
| 1to2-eval | 0.824 | 0.952 | clean |
| 1to2-test | 0.813 | 0.953 | clean |
| dedup-train | 0.861 | 0.969 | overlap (fit) |
| dedup-eval | 0.824 | 0.954 | clean |
| dedup-test | 0.820 | 0.952 | clean |
- Overlays vs A1 vs HF baseline (ROC/PR per column): `plots/`.
- Train fit (post-hoc forward 226271 rows, WS): AUC 0.9363 / MCC 0.7263 / ACC 0.8624 — at/below val (sigma70 0.945), no overfit.
  - by component: hf-train 0.9355/0.724 (n=223092); 1to2-train 0.9720/0.879 (n=2937); dedup-unseen n=242 all-negatives (no AUC possible).
- Clean dedup slices (0 overlap by construction): eval MCC 0.824 / test MCC 0.820 — below pneumo specialists (promoter 0.867, mini-d39v 0.848 on dedup-test).
