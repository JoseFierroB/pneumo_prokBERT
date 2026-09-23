# hfpromoter-baseline

- **Init:** `neuralbioinfo/prokbert-mini-promoter` @ `8502f02d1019`
- **Status:** exists
- **Train data:** None
- **Script:** `scripts/eval_matrix.py (inference only)`
- **Artifact:** `HF hub, pinned rev`
- **sha256:** `pending`

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.9239 | 0.8480 | 0.9791 | 0.9817 | 0.9266 | 0.8535 | 0.605 | train_overlap |
| ecoli-sigma70 | 1864 | 0.8509 | 0.7078 | 0.9373 | 0.9324 | 0.8659 | 0.7301 | 0.870 | clean |
| multispecies | 22582 | 0.7950 | 0.5908 | 0.8810 | 0.8867 | 0.8021 | 0.6080 | 0.761 | clean |
| d39v | 1989 | 0.8859 | 0.7787 | 0.9700 | 0.9780 | 0.9487 | 0.8980 | 0.974 | clean |
| tigr4-high | 1476 | 0.8306 | 0.6654 | 0.9209 | 0.9416 | 0.8963 | 0.7991 | 0.984 | clean |
| d39v-tigr4 | 3465 | 0.8623 | 0.7303 | 0.9493 | 0.9628 | 0.9255 | 0.8531 | 0.974 | clean |
| d39v-tigr4-dedup | 3135 | 0.8510 | 0.7102 | 0.9417 | 0.9507 | 0.9206 | 0.8402 | 0.974 | clean |
