# promoter-baseline

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
| d39v-tigr4-dedup | 3135 | 0.8510 | 0.7102 | 0.9417 | 0.9507 | 0.9206 | 0.8402 | 0.974 | clean |
