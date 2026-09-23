# paper223k-replica

- **Init:** `neuralbioinfo/prokbert-mini` @ `feb2520a43cd`
- **Status:** exists
- **Train data:** hf_train_80_178473
- **Script:** `legacy:/tmp/paper_faithful_2ep.py (mirrored by paper_replica.py)`
- **Artifact:** `finetuning_outputs/paper_80_10_10_2ep/best`
- **sha256:** `ffb2fd34f2d498a140ad23e8be2419c1118559cd0fc5b77fcade64378887b85f`

## Results matrix (@0.5 / Youden)
| Dataset | N | ACC | MCC | AUC | AP | ACC@Y | MCC@Y | Thr | Leakage |
|---|---|---|---|---|---|---|---|---|---|
| paper-test | 22310 | 0.7961 | 0.5941 | 0.8800 | 0.8951 | 0.7968 | 0.5959 | 0.510 | clean |
| ecoli-sigma70 | 1864 | 0.8492 | 0.7052 | 0.9318 | 0.9266 | 0.8600 | 0.7192 | 0.658 | clean |
| multispecies | 22582 | 0.7934 | 0.5882 | 0.8788 | 0.8942 | 0.7939 | 0.5906 | 0.529 | clean |
| d39v | 1989 | 0.9080 | 0.8192 | 0.9761 | 0.9832 | 0.9563 | 0.9137 | 0.937 | clean |
| tigr4-high | 1476 | 0.8604 | 0.7216 | 0.9289 | 0.9533 | 0.9085 | 0.8291 | 0.977 | clean |
| d39v-tigr4 | 3465 | 0.8877 | 0.7774 | 0.9563 | 0.9708 | 0.9356 | 0.8749 | 0.935 | clean |
| d39v-tigr4-dedup | 3135 | 0.8785 | 0.7589 | 0.9494 | 0.9611 | 0.9321 | 0.8652 | 0.935 | clean |
