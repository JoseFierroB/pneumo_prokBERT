# Evaluation Report — 5 Models × 5 Datasets (5090, k6s1, @0.5 / Youden)

Models: HF mini-promoter (baseline), Paper replica 80/10/10 2ep (223k), Mix 223k+d39v 2ep (225k), D39V from promoter 2ep (989), D39V base 989 paper 2ep (1973, base-only pneumo).
Datasets: `1_d39v 1989` (D39V 989+1000), `2_tigr4_high 1476`, `4_d39v_tigr4_high 3465` (naive), `dedup_3135 3135` (1399+1736 dedup), `test_sigma70 1864` (E.coli), `test_multispecies 22582` (33 species). Metrics @0.5 (Trainer/paper) and @Youden (benchmark). Data local `data/bacterial_promoters/*.parquet` 223k/1864/22582.

| Model | Dataset | N | ACC@0.5 | MCC@0.5 | AUC | ACC@Youden | MCC@Youden | Thr Youden |
|---|---|---|---|---|---|---|---|
| HF mini-promoter (baseline) | d39v_1989 | 1989 | 0.8859 | 0.7787 | 0.9700 | 0.9487 | 0.8980 | 0.974 |
| HF mini-promoter (baseline) | tigr4_1476 | 1476 | 0.8306 | 0.6654 | 0.9209 | 0.8963 | 0.7991 | 0.984 |
| HF mini-promoter (baseline) | mix_3465 | 3465 | 0.8623 | 0.7303 | 0.9493 | 0.9255 | 0.8531 | 0.974 |
| HF mini-promoter (baseline) | sigma70_1864 | 1864 | 0.8509 | 0.7078 | 0.9373 | 0.8659 | 0.7301 | 0.870 |
| HF mini-promoter (baseline) | multispecies_22582 | 22582 | 0.7950 | 0.5908 | 0.8810 | 0.8021 | 0.6080 | 0.761 |
| Paper replica 80/10/10 2ep (223k) | d39v_1989 | 1989 | 0.9080 | 0.8192 | 0.9761 | 0.9563 | 0.9137 | 0.937 |
| Paper replica 80/10/10 2ep (223k) | tigr4_1476 | 1476 | 0.8604 | 0.7216 | 0.9289 | 0.9085 | 0.8291 | 0.977 |
| Paper replica 80/10/10 2ep (223k) | mix_3465 | 3465 | 0.8877 | 0.7774 | 0.9563 | 0.9356 | 0.8749 | 0.935 |
| Paper replica 80/10/10 2ep (223k) | sigma70_1864 | 1864 | 0.8492 | 0.7052 | 0.9318 | 0.8600 | 0.7192 | 0.658 |
| Paper replica 80/10/10 2ep (223k) | multispecies_22582 | 22582 | 0.7934 | 0.5882 | 0.8788 | 0.7939 | 0.5906 | 0.529 |
| Mix 223k+d39v 2ep (225k) | d39v_1989 | 1989 | 0.9417 | 0.8834 | 0.9736 | 0.9517 | 0.9046 | 0.748 |
| Mix 223k+d39v 2ep (225k) | tigr4_1476 | 1476 | 0.8753 | 0.7514 | 0.9211 | 0.8977 | 0.8042 | 0.806 |
| Mix 223k+d39v 2ep (225k) | mix_3465 | 3465 | 0.9134 | 0.8270 | 0.9523 | 0.9284 | 0.8603 | 0.761 |
| Mix 223k+d39v 2ep (225k) | sigma70_1864 | 1864 | 0.8476 | 0.6965 | 0.9253 | 0.8627 | 0.7236 | 0.378 |
| Mix 223k+d39v 2ep (225k) | multispecies_22582 | 22582 | 0.7773 | 0.5604 | 0.8608 | 0.7783 | 0.5603 | 0.483 |
| D39V from promoter 2ep (989) | d39v_1989 | 1989 | 0.9326 | 0.8656 | 0.9759 | 0.9553 | 0.9115 | 0.955 |
| D39V from promoter 2ep (989) | tigr4_1476 | 1476 | 0.8686 | 0.7371 | 0.9239 | 0.9045 | 0.8157 | 0.968 |
| D39V from promoter 2ep (989) | mix_3465 | 3465 | 0.9053 | 0.8108 | 0.9540 | 0.9333 | 0.8697 | 0.966 |
| D39V from promoter 2ep (989) | sigma70_1864 | 1864 | 0.8578 | 0.7153 | 0.9378 | 0.8680 | 0.7352 | 0.780 |
| D39V from promoter 2ep (989) | multispecies_22582 | 22582 | 0.8030 | 0.6077 | 0.8830 | 0.8039 | 0.6109 | 0.545 |
| D39V base 989 paper 2ep (1973) | d39v_1989 | 1989 | 0.9331 | 0.8665 | 0.9665 | 0.9341 | 0.8685 | 0.500 |
| D39V base 989 paper 2ep (1973) | tigr4_1476 | 1476 | — | — | — | — | — | — |
| D39V base 989 paper 2ep (1973) | mix_3465 | 3465 | — | — | — | — | — | — |
| D39V base 989 paper 2ep (1973) | dedup_3135 | 3135 | 0.9116 | 0.8215 | 0.9411 | 0.9123 | 0.8228 | 0.500 |
| D39V base 989 paper 2ep (1973) | sigma70_1864 | 1864 | 0.4372 | -0.1026 | 0.4599 | 0.5483 | 0.0637 | 0.586 |
| D39V base 989 paper 2ep (1973) | multispecies_22582 | 22582 | 0.5085 | 0.0172 | 0.5295 | 0.5249 | 0.0587 | 0.444 |

## Notes
- HF baseline is zero-shot `neuralbioinfo/prokbert-mini-promoter` (no fine-tuning).
- Paper replica is `prokbert-mini` base fine-tuned on `bacterial_promoters train 223092` with paper `80/10/10 2ep bs128 lr0.0004 wd0.01 warmup500 betas 0.95/0.98 eps 5e-05`.
- Mix is `223k + d39v 1973 = 225k` fine-tuned 2ep same hyperparams.
- D39V from promoter is `mini-promoter` continual fine-tuned on `d39v_full_989 1973` 2ep.
- D39V base 989 is `prokbert-mini` base fine-tuned only on `d39v_full_989 1973` paper 2ep bs128 — pneumo-specific, collapses on E.coli (0.459/0.529 AUC) as expected, 0.966 on D39V, 0.941 on dedup 3135. Each in its own `finetuning_outputs/*/` folder (deletable).
- Youden = max(TPR-FPR) on test set (optimistic, for benchmark comparison).
- Data local: `data/bacterial_promoters/{train,test_sigma70,test_multispecies}.{parquet,csv,fasta}` 223k/1864/22582 + `datasets/d39v_full_989` 1973 and `dedup_3135` 3135.

## Model Paths
- `HF mini-promoter (baseline)`: `neuralbioinfo/prokbert-mini-promoter`
- `Paper replica 80/10/10 2ep (223k)`: `finetuning_outputs/paper_80_10_10_2ep/best`
- `Mix 223k+d39v 2ep (225k)`: `finetuning_outputs/mix_223k_plus_d39v_2ep/best`
- `D39V from promoter 2ep (989)`: `finetuning_outputs/d39v_from_promoter_2ep/best`
- `D39V base 989 paper 2ep (1973)`: `finetuning_outputs/d39v_989_paper_2ep/best` (22 steps 2.1s 5090, eval 0.959/0.918/0.980, test 0.912/0.824/0.955)
