# Resultados finetuning pneumo_prokBERT — 5090 solo cuda:0

## 1. HF oficial bacterial_promoters (test_sigma70 1864, E.coli σ70) vs nuestros modelos pneumo

**HF es multi-especie (223k train) con σ70 E.coli, no pneumo.** Nuestros finetunes D39V pierden generalización a HF (AUC 0.44-0.53, MCC negativo), esperado por domain shift. BASE HF mini tampoco supera 0.44 en nuestro eval 5090 k6s1 (mismatch o HF test no es nuestro target).

| modelo | HF sigma70 1864 mcc/acc/auc/f1 |
|---|---|
| BASE_mini (local) | -0.14 /0.42 /0.44 /0.55 |
| TUNE_668_default SigA+SigX 836 3ep | -0.04 /0.47 /0.48 /0.51 |
| TUNE_668_optuna 836 6ep | -0.04 /0.46 /0.48 /0.50 |
| TUNE_full_dedup100 3135 6ep | -0.06 /0.45 /0.49 /0.55 |
| TUNE_3way SigA/SigX/None 973 | 0.00 /0.53 /0.53 /0.00 |
| TUNE_SigA_Neg 1390 | -0.07 /0.45 /0.47 /0.47 |

*Archivo:* `results/hf_eval/sigma70_test1864.csv` — 1864 test_sigma70, 5090 fp16.

**Conclusión:** Para HF usar modelo base sin finetune pneumo; para pneumo D39V/TIGR4 usar finetunes D39V (ver abajo).

## 2. Finetuning pneumo propios (5090 solo cuda:0, fp16, WeightedTrainer para desbalance)

**Datasets 81bp [-60,+20] dedup 100% D39V, nombres claros:**

- `datasets/d39v_3way_SigA_SigX_None` 973 (681/146/146) 3way 0:SigA 390 1:SigX 21 2:None 562
- `datasets/d39v_binary_SigA_vs_ComX` 411 (287/62/62) binary 0:SigA 1:ComX 19:1
- `datasets/d39v_binary_SigA_vs_Neg` 1390 (973/208/209) binary 0:Neg 1:SigA
- `train_full_dedup100.csv` 3135 (2194/470/471) D39V+TIGR4 high dedup100
- `train_siga_sigx_668.csv` 836 (668/84/84) SigA+SigX piloto

**Modelos 5090:**

| modelo | datos | hparams | test mcc/acc/auc |
|---|---|---|---|
| d39v_3way default | 3way 973 3ep lr5e-05 | 8s | 0.13/0.58/0.65 |
| d39v_3way optuna_best (20 trials) | 3way | 40s | **0.36/0.65/0.74** |
| SigA_ComX default | 411 class_weights | 3.3s | 0.56/0.96/0.71 |
| SigA_ComX optuna | 411 | 30s | **0.64/0.96/0.75** |
| SigA_Neg default | 1390 | 5.4s | **0.89/0.95/0.95** |
| SigA_Neg optuna | 1390 4ep | 50s | 0.86/0.94/0.94 |
| FULL dedup100 6ep | 3135 | 26.5s | 0.85/0.92/0.95 |
| SigA+SigX 836 3ep | 836 | 5.3s | 0.92/0.96/0.99 |
| SigA+SigX optuna 6ep | 836 | 40s | 0.92/0.96/0.99 |

Carpeta `results/finetuning/`: trials csv (20 por dataset), `models/*_final` (79M, no pusheado), `datasets/` CSVs.

**Comandos reproduce:**
```bash
ssh ws; cd ~/pneumo_prokBERT
CUDA_VISIBLE_DEVICES=0 pixi run python train_d39v_all.py --all --optuna --trials 20
CUDA_VISIBLE_DEVICES=0 pixi run python eval_hf_sigma70.py
```
