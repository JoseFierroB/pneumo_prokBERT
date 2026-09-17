# D39V Finetuning ordenado — 3 datasets × (default + optuna 20) — 5090 solo cuda:0

**Fidelidad oficial** `finetuning_official.py` + `train_d39v_all.py` (WeightedTrainer, ProkBERTTokenizer k6s1, 5090 fp16)

## Datasets (81bp dedup, nombres claros)

| Dataset | N total (train/eval/test) | Clases | y | Split |
|---|---|---|---|
| `d39v_3way_SigA_SigX_None` | 973 (681/146/146) | 3way 0:SigA 390 1:SigX 21 2:None 562 | 0/1/2 | 70/15/15 stratify |
| `d39v_binary_SigA_vs_ComX` | 411 (287/62/62) | binary 0:SigA 390 1:ComX 21 | 0/1 | 70/15/15 |
| `d39v_binary_SigA_vs_Neg` | 1390 (973/208/209) | binary 0:Neg 1000 1:SigA 390 | 0/1 | 70/15/15 |

Todos `D39V 81bp [-60,+20]` dedup 100% (16 dups internos), sin fuga train-eval/test.

## Modelos (6) — nombres `models/<dataset>__<default|optuna_best>_final`

| Modelo | Datos | Hparams | Tiempo 5090 | Test mcc/acc/auc/f1 |
|---|---|---|---|---|
| `d39v_3way_SigA_SigX_None__default` | 3way 973 | 3ep lr5e-05 wd0 warmup0 bs16 | ~8s | 0.1306 /0.589 /0.658 /0.559 |
| `d39v_3way_SigA_SigX_None__optuna_best` | 3way 973 | **best trial** 20 optuna (mcc 0.36) | ~40s | **0.3627 /0.657 /0.744 /0.664** |
| `d39v_binary_SigA_vs_ComX__default` | binary 411 | 3ep lr5e-05 + class_weights [0.52,9.56] | 3.3s | 0.5678 /0.967 /0.717 /0.959 |
| `d39v_binary_SigA_vs_ComX__optuna_best` | binary 411 | optuna 20 best | ~30s | **0.6497 /0.967 /0.757 /0.967** |
| `d39v_binary_SigA_vs_Neg__default` | binary 1390 | 3ep lr5e-05 | 5.4s | **0.8932 /0.956 /0.955 /0.956** |
| `d39v_binary_SigA_vs_Neg__optuna_best` | binary 1390 | optuna 20 best lr1.5e-05 wd0.007 4ep | ~50s | 0.8683 /0.947 /0.944 /0.946 |

**Veredicto:** 
- 3way muy dificil (SigX 21) → optuna mejora 0.13→0.36 pero sigue bajo, necesita más SigX o oversampling adicional.
- SigA vs ComX desbalanceado 19:1 → WeightedTrainer mejora 0.56→0.64 con optuna.
- SigA vs Neg excelente 0.89 default ya supera optuna, modelo listo para deploy.

**Comandos**
```bash
ssh ws
cd ~/pneumo_prokBERT
CUDA_VISIBLE_DEVICES=0 pixi run python train_d39v_all.py --dataset d39v_3way_SigA_SigX_None        # default solo ese
CUDA_VISIBLE_DEVICES=0 pixi run python train_d39v_all.py --dataset d39v_3way_SigA_SigX_None --optuna --trials 20  # +optuna
CUDA_VISIBLE_DEVICES=0 pixi run python train_d39v_all.py --all --optuna --trials 20                # los 6
```
