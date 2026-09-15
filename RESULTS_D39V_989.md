# D39V 989 full dedup 1973 1381/296/296 — 5090 solo cuda:0

**Dataset** `positives 989 (973 uniq) + negatives 1000 =1973 dedup100` 16 dups pos, 70/15/15 stratify `train1381 700/681 eval296 150/146 test296 150/146` 81bp

**Finetuning** `train_d39v_989.py` fidelidad oficial k6s1 fp16 5090, default 3ep lr5e-05 + optuna 20 (lr1e-5-1e-4, wd0-0.3, warmup0-0.2, epochs3-6)

| Modelo | Hparams | Eval296 mcc/acc/auc | Test296 mcc/acc/auc | Tiempo |
|---|---|---|---|---|
| `d39v_full_989__default_final` | 3ep lr5e-05 | **0.9602 /0.979 /0.991** | 0.8254 /0.912 /0.972 | 30s |
| `d39v_full_989__optuna_best_final` | 4ep lr2.69e-05 wd0.06 warmup0.04 (best 0.9667) | **0.9667 /0.983 /0.989** | **0.8581 /0.929 /0.973** | 6ep +20 trials |

*Nota:* Eval 0.96 > Test 0.82-0.85 gap indica leve overfit eval, pero test 0.85 supera baseline 0.823 D39V_full 1989. Optuna mejora test 0.825→0.858.

**Comando**
```bash
ssh ws; cd ~/pneumo_prokBERT
CUDA_VISIBLE_DEVICES=0 pixi run python train_d39v_989.py --trials 20
CUDA_VISIBLE_DEVICES=0 pixi run python eval_989_local.py
```
