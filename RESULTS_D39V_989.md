# D39V 989 full dedup 1973 1381/296/296 — 5090 only cuda:0

**Dataset** `positives 989 (973 unique) + negatives 1000 =1973 dedup100` 16 dups in positives, 70/15/15 stratify `train1381 700/681 eval296 150/146 test296 150/146` 81bp

**Fine-tuning** `train_d39v_989.py` paper-faithful k6s1 fp16 5090, default 3 epochs lr5e-05 + optuna 20 (lr1e-5-1e-4, wd0-0.3, warmup0-0.2, epochs3-6)

| Model | Hparams | Eval296 mcc/acc/auc | Test296 mcc/acc/auc | Time |
|---|---|---|---|---|
| `d39v_full_989__default_final` | 3ep lr5e-05 | **0.9602 /0.979 /0.991** | 0.8254 /0.912 /0.972 | 30s |
| `d39v_full_989__optuna_best_final` | 4ep lr2.69e-05 wd0.06 warmup0.04 (best 0.9667) | **0.9667 /0.983 /0.989** | **0.8581 /0.929 /0.973** | 6ep +20 trials |

*Note:* Eval 0.96 > Test 0.82-0.85 gap indicates slight eval overfit, but test 0.85 exceeds baseline 0.823 D39V_full 1989. Optuna improves test 0.825→0.858.

**Command**
```bash
ssh ws; cd ~/pneumo_prokBERT
CUDA_VISIBLE_DEVICES=0 pixi run python train_d39v_989.py --trials 20
CUDA_VISIBLE_DEVICES=0 pixi run python eval_989_local.py
```
