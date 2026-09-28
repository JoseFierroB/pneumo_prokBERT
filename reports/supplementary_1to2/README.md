# Supplementary 1to2 (continues the mix on pneumo-only 1:2; NOT in the matrix)

## mini-223k-d39v-tigr4-1to2 (WS 5090)
- **Recipe**: `finetuning.py --model-name finetuning_outputs/mini_223k_d39v_tigr4_2ep/best --dataset-dir datasets/d39v_tigr4_high_1to2 --output-dir finetuning_outputs/mini_223k_d39v_tigr4_1to2_2ep --warmup-ratio 0.1` (2ep, 46 steps, fp16).
- **run_info**: test MCC 0.8362 / AUC 0.9676 (1to2 test, clean).
- **4-column eval** (`eval_single.py`, leakage from `mini-223k-d39v-tigr4` + dedup override): JSON + 4 CSVs here.
- **Finding**: pneumo-only continuation narrows the model — dedup 0.79→0.90 MCC but paper-test 0.59→0.43, sigma70 0.74→0.41, multispecies 0.58→0.43 (all clean). General promoter features partially forgotten.
