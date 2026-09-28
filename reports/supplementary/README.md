# Supplementary (provenance for the promoted model)

## promoter-d39v-tigr4 — PROMOTED to the matrix 2026-09-28
- **Recipe**: `finetuning.py --model-name neuralbioinfo/prokbert-mini-promoter --dataset-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/promoter_d39v_tigr4_2ep --cpu` (mini-promoter init, fp32, 2ep bs128 lr4e-4 wd0.01 warmup 500, CPU).
- **run_info**: `finetuning_outputs/promoter_d39v_tigr4_2ep/run_info.json` → eval 0.8464/0.9490, test 0.8667/0.9474 (MCC/AUC, dedup eval/test).
- **4-column eval** (`scripts/eval_single.py`): `promoter-d39v-tigr4.json` (merged into `reports/model_matrix/metrics.json` + `finetuning_outputs/eval_matrix_probs/` on promotion; same `cell_metrics`, same labels; per-column CSVs deleted as matrix duplicates).
- **Leakage**: d39v/tigr4/mix/dedup = `train_overlap` (trained on dedup); paper-test/paper-val = `train_overlap` (upstream init trained on 223k); sigma70/multi = `clean`.
Evaluated with the same evaluator and metrics as the matrix; part of it since the 5×4 cleanup (`models.json`, `datasets_eval.json`). Old overlays deleted (duplicated `reports/model_matrix/plots/`).
