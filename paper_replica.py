#!/usr/bin/env python3
"""
Paper replica / adaptation — ProkBERT fine-tuning aligned with the paper, from scratch with our sequences
- Based on tools/prokbert/examples/finetuning.py (133 lines, helper + ProkBERTConfig + training_utils)
  * Upstream file has a bug: `from prokbert.models import BertForBinaryClassificationWithPooling` does not exist
    (correct class is `ProkBertForSequenceClassification` in `prokbert/models/megatronbert.py:1710`), so it cannot be executed without a shim.
- This replica fixes that shim, removes colab parts (!pip, raise), and keeps the same official logic.
- Minimal adaptation: HF dataset bacterial_promoters (223k) -> our own 81 bp pneumo CSVs
  (D39V+TIGR4 dedup100 3135 or D39V 989) keeping the official BertForBinaryClassificationWithPooling (via ProkBertForSequenceClassification)
"""
import os
import sys
sys.path.insert(0, "./prokbert")
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer

from prokbert import helper
from prokbert.config_utils import ProkBERTConfig
from prokbert.training_utils import (
    get_default_pretrained_model_parameters,
    get_torch_data_from_segmentdb_classification,
    compute_metrics_eval_prediction,
    check_nvidia_gpu,
)
from prokbert.models import BertForBinaryClassificationWithPooling
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT

seed = 42
helper.set_seed(seed)

def load_pneumo_csv_as_segmentdb(csv_path):
    """
    Load pneumo CSV (segment,y,id,sigma,strain,split) and convert to the official segmentdb format
    expected by get_torch_data_from_segmentdb_classification:
    columns: segment, segment_id, sequence_id, y, label
    """
    df = pd.read_csv(csv_path)
    # Official segmentdb uses segment_id and sequence_id, label = y
    df["segment_id"] = df["id"].astype(str)
    df["sequence_id"] = df["id"].astype(str)
    df["label"] = df["y"].astype(int)
    # Ensure required columns
    # segment is already 81 bp, y is int
    # Some functions expect 'segment' not 'segment_id' for tokenization, but get_torch... uses segmentdb[segment]
    # sequtils.batch_tokenize_segments_with_ids uses segmentdb['segment'] and segment_id
    return df

def main():
    check_nvidia_gpu()
    # --- Official config: ProkBERTConfig with defaults from pretraining.yaml + sequence_processing.yaml ---
    prokbert_config = ProkBERTConfig()
    # Override model: use local HF mini 6/1 (same weights as pneumo_prokBERT base)
    # Equivalent to --model_name neuralbioinfo/prokbert-mini
    prokbert_config.get_and_set_model_parameters({"model_name": "neuralbioinfo/prokbert-mini"})
    # Official tokenization k6s1 (already default in config)
    prokbert_config.get_and_set_tokenization_parameters({"kmer": 6, "shift": 1})
    # Fine-tuning params: same as our 5090 best Optuna, but via ProkBERTConfig
    # For fidelity, adjust pretraining_params that Trainer will use
    # NOTE: examples/finetuning.py uses pretraining_params for Trainer, not finetuning_params directly
    # Minimal adjustment here: 6 epochs, bs16/32, lr7.7e-05, wd0.04, warmup via TrainingArguments directly
    # Keep compatibility: set via pretraining_params
    prokbert_config.pretraining_params["output_dir"] = "./finetuning_official_out"
    prokbert_config.pretraining_params["per_device_train_batch_size"] = 16
    prokbert_config.pretraining_params["per_device_eval_batch_size"] = 32
    prokbert_config.pretraining_params["num_train_epochs"] = 6
    prokbert_config.pretraining_params["learning_rate"] = 7.747486422728904e-05
    prokbert_config.pretraining_params["weight_decay"] = 0.040280909839829605
    prokbert_config.pretraining_params["warmup_ratio"] = 0.12725631995134956
    prokbert_config.pretraining_params["eval_strategy"] = "epoch"
    prokbert_config.pretraining_params["save_strategy"] = "epoch"
    prokbert_config.pretraining_params["load_best_model_at_end"] = True
    prokbert_config.pretraining_params["metric_for_best_model"] = "mcc"
    prokbert_config.pretraining_params["greater_is_better"] = True
    prokbert_config.pretraining_params["report_to"] = "none"
    prokbert_config.pretraining_params["seed"] = 42
    prokbert_config.pretraining_params["logging_steps"] = 10
    prokbert_config.pretraining_params["save_total_limit"] = 2
    prokbert_config.pretraining_params["dataloader_num_workers"] = 0
    # 5090 Blackwell: fp16 auto
    USE_CUDA = torch.cuda.is_available()
    prokbert_config.pretraining_params["fp16"] = USE_CUDA
    prokbert_config.pretraining_params["bf16"] = False
    print("Pretraining params (Trainer):", {k: prokbert_config.pretraining_params[k] for k in ["output_dir","per_device_train_batch_size","num_train_epochs","learning_rate","weight_decay"]})

    # --- Official model: BertForBinaryClassificationWithPooling ---
    model_name_path = prokbert_config.model_params["model_name"]
    print(f"Model: {model_name_path}")
    pretrained_model, tokenizer = get_default_pretrained_model_parameters(
        model_name=model_name_path,
        model_class="MegatronBertModel",
        output_hidden_states=False,
        output_attentions=False,
        move_to_gpu=False,
    )
    fine_tuned_model = BertForBinaryClassificationWithPooling(pretrained_model)
    print(f"Tokenizer k={tokenizer.tokenization_params['kmer']} shift={tokenizer.tokenization_params['shift']} ready")

    # --- Own data from scratch: choose dataset ---
    # Option A: FULL dedup100 3135 (2194/470/471) D39V+TIGR4
    # Option B: SIGA 836 (668/84/84) D39V SigA+SigX pilot
    # Change here per experiment; default is FULL
    USE_FULL = True
    if USE_FULL:
        train_csv = "train_full_dedup100.csv"
        eval_csv = "eval_full_dedup100.csv"
        test_csv = "test_full_dedup100.csv"
    else:
        train_csv = "train_siga_sigx_668.csv"
        eval_csv = "eval_siga_sigx_84.csv"
        test_csv = "test_siga_sigx_84.csv"

    train_db = load_pneumo_csv_as_segmentdb(train_csv)
    eval_db = load_pneumo_csv_as_segmentdb(eval_csv)
    test_db = load_pneumo_csv_as_segmentdb(test_csv)
    print(f"Train {len(train_db)} Eval {len(eval_db)} Test {len(test_db)}")
    print(train_db["y"].value_counts().to_dict())

    # --- Create official torch datasets ---
    print("Processing train database!")
    X_train, y_train, torchdb_train = get_torch_data_from_segmentdb_classification(tokenizer, train_db)
    print("Processing eval database!")
    X_val, y_val, torchdb_val = get_torch_data_from_segmentdb_classification(tokenizer, eval_db)
    print("Processing test database!")
    X_test, y_test, torchdb_test = get_torch_data_from_segmentdb_classification(tokenizer, test_db)

    train_ds = ProkBERTTrainingDatasetPT(X_train, y_train, AddAttentionMask=True)
    val_ds = ProkBERTTrainingDatasetPT(X_val, y_val, AddAttentionMask=True)
    test_ds = ProkBERTTrainingDatasetPT(X_test, y_test, AddAttentionMask=True)
    print(f"Datasets torch: train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")

    # --- Official Trainer ---
    training_args = TrainingArguments(**prokbert_config.pretraining_params)
    trainer = Trainer(
        model=fine_tuned_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_eval_prediction,
    )
    print("=== OFFICIAL Training from scratch ===")
    trainer.train()
    print("=== Eval ===")
    print("eval:", trainer.evaluate(val_ds))
    print("test:", trainer.evaluate(test_ds))
    final_model_output = "./finetuning_official_final"
    fine_tuned_model.save_pretrained(final_model_output)
    print(f"Saved to {final_model_output}")

if __name__ == "__main__":
    main()
