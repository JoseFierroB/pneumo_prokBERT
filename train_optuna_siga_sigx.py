#!/usr/bin/env python3
"""
Optuna finetuning ProkBERT SigA+SigX 668/84/84 - cambios MINIMOS sobre train_siga_sigx.py original
- Reusa tokenizer patch, tokenizacion y model load identicos
- Solo anade objective() + study Optuna para hiperparametros
- Datos propios 81bp, 836 seqs, sin tocar logica original
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
sys.path.insert(0, ".")
import torch
# 5090: respetar 4 threads host, Trainer usara cuda automaticamente
torch.set_num_threads(4)
print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} -> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
USE_CUDA = torch.cuda.is_available()
print(f"USE_CUDA={USE_CUDA}")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import numpy as np
import optuna
from pathlib import Path

set_seed(42)

# --- tokenizer patch identico a train_siga_sigx.py:28-52 ---
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
ProkBERTTokenizer.vocab_size = property(lambda self: len(self.get_vocab()) if hasattr(self, 'get_vocab') else 4101)
_orig_save = ProkBERTTokenizer.save_pretrained
def _patched_save(self, save_directory, **kwargs):
    from pathlib import Path
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    try:
        return _orig_save(self, save_directory, **kwargs)
    except NotImplementedError:
        (Path(save_directory) / "tokenizer_config.json").write_text('{"tokenizer_class":"ProkBERTTokenizer","kmer":6,"shift":1}')
        return ()
ProkBERTTokenizer.save_pretrained = _patched_save
tokenizer = ProkBERTTokenizer(tokenization_params={'kmer':6,'shift':1}, operation_space='sequence')
print("Tokenizer k=6 s=1 ready")

# --- data identico a train_siga_sigx.py:58-71 ---
ds = load_dataset("csv", data_files={
    "train": "train_siga_sigx_668.csv",
    "eval": "eval_siga_sigx_84.csv",
    "test": "test_siga_sigx_84.csv",
})
print(ds)
def tokenize_function(examples):
    enc = tokenizer(examples["segment"], add_special_tokens=True, padding=False, truncation=False)
    enc["labels"] = examples["y"]
    return enc
tokenized = ds.map(tokenize_function, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
print(tokenized)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return {
        "mcc": matthews_corrcoef(labels, preds),
        "accuracy": accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:,1]),
    }

# --- Optuna objective: solo hiperparametros, logica Trainer identica ---
# Hiperparametros a optimizar (rango conservador, no toca arquitectura):
# - learning_rate: 1e-5 .. 1e-4 log
# - weight_decay: 0.0 .. 0.3
# - warmup_ratio: 0.0 .. 0.2
# - num_train_epochs: 3 .. 6 (discreto)
# Batch 16 fijo para respetar original (5090 lo soporta en fp16, no riesgo HW)
def objective(trial: optuna.Trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.3)
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    epochs = trial.suggest_int("num_train_epochs", 3, 6)

    # Cada trial carga modelo fresco desde base (".") para no contaminar
    model = AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True)

    args = TrainingArguments(
        output_dir=f"./optuna_trial_{trial.number}",
        per_device_train_batch_size=16,  # respetar original
        per_device_eval_batch_size=32,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",  # optuna no guarda checkpoints por trial (ahorra disco)
        load_best_model_at_end=False,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        fp16=USE_CUDA,  # 5090 auto
        bf16=False,
        seed=42,
        logging_steps=10,
        save_total_limit=1,
        disable_tqdm=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(tokenized["eval"])
    # Optuna maximiza MCC (principal), loggea tambien AUC
    trial.set_user_attr("eval_accuracy", metrics["eval_accuracy"])
    trial.set_user_attr("eval_roc_auc", metrics["eval_roc_auc"])
    # limpieza
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics["eval_mcc"]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Optuna ProkBERT SigA+SigX - cambios minimos")
    ap.add_argument("--trials", type=int, default=20, help="numero de trials Optuna")
    ap.add_argument("--study-name", default="siga_sigx_optuna")
    args_cli = ap.parse_args()

    study = optuna.create_study(direction="maximize", study_name=args_cli.study_name)
    print(f"=== Optuna {args_cli.trials} trials, direccion maximize MCC ===")
    study.optimize(objective, n_trials=args_cli.trials)

    print("=== Mejores hiperparametros ===")
    print(f"Best MCC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Best trial: {study.best_trial.number}")

    # Re-entrenar mejor modelo en train+eval y evaluar test con params optimos (opcional)
    print("=== Re-entrenando mejor modelo (train 668) y evaluando test 83 ===")
    best = study.best_params
    model = AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True)
    final_args = TrainingArguments(
        output_dir="./siga_sigx_optuna_best",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=best["learning_rate"],
        weight_decay=best["weight_decay"],
        warmup_ratio=best["warmup_ratio"],
        num_train_epochs=best["num_train_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        fp16=USE_CUDA,
        bf16=False,
        seed=42,
        logging_steps=10,
        save_total_limit=2,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=final_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("eval:", trainer.evaluate(tokenized["eval"]))
    print("test:", trainer.evaluate(tokenized["test"]))
    trainer.save_model("./siga_sigx_optuna_best")
    print("Best model guardado en ./siga_sigx_optuna_best")
    # guardar estudio
    try:
        import pandas as pd
        df = study.trials_dataframe()
        df.to_csv("optuna_trials.csv", index=False)
        print("Trials guardados en optuna_trials.csv")
    except Exception as e:
        print(f"No se pudo guardar dataframe: {e}")
