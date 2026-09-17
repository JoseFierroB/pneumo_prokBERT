#!/usr/bin/env python3
"""
Finetuning D39V 989 full dedup 1973 1381/296/296 - binary promotor vs no-promotor
- Fidelidad oficial: ProkBERTTokenizer k6s1 + AutoModel 2 labels + Trainer
- Dataset: positives 989 (973 uniq) + negatives 1000 =1973 dedup 100%, 70/15/15
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import sys
sys.path.insert(0,".")
import torch
torch.set_num_threads(4)
print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} -> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
USE_CUDA=torch.cuda.is_available()
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import numpy as np
import optuna
from pathlib import Path
set_seed(42)
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
ProkBERTTokenizer.vocab_size=property(lambda self: len(self.get_vocab()) if hasattr(self, 'get_vocab') else 4101)
_orig_save=ProkBERTTokenizer.save_pretrained
def _patched_save(self, save_directory, **kwargs):
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    try:
        return _orig_save(self, save_directory, **kwargs)
    except NotImplementedError:
        (Path(save_directory)/"tokenizer_config.json").write_text('{"tokenizer_class":"ProkBERTTokenizer","kmer":6,"shift":1}')
        return ()
ProkBERTTokenizer.save_pretrained=_patched_save
tokenizer=ProkBERTTokenizer(tokenization_params={'kmer':6,'shift':1}, operation_space='sequence')
print("Tokenizer k6s1 ready")
ds=load_dataset("csv", data_files={
    "train": "datasets/d39v_full_989/train.csv",
    "eval": "datasets/d39v_full_989/eval.csv",
    "test": "datasets/d39v_full_989/test.csv",
})
print(ds)
def tok_fn(ex):
    enc=tokenizer(ex["segment"], add_special_tokens=True, padding=False, truncation=False)
    enc["labels"]=ex["y"]
    return enc
tokenized=ds.map(tok_fn, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
print(tokenized)
model=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=2)
print(f"Model {model.__class__.__name__}")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds=np.argmax(logits, axis=-1)
    probs=np.exp(logits)/np.sum(np.exp(logits),axis=-1,keepdims=True)
    return {"mcc":matthews_corrcoef(labels,preds),"accuracy":accuracy_score(labels,preds),"roc_auc":roc_auc_score(labels,probs[:,1])}

# Optuna 20 + default
def objective(trial):
    lr=trial.suggest_float("learning_rate",1e-5,1e-4,log=True)
    wd=trial.suggest_float("weight_decay",0.0,0.3)
    warmup=trial.suggest_float("warmup_ratio",0.0,0.2)
    epochs=trial.suggest_int("num_train_epochs",3,6)
    m=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=2)
    args=TrainingArguments(
        output_dir=f"./optuna_d39v989/trial_{trial.number}",
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=lr, weight_decay=wd, warmup_ratio=warmup, num_train_epochs=epochs,
        eval_strategy="epoch", save_strategy="no", load_best_model_at_end=False,
        report_to="none", fp16=USE_CUDA, bf16=False, seed=42, logging_steps=10, save_total_limit=1, disable_tqdm=True, dataloader_num_workers=0,
    )
    trainer=Trainer(model=m, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    metrics=trainer.evaluate(tokenized["eval"])
    trial.set_user_attr("acc", metrics["eval_accuracy"])
    trial.set_user_attr("auc", metrics["eval_roc_auc"])
    mcc=metrics["eval_mcc"]
    del trainer, m
    if USE_CUDA: torch.cuda.empty_cache()
    return mcc

import argparse
ap=argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--no-optuna", action="store_true")
args=ap.parse_args()

# Default 3ep
print("=== DEFAULT 3ep ===")
model=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=2)
training_args=TrainingArguments(
    output_dir="./models/d39v_full_989__default",
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=5e-05, weight_decay=0.0, warmup_ratio=0.0, num_train_epochs=3,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="mcc", greater_is_better=True,
    report_to="none", fp16=USE_CUDA, bf16=False, seed=42, logging_steps=10, save_total_limit=2, disable_tqdm=False, dataloader_num_workers=0,
)
trainer=Trainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
trainer.train()
print("eval:", trainer.evaluate(tokenized["eval"]))
print("test:", trainer.evaluate(tokenized["test"]))
trainer.save_model("./models/d39v_full_989__default_final")
print("Saved default")

if not args.no_optuna:
    import optuna
    study=optuna.create_study(direction="maximize")
    print(f"=== Optuna {args.trials} trials ===")
    study.optimize(objective, n_trials=args.trials)
    print(f"Best MCC {study.best_value:.4f} params {study.best_params}")
    best=study.best_params
    m=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=2)
    args2=TrainingArguments(
        output_dir="./models/d39v_full_989__optuna_best",
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=best["learning_rate"], weight_decay=best["weight_decay"], warmup_ratio=best["warmup_ratio"], num_train_epochs=best["num_train_epochs"],
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="mcc", greater_is_better=True,
        report_to="none", fp16=USE_CUDA, bf16=False, seed=42, logging_steps=10, save_total_limit=2, disable_tqdm=False, dataloader_num_workers=0,
    )
    trainer=Trainer(model=m, args=args2, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    print("eval:", trainer.evaluate(tokenized["eval"]))
    print("test:", trainer.evaluate(tokenized["test"]))
    trainer.save_model("./models/d39v_full_989__optuna_best_final")
    print("Saved optuna best")
    try:
        df=study.trials_dataframe()
        df.to_csv("./models/d39v_full_989__optuna_best_trials.csv", index=False)
        print("Trials saved")
    except Exception as e:
        print(e)
