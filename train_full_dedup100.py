#!/usr/bin/env python3
"""
Finetuning ProkBERT FULL D39V+TIGR4 dedup100 2194/470/471 - minimal cambios sobre train_siga_sigx.py
- Mismo tokenizer patch, tokenizacion, model load, compute_metrics
- Solo cambia CSVs a full dedup100 (3135 uniq, 1399 pos /1736 neg, 44.6% pos)
- Hiperparametros best Optuna (lr 7.7e-05 wd0.04 warmup0.12 epochs6) o default 3ep si se quiere original
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
sys.path.insert(0, ".")
import torch
torch.set_num_threads(4)
print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} -> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
USE_CUDA = torch.cuda.is_available()
print(f"USE_CUDA={USE_CUDA}")
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import numpy as np
set_seed(42)
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
# --- data FULL dedup100 ---
ds = load_dataset("csv", data_files={
    "train": "train_full_dedup100.csv",
    "eval": "eval_full_dedup100.csv",
    "test": "test_full_dedup100.csv",
})
print(ds)
def tokenize_function(examples):
    enc = tokenizer(examples["segment"], add_special_tokens=True, padding=False, truncation=False)
    enc["labels"] = examples["y"]
    return enc
tokenized = ds.map(tokenize_function, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
print(tokenized)
model = AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True)
print(f"Model loaded: {model.__class__.__name__}")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return {"mcc": matthews_corrcoef(labels, preds), "accuracy": accuracy_score(labels, preds), "roc_auc": roc_auc_score(labels, probs[:,1])}
# Hiperparametros: best Optuna (para full, 6ep) - si quieres original 3ep cambia epochs=3 lr=5e-5
args = TrainingArguments(
    output_dir="./full_dedup100_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=7.747486422728904e-05,  # best Optuna
    weight_decay=0.040280909839829605,
    warmup_ratio=0.12725631995134956,
    num_train_epochs=6,
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
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
print("=== Training FULL dedup100 2194 train ===")
trainer.train()
print("=== Eval ===")
print("eval:", trainer.evaluate(tokenized["eval"]))
print("test:", trainer.evaluate(tokenized["test"]))
trainer.save_model("./full_dedup100_final")
print("Saved to ./full_dedup100_final")
