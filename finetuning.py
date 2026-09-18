#!/usr/bin/env python3
"""
D39V adaptation — Train ProkBERT on D39V 989 dedup (1973 total: 989 pos + 1000 neg, 81 bp)
Uses paper parameters: 2 epochs, batch 128, lr 0.0004, wd 0.01, warmup 500, betas 0.95/0.98
Dataset: datasets/d39v_full_989 (train 1381 / eval 296 / test 296, stratify 70/15/15)
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
print(f"torch {torch.__version__} threads {torch.get_num_threads()} cuda {torch.cuda.is_available()}")
USE_CUDA = torch.cuda.is_available()
print(f"USE_CUDA={USE_CUDA} device={'cuda' if USE_CUDA else 'cpu'}")
# 5090 Blackwell: auto fp16 if CUDA is available, without touching the rest of the code

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import numpy as np

set_seed(42)

# --- tokenizer (patch vocab_size/save for Trainer) ---
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
# Patch vocab_size
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

# --- data: D39V 989 dedup 1973 ---
ds = load_dataset("csv", data_files={
    "train": "datasets/d39v_full_989/train.csv",
    "eval": "datasets/d39v_full_989/eval.csv",
    "test": "datasets/d39v_full_989/test.csv",
})
print(ds)

def tokenize_function(examples):
    enc = tokenizer(examples["segment"], add_special_tokens=True, padding=False, truncation=False)
    enc["labels"] = examples["y"]
    return enc

tokenized = ds.map(tokenize_function, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
print(tokenized)

# --- model local ---
model = AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True)
print(f"Model loaded: {model.__class__.__name__}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return {
        "mcc": matthews_corrcoef(labels, preds),
        "accuracy": accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:,1]),
    }

# Paper aligned hyperparams for D39V (same as upstream 223k: 2ep, bs128, lr0.0004, wd0.01, warmup500, betas 0.95/0.98)
args = TrainingArguments(
    output_dir="./finetuning_outputs/d39v_989_paper_2ep",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=0.0004,
    weight_decay=0.01,
    num_train_epochs=2,
    warmup_steps=500,
    adam_beta1=0.95,
    adam_beta2=0.98,
    adam_epsilon=5e-05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mcc",
    greater_is_better=True,
    report_to="none",
    fp16=USE_CUDA,
    bf16=False,
    seed=42,
    logging_steps=50,
    save_total_limit=2,
    disable_tqdm=False,
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

print("=== Training D39V 989 paper params 2ep ===")
trainer.train()
print("=== Eval ===")
print("eval:", trainer.evaluate(tokenized["eval"]))
print("test:", trainer.evaluate(tokenized["test"]))
trainer.save_model("./finetuning_outputs/d39v_989_paper_2ep/best")
print("Saved to ./finetuning_outputs/d39v_989_paper_2ep/best")
