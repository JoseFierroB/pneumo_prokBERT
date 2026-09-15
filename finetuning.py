#!/usr/bin/env python3
"""
Finetuning ProkBERT — colab Finetuning.ipynb convertido a .py limpio
- Sin !pip, sin markdown colab, sin raise SystemError colab
- Misma logica: AutoModel + tokenize_function + compute_metrics + TrainingArguments + Trainer
- Para datos propios: descomentar bloque CSV (train_full_dedup100.csv) y comentar HF bacterial_promoters
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, recall_score
import numpy as np

# --- GPU check (colab: if not cuda raise) -> aqui solo info, no raise ---
if torch.cuda.is_available():
    print(f"Found GPU at: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found, using CPU")
print(f"CPU cores: {os.cpu_count()}")

# --- Modelo y tokenizer (colab: neuralbioinfo/prokbert-mini + AutoTokenizer) ---
# Para pneumo 81bp usamos mini-promoter k6s1 local (mismo peso) con ProkBERTTokenizer
# Si quieres HF exacto colab, cambia a: model_name_path='neuralbioinfo/prokbert-mini' + AutoTokenizer
model_name_path = 'neuralbioinfo/prokbert-mini'
# Opcion pneumo local (recomendado, offline, kmer6):
USE_LOCAL = True
if USE_LOCAL:
    import sys
    sys.path.insert(0, ".")
    from prokbert.prokbert_tokenizer import ProkBERTTokenizer
    # patch vocab_size para Trainer.save (igual que train_siga_sigx.py)
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
    model = AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True)  # local pneumo_prokBERT base
else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_path, trust_remote_code=True)

# --- Datos: colab usa HF bacterial_promoters, aqui CSVs propios (descomentar) ---
# COLAB ORIGINAL:
# dataset = load_dataset("neuralbioinfo/bacterial_promoters")
# randomized_dataset = dataset.shuffle(seed=42)
# tokenized_dataset = randomized_dataset.map(tokenize_function, batched=True, batch_size=1000, num_proc=os.cpu_count(), remove_columns=dataset["train"].column_names)
# NUESTROS DATOS (mismo formato y=0/1, segment 81bp):
USE_OUR_DATA = True
if USE_OUR_DATA:
    # 836 SigA+SigX piloto o 3135 full dedup100 — cambia aqui
    dataset = load_dataset("csv", data_files={
        "train": "train_full_dedup100.csv",  # o train_siga_sigx_668.csv
        "eval": "eval_full_dedup100.csv",
        "test": "test_full_dedup100.csv",
    })
    # para compatibilidad colab, renombrar eval->test_sigma70
    dataset["test_sigma70"] = dataset["eval"]
else:
    dataset = load_dataset("neuralbioinfo/bacterial_promoters")
    dataset = dataset.shuffle(seed=42)

def tokenize_function(examples):
    encoded = tokenizer(
        examples["segment"],
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    encoded["labels"] = examples["y"]
    return encoded

# colab: tokenized_dataset = randomized_dataset.map(..., batch_size=1000, num_proc=num_cores, remove_columns=...)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset["train"].column_names,
)

# --- Metrics colab: mcc/acc/recall/roc_auc (binario vs multiclass) ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    mcc = matthews_corrcoef(labels, predictions)
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    try:
        if logits.shape[1] == 2:
            roc_auc = roc_auc_score(labels, logits[:, 1])
        else:
            roc_auc = roc_auc_score(labels, logits, multi_class='ovr')
    except ValueError:
        roc_auc = float('nan')
    return {"mcc": mcc, "accuracy": acc, "recall": recall, "roc_auc": roc_auc}

# --- Training colab: TrainingArguments bs128 1ep lr0.0004 steps 50 ---
# Para pneumo usamos bs16/32 3-6ep lr5e-05 + fp16 auto 5090 (cambios minimos)
USE_CUDA = torch.cuda.is_available()
training_args = TrainingArguments(
    output_dir="./results",  # colab: ./results  nosotros: ./siga_sigx_out o ./full_dedup100_out
    per_device_train_batch_size=16,  # colab 128
    per_device_eval_batch_size=32,   # colab 128
    report_to="none",
    eval_strategy="epoch",  # colab steps 50
    save_strategy="epoch",
    num_train_epochs=3,  # colab 1
    learning_rate=5e-05,  # colab 0.0004
    logging_steps=10,  # colab 50
    save_total_limit=2,  # colab 1
    load_best_model_at_end=True,
    metric_for_best_model="mcc",
    greater_is_better=True,
    fp16=USE_CUDA,
    bf16=False,
    seed=42,
    dataloader_num_workers=0,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test_sigma70'] if 'test_sigma70' in tokenized_dataset else tokenized_dataset['eval'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    # eval final como colab
    print(trainer.evaluate())
