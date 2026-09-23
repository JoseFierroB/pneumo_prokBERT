#!/usr/bin/env python3
"""
Paper replica — ProkBERT fine-tuning reproducing the bacterial_promoters setup
(neuralbioinfo/bacterial_promoters, 223,092 train, 81 bp, k=6 s=1).

Paper-aligned defaults (Frontiers §2.3.2):
  AdamW lr 0.0004, betas 0.95/0.98, eps 5e-05, wd 0.01, warmup 500, 2 epochs, bs 128/GPU
  Reproduces the 80/10/10 split of train 223k (seed 42) and evaluates sigma70 / multispecies.

Notes:
- Uses the official class ProkBertForSequenceClassification (prokbert/models/megatronbert.py:1710);
  upstream examples/finetuning.py imports BertForBinaryClassificationWithPooling, which does not exist.
- Runtime patches for the vendored code live in prokbert_patches.py (upstream prokbert/ untouched).
- Designed to run offline: defaults to the local materialized copy data/bacterial_promoters
  (train.parquet 223092 + test_sigma70.parquet 1864 + test_multispecies.parquet 22582).

Usage:
  pixi run python paper_replica.py
  pixi run python paper_replica.py --dataset_dir data/bacterial_promoters --output_dir finetuning_outputs/paper_80_10_10_2ep
  pixi run python paper_replica.py --model_name neuralbioinfo/prokbert-mini-promoter --num_train_epochs 3 --output_dir finetuning_outputs/paper_promoter_3ep
  pixi run python paper_replica.py --help
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
sys.path.insert(0, ".")
import torch
torch.set_num_threads(4)
print(f"torch {torch.__version__} threads {torch.get_num_threads()} cuda {torch.cuda.is_available()}")
from scripts.common import add_device_args, resolve_device

from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, f1_score
import numpy as np

from prokbert_patches import apply_tokenizer_patches, write_run_info
apply_tokenizer_patches()

from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.models.megatronbert import ProkBertForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser(description="ProkBERT paper replica / adaptation (bacterial_promoters, k=6 s=1, 81 bp)")
    # data
    p.add_argument("--dataset_dir", type=str, default="data/bacterial_promoters",
                   help="Local dir with train/test_sigma70/test_multispecies .parquet or .csv; if missing, uses HF neuralbioinfo/bacterial_promoters")
    p.add_argument("--seed", type=int, default=42, help="Seed for split and training")
    # model / output
    p.add_argument("--model_name", type=str, default="neuralbioinfo/prokbert-mini",
                   help="HF model id or local path (default: base neuralbioinfo/prokbert-mini)")
    p.add_argument("--output_dir", type=str, default="finetuning_outputs/paper_80_10_10_2ep",
                   help="Output dir (best saved to OUTPUT_DIR/best; one folder per run, deletable)")
    # paper hyperparams (defaults §2.3.2)
    p.add_argument("--num_train_epochs", type=int, default=2, help="Paper: 2 epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=128, help="Paper: 128/GPU")
    p.add_argument("--per_device_eval_batch_size", type=int, default=128, help="Eval batch size")
    p.add_argument("--learning_rate", type=float, default=0.0004, help="Paper: 0.0004 (AdamW)")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Paper: 0.01")
    p.add_argument("--warmup_steps", type=int, default=500, help="Paper: linear warmup 500")
    p.add_argument("--adam_beta1", type=float, default=0.95, help="Paper: 0.95")
    p.add_argument("--adam_beta2", type=float, default=0.98, help="Paper: 0.98")
    p.add_argument("--adam_epsilon", type=float, default=5e-05, help="Paper: 5e-05")
    p.add_argument("--eval_steps", type=int, default=50, help="Eval every N steps (as in the 223k replica)")
    p.add_argument("--save_steps", type=int, default=500, help="Save every N steps (multiple of eval_steps)")
    p.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    p.add_argument("--kmer", type=int, default=6, help="LCA kmer (default 6)")
    p.add_argument("--shift", type=int, default=1, help="LCA shift (default 1)")
    p.add_argument("--fp16", action="store_true", help="Enable fp16 (paper used fp32 on A100)")
    p.add_argument("--bf16", action="store_true", help="Enable bf16")
    add_device_args(p)
    return p.parse_args()


def load_bacterial_promoters(dataset_dir):
    from pathlib import Path
    d = Path(dataset_dir)
    if d.exists() and (d / "train.parquet").exists():
        files = {"train": str(d / "train.parquet")}
        for split in ("test_sigma70", "test_multispecies"):
            if (d / f"{split}.parquet").exists():
                files[split] = str(d / f"{split}.parquet")
        return load_dataset("parquet", data_files=files), f"local:{d}"
    return load_dataset("neuralbioinfo/bacterial_promoters"), "hf:neuralbioinfo/bacterial_promoters"


def main():
    args = parse_args()
    set_seed(args.seed)
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} (auto-detect; override with --gpu N or --cpu)")

    ds, data_source = load_bacterial_promoters(args.dataset_dir)
    print(f"Data source: {data_source}")
    print(ds)

    train_full = ds["train"]
    tmp = train_full.train_test_split(test_size=0.2, seed=args.seed)
    train_split, temp = tmp["train"], tmp["test"]
    val_test = temp.train_test_split(test_size=0.5, seed=args.seed)
    val_split, test_split = val_test["train"], val_test["test"]
    print(f"Split 80/10/10 seed {args.seed}: train {len(train_split)} val {len(val_split)} test {len(test_split)}")

    independent = {k: ds[k] for k in ("test_sigma70", "test_multispecies") if k in ds}
    print(f"Independent tests: {[(k, len(v)) for k, v in independent.items()]}")

    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': args.kmer, 'shift': args.shift}, operation_space='sequence')
    print(f"Tokenizer k={args.kmer} s={args.shift} ready, vocab {len(tokenizer.get_vocab())}")

    def tokenize_function(examples):
        enc = tokenizer(examples["segment"], add_special_tokens=True, padding=False, truncation=False)
        enc["labels"] = examples["y"]
        return enc

    tokenized = {}
    for name, split_ds in (("train", train_split), ("val", val_split), ("test", test_split), *independent.items()):
        print(f"Tokenizing {name}...")
        tokenized[name] = split_ds.map(tokenize_function, batched=True, batch_size=1000, remove_columns=split_ds.column_names)

    model = ProkBertForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True, num_labels=2)
    print(f"Model loaded: {model.__class__.__name__} hidden {model.config.hidden_size} layers {model.config.num_hidden_layers}")
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = float("nan")
        return {
            "mcc": matthews_corrcoef(labels, preds),
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
            "roc_auc": auc,
        }

    if args.bf16:
        precision = {"bf16": True, "fp16": False}
    elif args.fp16:
        precision = {"fp16": True, "bf16": False}
    else:
        precision = {"fp16": False, "bf16": False}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        disable_tqdm=False,
        **precision,
    )

    write_run_info(args.output_dir, {
        "script": "paper_replica.py",
        "model_name": args.model_name,
        "model_name_or_path": getattr(model.config, "_name_or_path", args.model_name),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "data_source": data_source,
        "n_train": len(tokenized["train"]), "n_val": len(tokenized["val"]), "n_test": len(tokenized["test"]),
        "kmer": args.kmer, "shift": args.shift, "seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "adam_beta1": args.adam_beta1, "adam_beta2": args.adam_beta2, "adam_epsilon": args.adam_epsilon,
        **precision,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
    })
    print(f"run_info.json written to {args.output_dir}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"=== Training paper replica {args.num_train_epochs}ep from {args.model_name} ===")
    trainer.train()
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    print(f"Saved to {best}")

    print("=== Eval ===")
    for name in ("val", "test", *independent.keys()):
        print(f"{name}:", trainer.evaluate(tokenized[name]))


if __name__ == "__main__":
    main()
