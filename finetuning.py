#!/usr/bin/env python3
"""
D39V adaptation — Train ProkBERT on S. pneumoniae promoter data (81 bp)

Paper-aligned defaults (Frontiers §2.3.2):
  AdamW lr 0.0004, betas 0.95/0.98, eps 5e-05, wd 0.01, warmup 500, 2 epochs, bs 128/GPU

Default dataset: datasets/d39v_full_989 (989 pos SigA+SigX + 1000 neg = 1973, 81 bp, train 1381/eval 296/test 296)
Output: finetuning_outputs/d39v_989_paper_2ep/best (one folder per run, deletable)

Usage:
  pixi run python finetuning.py
  pixi run python finetuning.py --dataset_dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/dedup_3135_2ep
  pixi run python finetuning.py --train_csv datasets/d39v_full_989/train.csv --eval_csv datasets/d39v_full_989/eval.csv --test_csv datasets/d39v_full_989/test.csv --output_dir finetuning_outputs/custom --num_train_epochs 3 --per_device_train_batch_size 64
  pixi run python finetuning.py --help
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
USE_CUDA = torch.cuda.is_available()
print(f"USE_CUDA={USE_CUDA} device={'cuda' if USE_CUDA else 'cpu'}")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import numpy as np

# --- tokenizer (patch vocab_size/save for Trainer) ---
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

def parse_args():
    p = argparse.ArgumentParser(description="ProkBERT D39V adaptation — paper-aligned fine-tuning (k=6 s=1, 81 bp)")
    # data
    p.add_argument("--dataset_dir", type=str, default="datasets/d39v_full_989",
                   help="Folder with train.csv/eval.csv/test.csv (each with segment,y). Overrides --train_csv etc if provided.")
    p.add_argument("--train_csv", type=str, default=None, help="Path to train.csv (with segment,y). If not set, uses dataset_dir/train.csv")
    p.add_argument("--eval_csv", type=str, default=None, help="Path to eval.csv")
    p.add_argument("--test_csv", type=str, default=None, help="Path to test.csv")
    # model / output
    p.add_argument("--model_name", type=str, default=".", help="HF model id or local path (default: . for repo root model.safetensors)")
    p.add_argument("--output_dir", type=str, default="finetuning_outputs/d39v_989_paper_2ep",
                   help="Output dir for checkpoints (best saved to OUTPUT_DIR/best)")
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
    p.add_argument("--seed", type=int, default=42, help="Seed")
    p.add_argument("--kmer", type=int, default=6, help="LCA kmer (default 6)")
    p.add_argument("--shift", type=int, default=1, help="LCA shift (default 1)")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve data files
    if args.train_csv and args.eval_csv and args.test_csv:
        train_csv, eval_csv, test_csv = args.train_csv, args.eval_csv, args.test_csv
    else:
        from pathlib import Path
        d = Path(args.dataset_dir)
        train_csv, eval_csv, test_csv = str(d / "train.csv"), str(d / "eval.csv"), str(d / "test.csv")
    print(f"Data: train={train_csv} eval={eval_csv} test={test_csv} output={args.output_dir} model={args.model_name}")
    print(f"Hyperparams paper: epochs={args.num_train_epochs} bs={args.per_device_train_batch_size} lr={args.learning_rate} wd={args.weight_decay} warmup={args.warmup_steps} betas={args.adam_beta1}/{args.adam_beta2} eps={args.adam_epsilon} seed={args.seed} k={args.kmer} s={args.shift}")

    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': args.kmer, 'shift': args.shift}, operation_space='sequence')
    print(f"Tokenizer k={args.kmer} s={args.shift} ready")

    ds = load_dataset("csv", data_files={"train": train_csv, "eval": eval_csv, "test": test_csv})
    print(ds)

    def tokenize_function(examples):
        enc = tokenizer(examples["segment"], add_special_tokens=True, padding=False, truncation=False)
        enc["labels"] = examples["y"]
        return enc

    tokenized = ds.map(tokenize_function, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
    print(tokenized)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"Model loaded: {model.__class__.__name__} from {args.model_name}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return {
            "mcc": matthews_corrcoef(labels, preds),
            "accuracy": accuracy_score(labels, preds),
            "roc_auc": roc_auc_score(labels, probs[:, 1]),
        }

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        fp16=USE_CUDA,
        bf16=False,
        seed=args.seed,
        logging_steps=50,
        save_total_limit=2,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"=== Training {args.dataset_dir} paper params {args.num_train_epochs}ep ===")
    trainer.train()
    print("=== Eval ===")
    print("eval:", trainer.evaluate(tokenized["eval"]))
    print("test:", trainer.evaluate(tokenized["test"]))
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    print(f"Saved to {best}")

if __name__ == "__main__":
    main()
