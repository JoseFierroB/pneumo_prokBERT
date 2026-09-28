#!/usr/bin/env python3
"""
D39V adaptation — Train ProkBERT on S. pneumoniae promoter data (81 bp)

Paper-aligned defaults (Frontiers §2.3.2):
  AdamW lr 0.0004, betas 0.95/0.98, eps 5e-05, wd 0.01, warmup 500, 2 epochs, bs 128/GPU

Default dataset: datasets/d39v_tigr4_dedup_3135 (1399 pos + 1736 neg = 3135 unique, 81 bp, 2194/470/471)
Default model: neuralbioinfo/prokbert-mini (base MLM, backbone loaded, classification head fresh).
  --model-name also accepts a local checkpoint dir.
Output: finetuning_outputs/<run>/best (one folder per run, deletable; run_info.json with provenance)
  Default output dir: finetuning_outputs/dedup_3135_2ep

Warmup: paper uses warmup_steps=500 for full runs (223k, ~2790 steps). On small datasets
  (22 steps) warmup 500 caps the LR at 4e-4*22/500 = 1.76e-5; use --warmup-ratio 0.1 there.

Usage:
  pixi run python finetuning.py
  pixi run python finetuning.py --dataset-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/dedup_3135_2ep
  pixi run python finetuning.py --model-name neuralbioinfo/prokbert-mini-promoter --warmup-ratio 0.1 --output-dir finetuning_outputs/dedup_from_promoter_2ep
  pixi run python finetuning.py --train-csv datasets/d39v_tigr4_dedup_3135/train.csv --eval-csv datasets/d39v_tigr4_dedup_3135/eval.csv --test-csv datasets/d39v_tigr4_dedup_3135/test.csv --output-dir finetuning_outputs/custom --num-train-epochs 3 --per-device-train-batch-size 64
  pixi run python finetuning.py --help
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import sys
sys.path.insert(0, ".")
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
print(f"torch {torch.__version__} threads {torch.get_num_threads()} cuda {torch.cuda.is_available()}")
from scripts.common import add_device_args, resolve_device
from scripts.metrics import trainer_compute_metrics

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed


class WeightedTrainer(Trainer):
    """Trainer with optional class weights (None = plain CE, identical to Trainer)."""

    def __init__(self, *args, class_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = torch.tensor(class_weight, dtype=torch.float) if class_weight else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        w = self.class_weight.to(logits.device) if self.class_weight is not None else None
        loss = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), weight=w)
        return (loss, outputs) if return_outputs else loss

# --- tokenizer runtime patches (upstream prokbert/ left untouched) ---
from prokbert_patches import apply_tokenizer_patches, write_run_info
apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

def parse_args():
    p = argparse.ArgumentParser(description="ProkBERT D39V adaptation — paper-aligned fine-tuning (k=6 s=1, 81 bp)")
    # data
    p.add_argument("--dataset-dir", type=str, default="datasets/d39v_tigr4_dedup_3135",
                   help="Folder with train.csv/eval.csv/test.csv (each with segment,y). Overrides --train-csv etc if provided.")
    p.add_argument("--train-csv", type=str, default=None, help="Path to train.csv (with segment,y). If not set, uses dataset_dir/train.csv")
    p.add_argument("--eval-csv", type=str, default=None, help="Path to eval.csv")
    p.add_argument("--test-csv", type=str, default=None, help="Path to test.csv")
    # model / output
    p.add_argument("--model-name", type=str, default="neuralbioinfo/prokbert-mini",
                   help="HF model id or local path. Default: base neuralbioinfo/prokbert-mini (backbone + fresh head)")
    p.add_argument("--output-dir", type=str, default="finetuning_outputs/dedup_3135_2ep",
                   help="Output dir for checkpoints (best saved to OUTPUT_DIR/best)")
    # paper hyperparams (defaults §2.3.2)
    p.add_argument("--num-train-epochs", type=int, default=2, help="Paper: 2 epochs")
    p.add_argument("--per-device-train-batch-size", type=int, default=128, help="Paper: 128/GPU")
    p.add_argument("--per-device-eval-batch-size", type=int, default=128, help="Eval batch size")
    p.add_argument("--learning-rate", type=float, default=0.0004, help="Paper: 0.0004 (AdamW)")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Paper: 0.01")
    p.add_argument("--warmup-steps", type=int, default=500, help="Paper: linear warmup 500 (used when --warmup-ratio is not set)")
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="If set, warmup_steps=0 and this ratio of total steps is used (e.g. 0.1 for small datasets)")
    p.add_argument("--adam-beta1", type=float, default=0.95, help="Paper: 0.95")
    p.add_argument("--adam-beta2", type=float, default=0.98, help="Paper: 0.98")
    p.add_argument("--adam-epsilon", type=float, default=5e-05, help="Paper: 5e-05")
    p.add_argument("--seed", type=int, default=42, help="Seed")
    p.add_argument("--kmer", type=int, default=6, help="LCA kmer (default 6)")
    p.add_argument("--shift", type=int, default=1, help="LCA shift (default 1)")
    p.add_argument("--class-weight", type=str, default=None,
                   help="e.g. '0.5,9.5' (majority,minority) for WeightedTrainer; omit for plain CE")
    add_device_args(p)
    return p.parse_args()

def resolve_data_files(args):
    """Resolve (train, eval, test) CSV paths from explicit flags or --dataset-dir."""
    if args.train_csv and args.eval_csv and args.test_csv:
        files = (args.train_csv, args.eval_csv, args.test_csv)
    else:
        d = Path(args.dataset_dir)
        files = (str(d / "train.csv"), str(d / "eval.csv"), str(d / "test.csv"))
    print(f"Data: train={files[0]} eval={files[1]} test={files[2]} output={args.output_dir} model={args.model_name}")
    print(f"Hyperparams paper: epochs={args.num_train_epochs} bs={args.per_device_train_batch_size} lr={args.learning_rate} wd={args.weight_decay} warmup={args.warmup_steps} betas={args.adam_beta1}/{args.adam_beta2} eps={args.adam_epsilon} seed={args.seed} k={args.kmer} s={args.shift}")
    return files


def build_tokenizer(args):
    tok = ProkBERTTokenizer(tokenization_params={'kmer': args.kmer, 'shift': args.shift}, operation_space='sequence')
    print(f"Tokenizer k={args.kmer} s={args.shift} ready")
    return tok


def tokenize_csvs(tokenizer, files):
    train_csv, eval_csv, test_csv = files
    ds = load_dataset("csv", data_files={"train": train_csv, "eval": eval_csv, "test": test_csv})
    print(ds)

    def tokenize_function(examples):
        enc = tokenizer(examples["segment"], add_special_tokens=True, padding=False, truncation=False)
        enc["labels"] = examples["y"]
        return enc

    tokenized = ds.map(tokenize_function, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
    print(tokenized)
    return tokenized


def resolve_warmup(args):
    """Paper warmup_steps=500, or ratio-based for small runs (22-36 steps)."""
    if args.warmup_ratio is not None:
        warmup = (0, args.warmup_ratio)
    else:
        warmup = (args.warmup_steps, 0.0)
    print(f"Warmup: steps={warmup[0]} ratio={warmup[1]}")
    return warmup


def build_run_info(args, model, tokenized, files, warmup, device, use_cuda):
    """Provenance for the run (init + data + hyperparams). Written before and after training."""
    train_csv, eval_csv, test_csv = files
    run_info = {
        "script": "finetuning.py",
        "model_name": args.model_name,
        "model_name_or_path": getattr(model.config, "_name_or_path", args.model_name),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "dataset_dir": args.dataset_dir,
        "train_csv": train_csv, "eval_csv": eval_csv, "test_csv": test_csv,
        "n_train": len(tokenized["train"]), "n_eval": len(tokenized["eval"]), "n_test": len(tokenized["test"]),
        "kmer": args.kmer, "shift": args.shift, "seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": warmup[0], "warmup_ratio": warmup[1],
        "adam_beta1": args.adam_beta1, "adam_beta2": args.adam_beta2, "adam_epsilon": args.adam_epsilon,
        "class_weight": args.class_weight,
        "fp16": use_cuda, "bf16": False,
        "device": device,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
    }
    local_ckpt = Path(args.model_name) / "model.safetensors"
    if local_ckpt.exists():
        run_info["init_weights_sha256"] = hashlib.sha256(local_ckpt.read_bytes()).hexdigest()
    return run_info


def build_training_args(args, warmup_steps, warmup_ratio, use_cuda):
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        fp16=use_cuda,
        bf16=False,
        seed=args.seed,
        logging_steps=50,
        save_total_limit=2,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device, use_cuda = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} (auto-detect; override with --gpu N or --cpu)")

    files = resolve_data_files(args)
    tokenizer = build_tokenizer(args)
    tokenized = tokenize_csvs(tokenizer, files)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True)
    print(f"Model loaded: {model.__class__.__name__} from {args.model_name}")

    warmup_steps, warmup_ratio = resolve_warmup(args)

    run_info = build_run_info(args, model, tokenized, files, (warmup_steps, warmup_ratio), device, use_cuda)
    write_run_info(args.output_dir, run_info)
    print(f"run_info.json written to {args.output_dir}")

    training_args = build_training_args(args, warmup_steps, warmup_ratio, use_cuda)

    cw = None
    if args.class_weight:
        cw = [float(x) for x in args.class_weight.split(",")]
        assert len(cw) == 2, "--class-weight needs 2 values 'w0,w1'"
        print(f"Class weights: {cw}")

    trainer = WeightedTrainer(
        model=model,
        class_weight=cw,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_compute_metrics,
    )

    print(f"=== Training {args.dataset_dir} {args.num_train_epochs}ep from {args.model_name} ===")
    trainer.train()
    print("=== Eval ===")
    eval_metrics = trainer.evaluate(tokenized["eval"])
    test_metrics = trainer.evaluate(tokenized["test"])
    print("eval:", eval_metrics)
    print("test:", test_metrics)
    run_info["eval"] = eval_metrics
    run_info["test"] = test_metrics
    write_run_info(args.output_dir, run_info)
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    print(f"Saved to {best}")

if __name__ == "__main__":
    main()
