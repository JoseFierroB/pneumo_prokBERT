#!/usr/bin/env python3
"""
Mixed training: HF bacterial_promoters train_80 + pneumo train, parallel protocol.

Fit:      HF train_80 (178473) + pneumo train (d39v 1381 | dedup 2194), shuffled seed 42
Select:   combined val = HF val_10 (22309) + pneumo eval (296 | 470), best by MCC
Locked:   HF test_10 (22310), data test, sigma70, multispecies (eval only)

Mirrors the legacy mix recipe (paper hyperparams, fp32, warmup 500, eval/save steps 50/500)
but leaves a true in-dist holdout untouched. Run from the repo root.

Usage:
  pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_full_989 --output_dir finetuning_outputs/base_223k_d39v_2ep
  pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/base_223k_d39v_tigr4_2ep
  pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_full_989 --dry_run   # validate data/splits only

--data-dir holds train.csv/eval.csv/test.csv with segment,y (+id); the run name is its basename.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from prokbert_patches import apply_tokenizer_patches, write_run_info

apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

BASE_MODEL = "neuralbioinfo/prokbert-mini"
HF_DATASET = "neuralbioinfo/bacterial_promoters"
EXPECTED = {
    "train_80": 178473, "val_10": 22309, "test_10": 22310,
    "val_sha": "716c3c3320d22c23", "test_sha": "49a61e636397556b",
}
def parse_args():
    p = argparse.ArgumentParser(description="Mixed 223k + extra-data training, parallel protocol (paper hyperparams)")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Folder with train.csv/eval.csv/test.csv (segment,y +id); run name is its basename")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=0.0004)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--dry_run", action="store_true", help="build data/splits, verify hashes, exit before training")
    add_device_args(p)
    return p.parse_args()


def normalize(s):
    return str(s).upper().replace("U", "T").strip()


def sha_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def main():
    args = parse_args()
    set_seed(args.seed)
    data_name = Path(args.data_dir).name
    if args.output_dir is None:
        args.output_dir = f"finetuning_outputs/base_223k_{data_name}_2ep"
    device, use_cuda = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} (auto-detect; override with --gpu N or --cpu) | data={data_name} | out={args.output_dir}")

    # --- HF train + fixed 80/10/10 split (identical code as paper_replica.py) ---
    ds = load_dataset(HF_DATASET)
    assert len(ds["train"]) == 223092, f"unexpected HF train size {len(ds['train'])}"
    tmp = ds["train"].train_test_split(test_size=0.2, seed=args.seed)
    train_80, temp = tmp["train"], tmp["test"]
    vt = temp.train_test_split(test_size=0.5, seed=args.seed)
    val_10, test_10 = vt["train"], vt["test"]
    assert (len(train_80), len(val_10), len(test_10)) == (EXPECTED["train_80"], EXPECTED["val_10"], EXPECTED["test_10"])
    assert sha_ids(val_10["segment_id"])[:16] == EXPECTED["val_sha"], "val split IDs differ from manifest"
    assert sha_ids(test_10["segment_id"])[:16] == EXPECTED["test_sha"], "test split IDs differ from manifest"
    print(f"HF split OK: train {len(train_80)} val {len(val_10)} test {len(test_10)}")

    # --- extra-data splits (train.csv/eval.csv/test.csv), mapped to the same minimal schema ---
    pneu_raw = load_dataset("csv", data_files={
        "train": f"{args.data_dir}/train.csv",
        "eval": f"{args.data_dir}/eval.csv",
        "test": f"{args.data_dir}/test.csv",
    })
    for k in ("train", "eval", "test"):
        cols = set(pneu_raw[k].column_names)
        assert {"segment", "y"} <= cols, f"{args.data_dir}/{k}.csv must have segment,y columns (has {sorted(cols)})"
    _tr, _ev, _te = (set(pneu_raw["train"]["segment"]), set(pneu_raw["eval"]["segment"]), set(pneu_raw["test"]["segment"]))
    assert not (_tr & _ev) and not (_tr & _te) and not (_ev & _te), "train/eval/test splits overlap in --data-dir"
    def _sid(ex, i, col):
        return str(ex[col]) if col in ex else f"row-{i}"
    pneu = {}
    for k in ("train", "eval", "test"):
        has_id = "id" in pneu_raw[k].column_names
        pneu[k] = pneu_raw[k].map(
            lambda ex, i, _h=has_id: {"segment_id": str(ex["id"]) if _h else f"row-{i}",
                                      "segment": ex["segment"], "y": ex["y"]},
            with_indices=True, remove_columns=pneu_raw[k].column_names)
    print(f"Data OK ({data_name}): train {len(pneu['train'])} eval {len(pneu['eval'])} test {len(pneu['test'])}")

    def minimal(hf_split):
        return hf_split.map(lambda ex: {"segment_id": str(ex["segment_id"]), "segment": ex["segment"], "y": ex["y"]},
                            remove_columns=[c for c in hf_split.column_names if c not in ("segment_id", "segment", "y")])

    # --- parallel composition ---
    train_ds = concatenate_datasets([minimal(train_80), pneu["train"]]).shuffle(seed=args.seed)
    val_ds = concatenate_datasets([minimal(val_10), pneu["eval"]])
    print(f"Combined: train {len(train_ds)} val {len(val_ds)} (+locked test_10 {len(test_10)}, data test {len(pneu['test'])})")

    manifest = {
        "hf_dataset": HF_DATASET, "hf_train": 223092,
        "hf_split": {"train_80": len(train_80), "val_10": len(val_10), "test_10": len(test_10),
                     "val_sha16": EXPECTED["val_sha"], "test_sha16": EXPECTED["test_sha"], "seed": args.seed},
        "data_dir": args.data_dir, "data_name": data_name,
        "data_splits": {k: {"n": len(pneu_raw[k]), "sha256": hashlib.sha256(
            open(f"{args.data_dir}/{k}.csv", "rb").read()).hexdigest()} for k in ("train", "eval", "test")},
        "combined": {"train": len(train_ds), "val": len(val_ds)},
    }

    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    print(f"Tokenizer k=6 s=1 ready, vocab {len(tokenizer.get_vocab())}")

    def tokenize_fn(examples):
        enc = tokenizer([normalize(s) for s in examples["segment"]],
                        add_special_tokens=True, padding=False, truncation=False)
        enc["labels"] = examples["y"]
        return enc

    cols = train_ds.column_names
    tok_train = train_ds.map(tokenize_fn, batched=True, batch_size=1000, remove_columns=cols)
    tok_val = val_ds.map(tokenize_fn, batched=True, batch_size=1000, remove_columns=val_ds.column_names)
    print(f"Tokenized: train {len(tok_train)} val {len(tok_val)}")

    if args.dry_run:
        print("DRY RUN OK — data, splits and hashes verified, no training started")
        print(json.dumps(manifest, indent=2))
        return

    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    config.num_labels = 2
    config.classifier_dropout = 0.1
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    config.problem_type = "single_label_classification"
    config.label2id = {"0": 0, "1": 1}
    config.id2label = {0: "0", 1: "1"}
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config, trust_remote_code=True)
    print(f"Model: {model.__class__.__name__} from {BASE_MODEL}")
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = float("nan")
        return {"mcc": matthews_corrcoef(labels, preds), "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds), "roc_auc": auc}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=128,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        adam_beta1=0.95, adam_beta2=0.98, adam_epsilon=5e-05,
        logging_steps=50, save_steps=500,
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="mcc", greater_is_better=True,
        report_to="none", seed=args.seed,
        bf16=False, fp16=False,
        dataloader_num_workers=0, optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tok_train, eval_dataset=tok_val,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_cuda else None),
                      compute_metrics=compute_metrics)

    write_run_info(args.output_dir, {
        "script": "scripts/train_mix223k.py", "model_name": BASE_MODEL,
        "model_name_or_path": getattr(model.config, "_name_or_path", BASE_MODEL),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "data_dir": args.data_dir, "data_name": data_name, "device": device, "seed": args.seed,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "data_manifest": manifest,
    })

    print(f"=== Training {data_name} mix {args.num_train_epochs}ep ===")
    trainer.train()
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    print(f"Saved to {best}")

    def tok_split(split_ds, cols0):
        return split_ds.map(tokenize_fn, batched=True, batch_size=1000, remove_columns=cols0)

    print("=== Locked eval (report only) ===")
    locked = {}
    for name, sds, c0 in [("val_combined", val_ds, val_ds.column_names),
                          ("test_10", test_10, test_10.column_names),
                          ("data_test", pneu["test"], pneu["test"].column_names),
                          ("sigma70", ds["test_sigma70"], ds["test_sigma70"].column_names),
                          ("multispecies", ds["test_multispecies"], ds["test_multispecies"].column_names)]:
        m = trainer.evaluate(tok_split(sds, c0))
        locked[name] = m
        print(name, m)
    import json as _json
    _ri = _json.load(open(f"{args.output_dir.rstrip('/')}/run_info.json"))
    _ri["locked_eval"] = locked
    write_run_info(args.output_dir, _ri)


if __name__ == "__main__":
    main()
