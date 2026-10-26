#!/usr/bin/env python3
"""
Exact-replica fine-tuning of ProkBERT-mini on HF bacterial_promoters (paper-223k only).

Two recipes, identical in everything except the optimizer/scheduler source of truth:
  --recipe code : upstream ProkBERTTrainer (torch AdamW lr/eps/wd, warmup=0,
                  betas ignored -> torch defaults 0.9/0.999). Matches THEIR code.
  --recipe paper: HF Trainer with paper hyperparameters (AdamW lr 4e-4,
                  betas 0.95/0.98, eps 5e-5, wd 0.01, linear warmup 500). Matches §2.3.2.

Shared exactness closures (both recipes):
  - Official head ProkBertForSequenceClassification (never AutoModel -> Ext),
    fresh init, dropout 0.1 on classifier+hidden+attention (held constant).
  - Upstream tokenization path: ProkBERTTokenizer k=6 s=1, text.upper() ONLY
    (no U->T; the tokenizer uppercases internally), add_special_tokens, no
    map-time padding (dynamic per-batch pad, same tensors as upstream
    padding=True); attention_mask zeroed at token ids 2|3 (upstream rule).
  - Splits: HF 80/10/10 seed 42, SHA-asserted (same as paper_replica/train_mix).
  - Plain CrossEntropyLoss (no smoothing/focal/weights), fp32.
  - Select best by val_10 ROC-AUC (their early-stopping metric family);
    test_10 evaluated ONCE post-hoc (locked report). Probs CSVs saved for bootstrap.

Usage:
  pixi run python scripts/train_replica.py --recipe code --output-dir finetuning_outputs/replica_code_2ep
  pixi run python scripts/train_replica.py --recipe paper --output-dir finetuning_outputs/replica_paper_2ep
  pixi run python scripts/train_replica.py --recipe code --dry-run
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys

import numpy as np
import torch

print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from scripts.metrics import trainer_compute_metrics, save_probs_csv
from scripts.train_mix223k import (
    BASE_MODEL,
    EXPECTED,
    HF_DATASET,
    load_hf_split,
    minimal,
    train_seed as _mix_train_seed,
)

from prokbert_patches import apply_tokenizer_patches, write_run_info

apply_tokenizer_patches()
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.models.megatronbert import ProkBertForSequenceClassification
from prokbert.training_utils import ProkBERTTrainer

RECIPES = ("code", "paper")


def parse_args():
    p = argparse.ArgumentParser(description="Exact-replica ProkBERT fine-tune (paper-223k only)")
    p.add_argument("--recipe", choices=RECIPES, required=True,
                   help="code = upstream ProkBERTTrainer (warmup 0, torch betas); "
                        "paper = HF Trainer (§2.3.2: warmup 500, betas 0.95/0.98)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-seed", type=int, default=None,
                   help="training randomness (shuffle/init/order); default = --seed")
    p.add_argument("--num-train-epochs", type=int, default=2)
    p.add_argument("--per-device-train-batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.0004)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1,
                   help="shared held-constant dropout (classifier+hidden+attention)")
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="THEIR real practice (#133): ratio of total steps (e.g. 0.1). "
                        "If set, warmup_steps forced to 0. Code recipe then matches "
                        "upstream behavior (no fixed-step warmup).")
    p.add_argument("--metric-for-best", type=str, default="roc_auc",
                   help="selection metric: roc_auc (frozen default) or mcc (their #133 practice)")
    p.add_argument("--full-train", action="store_true",
                   help="(a) fit FULL HF train (223092, no holdout); selection on "
                        "external sigma70 by AUC (their published setup has no holdout)")
    p.add_argument("--model-loader", choices=("official", "auto"), default="official",
                   help="(c) auto = AutoModelForSequenceClassification (as our mix "
                        "used; may resolve to Ext head); official = local class")
    p.add_argument("--dry-run", action="store_true")
    add_device_args(p)
    return p.parse_args()


def train_seed(args):
    return args.train_seed if args.train_seed is not None else args.seed


def tokenize_fn(examples, tokenizer):
    # Upstream rule: text.upper() only (tokenizer uppercases internally;
    # NO U->T). Mask token ids 2|3 in attention (upstream tokenize_function_prokbert).
    enc = tokenizer([str(s).upper() for s in examples["segment"]],
                    add_special_tokens=True, padding=False, truncation=False)
    ids = [np.array(r, dtype=np.int64) for r in enc["input_ids"]]
    mask = [np.where((r == 2) | (r == 3), 0, 1).tolist() for r in ids]
    enc["attention_mask"] = mask
    enc["labels"] = examples["y"]
    return enc


def build_model(device, dropout, loader="official"):
    if loader == "auto":
        # (c) As our mix used: AutoModel resolution (may bind Ext head).
        from transformers import AutoModelForSequenceClassification as _Auto
        from transformers import AutoConfig as _AutoConfig
        config = _AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        config.num_labels = 2
        config.classifier_dropout = dropout
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        config.problem_type = "single_label_classification"
        model = _Auto.from_pretrained(BASE_MODEL, config=config, trust_remote_code=True)
        print(f"Loader: auto -> {model.__class__.__name__} (fresh head expected)")
        model.to(device)
        return model
    # Manual load: local class + local config (their custom from_pretrained
    # re-resolves the REMOTE ProkBertConfig, which the local head rejects).
    # Backbone weights come from the hub; the head stays freshly initialized.
    # Manual load: local class + local config (their custom from_pretrained
    # re-resolves the REMOTE ProkBertConfig, which the local head rejects).
    # Backbone weights come from the hub; the head stays freshly initialized.
    from prokbert.models.megatronbert import ProkBertConfig
    from transformers import AutoConfig as _AutoConfig
    remote = _AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    config = ProkBertConfig.from_dict(remote.to_dict())
    assert type(config).__module__.startswith("prokbert."), \
        f"config class leaked: {type(config).__module__}"
    config.num_labels = 2
    config.classifier_dropout = dropout
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    config.problem_type = "single_label_classification"
    config.label2id = {"0": 0, "1": 1}
    config.id2label = {0: "0", 1: "1"}
    model = ProkBertForSequenceClassification(config)
    print(f"Loader: official ({model.__class__.__name__}, fresh head)")

    from huggingface_hub import hf_hub_download
    try:
        wpath = hf_hub_download(BASE_MODEL, filename="model.safetensors")
        from safetensors.torch import load_file
        sd = load_file(wpath)
    except Exception:
        wpath = hf_hub_download(BASE_MODEL, filename="pytorch_model.bin")
        sd = torch.load(wpath, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Weights from {wpath.split('/')[-1]}: missing(head, fresh)=[{', '.join(sorted(missing))}] "
          f"unexpected={sorted(unexpected)}")
    assert all(k == "cls.seq_relationship.weight" or k == "cls.seq_relationship.bias"
               or k.startswith("cls.predictions.") for k in unexpected), \
        f"non-MLM unexpected keys: {unexpected}"
    assert missing and all(k.split(".")[0] in ("classifier", "weighting_layer", "pooler", "score")
                           for k in missing), f"non-head keys missing: {missing}"
    print(f"Model: {model.__class__.__name__} (official, fresh head) from {BASE_MODEL}")
    model.to(device)
    return model


def build_training_args(args):
    if args.warmup_ratio is not None:
        warmup_steps, warmup_ratio = 0, args.warmup_ratio
    elif args.recipe == "paper":
        # Paper §2.3.2 optimizer: betas 0.95/0.98 + linear warmup 500.
        warmup_steps, warmup_ratio = 500, 0.0
    else:
        warmup_steps, warmup_ratio = 0, 0.0
    kw = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=128,
        learning_rate=args.learning_rate,
        adam_epsilon=5e-05,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps, warmup_ratio=warmup_ratio,
        logging_steps=50, save_steps=500,
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model=args.metric_for_best,
        greater_is_better=True,
        report_to="none", seed=train_seed(args),
        bf16=False, fp16=False,
        dataloader_num_workers=0, optim="adamw_torch",
    )
    if args.recipe == "paper":
        kw.update(adam_beta1=0.95, adam_beta2=0.98)
    return TrainingArguments(**kw)


def dump_probs(trainer, split, path):
    from scipy.special import softmax
    pred = trainer.predict(split)
    scores = softmax(pred.predictions, axis=-1)[:, 1]
    save_probs_csv(path, np.asarray(pred.label_ids, dtype=int), np.asarray(scores))
    return path


def main():
    args = parse_args()
    set_seed(train_seed(args))
    if args.output_dir is None:
        args.output_dir = f"finetuning_outputs/replica_{args.recipe}_2ep"
    device, use_cuda = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} | recipe={args.recipe} | out={args.output_dir}")

    _, train_80, val_10, test_10 = load_hf_split(args.seed, "80/10/10")
    from datasets import load_dataset as _load_ds
    if args.full_train:
        # (a) FULL HF train (223092, no holdout); select on external sigma70.
        ds = _load_ds(HF_DATASET)
        assert len(ds["train"]) == 223092
        train_ds = minimal(ds["train"])
        val_ds = minimal(ds["test_sigma70"])
        select_sets = [("val_sigma70", val_ds)]
        print(f"Replica data FULL: train {len(train_ds)} val-sigma70 {len(val_ds)} "
              f"(test_10/sigma70/multi reported locked post-hoc)")
    else:
        train_ds, val_ds = minimal(train_80), minimal(val_10)
        select_sets = []
        print(f"Replica data: train {len(train_ds)} val {len(val_ds)} (+locked test_10 {len(test_10)})")
    train_ds = train_ds.shuffle(seed=train_seed(args))

    tokenizer = ProkBERTTokenizer(tokenization_params={"kmer": 6, "shift": 1},
                                  operation_space="sequence")
    kw = {"fn_kwargs": {"tokenizer": tokenizer}, "batched": True, "batch_size": 1000}
    tok_train = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, **kw)
    tok_val = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names, **kw)
    print(f"Tokenized: train {len(tok_train)} val {len(tok_val)}")

    if args.dry_run:
        print("DRY RUN OK — splits/hashes/tokenizer verified, no training started")
        return

    model = build_model(device, args.dropout, args.model_loader)
    training_args = build_training_args(args)
    trainer_cls = ProkBERTTrainer if args.recipe == "code" else Trainer
    print(f"Trainer: {trainer_cls.__name__}")
    trainer = trainer_cls(
        model=model, args=training_args, train_dataset=tok_train, eval_dataset=tok_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_compute_metrics)

    write_run_info(args.output_dir, {
        "script": "scripts/train_replica.py", "recipe": args.recipe,
        "model_name": BASE_MODEL,
        "model_class": model.__class__.__name__,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "hf_dataset": HF_DATASET,
        "hf_split": {"train_80": len(train_80), "val_10": len(val_10), "test_10": len(test_10),
                     "val_sha16": EXPECTED["val_sha"], "test_sha16": EXPECTED["test_sha"],
                     "seed": args.seed},
        "device": device, "seed": args.seed, "train_seed": train_seed(args),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "adam_epsilon": 5e-05,
        "adam_betas": [0.95, 0.98] if args.recipe == "paper" else "torch-default (code ignores)",
        "warmup_steps": training_args.warmup_steps, "warmup_ratio": args.warmup_ratio,
        "dropout": args.dropout, "select": f"val_10 {args.metric_for_best}",
        "full_train": args.full_train, "model_loader": args.model_loader,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
    })

    print("=== Training replica_%s%s %sep (loader=%s) ===" % (
        args.recipe, "+full" if args.full_train else "", args.num_train_epochs, args.model_loader))
    trainer.train()
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    print(f"Saved to {best}")

    from datasets import load_dataset as _load_ds2
    _ds = _load_ds2(HF_DATASET)
    print("=== Locked eval (test_10 + sigma70 + multispecies report) ===")
    locked = {}
    for name, split in (("test_10", test_10), ("sigma70", minimal(_ds["test_sigma70"])),
                        ("multispecies", minimal(_ds["test_multispecies"]))):
        tok = split.map(tokenize_fn, batched=True, batch_size=1000,
                        remove_columns=split.column_names,
                        fn_kwargs={"tokenizer": tokenizer})
        m = trainer.evaluate(tok)
        locked[name] = m
        print(name, m)
        dump_probs(trainer, tok, f"{args.output_dir.rstrip('/')}/{name}_probs.csv")
    _ri = json.load(open(f"{args.output_dir.rstrip('/')}/run_info.json"))
    _ri["locked_eval"] = locked
    _ri["locked_leakage"] = ({"test_10": "train_overlap", "sigma70": "selection-set",
                              "multispecies": "clean"} if args.full_train else
                             {"test_10": "clean", "sigma70": "clean", "multispecies": "clean"})
    write_run_info(args.output_dir, _ri)


if __name__ == "__main__":
    main()
