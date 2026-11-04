#!/usr/bin/env python3
"""Training orchestration: same stage order as scripts/train_mix223k.main."""
import json
from pathlib import Path

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "tunable")


def run(args):
    """Full run: data -> tokenize -> model -> trainer -> best + locked evals.

    Returns {"output_dir": ..., "manifest": ..., "locked": ...}.
    """
    import torch
    from transformers import (DataCollatorWithPadding, EarlyStoppingCallback,
                              set_seed)
    from scripts import train_mix223k as T
    from scripts.common import resolve_device
    from scripts.metrics import trainer_compute_metrics
    from prokbert_patches import write_run_info

    import tunable.data as tdata
    import tunable.model as tmodel

    set_seed(T.train_seed(args))
    if args.output_dir is None:
        args.output_dir = f"finetuning_outputs/tunable_{Path(args.data_dir).name}"
    device, use_cuda = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} | data={Path(args.data_dir).name} | out={args.output_dir}")

    d = tdata.load_all(args)
    data_name, manifest = d["data_name"], d["manifest"]
    train_ds, val_ds, locked_sets = d["train_ds"], d["val_ds"], d["locked_sets"]

    from scripts.train_mix223k import build_tokenizer, tokenize_splits
    tokenizer = build_tokenizer()
    tok_train, tok_val = tokenize_splits(tokenizer, train_ds, val_ds)

    if args.dry_run:
        print("DRY RUN OK -- data, splits and hashes verified, no training started")
        print(json.dumps(manifest, indent=2))
        return {"output_dir": args.output_dir, "manifest": manifest, "locked": None, "dry_run": True}

    model, minfo = tmodel.build_model(args, device)
    training_args = T.build_training_args(args, use_cuda)
    diff_opt = T.build_diff_lr_optimizer_(model, args)
    trainer = T.LabelSmoothTrainer(
        model=model, args=training_args, train_dataset=tok_train, eval_dataset=tok_val,
        ls=args.label_smoothing,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_cuda else None),
        compute_metrics=trainer_compute_metrics,
        optimizers=(diff_opt, None) if diff_opt else (None, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience)]
        if args.es_patience else None)

    write_run_info(args.output_dir, {
        "script": "tunable/train.py", "model_name": T.BASE_MODEL,
        "model_name_or_path": getattr(model.config, "_name_or_path", T.BASE_MODEL),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "mode": "full-train" if args.full_train else "parallel",
        "data_dir": args.data_dir, "data_name": data_name, "device": device, "seed": args.seed,
        "hf_split_spec": args.hf_split_spec, "train_seed": T.train_seed(args),
        "pneumo_upsample": getattr(args, "pneumo_upsample", 1),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "warmup_ratio": args.warmup_ratio,
        "es_patience": args.es_patience, "select_metric": args.select_metric, "dropout": args.dropout,
        "dropout_head": minfo["dropout_head"], "dropout_hidden": minfo["dropout_hidden"],
        "dropout_attn": minfo["dropout_attn"], "dropout_layers": minfo["dropout_layers"],
        "label_smoothing": args.label_smoothing,
        "freeze_backbone": args.freeze_backbone, "freeze_bottom_k": args.freeze_bottom_k,
        "freeze_top_k": args.freeze_top_k, "mlp_head": args.mlp_head,
        "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
        "probe_checkpoint": args.probe_checkpoint,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "data_manifest": manifest,
    })

    print(f"=== Training {data_name} mix {args.num_train_epochs}ep (tunable) ===")
    trainer.train()
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    print(f"Saved to {best}")

    T.report_locked_eval(trainer, tokenizer, args.output_dir, locked_sets)
    return {"output_dir": args.output_dir, "manifest": manifest, "locked": [n for n, _ in locked_sets],
            "dry_run": False}
