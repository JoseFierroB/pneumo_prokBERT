#!/usr/bin/env python3
"""Hyperparameters: paper-replica defaults + argparse-only overrides.

Same flag names, types and defaults as scripts/train_mix223k.py so the
resulting Namespace is drop-in compatible with its pipeline functions.
Presets only change parser defaults; any CLI flag overrides the preset.
"""
import argparse

import sys
sys.path.insert(0, ".")
from scripts.common import add_device_args

# Paper Frontiers 2.3.2 (also the OFAT replica center, except epochs).
PAPER_REPLICA = {
    "learning_rate": 0.0004,
    "per_device_train_batch_size": 128,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "warmup_ratio": None,
    "num_train_epochs": 2,
    "dropout": 0.1,
    "dropout_head": None,
    "dropout_hidden": None,
    "dropout_attn": None,
    "seed": 42,
    "select_metric": "mcc",
    "label_smoothing": 0.0,
    "es_patience": None,
}

PRESETS = {
    "replica": dict(PAPER_REPLICA),
    "replica-1ep": dict(PAPER_REPLICA, num_train_epochs=1),
    "ofat_center": dict(PAPER_REPLICA, num_train_epochs=1),
}


def build_parser(preset=None):
    p = argparse.ArgumentParser(description="Tunable ProkBERT fine-tuning (modular pipeline)")
    p.add_argument("--preset", type=str, default=preset, choices=[None, "replica", "replica-1ep", "ofat_center"],
                   help="preset filling parser defaults (any flag below overrides it)")
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", type=str, required=True,
                   help="Folder with train.csv/eval.csv/test.csv (segment,y +id)")
    g.add_argument("--output-dir", type=str, default=None)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--hf-split-spec", type=str, default="80/10/10")
    g.add_argument("--hf-subset", type=int, default=None)
    g.add_argument("--train-seed", type=int, default=None)
    g.add_argument("--pneumo-upsample", type=int, default=1)
    g.add_argument("--hf-fold-dir", type=str, default=None)
    g.add_argument("--full-train", action="store_true")
    g.add_argument("--dedup-dir", type=str, default="datasets/d39v_tigr4_dedup_3135")
    g = p.add_argument_group("optim")
    g.add_argument("--num-train-epochs", type=float, default=2)
    g.add_argument("--per-device-train-batch-size", type=int, default=128)
    g.add_argument("--learning-rate", type=float, default=0.0004)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--warmup-ratio", type=float, default=None)
    g.add_argument("--es-patience", type=int, default=None)
    g.add_argument("--select-metric", type=str, default="mcc")
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--lr-backbone", type=float, default=None)
    g.add_argument("--lr-head", type=float, default=None)
    g = p.add_argument_group("model")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--dropout-head", type=float, default=None)
    g.add_argument("--dropout-hidden", type=float, default=None)
    g.add_argument("--dropout-attn", type=float, default=None)
    g.add_argument("--dropout-layers", type=str, default=None)
    g.add_argument("--freeze-backbone", action="store_true")
    g.add_argument("--freeze-bottom-k", type=int, default=None, metavar="K")
    g.add_argument("--freeze-top-k", type=int, default=None, metavar="K")
    g.add_argument("--mlp-head", action="store_true")
    g.add_argument("--probe-checkpoint", type=str, default=None)
    g = p.add_argument_group("run")
    g.add_argument("--dry-run", action="store_true", help="build data/splits, verify hashes, exit before training")
    g.add_argument("--gap", action="store_true", help="run train-vs-test gap after training")
    g.add_argument("--gap-test-csv", type=str, default=None)
    g.add_argument("--gap-test-name", type=str, default="test-set")
    g.add_argument("--gap-out-dir", type=str, default=None)
    g.add_argument("--gap-batch-size", type=int, default=512)
    g.add_argument("--loss-overlay", action="store_true", default=True,
                   help="combined train-vs-val loss PNG after training")
    g.add_argument("--no-loss-overlay", dest="loss_overlay", action="store_false")
    add_device_args(p)
    return p


def parse_args(argv=None, preset=None):
    p = build_parser()
    args = p.parse_args(argv)
    name = args.preset or preset
    if name:
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name} (choices: {sorted(PRESETS)})")
        cli_given = set()
        raw = argv if argv is not None else sys.argv[1:]
        for tok in raw:
            if tok.startswith("--"):
                cli_given.add(tok[2:].split("=")[0].replace("-", "_"))
        cli_given.add("preset")
        for k, v in PRESETS[name].items():
            if k not in cli_given:
                setattr(args, k, v)
    return args
