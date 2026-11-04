"""Tunable fine-tuning package: modular, argparse-only, repo-convention compatible.

Modules mirror the pipeline stages of scripts/train_mix223k.py but factored
for reuse: config (hyperparams) -> data -> model -> train -> evaluate.
New code only; existing scripts/* are imported, never modified.
"""
__version__ = "0.1.0"
