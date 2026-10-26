#!/usr/bin/env python3
"""
Mixed training: HF bacterial_promoters train_80 + pneumo train, parallel protocol.

Fit:      HF train_80 (178473) + pneumo train (d39v 1381 | dedup 2194), shuffled seed 42
Select:   combined val = HF val_10 (22309) + pneumo eval (296 | 470), best by MCC
Locked:   HF test_10 (22310), data test, sigma70, multispecies (eval only)

Mirrors the legacy mix recipe (paper hyperparams, fp32, warmup 500, eval/save steps 50/500)
but leaves a true in-dist holdout untouched. Run from the repo root.

Full-train mode (--full-train): fit FULL HF train (223092) + 1to2-train + dedup-train
segments unseen in 1to2 (no duplicates); select best by MCC on sigma70 (external,
clean); paper-test/test_10 reported as train_overlap diagnostics. HF-promoter analog.

Usage:
  pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/mini_223k_d39v_tigr4_2ep
  pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --dry-run   # validate data/splits only
  pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_high_1to2 --full-train --output-dir finetuning_outputs/mini_223kfull_1to2_dedup_2ep

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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from scripts.metrics import trainer_compute_metrics


class LabelSmoothTrainer(Trainer):
    """Trainer with optional cross-entropy label smoothing (ls=0: identical to Trainer)."""

    def __init__(self, *a, ls=0.0, **k):
        super().__init__(*a, **k)
        self._ls = ls

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not self._ls:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)
        import torch.nn.functional as F
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        loss = F.cross_entropy(outputs.logits.view(-1, model.config.num_labels), labels.view(-1),
                               label_smoothing=self._ls)
        return (loss, outputs) if return_outputs else loss
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
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-split-spec", type=str, default="80/10/10",
                   help="HF split: 80/10/10 (fixed, SHA-asserted), 75/15/10, 75/10/15, "
                        "species80/10/10 (species-balanced 80/10/10, rares<10 grouped as OTHER), "
                        "or local:<spec> (offline versioned split from datasets/hf_splits/<spec>, "
                        "SHA-verified against its manifest; e.g. local:70/15/15)")
    p.add_argument("--hf-subset", type=int, default=None,
                   help="cap-mode subset of HF train_80 to N rows (50/50: species-cap "
                        "positives + subtype-proportional negatives; parallel-mode, "
                        "80/10/10 base only)")
    p.add_argument("--train-seed", type=int, default=None,
                   help="training randomness (shuffle/init/order); default = --seed. Splits always use --seed.")
    p.add_argument("--num-train-epochs", type=float, default=2,
                   help="epochs, float allowed (e.g. 1.5, 2.5) to walk the gap ladder between integers")
    p.add_argument("--per-device-train-batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.0004)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="if set, warmup_steps forced to 0 and this fraction of total "
                        "steps used (e.g. 0.1); fixed steps confound epoch/subset comparisons")
    p.add_argument("--es-patience", type=int, default=None,
                   help="early stopping patience in evals (e.g. 3 = stop if no MCC gain "
                        "in 3 consecutive evals of 50 steps); needs load_best (always on). "
                        "Set epochs high (e.g. 6) and let the val curve decide the stop")
    p.add_argument("--hf-fold-dir", type=str, default=None,
                   help="joint-CV mode: folder with hf_train.csv/hf_eval.csv/hf_test.csv "
                        "(segment_id,segment,y) replacing the fixed 80/10/10 split; "
                        "combine with --data-dir pneumo fold for aligned full-data CV")
    p.add_argument("--select-metric", type=str, default="mcc",
                   help="metric for best-checkpoint and ES: mcc (deployment objective, "
                        "historic default) or roc_auc (threshold-free, stable; use for gap work)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="classifier+hidden+attention dropout (default 0.1)")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="freeze everything except the classification head "
                        "(classifier/score/weighting_layer/pooler stay trainable); "
                        "dropout noise stays active, only weight updates stop")
    p.add_argument("--probe-checkpoint", type=str, default=None,
                   help="with --freeze-backbone: load backbone weights from this "
                        "checkpoint dir instead of the pretrained base, then re-init "
                        "the head fresh (linear probe of learned representations)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="cross-entropy label smoothing (default 0 = off)")
    p.add_argument("--dry-run", action="store_true", help="build data/splits, verify hashes, exit before training")
    p.add_argument("--full-train", action="store_true",
                   help="fit FULL HF train + data train + dedup-unseen rows; select on sigma70 (no in-dist holdout)")
    p.add_argument("--dedup-dir", type=str, default="datasets/d39v_tigr4_dedup_3135",
                   help="dedup dir for unseen extra rows in --full-train mode")
    add_device_args(p)
    return p.parse_args()


def normalize(s):
    return str(s).upper().replace("U", "T").strip()


def sha_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def minimal(hf_split):
    """Map an HF split to the minimal {segment_id, segment, y} schema."""
    return hf_split.map(lambda ex: {"segment_id": str(ex["segment_id"]), "segment": ex["segment"], "y": ex["y"]},
                        remove_columns=[c for c in hf_split.column_names if c not in ("segment_id", "segment", "y")])


def tokenize_fn(examples, tokenizer):
    enc = tokenizer([normalize(s) for s in examples["segment"]],
                    add_special_tokens=True, padding=False, truncation=False)
    enc["labels"] = examples["y"]
    return enc


def parse_ratio(spec):
    """'75/15/10' -> (0.75, 0.15, 0.10); validates sum 100."""
    parts = [float(x) for x in spec.split("/")]
    assert len(parts) == 3 and abs(sum(parts) - 100) < 1e-9, f"bad ratio {spec}"
    return parts[0] / 100, parts[1] / 100, parts[2] / 100


def load_hf_split(seed, spec="80/10/10"):
    """HF train split. Fixed 80/10/10 (identical code as paper_replica.py, SHA-asserted);
    75/15/10, 75/10/15 (same code path, new SHAs recorded); species80/10/10
    (species-balanced via local parquet, rares<10 grouped as OTHER)."""
    if spec == "80/10/10":
        ds = load_dataset(HF_DATASET)
        assert len(ds["train"]) == 223092, f"unexpected HF train size {len(ds['train'])}"
        tmp = ds["train"].train_test_split(test_size=0.2, seed=seed)
        train_80, temp = tmp["train"], tmp["test"]
        vt = temp.train_test_split(test_size=0.5, seed=seed)
        val_10, test_10 = vt["train"], vt["test"]
        assert (len(train_80), len(val_10), len(test_10)) == (EXPECTED["train_80"], EXPECTED["val_10"], EXPECTED["test_10"])
        assert sha_ids(val_10["segment_id"])[:16] == EXPECTED["val_sha"], "val split IDs differ from manifest"
        assert sha_ids(test_10["segment_id"])[:16] == EXPECTED["test_sha"], "test split IDs differ from manifest"
        print(f"HF split OK: train {len(train_80)} val {len(val_10)} test {len(test_10)}")
        return ds, train_80, val_10, test_10
    if spec.startswith("species"):
        import pandas as pd
        from datasets import Dataset
        ratio = spec[len("species"):]
        ftr, fva, _ = parse_ratio(ratio)
        loc = Path("data/bacterial_promoters/train.parquet")
        assert loc.exists(), "species mode needs local train.parquet"
        df = pd.read_parquet(loc)
        assert len(df) == 223092, f"unexpected parquet size {len(df)}"
        df["ppd_original_SpeciesName"] = df["ppd_original_SpeciesName"].fillna("UNKNOWN")
        counts = df["ppd_original_SpeciesName"].value_counts()
        rare = set(counts[counts < 10].index)
        df["sp_group"] = df["ppd_original_SpeciesName"].where(~df["ppd_original_SpeciesName"].isin(rare), "OTHER")
        print(f"species groups: {df['sp_group'].nunique()} (OTHER n={(df['sp_group'] == 'OTHER').sum()})")
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        i_tr, i_tmp = train_test_split(idx, test_size=1 - ftr, random_state=seed, stratify=df["sp_group"])
        i_va, i_te = train_test_split(i_tmp, test_size=fva / (1 - ftr), random_state=seed,
                                      stratify=df.iloc[i_tmp]["sp_group"])
        tr = Dataset.from_pandas(df.iloc[i_tr].reset_index(drop=True))
        va = Dataset.from_pandas(df.iloc[i_va].reset_index(drop=True))
        te = Dataset.from_pandas(df.iloc[i_te].reset_index(drop=True))
        print(f"HF species split {ratio} OK: train {len(tr)} val {len(va)} test {len(te)} "
              f"(val_sha {sha_ids(va['segment_id'])[:16]}, test_sha {sha_ids(te['segment_id'])[:16]})")
        ds = load_dataset(HF_DATASET)
        return ds, tr, va, te
    ftr, fva, _ = parse_ratio(spec)
    ds = load_dataset(HF_DATASET)
    assert len(ds["train"]) == 223092, f"unexpected HF train size {len(ds['train'])}"
    tmp = ds["train"].train_test_split(test_size=1 - ftr, seed=seed)
    tr, temp = tmp["train"], tmp["test"]
    # vt["test"] holds fva/(1-ftr) of temp = val fraction of total
    vt = temp.train_test_split(test_size=fva / (1 - ftr), seed=seed)
    va, te = vt["test"], vt["train"]
    print(f"HF split {spec} OK: train {len(tr)} val {len(va)} test {len(te)} "
          f"(val_sha {sha_ids(va['segment_id'])[:16]}, test_sha {sha_ids(te['segment_id'])[:16]})")
    return ds, tr, va, te


def apply_hf_subset(train_80, n_total, seed):
    """Cap-mode subset of HF train_80 to n_total rows (50/50).

    Positives: per-species cap C binary-searched so kept ~= n_total//2;
    species with n<=C kept whole (floor: rares never cut).
    Negatives: proportional by prom_class subtype (CDS/comp_kt1_kc3/RND).
    Species/subtype mapped via local data/bacterial_promoters/train.parquet
    (same precedent as species80 mode). Deterministic in seed.
    Returns (subset_ds, info dict for run_info).
    """
    import collections
    import pandas as pd
    loc = Path("data/bacterial_promoters/train.parquet")
    assert loc.exists(), "hf-subset needs local train.parquet (species/subtype map)"
    pq = pd.read_parquet(loc, columns=["segment_id", "ppd_original_SpeciesName", "prom_class"])
    pq["segment_id"] = pq["segment_id"].astype(str)
    sp_map = dict(zip(pq["segment_id"], pq["ppd_original_SpeciesName"].fillna("UNKNOWN")))
    st_map = dict(zip(pq["segment_id"], pq["prom_class"]))
    ids = [str(i) for i in train_80["segment_id"]]
    y = np.array(train_80["y"])
    assert all(i in sp_map for i in ids), "parquet does not cover train_80 ids"
    n_pos, n_neg = n_total // 2, n_total - n_total // 2
    rng = np.random.RandomState(seed)
    pos_idx = np.array([i for i, yy in enumerate(y) if yy == 1])
    groups = collections.defaultdict(list)
    for ii in pos_idx:
        groups[sp_map[ids[ii]]].append(ii)
    counts = sorted(groups.items())
    ns = np.array([len(v) for _, v in counts])
    lo, hi = 1, int(ns.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if int(np.minimum(ns, mid).sum()) >= n_pos:
            hi = mid
        else:
            lo = mid + 1
    keep_pos, capped_pool, n_trim = [], [], 0
    for _, v in counts:
        v = np.array(v)
        if len(v) > lo:
            sel = rng.choice(v, size=lo, replace=False).tolist()
            keep_pos.extend(sel)
            capped_pool.extend(sel)
            n_trim += 1
        else:
            keep_pos.extend(v.tolist())
    excess = len(keep_pos) - n_pos
    assert excess >= 0, f"cap kept {len(keep_pos)} positives, want {n_pos}"
    if excess:
        drop = set(rng.choice(np.array(capped_pool), size=excess, replace=False).tolist())
        keep_pos = [i for i in keep_pos if i not in drop]
    assert len(keep_pos) == n_pos, f"cap kept {len(keep_pos)} positives, want {n_pos}"
    neg_idx = np.array([i for i, yy in enumerate(y) if yy == 0])
    sts = collections.defaultdict(list)
    for ii in neg_idx:
        sts[st_map[ids[ii]]].append(ii)
    st_counts = sorted(sts.items())
    quotas, rem = {}, n_neg
    for j, (s, v) in enumerate(st_counts):
        q = n_neg * len(v) // len(neg_idx) if j < len(st_counts) - 1 else rem
        quotas[s] = q
        rem -= q
    keep_neg = []
    for s, v in st_counts:
        v = np.array(v)
        q = quotas[s]
        keep_neg.extend(rng.choice(v, size=q, replace=False).tolist() if q < len(v) else v.tolist())
    assert len(keep_neg) == n_neg, f"kept {len(keep_neg)} negatives, want {n_neg}"
    keep = sorted(keep_pos + keep_neg)
    sub = train_80.select(keep)
    info = {"mode": "cap", "n": n_total, "seed": seed, "pos_cap": lo,
            "species_trimmed": n_trim, "species_intact": len(counts) - n_trim,
            "neg_subtypes": {s: quotas[s] for s, _ in st_counts},
            "subset_sha16": sha_ids([ids[i] for i in keep])[:16]}
    print(f"HF subset cap-mode: {len(sub)} rows (pos {n_pos} cap {lo}, "
          f"trimmed {n_trim}/{len(counts)} species; neg {quotas}); sha {info['subset_sha16']}")
    return sub, info


def load_data_splits(args, data_name):
    """Extra-data splits (train.csv/eval.csv/test.csv), mapped to the minimal schema.

    Splits load via pandas (the csv builder rejects 0-row files); an empty train
    means 223k-only (used by --full-train ablations).
    """
    import pandas as pd
    from datasets import Dataset
    pneu_raw = {}
    for k in ("train", "eval", "test"):
        df0 = pd.read_csv(f"{args.data_dir}/{k}.csv")
        assert {"segment", "y"} <= set(df0.columns), f"{args.data_dir}/{k}.csv must have segment,y columns"
        keep = ["segment", "y"] + (["id"] if "id" in df0.columns else [])
        pneu_raw[k] = Dataset.from_pandas(df0[keep], preserve_index=False)
    for k in ("train", "eval", "test"):
        cols = set(pneu_raw[k].column_names)
        assert {"segment", "y"} <= cols, f"{args.data_dir}/{k}.csv must have segment,y columns (has {sorted(cols)})"
    _tr, _ev, _te = (set(pneu_raw["train"]["segment"]), set(pneu_raw["eval"]["segment"]), set(pneu_raw["test"]["segment"]))
    assert not (_tr & _ev) and not (_tr & _te) and not (_ev & _te), "train/eval/test splits overlap in --data-dir"
    pneu = {}
    for k in ("train", "eval", "test"):
        has_id = "id" in pneu_raw[k].column_names
        pneu[k] = pneu_raw[k].map(
            lambda ex, i, _h=has_id: {"segment_id": str(ex["id"]) if _h else f"row-{i}",
                                      "segment": ex["segment"], "y": ex["y"]},
            with_indices=True, remove_columns=pneu_raw[k].column_names)
    print(f"Data OK ({data_name}): train {len(pneu['train'])} eval {len(pneu['eval'])} test {len(pneu['test'])}")
    return pneu_raw, pneu


def train_seed(args):
    """Training randomness; splits always use --seed (locked)."""
    return args.train_seed if args.train_seed is not None else args.seed


def compose_parallel(args, train_80, val_10, test_10, pneu):
    """Parallel composition: HF train_80 + pneumo train (fit); HF val_10 + pneumo eval (select)."""
    train_ds = concatenate_datasets([minimal(train_80), pneu["train"]]).shuffle(seed=train_seed(args))
    val_ds = concatenate_datasets([minimal(val_10), pneu["eval"]])
    print(f"Combined: train {len(train_ds)} val {len(val_ds)} (+locked test_10 {len(test_10)}, data test {len(pneu['test'])})")
    return train_ds, val_ds


def load_hf_fold(hf_fold_dir):
    """Joint-CV mode: HF fold CSVs (segment_id,segment,y) instead of fixed split."""
    import pandas as pd
    from datasets import Dataset
    out = {}
    for k in ("train", "eval", "test"):
        df = pd.read_csv(f"{hf_fold_dir}/hf_{k}.csv")
        assert {"segment_id", "segment", "y"} <= set(df.columns), f"{hf_fold_dir}/hf_{k}.csv schema"
        out[k] = Dataset.from_pandas(df[["segment_id", "segment", "y"]], preserve_index=False)
    _a, _b, _c = (set(out["train"]["segment"]), set(out["eval"]["segment"]), set(out["test"]["segment"]))
    assert not (_a & _b) and not (_a & _c) and not (_b & _c), "HF fold CSVs overlap"
    print(f"HF fold OK ({hf_fold_dir}): train {len(out['train'])} eval {len(out['eval'])} test {len(out['test'])}")
    return out["train"], out["eval"], out["test"]


def compose_full(args, ds, pneu):
    """Full-train composition (HF-promoter analog): FULL HF train + data train +
    dedup-train segments unseen in data train (no duplicates); select on sigma70."""
    import pandas as pd
    full = ds["train"]
    assert len(full) == 223092, f"unexpected HF train size {len(full)}"
    seen = {str(s).upper() for s in pneu["train"]["segment"]} if len(pneu["train"]) else set()
    ddf = pd.read_csv(f"{args.dedup_dir}/train.csv")
    extra = ddf[~ddf["segment"].str.upper().isin(seen)].copy()
    print(f"dedup-unseen extra rows: {len(extra)} (excluded {len(ddf) - len(extra)} duplicates)")
    extra_ds = extra.assign(segment_id=extra["id"].astype(str))[["segment_id", "segment", "y"]]
    from datasets import Dataset
    extra_min = Dataset.from_pandas(extra_ds, preserve_index=False)
    train_ds = concatenate_datasets([minimal(full), pneu["train"], extra_min]).shuffle(seed=train_seed(args))
    val_ds = minimal(ds["test_sigma70"])
    print(f"Full: train {len(train_ds)} (full {len(full)} + data {len(pneu['train'])} + dedup-unseen {len(extra_min)}) "
          f"val sigma70 {len(val_ds)}")
    return train_ds, val_ds, len(extra_min), len(ddf) - len(extra_min)


def build_manifest(args, data_name, train_80, val_10, test_10, pneu_raw, train_ds, val_ds, hf_subset_info=None):
    return {
        "hf_dataset": HF_DATASET, "hf_train": 223092,
        "hf_split": {"train_80": len(train_80), "val_10": len(val_10), "test_10": len(test_10),
                     "val_sha16": sha_ids(val_10["segment_id"])[:16],
                     "test_sha16": sha_ids(test_10["segment_id"])[:16], "seed": args.seed,
                     "spec": args.hf_split_spec},
        "hf_subset": hf_subset_info,
        "data_dir": args.data_dir, "data_name": data_name,
        "data_splits": {k: {"n": len(pneu_raw[k]), "sha256": hashlib.sha256(
            open(f"{args.data_dir}/{k}.csv", "rb").read()).hexdigest()} for k in ("train", "eval", "test")},
        "combined": {"train": len(train_ds), "val": len(val_ds)},
    }


def build_tokenizer():
    tok = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    print(f"Tokenizer k=6 s=1 ready, vocab {len(tok.get_vocab())}")
    return tok


def tokenize_splits(tokenizer, train_ds, val_ds):
    kw = {"fn_kwargs": {"tokenizer": tokenizer}, "batched": True, "batch_size": 1000}
    tok_train = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, **kw)
    tok_val = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names, **kw)
    print(f"Tokenized: train {len(tok_train)} val {len(tok_val)}")
    return tok_train, tok_val


def load_model(device, dropout=0.1):
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    config.num_labels = 2
    config.classifier_dropout = dropout
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    config.problem_type = "single_label_classification"
    config.label2id = {"0": 0, "1": 1}
    config.id2label = {0: "0", 1: "1"}
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config, trust_remote_code=True)
    print(f"Model: {model.__class__.__name__} from {BASE_MODEL}")
    model.to(device)
    return model


HEAD_KEYS = ("classifier", "score", "weighting_layer", "pooler")


def freeze_backbone_(model):
    """Freeze encoder/embeddings; head stays trainable. Returns (n_train, n_frozen)."""
    n_tr = n_fr = 0
    for name, p in model.named_parameters():
        if name.split(".")[0] in HEAD_KEYS:
            p.requires_grad = True
            n_tr += p.numel()
        else:
            p.requires_grad = False
            n_fr += p.numel()
    print(f"Frozen backbone: trainable {n_tr:,} / frozen {n_fr:,} params")
    assert n_tr > 0, "no trainable head params found — check HEAD_KEYS"
    return n_tr, n_fr


def load_probe_backbone_(model, ckpt_dir):
    """Load backbone weights from a trained checkpoint, then re-init head fresh."""
    import torch
    from safetensors.torch import load_file
    sd = load_file(f"{ckpt_dir.rstrip('/')}/model.safetensors")
    own = set(model.state_dict().keys())
    backbone = {k: v for k, v in sd.items() if k.split(".")[0] not in HEAD_KEYS and k in own}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    print(f"Probe backbone from {ckpt_dir}: loaded {len(backbone)} tensors")
    for mod_name in HEAD_KEYS:
        mod = getattr(model, mod_name, None)
        if mod is not None and hasattr(mod, "reset_parameters"):
            try:
                mod.reset_parameters()
            except Exception as e:
                print(f"  head {mod_name}: reset skipped ({e})")
    print("Probe: head re-initialized fresh")


def build_training_args(args, use_cuda):
    if args.warmup_ratio is not None:
        warmup_steps, warmup_ratio = 0, args.warmup_ratio
    else:
        warmup_steps, warmup_ratio = args.warmup_steps, 0.0
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=128,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps, warmup_ratio=warmup_ratio,
        adam_beta1=0.95, adam_beta2=0.98, adam_epsilon=5e-05,
        logging_steps=50, save_steps=500,
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model=args.select_metric, greater_is_better=True,
        report_to="none", seed=train_seed(args),
        bf16=False, fp16=False,
        dataloader_num_workers=0, optim="adamw_torch",
    )


def report_locked_eval(trainer, tokenizer, output_dir, locked_sets):
    """Locked eval (report only) over explicit (name, split) pairs."""
    print("=== Locked eval (report only) ===")
    locked = {}
    for name, sds in locked_sets:
        m = trainer.evaluate(sds.map(tokenize_fn, batched=True, batch_size=1000,
                                     remove_columns=sds.column_names,
                                     fn_kwargs={"tokenizer": tokenizer}))
        locked[name] = m
        print(name, m)
    _ri = json.load(open(f"{output_dir.rstrip('/')}/run_info.json"))
    _ri["locked_eval"] = locked
    write_run_info(output_dir, _ri)


def main():
    args = parse_args()
    set_seed(train_seed(args))
    data_name = Path(args.data_dir).name
    if args.output_dir is None:
        args.output_dir = f"finetuning_outputs/mini_223k_{data_name}_2ep"
    device, use_cuda = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device} (auto-detect; override with --gpu N or --cpu) | data={data_name} | out={args.output_dir}")

    if args.hf_split_spec.startswith("local:"):
        from scripts.hf_local import resolve_split
        from datasets import load_dataset
        spec = args.hf_split_spec.split("local:", 1)[1]
        ds = load_dataset(HF_DATASET)  # remote test splits stay remote (sigma70/multi referee)
        local = resolve_split(spec)
        train_80, val_10, test_10 = local["train"], local["eval"], local["test"]
    else:
        ds, train_80, val_10, test_10 = load_hf_split(args.seed, args.hf_split_spec)
    hf_subset_info = None
    joint_fold = None
    referee_10 = test_10
    if args.hf_subset:
        assert args.hf_split_spec in ("80/10/10", "local:80/10/10"), \
            "--hf-subset needs the SHA-asserted 80/10/10 base (fixed or local:)"
    if args.hf_fold_dir:
        assert not args.full_train and not args.hf_subset, "--hf-fold-dir is parallel-mode, no subset"
        joint_fold = args.hf_fold_dir
        train_80, val_10, test_10 = load_hf_fold(args.hf_fold_dir)
    if args.hf_subset:
        assert not args.full_train, "--hf-subset is parallel-mode only"
        train_80, hf_subset_info = apply_hf_subset(train_80, args.hf_subset, args.seed)
    pneu_raw, pneu = load_data_splits(args, data_name)
    if args.full_train:
        train_ds, val_ds, n_extra, n_excluded = compose_full(args, ds, pneu)
        manifest = {
            "mode": "full-train", "hf_dataset": HF_DATASET, "hf_full_train": 223092,
            "data_dir": args.data_dir, "data_name": data_name,
            "data_splits": {k: {"n": len(pneu_raw[k]), "sha256": hashlib.sha256(
                open(f"{args.data_dir}/{k}.csv", "rb").read()).hexdigest()} for k in ("train", "eval", "test")},
            "dedup_dir": args.dedup_dir, "dedup_unseen_extra": n_extra, "dedup_excluded_dup": n_excluded,
            "select": "sigma70 MCC (external, clean)",
            "combined": {"train": len(train_ds), "val": len(val_ds)},
        }
        locked_sets = [("val_sigma70", val_ds), ("data_test", pneu["test"]),
                       ("multispecies", ds["test_multispecies"])]
    else:
        train_ds, val_ds = compose_parallel(args, train_80, val_10, test_10, pneu)
        manifest = build_manifest(args, data_name, train_80, val_10, test_10, pneu_raw, train_ds, val_ds,
                                  hf_subset_info)
        locked_sets = [("val_combined", val_ds), ("test_10", test_10), ("data_test", pneu["test"]),
                       ("sigma70", ds["test_sigma70"]), ("multispecies", ds["test_multispecies"])]
        if joint_fold:
            manifest["hf_fold_dir"] = joint_fold
            # fold-test replaces test_10 as the report set; fixed test_10 stays as referee
            locked_sets = [("val_combined", val_ds), ("fold_test", test_10), ("data_test", pneu["test"]),
                           ("test_10", referee_10),
                           ("sigma70", ds["test_sigma70"]), ("multispecies", ds["test_multispecies"])]

    tokenizer = build_tokenizer()
    tok_train, tok_val = tokenize_splits(tokenizer, train_ds, val_ds)

    if args.dry_run:
        print("DRY RUN OK — data, splits and hashes verified, no training started")
        print(json.dumps(manifest, indent=2))
        return

    model = load_model(device, dropout=args.dropout)
    if args.probe_checkpoint and not args.freeze_backbone:
        raise SystemExit("--probe-checkpoint requires --freeze-backbone")
    if args.probe_checkpoint:
        load_probe_backbone_(model, args.probe_checkpoint)
    if args.freeze_backbone:
        freeze_backbone_(model)
    training_args = build_training_args(args, use_cuda)
    trainer = LabelSmoothTrainer(model=model, args=training_args, train_dataset=tok_train, eval_dataset=tok_val, ls=args.label_smoothing,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_cuda else None),
                      compute_metrics=trainer_compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.es_patience)]
                      if args.es_patience else None)

    write_run_info(args.output_dir, {
        "script": "scripts/train_mix223k.py", "model_name": BASE_MODEL,
        "model_name_or_path": getattr(model.config, "_name_or_path", BASE_MODEL),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "mode": "full-train" if args.full_train else "parallel",
        "data_dir": args.data_dir, "data_name": data_name, "device": device, "seed": args.seed,
        "hf_split_spec": args.hf_split_spec, "train_seed": train_seed(args),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "warmup_ratio": args.warmup_ratio,
        "es_patience": args.es_patience, "select_metric": args.select_metric, "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "freeze_backbone": args.freeze_backbone, "probe_checkpoint": args.probe_checkpoint,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "data_manifest": manifest,
    })

    print(f"=== Training {data_name} mix {args.num_train_epochs}ep ===")
    trainer.train()
    best = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best)
    tokenizer.save_pretrained(best)
    print(f"Saved to {best}")

    report_locked_eval(trainer, tokenizer, args.output_dir, locked_sets)


if __name__ == "__main__":
    main()
