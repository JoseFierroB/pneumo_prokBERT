#!/usr/bin/env python3
"""Pneumo-ProkBERT zero-shot inference (CPU).

Adapted from promoter-tools/src/runners/prokbert_mini.py for S. pneumoniae D39V/TIGR4 81 bp.
Supports official variants with the same protocol:
  mini  = neuralbioinfo/prokbert-mini-promoter      (kmer 6, shift 1, 78 tokens on 81 bp) — promoter fine-tune, default
  minic = neuralbioinfo/prokbert-mini-c-promoter    (kmer 1, shift 1, 83 tokens on 81 bp) — promoter fine-tune
  base  = neuralbioinfo/prokbert-mini               (kmer 6, shift 1, 78 tokens) — MLM base, for fine-tuning only (no promoter head)

Note: mini-long (kmer 6, shift 2) is currently excluded — missing models.py in the
upstream repository (neuralbioinfo/prokbert-mini-long) and 81 bp would be split into 2x40,
which requires custom pooling.

Usage:
  python infer_pneumo_prokbert.py --variant mini --pos positives_81bp.fasta --neg negatives_81bp.fasta -o predictions.tsv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO

try:
    from transformers import AutoModelForSequenceClassification
    from prokbert.prokbert_tokenizer import ProkBERTTokenizer
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall: pip install torch transformers biopython pandas scikit-learn\n"
             "And: pip install git+https://github.com/nbrg-ppcu/prokbert.git")

VARIANTS = {
    "mini": ("neuralbioinfo/prokbert-mini-promoter", 6, 1),  # promoter fine-tune (default for inference)
    "minic": ("neuralbioinfo/prokbert-mini-c-promoter", 1, 1),  # promoter fine-tune, k1
    "base": ("neuralbioinfo/prokbert-mini", 6, 1),  # MLM base, not for direct promoter inference — use for fine-tuning
}


def predict_fasta(model, tokenizer, fasta_path, device, batch_size=32):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    ids = [r.id for r in records]
    seqs = [str(r.seq).upper() for r in records]
    preds = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        enc = tokenizer.batch_encode_plus(batch, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()
            preds.extend(probs)
    return ids, preds


def main():
    ap = argparse.ArgumentParser(description="Pneumo-ProkBERT zero-shot (CPU)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="mini")
    ap.add_argument("--pos", required=True, help="Positives FASTA (81 bp)")
    ap.add_argument("--neg", required=True, help="Negatives FASTA (81 bp)")
    ap.add_argument("-o", "--output", required=True, help="Output TSV {ID,LABEL,PRED}")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_id, kmer, shift = VARIANTS[args.variant]
    if args.variant == "base":
        sys.exit("Variant 'base' is the MLM pre-trained model without a promoter head. "
                 "Use it for fine-tuning (see finetuning.py / paper_replica.py), not for direct promoter inference. "
                 "Choose --variant mini or minic for zero-shot promoter prediction.")
    t0 = time.time()
    tokenizer = ProkBERTTokenizer(tokenization_params={"kmer": kmer, "shift": shift},
                                   operation_space="sequence")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    pos_ids, pos_preds = predict_fasta(model, tokenizer, args.pos, args.device, args.batch_size)
    neg_ids, neg_preds = predict_fasta(model, tokenizer, args.neg, args.device, args.batch_size)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds}),
                    pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds})],
                   ignore_index=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[{args.variant}/{model_id}] {len(df)} sequences "
          f"({len(pos_ids)} pos / {len(neg_ids)} neg) in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
