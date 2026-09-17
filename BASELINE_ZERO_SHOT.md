# Pneumo-ProkBERT — zero-shot baseline (CPU)

Objective: Adapt ProkBERT to *Streptococcus pneumoniae* D39V / TIGR4 promoters (81 bp `[-60,+20]`).
Current phase: **zero-shot only**, no fine-tuning. Official models without retraining.

## Evaluated variants

| Variant | HF ID | kmer/shift | Tokens on 81 bp |
|---|---|---|---|
| mini | `neuralbioinfo/prokbert-mini-promoter` | 6 / 1 | 78 (76 + CLS/SEP) |
| minic (-c) | `neuralbioinfo/prokbert-mini-c-promoter` | 1 / 1 | 83 (81 + CLS/SEP) |
| minilong | `neuralbioinfo/prokbert-mini-long-promoter` | 6 / 2 | 2x40 segments — **blocked upstream**: missing `models.py` in `neuralbioinfo/prokbert-mini-long`, requires custom pooling for 81 bp |

Environment note: Use an environment with `MegatronBertModel` available (in promoter-tools it is the `tools/iPro-MP` env `ipro-mp`, not the pure `tools/prokbert` env). `torch CPU`, `transformers 4.57.6`.

## Zero-shot results (2026-09-11, CPU single thread)

Compatible output: TSV `{ID, LABEL, PRED}`.

| Dataset | N | mini AUC | minic AUC | mean_pos mini/minic | mean_neg |
|---|---|---|---|---|---|
| D39V full 989/1000 | 1989 | 0.970 | 0.970 | 0.951 / 0.946 | 0.222 / 0.236 |
| D39V SigA 397/1000 | 1397 | 0.991 | 0.995 | 0.989 / 0.988 | 0.222 / 0.236 |
| D39V SigX 21/1000 | 1021 | 0.935 | 0.987 | 0.865 / 0.947 | 0.222 / 0.236 |
| TIGR4 high 738/738 | 1476 | 0.921 | 0.923 | 0.889 / 0.880 | 0.255 / 0.256 |

Reference promoter-tools REPORT: iPro-MP 0.953, LCNN 0.949, MLDSPP75* 0.957 on D39V; iPro SigX 0.619.
Interpretation: ProkBERT zero-shot already exceeds the baseline on D39V; `-c` (kmer 1) generalizes better to the divergent SigX combox `TACGAATA`.

CPU timings: mini 84s / 1989 sequences (~23 seq/s), minic 115s (~17 seq/s). Peak RAM ~0.8-1.1 GB.

## Embeddings (probe)

`prokbert_pca_cls_421`: CLS 384d last layer, PCA 2D on D39V (SigA 150, SigX 21, None 100, Neg 150).
Centroids PC1: Neg -6.578 | SigX 0.312 | None 2.874 | SigA 4.618 — SigX intermediate, consistent with divergence.
See `output/plots/prokbert_pca_cls_421.{png,svg,pdf}` and `output/tables/prokbert_pca_cls_421.tsv` in promoter-tools.

## Usage

```bash
pip install torch transformers biopython pandas scikit-learn
pip install git+https://github.com/nbrg-ppcu/prokbert.git
python infer_pneumo_prokbert.py --variant mini --pos positives_81bp.fasta --neg negatives_81bp.fasta -o predictions.tsv
python infer_pneumo_prokbert.py --variant minic --pos positives_81bp_SigA.fasta --neg negatives_81bp.fasta -o pred_siga_minic.tsv
```

## Roadmap

1. Fix minilong (vendor `models.py` or wait for upstream fix + 2-segment mean pooling).
2. Binary pneumo-specific fine-tuning with `GroupKFold` by MMseqs2 clusters (2247 clusters) — SigX held-out only (N=21).
3. Multiclass SigA / SigX / non-promoter.
