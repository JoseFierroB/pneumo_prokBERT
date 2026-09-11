# Pneumo-ProkBERT — baseline zero-shot (CPU)

Objetivo: adaptar ProkBERT a promotores de *S. pneumoniae* D39V / TIGR4 (81bp `[-60,+20]`).
Fase actual: **solo zero-shot**, sin finetuning. Modelos oficiales sin reentrenar.

## Variantes evaluadas

| Variante | HF ID | kmer/shift | Tokens en 81bp |
|---|---|---|---|
| mini | `neuralbioinfo/prokbert-mini-promoter` | 6 / 1 | 78 (76 + CLS/SEP) |
| minic (-c) | `neuralbioinfo/prokbert-mini-c-promoter` | 1 / 1 | 83 (81 + CLS/SEP) |
| minilong | `neuralbioinfo/prokbert-mini-long-promoter` | 6 / 2 | 2x40 segmentos — **bloqueado upstream**: falta `models.py` en `neuralbioinfo/prokbert-mini-long`, y requiere pooling custom para 81bp |

Nota env: usar entorno con `MegatronBertModel` disponible (en promoter-tools es `tools/iPro-MP` env ipro-mp, no el env `tools/prokbert` puro). `torch CPU`, `transformers 4.57.6`.

## Resultados zero-shot (11-sep-2026, CPU 1 hilo)

Salida compatible: TSV `{ID, LABEL, PRED}`.

| Dataset | N | mini AUC | minic AUC | mean_pos mini/minic | mean_neg |
|---|---|---|---|---|---|
| D39V full 989/1000 | 1989 | 0.970 | 0.970 | 0.951 / 0.946 | 0.222 / 0.236 |
| D39V SigA 397/1000 | 1397 | 0.991 | 0.995 | 0.989 / 0.988 | 0.222 / 0.236 |
| D39V SigX 21/1000 | 1021 | 0.935 | 0.987 | 0.865 / 0.947 | 0.222 / 0.236 |
| TIGR4 high 738/738 | 1476 | 0.921 | 0.923 | 0.889 / 0.880 | 0.255 / 0.256 |

Referencia promoter-tools REPORT: iPro-MP 0.953, LCNN 0.949, MLDSPP75* 0.957 en D39V; iPro SigX 0.619.
Lectura: ProkBERT zero-shot ya supera el baseline en D39V; `-c` (kmer1) generaliza mejor al combox divergente SigX `TACGAATA`.

Tiempos CPU: mini 84s / 1989 seqs (~23 seq/s), minic 115s (~17 seq/s). Pico RAM ~0.8-1.1GB.

## Embeddings (sondeo)

`prokbert_pca_cls_421`: CLS 384d última capa, PCA 2D en D39V (SigA 150, SigX 21, None 100, Neg 150).
Centroides PC1: Neg -6.578 | SigX 0.312 | None 2.874 | SigA 4.618 — SigX intermedio, coherente con divergencia.
Ver `output/plots/prokbert_pca_cls_421.{png,svg,pdf}` y `output/tables/prokbert_pca_cls_421.tsv` en promoter-tools.

## Uso

```bash
pip install torch transformers biopython pandas scikit-learn
pip install git+https://github.com/nbrg-ppcu/prokbert.git
python infer_pneumo_prokbert.py --variant mini --pos positives_81bp.fasta --neg negatives_81bp.fasta -o predictions.tsv
python infer_pneumo_prokbert.py --variant minic --pos positives_81bp_SigA.fasta --neg negatives_81bp.fasta -o pred_siga_minic.tsv
```

## Roadmap

1. Resolver minilong (vendorizar `models.py` o esperar fix upstream + pooling media de 2 segmentos).
2. Finetuning piloto binario pneumo-específico con `GroupKFold` por clusters MMseqs2 (2247 clusters) — SigX solo held-out (N=21).
3. Multiclass SigA/SigX/no.
