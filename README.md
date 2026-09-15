# pneumo_prokBERT

Adaptación de ProkBERT (gLM) a promotores de *Streptococcus pneumoniae* D39V / TIGR4.

Fase actual: **zero-shot** (baseline en `BASELINE_ZERO_SHOT.md`) + **piloto experimental finetuning** SigA+SigX 81bp `[-60,+20]`.

```bash
pip install torch transformers biopython pandas scikit-learn
pip install git+https://github.com/nbrg-ppcu/prokbert.git
# zero-shot
python infer_pneumo_prokbert.py --variant mini --pos positives_81bp.fasta --neg negatives_81bp.fasta -o predictions.tsv
# finetuning piloto (CPU, 4 hilos, 3 epochs) — requiere model/vocab locales
python train_siga_sigx.py  # usa train_siga_sigx_668.csv / eval_84 / test_84
```

## Contenido

| Archivo | Qué es |
|---|---|
| `infer_pneumo_prokbert.py` | Inferencia zero-shot multi-variante (mini/minic), TSV `{ID,LABEL,PRED}` |
| `BASELINE_ZERO_SHOT.md` | AUCs D39V/TIGR4/SigA/SigX, tokenización, embeddings PCA |
| `train_siga_sigx.py` | **Experimental** — finetuning 3ep SigA+SigX (668/84/84), patch tokenizer para Trainer CPU |
| `train_siga_sigx_668.csv` / `eval_siga_sigx_84.csv` / `test_siga_sigx_84.csv` | Splits piloto (GroupShuffle por segmento, 836 seqs 418/418) |
| `siga_sigx_final/` | Checkpoint finetuned 3ep (experimental, sin evaluación final reportada) |
| `config.json` `model.safetensors` `vocab.txt` etc. | Pesos base ProkBERT-mini k6s1 locales (offline CPU) |
| `prokbert/` | Tokenizer LCA vendorizado (k=6 s=1) |

## Roadmap

1. Resolver `mini-long` (falta `models.py` upstream + pooling 2 segmentos).
2. Piloto SigA+SigX — evaluar hold-out real y ablations.
3. Multiclass SigA / SigX / no-promotor.

> Nota: resultados del piloto no se reportan como baseline definitivo — rama experimental.
