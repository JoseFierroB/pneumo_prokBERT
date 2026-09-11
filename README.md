# pneumo_prokBERT

Adaptación de ProkBERT (gLM) a promotores de *Streptococcus pneumoniae* D39V / TIGR4.

Fase actual: **zero-shot** con modelos oficiales (`mini` k6s1 y `minic` k1s1) sobre ventanas 81bp `[-60,+20]`.
Ver baseline y comparativa en `BASELINE_ZERO_SHOT.md`.

```bash
pip install torch transformers biopython pandas scikit-learn
pip install git+https://github.com/nbrg-ppcu/prokbert.git
python infer_pneumo_prokbert.py --variant mini --pos positives_81bp.fasta --neg negatives_81bp.fasta -o predictions.tsv
```

## Contenido

| Archivo | Qué es |
|---|---|
| `infer_pneumo_prokbert.py` | Inferencia zero-shot multi-variante (mini/minic), salida TSV `{ID,LABEL,PRED}` compatible con `promoter-tools/src/analysis` |
| `BASELINE_ZERO_SHOT.md` | AUCs D39V/TIGR4/SigA/SigX, tokenización, embeddings PCA, notas de minilong bloqueado |
| `requirements.txt` | Deps mínimas CPU |

## Roadmap

1. Resolver `mini-long` (falta `models.py` upstream + pooling 2 segmentos).
2. Finetuning piloto binario con `GroupKFold` MMseqs2 — SigX held-out.
3. Multiclass SigA / SigX / no-promotor.