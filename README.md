# pneumo_prokBERT

ProkBERT fine-tuning for *Streptococcus pneumoniae* D39V promoter classification.

## Overview

This repository adapts ProkBERT, a genomic language model, to binary promoter prediction in *S. pneumoniae* D39V. The task is promoter vs. non-promoter on 81 bp windows centered on the transcription start site `[-60, +20]`.

## Model

- **Base:** ProkBERT-mini, Local Context-Aware (LCA) tokenization `k=6, s=1`, `hidden size 384`, `vocabulary ~4101`.
- **Weights:** `config.json`, `model.safetensors`, and `vocab.txt` are included for offline use.
- **Tokenizer:** Vendored `prokbert/prokbert_tokenizer.py`; `Trainer`-compatible saving is patched for CPU/GPU.

## Data

- **Strains:** D39V (989 pos SigA+SigX pooled + 1000 neg = 1989, dedup) + dedup integrations with TIGR4.
- **Datasets (tracked, `datasets/` 0.5–0.7M):**
  - `d39v_full_989` — 989+1000=1973 unique (16 dups removed, train 1381/eval 296/test 296, 70/15/15 seed 42, 81 bp).
  - `d39v_tigr4_dedup_3135` — 1399 pos + 1736 neg = 3135 dedup exact (3465 naive −328 pos dups, 308 cross-strain).
  - `d39v_tigr4_dedup_balanced_2798` — 1399+1399 balanced.
- **Window:** 81 bp, `[-60, +20]` relative to TSS. Format: CSVs with `segment,y,id,sigma,strain,split`.
- **Upstream data for transfer (local, `data/bacterial_promoters/` 70M, not versioned, see `.gitignore`):** `neuralbioinfo/bacterial_promoters` (223,092 train / 1,864 sigma70 / 22,582 multispecies, 81 bp) materialized locally as `parquet+csv+fasta` (9.6M/29M/25M train) from HF cache `38f748a`. Use via `load_dataset` or `pd.read_parquet`.

## Usage

### Install

```bash
pixi install
```

Environment is defined in `pixi.toml` (Python 3.10, PyTorch cu128, transformers, datasets, scikit-learn).

### Fine-tuning (paper replica, upstream 223k)

```bash
# Paper replica / adaptation — aligned with tools/prokbert/examples/finetuning.py
# Note: Upstream examples/finetuning.py has a bug (BertForBinaryClassificationWithPooling not found,
# correct class is ProkBertForSequenceClassification in prokbert/models/megatronbert.py:1710)
# This replica fixes that shim and is executable.
pixi run python paper_replica.py \
  --model_name neuralbioinfo/prokbert-mini \
  --ftmodel mini_promoter \
  --model_outputpath finetuning_outputs \
  --num_train_epochs 2 \
  --per_device_train_batch_size 128 \
  --learning_rate 0.0004
```

Outputs are written to `finetuning_outputs/` (checkpoints, logs, and the final model). This follows the paper configuration (AdamW, linear warmup, 2 epochs, batch 128/GPU).

### Fine-tuning (D39V adaptation — paper params, configurable)

```bash
# Default: D39V 989 paper-aligned 2ep bs128 lr0.0004 wd0.01 warmup500 betas 0.95/0.98
pixi run python finetuning.py
# = pixi run python finetuning.py --dataset_dir datasets/d39v_full_989 --output_dir finetuning_outputs/d39v_989_paper_2ep --model_name . --num_train_epochs 2 --per_device_train_batch_size 128 --learning_rate 0.0004 --weight_decay 0.01 --warmup_steps 500

# Other datasets (each in its own output folder, deletable)
pixi run python finetuning.py --dataset_dir datasets/d39v_tigr4_dedup_3135 --output_dir finetuning_outputs/dedup_3135_2ep
pixi run python finetuning.py --dataset_dir datasets/d39v_tigr4_dedup_balanced_2798 --output_dir finetuning_outputs/balanced_2798_2ep --num_train_epochs 3

# Custom CSVs / hyperparams
pixi run python finetuning.py --train_csv my/train.csv --eval_csv my/eval.csv --test_csv my/test.csv --output_dir finetuning_outputs/custom --learning_rate 0.0008 --weight_decay 0.02
```

Defaults are paper §2.3.2 (AdamW 2 epochs, batch 128/GPU, wd 0.01, warmup 500, betas 0.95/0.98). All flags override defaults; no code edit needed. Uses `segment,y` 81 bp, LCA k=6 s=1, 78 tokens. The local D39V model is `base` `prokbert-mini` → `d39v_989_paper_2ep/best` (79M, 22 steps 2.1s 5090, eval 0.959/0.918/0.980).

## Pipeline

### Pseudocode

```python
# 1. Load data
dataset = load_dataset("neuralbioinfo/bacterial_promoters")  # or local CSVs: segment,y
train_db, eval_db = dataset["train"].to_pandas(), dataset["test_sigma70"].to_pandas()

# 2. Tokenization (LCA k=6 s=1) -> 76 k-mers + [CLS]/[SEP] = 78 tokens
tokenizer = ProkBERTTokenizer(tokenization_params={'kmer':6,'shift':1})
X, y, _ = get_torch_data_from_segmentdb_classification(tokenizer, train_db)
# X: (N, 78), y: (N,)  attention_mask = (input_ids != PAD)

# 3. Model
pretrained, tokenizer = get_default_pretrained_model_parameters("neuralbioinfo/prokbert-mini", "MegatronBertModel")
model = ProkBertForSequenceClassification(pretrained.config)  # weighting_layer + classifier
model.bert.load_state_dict(pretrained.bert.state_dict(), strict=False)

# 4. Trainer
args = TrainingArguments(per_device_train_batch_size=128, learning_rate=0.0004,
                         num_train_epochs=2, weight_decay=0.01, warmup_steps=500)
trainer = Trainer(model=model, args=args,
                  train_dataset=ProkBERTTrainingDatasetPT(X_train, y_train),
                  eval_dataset=ProkBERTTrainingDatasetPT(X_eval, y_eval),
                  compute_metrics=compute_metrics_eval_prediction)
trainer.train()
model.save_pretrained("finetuning_outputs/neuralbioinfo/prokbert-mini")
```

### Diagram

```
81 bp segment
    |
    v
[LCA k=6 s=1]  76 k-mers + [CLS]/[SEP] = 78 tokens (vocab 4101)
    |
    v
Rectangular batch  B x 78  (input_ids, attention_mask)
    |
    v
ProkBertModel (6 layers, hidden 384)
    |
    v
sequence_output  B x 78 x 384
    |
    v
weighting_layer Linear(384->1)  B x 78 x 1  (token scores)
    |
    v
masked_attention_pool  softmax(dim=1)  B x 78 x 1
    |
    v
pooled  B x 384  = sum(weights * sequence_output, dim=1)
    |
    v
classifier Linear(384->2)  B x 2  (logits)
    |
    v
CrossEntropy  promoter vs non-promoter
```

Shape notes: `s` in tokenization is the k-mer stride (`1`), `t=78` is the sequence length in tokens (dimension for softmax). `B` is the batch size.

## Evaluation

Metrics are Accuracy, MCC, and ROC-AUC computed on `eval`/`test`. The `compute_metrics` callback uses `argmax` (threshold 0.5) for MCC/Accuracy and `softmax(logits)[:,1]` for ROC-AUC.

Full results for the five models (HF mini-promoter baseline, paper replica 223k, mix 223k+d39v, D39V from promoter, D39V base 989) across datasets (D39V 1989/1973, TIGR4 1476, mix 3465, dedup 3135, sigma70 1864, multispecies 22582) are in [EVALUATION.md](EVALUATION.md) and [EVALUATION.json](EVALUATION.json) (5×5 table, @0.5 and @Youden, 5090 k6s1). The D39V base model (`d39v_989_paper_2ep/best`, 0.966 AUC on D39V, 0.941 on dedup, collapses to 0.459/0.529 on sigma70/multispecies) is the expected pneumo-specific overfit.

## License

MIT. ProkBERT is licensed under the terms provided by the original authors; see `tools/prokbert` for details.

## Citation

If you use this repository, please cite ProkBERT:

```bibtex
@article{ProkBERT2024,
  author  = {Ligeti et al.},
  journal = {Frontiers in Microbiology},
  title   = {ProkBERT family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14}
}
```
