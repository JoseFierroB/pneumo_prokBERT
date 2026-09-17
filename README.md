# pneumo_prokBERT

ProkBERT fine-tuning for *Streptococcus pneumoniae* D39V promoter classification.

## Overview

This repository adapts ProkBERT, a genomic language model, to binary promoter prediction in *S. pneumoniae* D39V. The task is promoter vs. non-promoter on 81 bp windows centered on the transcription start site `[-60, +20]`.

## Model

- **Base:** ProkBERT-mini, Local Context-Aware (LCA) tokenization `k=6, s=1`, `hidden size 384`, `vocabulary ~4101`.
- **Weights:** `config.json`, `model.safetensors`, and `vocab.txt` are included for offline use.
- **Tokenizer:** Vendored `prokbert/prokbert_tokenizer.py`; `Trainer`-compatible saving is patched for CPU/GPU.

## Data

- **Strain:** D39V, deduplicated.
- **Size:** 989 positive (SigA+SigX) + 1000 negative = 1989 total sequences.
- **Window:** 81 bp, `[-60, +20]` relative to TSS.
- **Format:** CSVs with `segment,y` (`train`, `eval`, `test` splits provided).
- **Upstream data for transfer:** `neuralbioinfo/bacterial_promoters` (223,092 train / 1,864 sigma70 / 22,582 multispecies, 81 bp) cached via Hugging Face Datasets.

## Usage

### Install

```bash
pixi install
```

Environment is defined in `pixi.toml` (Python 3.10, PyTorch cu128, transformers, datasets, scikit-learn).

### Fine-tuning (upstream)

```bash
pixi run python finetuning.py \
  --model_name neuralbioinfo/prokbert-mini \
  --ftmodel mini_promoter \
  --model_outputpath finetuning_outputs \
  --num_train_epochs 2 \
  --per_device_train_batch_size 128 \
  --learning_rate 0.0004
```

Outputs are written to `finetuning_outputs/` (checkpoints, logs, and the final model). This follows the paper configuration (AdamW, linear warmup, 2 epochs, batch 128/GPU).

### Fine-tuning (D39V)

```bash
pixi run python finetuning.py
```

Uses the local D39V CSVs and the ProkBERT-mini weights in the repository root.

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

Metrics are Accuracy, MCC, F1, and ROC-AUC computed on `eval`/`test`. The `compute_metrics` callback uses `argmax` (threshold 0.5) for MCC/Accuracy and `softmax(logits)[:,1]` for ROC-AUC.

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
