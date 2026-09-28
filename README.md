# pneumo_prokBERT

ProkBERT fine-tuning for *Streptococcus pneumoniae* (D39V, TIGR4) promoter classification.

Binary promoter vs. non-promoter on 81 bp windows centered on the TSS `[-60, +20]`.
Five variants in [MODELS.md](MODELS.md), compared on a 5×4 matrix ([EVALUATION.md](EVALUATION.md)) with per-cell leakage flags.

## Model

- **Base:** `neuralbioinfo/prokbert-mini` (MLM), LCA tokenization `k=6, s=1`, hidden 384, vocab 4101.
- **Init:** always from HF (`prokbert-mini` @ `feb2520a`, `mini-promoter` @ `8502f02`), pinned in `models.json`. Root `models.py` is retained (checkpoints load it via `auto_map`).
- **Tokenizer:** vendored `prokbert/` (upstream untouched); runtime fixes in `prokbert_patches.py`.

## Data

- D39V: 989 raw positives (973 unique) + 1000 negatives.
- Tracked (`datasets/`, ~5M):
  - `d39v_tigr4_dedup_3135` — 1399+1736=3135 (3465 naive − 330 dups; train 2194/eval 470/test 471).
  - `d39v_tigr4_high_1to2` — 1399+2798=4197, 1:2 pos:neg via within-split bootstrap (train 2937/eval 630/test 630).
  - `d39v_binary_siga_comx/` — SigA-vs-ComX binary (287/62/62) + MANIFEST.
  - `d39v_full_989` — 973+1000=1973. Tag source appendix (`sigma,strain,split` columns for ComX/sigma builders); not used for training.
  - `external/` — vendored TIGR4 sigma metadata (738 rows) + D39V genome; no sibling repo needed.
- Format: CSVs with `segment,y` (+`id`; +`sigma,strain,split` in `d39v_full_989`).
- Benchmark FASTAs, vendored (`datasets/reference/`, ~0.8M): `d39v` (989+1000), `tigr4-high` (738+738), `d39v-tigr4` (1727+1738), 81 bp. Self-contained: no sibling repo needed.
- Upstream `neuralbioinfo/bacterial_promoters` rev `38f748a` (223,092 train / 1,864 sigma70 / 22,582 multispecies). Local copy in `data/bacterial_promoters/` (70M, git-ignored); scripts fall back to HF. Materialize after cloning:
  ```bash
  pixi run python - <<'PY'
  from datasets import load_dataset
  ds = load_dataset("neuralbioinfo/bacterial_promoters")
  for split in ["train", "test_sigma70", "test_multispecies"]:
      ds[split].to_pandas().to_parquet(f"data/bacterial_promoters/{split}.parquet")
  PY
  ```

## Usage

```bash
pixi install
```
Env: `pixi.toml` (Python 3.10, PyTorch cu128, transformers, datasets, accelerate, biopython, scikit-learn, matplotlib).

Device auto-detects in all GPU scripts (best-free GPU, else CPU); override with `--gpu N` / `--cpu`.
Check data readiness: `pixi run python scripts/fetch_data.py --data-dir datasets/d39v_tigr4_dedup_3135` (repeatable).

### Paper replica (upstream 223k)

```bash
pixi run python paper_replica.py   # defaults = paper §2.3.2 (2ep, bs128, lr0.0004, wd0.01, warmup500)
```
80/10/10 split of the 223,092 train, seed 42. Output: checkpoints, `run_info.json`, `best/`.
Uses `ProkBertForSequenceClassification` (`prokbert/models/megatronbert.py`); upstream `examples/finetuning.py` imports `BertForBinaryClassificationWithPooling`, which does not exist.

### Pneumo fine-tuning

```bash
pixi run python finetuning.py --warmup-ratio 0.1   # default dataset d39v_tigr4_dedup_3135
pixi run python finetuning.py --dataset-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/mini_d39v_tigr4_2ep --warmup-ratio 0.1
pixi run python finetuning.py --train-csv my/train.csv --eval-csv my/eval.csv --test-csv my/test.csv --output-dir finetuning_outputs/custom --learning-rate 0.0008
```
Paper params except warmup: `--warmup-steps 500` suits full runs; on 22–36 steps it caps LR at 4e-4·steps/500, so small runs use `--warmup-ratio 0.1`. Every run writes `run_info.json` + `best/` in its own folder. See `finetuning.py --help` for all flags.

### Mixed training (223k + data dir)

```bash
# Fit HF train_80 + data train | select on combined val (MCC) | locked: HF test_10 + data test + sigma70 + multispecies
pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --output-dir finetuning_outputs/mini_223k_d39v_tigr4_2ep
pixi run python scripts/train_mix223k.py --data-dir datasets/d39v_tigr4_dedup_3135 --dry-run
```
`--data-dir` takes any folder with `train.csv/eval.csv/test.csv`; run name = its basename. Paper hyperparams (fp32, warmup 500, eval/save 50/500, best by MCC); writes `run_info.json` + `data_manifest.json`.

### Evaluation matrix (5 models × 4 datasets)

```bash
pixi run python scripts/eval_matrix.py
pixi run python scripts/eval_matrix.py --models mini-223k-d39v-tigr4 promoter-baseline --datasets ecoli-sigma70
pixi run python scripts/build_report.py   # regenerate EVALUATION.md/json + cards from metrics.json (same selectors)
```
Outputs: `reports/model_matrix/metrics.json`, `summary_metrics.csv`, `plots/roc__<dataset>.png` + `plots/pr__<dataset>.png` (one metric per file); per-cell `.csv` (`y_true,y_score`, ignored) in `finetuning_outputs/eval_matrix_probs/`. Extra spreadsheet tables (TSV by default; `--format csv|both`): `significance`, `sigma_imbalance/siga_vs_sigx`, `sigma_{counts,gc,fragile}`, `energy/results/energy_summary`.

## Pipeline

```python
ds = load_dataset("csv", data_files={"train": ..., "eval": ..., "test": ...})  # or bacterial_promoters
tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')  # 76 k-mers + CLS/SEP = 78 tokens
tokenized = ds.map(tokenize_function, batched=True)  # input_ids, attention_mask, labels
model = AutoModelForSequenceClassification.from_pretrained(init, trust_remote_code=True)  # head fresh (inherited for promoter-*)
args = TrainingArguments(bs128, lr0.0004, 2ep, wd0.01, warmup, seed42)  # warmup_ratio 0.1 on small datasets
trainer = Trainer(model, args, train_dataset=..., eval_dataset=...,  # best checkpoint by val MCC
                  data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
trainer.train()
trainer.save_model("finetuning_outputs/<run>/best")  # + run_info.json
```

```
81 bp → [LCA k6s1] 76 k-mers + [CLS]/[SEP] = 78 tokens (vocab 4101)
  → ProkBertModel (6L, hidden 384) → B×78×384
  → weighting_layer Linear(384→1) → masked_attention_pool softmax(dim=1)
  → pooled B×384 → classifier Linear(384→2) → CrossEntropy
```
`promoter-*` use the HF pooler variant instead (113 keys); documented per model in `reports/model_matrix/<alias>/model_card.md`.

### Code map
Shared code lives in `scripts/` — import it, do not copy it:
- `common.py`: device flags (`--gpu/--cpu`) + seed.
- `metrics.py`: `cell_metrics` (matrix cells), `cell_metrics_light` (probes), `trainer_compute_metrics` (Trainer callback).
- `reference_sets.py`: canonical FASTA loader shared by the matrix and `energy/` probes.
Left duplicated on purpose (different sources): HF split recompute vs parquet/hub reads, 2-line CSV loaders.
Verification ladder (no test suite, by design): `py_compile` → `--help` → `--dry-run` / `fetch_data` READY → subset run vs saved `metrics.json`.

## Evaluation

Per cell @0.5 plus tuned Youden threshold: ACC, balACC, MCC, F1, ROC-AUC, PR-AUC; leakage flag `clean` / `train_overlap` / `selection_overlap`. Full table: [EVALUATION.md](EVALUATION.md). Headlines (clean sets, paired bootstrap B=2000, all n.s.): sigma70 `mini-223k-d39v-tigr4` MCC +0.029 [-0.001,+0.061] (p=0.029, CI crosses 0); multispecies MCC -0.009 (tied); paper-test held by `mini-223k-replica` (MCC 0.592, clean); pneumo-only models collapse on sigma70/multispecies as expected.

Docs: [MODELS.md](MODELS.md), `models.json`, `datasets_eval.json`, `reports/model_matrix/`.

## License

Our code and docs: MIT (see `LICENSE`). `prokbert/` is vendored upstream code under its original terms.

## Citation

```bibtex
@article{ProkBERT2024,
  author  = {Ligeti et al.},
  journal = {Frontiers in Microbiology},
  title   = {ProkBERT family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14}
}
```
