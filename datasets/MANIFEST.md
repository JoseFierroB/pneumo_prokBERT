# datasets/ — manifest

Canonical pneumo + versioned HF splits only. Everything else was archived out on
2026-10-26 (`~/pneumo_prokBERT_archive/2026-10-26/datasets/`). CSV schema:
`segment,y,id` (+`src` where noted). Splits inside each dir are exact-duplicate
disjoint (asserted at build; CV folds additionally Hamming-capped variants were
retired with the ham track).

| dir | n (pos/neg) | what | used by |
|---|---|---|---|
| `d39v_tigr4_dedup_3135/` | 3135 (1399/1736) | deduped D39V+TIGR4 IGRs, train/eval/test | mix pneumo arm (original) |
| `d39v_tigr4_high_1to2/` | 4197 (1399/2798) | 1:2 bootstrap of dedup negs per split (`build_unified_1to2.py`), MANIFEST asserted zero cross-split overlap | CV classic folds source (2792 unique segs) |
| `d39v_tigr4_high5050/` | 4197 (1399/2798) | high-confidence set, negatives 50% CDS + 50% IGR (`build_high5050.py`) | **primary mix data** (fit 2937) |
| `pt_benchmark/` | 4197 (1399/2798) | clean balanced promoter-tools benchmark (`all.csv`: segment,y,id; `all_annotated.csv` adds sigma/src) | CV-pt_benchmark 5×1ep |
| `pneumo_5fold/` | 5×(fit ~1674/sel ~559/rep ~559) | nested 5-fold over the 2792 unique segs, stratified by y | classic CV 1ep/2ep |
| `hf_splits/{80/10/10,70/15/15,60/20/20}/` | 178473 / 156164 / 133855 train | versioned HF splits: ids + `manifest.json` (method, seed 42, per-split SHA16); 80/10/10 reproduces canonical SHAs | `train_mix --hf-split-spec local:<spec>`, joint CV |
| `external/` | TIGR4 sigma metadata + D39V genome | vendored reference, no sibling repo needed | sigma annotation joins |
| `reference/` | benchmark FASTAs (d39v, tigr4-high, d39v-tigr4) | 81bp sets used by matrix + probes | `reference_sets.py` |

Retired 2026-10-26 (archived, reproducible via builders): `pneumo_5fold_ham/`,
`kfold_5050/`, `*_75_10_15/`, `*_75_15_10/`, `*_sigbal/`, `cdsonly/`, `igronly/`,
`1to2_plus_dedup/`, `full_989/`, `binary_siga_sigx/`, `sigx_1to2/`.
