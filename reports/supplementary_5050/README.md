# Supplementary 5050 — CDS/IGR negatives head-to-head

`mini-223k-d39v-tigr4-high5050` (HF-train80 + `datasets/d39v_tigr4_high5050` train)
vs `mini-223k-d39v-tigr4`, identical protocol (2ep, seed 42, Lambda A10).

Fair slices (both-clean: unseen by both trains; splits share the pool):
- 5050-test clean n=327: new 0.6515 vs old 0.5678, Δ+0.084 CI[+0.024,+0.144] **SIG**
- dedup-test clean n=185: new 0.7924 vs old 0.7550, Δ+0.038 CI[-0.027,+0.099] n.s.
- paper-test Δ+0.022 **SIG**, multispecies Δ+0.032 **SIG**, ecoli Δ+0.015 n.s.

Rule met → **PROMOTED to primary mix model**. Numbers: `h2h.json`.
Builder: `scripts/build_high5050.py`. Matrix probs: `finetuning_outputs/eval_matrix_probs/mini-223k-d39v-tigr4-high5050__*.csv`.
