#!/usr/bin/env python3
"""Phase 2 (simple): SigA-family vs SigX (ComX) separation in existing models.

Uses P(promoter) from the promoter-vs-nonpromoter classifiers as the ranking
score with labels SigA-family=0, SigX=1. Diagnostic of impact, not a trained
capability: it shows whether current models separate sigma classes.

Sets (CPU-friendly):
  d39v-test  D39V test split only (honest; ~68 SigA-fam vs 4 SigX)
  d39v-all   all D39V positives (train_overlap for fine-tuned models)
  tigr4      TIGR4 positives via metadata join (733 vs 5; clean except dedup-trained)
  combined   d39v-all + tigr4

Usage:
  pixi run python scripts/eval_sigma.py
  pixi run python scripts/eval_sigma.py --models mini-d39v-tigr4 promoter-baseline
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, ".")
from scripts.common import add_device_args, resolve_device
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

SIGA = {"SigA", "SigA-LOWCONF"}


def parse_args():
    p = argparse.ArgumentParser(description="SigA vs SigX separation in existing models")
    p.add_argument("--models", nargs="*", default=None, help="subset of aliases (default: all in models.json)")
    p.add_argument("--tigr4-meta", type=str,
                   default="datasets/external/positives_high_sigma_assigned_metadata.tsv")
    p.add_argument("--out-dir", type=str, default="reports/sigma_imbalance")
    add_device_args(p)
    return p.parse_args()


def metrics(y, scores):
    """Per-sigma diagnostic cell (own variant on purpose: adds mean_pos/mean_neg,
    which scripts.metrics.cell_metrics_light does not report)."""
    pred = (scores >= 0.5).astype(int)
    return {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
            "auc": float(roc_auc_score(y, scores)), "ap": float(average_precision_score(y, scores)),
            "acc_05": float(accuracy_score(y, pred)), "mcc_05": float(matthews_corrcoef(y, pred)),
            "mean_pos": float(scores[y == 1].mean()), "mean_neg": float(scores[y == 0].mean())}


def main():
    args = parse_args()
    device, _ = resolve_device(args.gpu, args.cpu)
    print(f"Device: {device}")
    models = json.load(open("models.json"))
    aliases = args.models or [k for k in models if models[k]["status"] == "exists"]

    alldf = pd.read_csv("datasets/d39v_full_989/all.csv")
    pos = alldf[alldf.y == 1].copy()
    pos["tag"] = pos["id"].str.split("_").str[-1]
    te = pos[pos["split"] == "test"] if "split" in pos.columns else pos

    sets = {}
    sets["d39v-test"] = (te[te["tag"].isin(SIGA)].copy(), te[te["tag"] == "SigX"].copy())
    sets["d39v-all"] = (pos[pos["tag"].isin(SIGA)].copy(), pos[pos["tag"] == "SigX"].copy())

    tigr = None
    try:
        from Bio import SeqIO
        meta = pd.read_csv(args.tigr4_meta, sep="\t")
        mseq = {}
        for r in SeqIO.parse("datasets/reference/tigr4-high/positives_81bp.fasta", "fasta"):
            mseq[str(r.id)] = str(r.seq).upper()
        msig = dict(zip(meta["Sequence_ID"].astype(str), meta["Sigma_Factor"]))
        sa = [mseq[i] for i in msig if msig[i] in ("SigA", "SigA_Putative") and i in mseq]
        # v2 metadata: no SigX left; TIGR4 competence = ComX
        sx = [mseq[i] for i in msig if msig[i] in ("SigX", "ComX") and i in mseq]
        sets["tigr4"] = (pd.DataFrame({"segment": sa}), pd.DataFrame({"segment": sx}))
        tigr = {"n_siga": len(sa), "n_sigx": len(sx)}
        print(f"tigr4 join: SigA-fam {len(sa)}, competence-SigX/ComX {len(sx)}")
    except FileNotFoundError as e:
        print(f"WARNING: TIGR4 metadata/reference missing ({e}); skipping tigr4/combined")

    if tigr is not None:
        sets["combined"] = (pd.concat([sets["d39v-all"][0], sets["tigr4"][0]]),
                            pd.concat([sets["d39v-all"][1], sets["tigr4"][1]]))

    leak = {"d39v-test": "clean", "d39v-all": "train_overlap", "tigr4": "clean-except-dedup-trained",
            "combined": "mixed"}
    tok = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    from transformers import AutoModelForSequenceClassification
    out = {"task": "SigA-family(0) vs SigX(1) ranked by P(promoter); diagnostic only",
           "models": {}}
    for alias in aliases:
        entry = models[alias]
        src = entry["init"]["hf_id"] if alias == "promoter-baseline" else entry["artifact"]
        print(f"=== {alias} ===")
        model = AutoModelForSequenceClassification.from_pretrained(src, trust_remote_code=True).to(device).eval()
        out["models"][alias] = {}
        with torch.no_grad():
            for name, (df0, df1) in sets.items():
                seqs = df0["segment"].str.upper().tolist() + df1["segment"].str.upper().tolist()
                y = np.array([0] * len(df0) + [1] * len(df1))
                scores = []
                for i in range(0, len(seqs), 256):
                    enc = tok(seqs[i:i + 256], add_special_tokens=True, padding=True,
                              truncation=False, return_tensors="pt")
                    kw = {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}
                    if "token_type_ids" in enc:
                        kw["token_type_ids"] = enc["token_type_ids"].to(device)
                    scores.extend(torch.softmax(model(**kw).logits, -1)[:, 1].cpu().numpy().tolist())
                m = metrics(y, np.array(scores))
                m["leakage"] = leak[name]
                out["models"][alias][name] = m
                print(f"  {name:12s} n0={m['n_neg']:4d} n1={m['n_pos']:3d} AUC {m['auc']:.4f} AP {m['ap']:.4f} "
                      f"MCC@0.5 {m['mcc_05']:+.4f} mean_neg {m['mean_neg']:.3f} mean_pos {m['mean_pos']:.3f} [{m['leakage']}]",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    outp = Path(args.out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp / "siga_vs_sigx.json", "w"), indent=2)
    L = ["# SigA-family vs SigX (ComX) separation — diagnostic (no training)", "",
         "P(promoter) from promoter-vs-nonpromoter models used as ranking score, labels SigA-family=0 / SigX=1.",
         "d39v-test is honest (unseen); d39v-all is train_overlap; tigr4 clean except dedup-trained models.", "",
         "| model | set | n0 | n1 | AUC | AP | MCC@0.5 | mean_neg | mean_pos | leakage |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for alias in aliases:
        for name in out["models"][alias]:
            m = out["models"][alias][name]
            L.append(f"| {alias} | {name} | {m['n_neg']} | {m['n_pos']} | {m['auc']:.4f} | {m['ap']:.4f} | "
                     f"{m['mcc_05']:+.4f} | {m['mean_neg']:.3f} | {m['mean_pos']:.3f} | {m['leakage']} |")
    open(outp / "siga_vs_sigx.md", "w").write("\n".join(L) + "\n")
    print(f"saved {outp}/siga_vs_sigx.json + siga_vs_sigx.md")


if __name__ == "__main__":
    main()
