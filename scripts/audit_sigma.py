#!/usr/bin/env python3
"""Phase 1: sigma-factor imbalance audit (no training).

- Per-tag counts x split for d39v_full_989 (tags parsed from id suffix).
- Fragile cells (eval/test n<5), singletons, LOWCONF summary.
- Stratification check: observed vs proportional expectation.
- GC composition per sigma group (confound check).
- TIGR4 join: positives_high (738) + dedup TIGR4 rows via Sequence_ID
  against positives_high_sigma_assigned_metadata.tsv.

Writes reports/sigma_imbalance/descriptives.json + descriptives.md.

Usage:
  pixi run python scripts/audit_sigma.py
  pixi run python scripts/audit_sigma.py --tigr4-meta /path/to/positives_high_sigma_assigned_metadata.tsv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SPLITS = ["train", "eval", "test"]
FRAGILE_N = 5


def parse_args():
    p = argparse.ArgumentParser(description="Sigma imbalance descriptive audit")
    p.add_argument("--tigr4-meta", type=str,
                   default="datasets/external/positives_high_sigma_assigned_metadata.tsv")
    p.add_argument("--out-dir", type=str, default="reports/sigma_imbalance")
    return p.parse_args()


def tag_of(df):
    return df["id"].str.split("_").str[-1]


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"splits": {}, "fragile": [], "singletons": [], "lowconf": {},
           "stratification": {}, "gc": {}, "tigr4": {}}

    dfs = {s: pd.read_csv(f"datasets/d39v_full_989/{s}.csv") for s in SPLITS}
    alldf = pd.read_csv("datasets/d39v_full_989/all.csv")
    N = len(alldf)

    for s in SPLITS + ["all"]:
        df = alldf if s == "all" else dfs[s]
        pos = df[df.y == 1]
        tags = tag_of(pos).value_counts()
        rep["splits"][s] = {"n": len(df), "n_pos": len(pos), "tags": {k: int(v) for k, v in tags.items()}}
        if s in SPLITS:
            for tag, n in tags.items():
                if n < FRAGILE_N:
                    rep["fragile"].append({"split": s, "tag": tag, "n": int(n)})

    # singletons + LOWCONF (all)
    pos = alldf[alldf.y == 1]
    vc = tag_of(pos).value_counts()
    rep["singletons"] = sorted([t for t, n in vc.items() if n == 1])
    for t in sorted([t for t in vc.index if "LOWCONF" in t]):
        rep["lowconf"][t] = int(vc[t])

    # stratification: observed vs proportional expectation
    for s in SPLITS:
        frac = len(dfs[s]) / N
        obs = tag_of(dfs[s][dfs[s].y == 1]).value_counts()
        exp = (tag_of(pos).value_counts() * frac).round(2)
        common = ["SigA", "SigX", "None"]
        rep["stratification"][s] = {
            "frac": round(frac, 4),
            "observed": {t: int(obs.get(t, 0)) for t in common},
            "expected": {t: float(exp.get(t, 0.0)) for t in common},
        }

    # GC per sigma group (all)
    tags_all = tag_of(alldf)
    for grp, sel in [("SigA", tags_all == "SigA"), ("SigX", tags_all == "SigX"),
                     ("None", tags_all == "None"),
                     ("neg", alldf.y == 0)]:
        seqs = alldf[sel]["segment"]
        gc = seqs.apply(lambda x: (x.count("G") + x.count("C")) / len(x))
        rep["gc"][grp] = {"n": int(len(gc)), "mean": round(float(gc.mean()), 4),
                          "std": round(float(gc.std()), 4)}

    # TIGR4 join
    meta = pd.read_csv(args.tigr4_meta, sep="\t")
    rep["tigr4"]["metadata_rows"] = len(meta)
    rep["tigr4"]["sigma"] = {k: int(v) for k, v in meta["Sigma_Factor"].value_counts().items()}
    # Minus_10_Seq is optional (upstream metadata dropped the column 2026-09-28)
    if "Minus_10_Seq" in meta.columns:
        m10 = meta["Minus_10_Seq"]
        rep["tigr4"]["minus10_available"] = int((m10.notna() & (m10 != "NA")).sum())
    else:
        rep["tigr4"]["minus10_available"] = 0
    dedup = pd.read_csv("datasets/d39v_tigr4_dedup_3135/all.csv")
    tigr = dedup[~dedup["id"].str.contains("D39V")]
    key = tigr["id"]
    hit = key.isin(set(meta["Sequence_ID"]))
    tigr_pos = tigr[tigr.y == 1]
    hit_pos = tigr_pos["id"].isin(set(meta["Sequence_ID"]))
    rep["tigr4"]["dedup_tigr4_rows"] = int(len(tigr))
    rep["tigr4"]["dedup_join_rate_all"] = round(float(hit.mean()), 4)
    rep["tigr4"]["dedup_join_rate_pos"] = round(float(hit_pos.mean()), 4)
    rep["tigr4"]["note"] = "metadata covers positives only; negatives cannot join by design"
    sigx_d39v = int((tag_of(pos) == "SigX").sum())
    # v2 metadata has no SigX left (real ortholog assignment); TIGR4 competence = ComX (6)
    comx_tigr = int((meta["Sigma_Factor"] == "ComX").sum())
    rep["tigr4"]["sigx_pool"] = {"d39v_sigx": sigx_d39v, "tigr4_comx": comx_tigr,
                                 "combined": sigx_d39v + comx_tigr}

    json.dump(rep, open(out / "descriptives.json", "w"), indent=2)

    L = ["# Sigma imbalance audit — d39v_full_989 (+ TIGR4 join)", "",
         "No training performed. Tags parsed from `id` suffix; splits stratified by `y` only.", "",
         "## Counts per tag x split (positives)", "",
         "| split | n | pos | top tags |", "|---|---|---|---|"]
    for s in SPLITS + ["all"]:
        r = rep["splits"][s]
        top = ", ".join(f"{k} {v}" for k, v in sorted(r["tags"].items(), key=lambda x: -x[1])[:6])
        L.append(f"| {s} | {r['n']} | {r['n_pos']} | {top} |")
    L += ["", "## Fragile cells (eval/test n<5)",
          "| split | tag | n |", "|---|---|---|"]
    for f in rep["fragile"]:
        L.append(f"| {f['split']} | {f['tag']} | {f['n']} |")
    L += ["",
          f"Singletons (n=1 in all): {', '.join(rep['singletons'])}",
          f"LOWCONF: {rep['lowconf']}", "",
          "## Stratification check (observed vs proportional)",
          "| split | group | obs | exp |", "|---|---|---|---|"]
    for s in SPLITS:
        st = rep["stratification"][s]
        for t in ("SigA", "SigX", "None"):
            L.append(f"| {s} | {t} | {st['observed'][t]} | {st['expected'][t]} |")
    L += ["", "## GC composition (confound check)",
          "| group | n | mean | std |", "|---|---|---|---|"]
    for g, v in rep["gc"].items():
        L.append(f"| {g} | {v['n']} | {v['mean']} | {v['std']} |")
    t = rep["tigr4"]
    L += ["", "## TIGR4 join",
          f"- metadata rows: {t['metadata_rows']}; sigma: {t['sigma']}",
          f"- minus-10 available: {t['minus10_available']}/738",
          f"- dedup TIGR4 rows: {t['dedup_tigr4_rows']}; join rate pos-only: {t['dedup_join_rate_pos']} "
          f"(all: {t['dedup_join_rate_all']}; negatives have no metadata by design)",
          f"- Competence pool: D39V SigX {t['sigx_pool']['d39v_sigx']} + TIGR4 ComX {t['sigx_pool']['tigr4_comx']} = {t['sigx_pool']['combined']}", "",
          "## Proposed taxonomy (pending decision)",
          "- SigA (±LOWCONF) / competence SigX+ComX (27) / None / Other-TF / Neg, with LOWCONF in/out sensitivity.",
          "- Singletons: keep in train, drop from eval metrics."]
    open(out / "descriptives.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {out}/descriptives.json + descriptives.md")
    print(f"fragile cells: {len(rep['fragile'])}, singletons: {len(rep['singletons'])}, "
          f"competence pool: {t['sigx_pool']['combined']}")


if __name__ == "__main__":
    main()
