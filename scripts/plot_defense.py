#!/usr/bin/env python3
"""Defense plots for tomorrow's report — one metric per file (verified numbers only).

Writes reports/defense/{sigma_dist,e13_ablation,energy_compare,throughput,h2h_forest}.png
+ defense.json (source data).

Usage:
  pixi run python scripts/plot_defense.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/defense")
DATA = {
    "sigma_dist": {"labels": ["desconocido", "SigA", "SigX-D39V", "SigX-TIGR4"],
                   "d39v": [562, 389, 21, 0], "tigr4": [291, 130, 0, 6]},
    "e13": {"groups": ["own-test", "mixed-5050"],
            "cds-only": [0.7558, 0.6736], "igr-only": [0.7189, 0.7189], "mix-5050": [0.7162, 0.7162]},
    "energy": {"labels": ["paper-test", "ecoli-sigma70"],
               "energy_only_mcc": [0.4416, 0.5712], "blend_mcc": [0.5956, 0.7107]},
    "throughput": {"labels": ["eval fp32\nbs512 A10/5090", "train\nbs128", "infer CPU\nbs64", "tokenize Pool\nbs512"],
                   "seq_s": [5000, 3200, 58, 250], "note": "measured; tokenize ~250/s approx"},
    "h2h": {"labels": ["5050-clean\n327", "dedup-clean\n185", "paper\n22310", "multi\n22582",
                       "ecoli\n1864", "3v2 paper", "3v2 multi", "spec-3ep\n185"],
            "delta": [0.0836, 0.0375, 0.0224, 0.0318, 0.0153, 0.0285, 0.0286, 0.0941],
            "lo": [0.0240, -0.0272, 0.0137, 0.0228, -0.0101, 0.0201, 0.0204, 0.0051],
            "hi": [0.1443, 0.0991, 0.0311, 0.0408, 0.0402, 0.0366, 0.0367, 0.1821]},
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)
    print(f"{name} OK")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(DATA, open(OUT / "defense.json", "w"), indent=1)
    d = DATA["sigma_dist"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(d["labels"]))
    ax.bar(x, d["d39v"], label="D39V")
    ax.bar(x, d["tigr4"], bottom=d["d39v"], label="TIGR4")
    ax.set_xticks(x, d["labels"], fontsize=8)
    ax.set_ylabel("positives")
    ax.set_title("Sigma distribution — 1399 pneumo positives")
    ax.legend(fontsize=8)
    save(fig, "sigma_dist.png")
    d = DATA["e13"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(d["groups"]))
    w = 0.25
    for i, k in enumerate(["cds-only", "igr-only", "mix-5050"]):
        ax.bar(x + (i - 1) * w, d[k], w, label=k)
    ax.set_xticks(x, d["groups"], fontsize=8)
    ax.set_ylabel("MCC")
    ax.set_title("E13 source ablation (locked MCC)")
    ax.legend(fontsize=8)
    save(fig, "e13_ablation.png")
    d = DATA["energy"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(d["labels"]))
    ax.bar(x - 0.2, d["energy_only_mcc"], 0.4, label="energy only")
    ax.bar(x + 0.2, d["blend_mcc"], 0.4, label="with BERT")
    ax.set_xticks(x, d["labels"], fontsize=8)
    ax.set_ylabel("MCC")
    ax.set_title("Energies a la MLDSPP (parked: redundant)")
    ax.legend(fontsize=8)
    save(fig, "energy_compare.png")
    d = DATA["throughput"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(d["labels"], d["seq_s"])
    ax.set_yscale("log")
    ax.set_ylabel("seq/s (log)")
    ax.set_title("Measured throughput (no Rust needed)")
    for i, v in enumerate(d["seq_s"]):
        ax.text(i, v * 1.2, str(v), ha="center", fontsize=8)
    save(fig, "throughput.png")
    d = DATA["h2h"]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(d["labels"]))
    lo = np.array(d["delta"]) - np.array(d["lo"])
    hi = np.array(d["hi"]) - np.array(d["delta"])
    colors = ["C0" if l > 0 else "gray" for l in d["lo"]]
    ax.errorbar(d["delta"], y, xerr=[lo, hi], fmt="o", ecolor="k", capsize=3)
    for i, c in enumerate(colors):
        ax.plot(d["delta"][i], y[i], "o", color=c)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, d["labels"], fontsize=7)
    ax.set_xlabel("ΔMCC (paired bootstrap 95% CI)")
    ax.set_title("Head-to-head forest (new-old)")
    save(fig, "h2h_forest.png")


if __name__ == "__main__":
    main()
