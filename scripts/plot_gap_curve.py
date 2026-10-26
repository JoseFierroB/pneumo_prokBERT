#!/usr/bin/env python3
"""Train/test MCC + gap vs epochs from a JSON table (overfit evidence plot).

Usage:
  pixi run python scripts/plot_gap_curve.py --in reports/supplementary_5050/gaps.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Gap-vs-epoch plot")
    p.add_argument("--in", dest="inp", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    d = json.load(open(args.inp))
    out = Path(args.inp).parent / "gap_vs_epoch.png"
    num = [e for e in d["rows"] if isinstance(e["epoch"], int)]
    eps = [e["epoch"] for e in num]
    fig, ax = plt.subplots(figsize=(6, 4))
    for side, key_tr, key_te, col in [("223k", "train223k", "test223k", "C0"),
                                      ("pneumo", "trainpn", "testpn", "C1")]:
        ax.plot(eps, [e[key_tr] for e in num], "o--", color=col, lw=1.2, label=f"train {side}")
        ax.plot(eps, [e[key_te] for e in num], "s-", color=col, lw=1.5, label=f"test {side}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="epochs", ylabel="MCC", title="Train/test MCC vs epochs — high5050")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"{out} OK")
    # frontier: test MCC vs gap (Pareto defense — no config has lower gap at equal test)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for e in d["rows"]:
        if e["gap223k"] is None:
            continue
        ax2.plot(e["gap223k"], e["test223k"], "o", ms=7)
        ax2.text(e["gap223k"] + 0.002, e["test223k"] + 0.001, str(e["epoch"]), fontsize=8)
    ax2.set(xlabel="train-test gap MCC (223k)", ylabel="paper-test MCC",
            title="Frontier: test improves with gap; 4ep off-frontier")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    f2 = Path(args.inp).parent / "gap_frontier.png"
    fig2.savefig(f2, dpi=120)
    plt.close(fig2)
    print(f"{f2} OK")


if __name__ == "__main__":
    main()
