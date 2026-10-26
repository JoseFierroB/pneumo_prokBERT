#!/usr/bin/env python3
"""Screening plots from screening.json — one metric per file.

Reads reports/supplementary_5050/screening.json, writes pr_curve.png,
recall_vs_divergence.png (F1), recall_vs_shift.png (F3) next to it.

Usage:
  pixi run python scripts/plot_screening.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("reports/supplementary_5050/screening.json")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(SRC.parent / name, dpi=120)
    plt.close(fig)
    print(f"{name} OK")


def main():
    d = json.load(open(SRC))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(d["pr_curve"]["rec"], d["pr_curve"]["prec"], lw=1.5)
    ax.set(xlabel="recall", ylabel="precision", title="PR — 3ep 5050-test")
    ax.grid(alpha=0.3)
    save(fig, "pr_curve.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    f1 = d["f1_divergence"]
    ax.plot([r * 100 for r in f1["rate"]], f1["recall"], "o-", lw=1.5)
    ax.set(xlabel="divergence %", ylabel="recall@0.5", title="F1 robustness — 3ep")
    ax.grid(alpha=0.3)
    save(fig, "recall_vs_divergence.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    f3 = d["f3_shift"]
    ax.plot(f3["shift"], f3["recall"], "o-", lw=1.5)
    ax.set(xlabel="shift bp", ylabel="recall@0.5", title="F3 position sensitivity — 3ep")
    ax.grid(alpha=0.3)
    save(fig, "recall_vs_shift.png")


if __name__ == "__main__":
    main()
