#!/usr/bin/env python3
"""Training curves for 1ep + dropout arms from WS train logs.

Reads /tmp/curve_logs/train_*.log (loss/eval dicts), writes per-model
loss+MCC+AUC plots + a cross-model AUC-val overlay into
reports/supplementary_5050/curves_1ep/ + curves_1ep.json (step of max val-MCC/AUC).

Usage:
  pixi run python scripts/parse_curves_1ep.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = {
    "base@1ep": "/tmp/curve_logs/train_base1ep.log",
    "do005@1ep": "/tmp/curve_logs/train_do005_1ep.log",
    "do012@1ep": "/tmp/curve_logs/train_do012_1ep.log",
    "do015@1ep": "/tmp/curve_logs/train_do015_1ep.log",
    "do005@1ep-s44": "/tmp/curve_logs/train_do005_s44_1ep.log",
    "do020@1ep": "/tmp/curve_logs/cell_do020_1ep.log",
}
OUT = Path("reports/supplementary_5050/curves_1ep")


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", t).replace(chr(13), chr(10))
    loss = [(int(float(e) * 100), float(v)) for v, e in
            re.findall(r"'loss': ([0-9.]+).*?'epoch': ([0-9.]+)\}", t)]
    mcc = [float(v) for v in re.findall(r"'eval_mcc': ([0-9.\-e]+)", t)]
    auc = [float(v) for v in re.findall(r"'eval_roc_auc': ([0-9.\-ena]+)", t) if v != "nan"]
    steps = sorted({s for s, _ in loss})
    epochs = np.array(steps) / 100.0
    return epochs, [v for _, v in sorted(loss)], mcc, auc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, overlay = {}, {}
    for alias, path in LOGS.items():
        if not Path(path).exists():
            print(f"SKIP {alias} (no log)"); continue
        xs, loss, mcc, auc = parse(path)
        n = min(len(xs), len(mcc), len(auc))
        xs = np.array(xs[:n])
        mcc, auc = np.array(mcc[:n]), np.array(auc[:n])
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(xs, loss[:n] if len(loss) >= n else loss, "k-", lw=1, label="train loss")
        ax1.set(xlabel="epoch", ylabel="loss")
        ax1.set_ylim(bottom=0.4)
        ax2 = ax1.twinx()
        ax2.plot(xs, mcc, "b--", lw=1.2, label="val MCC")
        ax2.plot(xs, auc, "g-.", lw=1.2, label="val AUC")
        ax2.set_ylabel("MCC / AUC")
        ax2.set_ylim(0.0, 1.0)
        ax1.set_title(f"train curve — {alias}")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"curve__{alias}.png", dpi=120)
        plt.close(fig)
        i_m, i_a = int(np.argmax(mcc)), int(np.argmax(auc))
        table[alias] = {"n_eval": n, "max_mcc": round(float(mcc[i_m]), 4),
                        "epoch_max_mcc": round(float(xs[i_m]), 2),
                        "max_auc": round(float(auc[i_a]), 4),
                        "epoch_max_auc": round(float(xs[i_a]), 2)}
        overlay[alias] = (xs.tolist(), auc.tolist())
        print(alias, table[alias])
    fig, ax = plt.subplots(figsize=(6, 4))
    for alias, (xs, auc) in overlay.items():
        ax.plot(xs, auc, lw=1.4, label=alias)
    ax.set(xlabel="epoch", ylabel="val AUC", title="val-AUC overlay — 1ep dropout arms")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "overlay__auc.png", dpi=120)
    plt.close(fig)
    json.dump(table, open(OUT / "curves_1ep.json", "w"), indent=1)
    print("curves_1ep.json OK")


if __name__ == "__main__":
    main()
