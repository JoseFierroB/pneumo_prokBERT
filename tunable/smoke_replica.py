#!/usr/bin/env python3
"""Smoke experiment: paper-replica via tunable/ on the new data regime.

Acceptance (must match the OFAT center ofat_lr4e4, deterministic):
  data_test MCC == 0.6483, data_test AUC == 0.9124, gap AUC == +0.0096
Tolerances are tight on purpose: same code path + same seed must reproduce.
Exit nonzero on mismatch (gate for tunable/ adoption).

Usage:
  pixi run python tunable/smoke_replica.py [--output-dir finetuning_outputs/tunable_smoke]
"""
import re
import subprocess
import sys

EXPECTED = {"data_test_mcc": 0.6483169198887292, "data_test_auc": 0.9124489795918367,
            "gap_auc": 0.0096}
TOL = {"data_test_mcc": 1e-4, "data_test_auc": 1e-4, "gap_auc": 1e-3}

BASE = ["pixi", "run", "python", "-m", "tunable.cli", "--preset", "replica-1ep",
        "--data-dir", "datasets/high5050_stratified_801010",
        "--hf-split-spec", "local:species801010", "--seed", "42",
        "--gap", "--gap-test-csv", "datasets/hf_splits/species801010/test_segments.csv",
        "--gap-test-name", "test-set", "--gap-out-dir", "reports/gaps/tunable"]


def sh(cmd, log):
    print("+", " ".join(cmd))
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(r.stdout)
    print(f"rc={r.returncode} (log {log})")
    if r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {' '.join(cmd)} (see {log})")
    return r.stdout


def main():
    out = sys.argv[sys.argv.index("--output-dir") + 1] if "--output-dir" in sys.argv \
        else "finetuning_outputs/tunable_smoke"
    train_log = "/tmp/tunable_smoke_train.log"
    out_txt = sh(BASE + ["--output-dir", out], train_log)
    flat = out_txt.replace("\r", "\n")
    m = re.search(r"^data_test \{(.*?)\}", flat, re.M)
    if not m:
        raise SystemExit("data_test line not found in train log")
    body = m.group(1)
    got = {"data_test_mcc": float(re.search(r"eval_mcc': ([0-9.]+)", body).group(1)),
           "data_test_auc": float(re.search(r"eval_roc_auc': ([0-9.]+)", body).group(1))}
    mg = re.search(r"gap AUC ([+-][0-9.]+)", flat)
    if mg:
        got["gap_auc"] = float(mg.group(1))
    else:
        gap_log = f"{out.rstrip('/')}/gap.log"
        gtxt = open(gap_log).read().replace("\r", "\n")
        mg2 = re.search(r"gap AUC ([+-][0-9.]+)", gtxt)
        if not mg2:
            raise SystemExit("gap AUC not found (stdout nor gap.log)")
        got["gap_auc"] = float(mg2.group(1))
    print("smoke got:", got)
    print("expected :", EXPECTED)
    bad = [k for k in EXPECTED if abs(got[k] - EXPECTED[k]) > TOL[k]]
    if bad:
        raise SystemExit(f"SMOKE MISMATCH on {bad}: got {got} vs {EXPECTED}")
    print("SMOKE OK -- tunable/ reproduces ofat_lr4e4")


if __name__ == "__main__":
    main()
