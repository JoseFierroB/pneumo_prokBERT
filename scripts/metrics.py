"""Shared classification metrics (single implementation; import, do not copy).

Two reporting levels:
- cell_metrics: full cell for reports/model_matrix (confusion matrix, rates, Youden).
- cell_metrics_light: small cell for energy/ probes (Youden only, no confusion matrix).
- trainer_compute_metrics: HF Trainer callback (mcc/accuracy/f1/roc_auc, stable softmax).

Usage:
  from scripts.metrics import cell_metrics, cell_metrics_light, trainer_compute_metrics
  m = cell_metrics(y_true, scores)
  trainer = Trainer(..., compute_metrics=trainer_compute_metrics)
  save_probs_csv("probs/foo__bar.csv", y_true, scores)
  y, s = load_probs_csv("probs/foo__bar.csv")
"""
import csv

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve)


def cell_metrics(y: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> dict:
    """Full metric cell: fixed-threshold (0.5) + ranking (AUC/AP) + Youden-threshold."""
    pred = (scores >= thr).astype(int)
    fpr, tpr, ths = roc_curve(y, scores)
    idx = int(np.argmax(tpr - fpr))
    ythr = float(ths[idx])
    pred_y = (scores >= ythr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel().tolist()
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "cm_05": [[tn, fp], [fn, tp]],
        "acc_05": float(accuracy_score(y, pred)),
        "balacc_05": float(balanced_accuracy_score(y, pred)),
        "mcc_05": float(matthews_corrcoef(y, pred)),
        "f1_05": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "tpr_05": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tnr_05": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ppv_05": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "pred_pos_rate_05": float(pred.mean()),
        "youden_thr": ythr,
        "acc_y": float(accuracy_score(y, pred_y)),
        "mcc_y": float(matthews_corrcoef(y, pred_y)),
    }


def cell_metrics_light(y: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> dict:
    """Small metric cell for probes: fixed-threshold + ranking + Youden (no confusion matrix)."""
    pred = (scores >= thr).astype(int)
    fpr, tpr, ths = roc_curve(y, scores)
    idx = int(np.argmax(tpr - fpr))
    ythr = float(ths[idx])
    pred_y = (scores >= ythr).astype(int)
    return {"n": int(len(y)), "acc_05": float(accuracy_score(y, pred)),
            "mcc_05": float(matthews_corrcoef(y, pred)), "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "acc_y": float(accuracy_score(y, pred_y)), "mcc_y": float(matthews_corrcoef(y, pred_y)),
            "youden_thr": ythr}


def trainer_compute_metrics(eval_pred) -> dict:
    """HF Trainer metric callback: argmax class + stable-softmax AUC (NaN if single class)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return {"mcc": matthews_corrcoef(labels, preds), "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds), "roc_auc": auc}


def save_probs_csv(path, y: np.ndarray, scores: np.ndarray) -> None:
    """Write per-cell probabilities as CSV (y_true,y_score), human-readable cache."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "y_score"])
        w.writerows((int(t), repr(float(p))) for t, p in zip(y, s))


def load_probs_csv(path) -> tuple:
    """Read per-cell probabilities CSV. Returns (y_true int array, y_score float array)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows and rows[0] == ["y_true", "y_score"], f"bad header in {path}"
    y = np.array([int(r[0]) for r in rows[1:]])
    s = np.array([float(r[1]) for r in rows[1:]])
    return y, s
