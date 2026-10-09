"""Competition metrics. Add the official one here the moment the problem drops,
and copy its formula *exactly* (edge cases like 0/0 matter for SMAPE/MAPE)."""
from __future__ import annotations

import numpy as np


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = (np.abs(y) + np.abs(p)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom == 0, 0.0, np.abs(y - p) / denom)
    return 100 * r.mean()


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 100 * np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-9))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro"))


def extraction_f1(y, p):
    """2024-style: TP = pred matches non-empty GT; FP = wrong/extra pred; FN = missed GT."""
    tp = fp = fn = 0
    for gt, pr in zip(y, p):
        gt = "" if gt is None or (isinstance(gt, float) and np.isnan(gt)) else str(gt).strip()
        pr = "" if pr is None or (isinstance(pr, float) and np.isnan(pr)) else str(pr).strip()
        if pr and gt:
            tp += pr == gt
            fp += pr != gt
        elif pr and not gt:
            fp += 1
        elif gt and not pr:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


# name -> (fn, greater_is_better)
METRICS = {
    "smape": (smape, False),
    "mape": (mape, False),
    "mae": (mae, False),
    "rmse": (rmse, False),
    "accuracy": (accuracy, True),
    "macro_f1": (macro_f1, True),
    "extraction_f1": (extraction_f1, True),
}


def score(y, p, name: str | None = None) -> float:
    from .data import cfg
    fn, _ = METRICS[name or cfg()["metric"]]
    return float(fn(y, p))


def greater_is_better(name: str | None = None) -> bool:
    from .data import cfg
    return METRICS[name or cfg()["metric"]][1]
