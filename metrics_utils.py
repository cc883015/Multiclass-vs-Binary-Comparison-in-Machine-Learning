# metrics_utils.py
# -----------------------------------------------------------
# 统一指标：
# - accuracy, macro_precision, macro_recall, macro_f1, macro_fpr
# - latency_ms_per_sample（predict 批量耗时 / 样本数）
# - model_size_mb（序列化大小）
# -----------------------------------------------------------

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import time, os, joblib, tempfile

def macro_fpr(y_true, y_pred, labels: List) -> float:
    fprs = []
    for k in labels:
        y_t = (y_true == k).astype(int)
        y_p = (y_pred == k).astype(int)
        cm = confusion_matrix(y_t, y_p, labels=[1,0])
        TP, FN, FP, TN = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        denom = FP + TN
        fprs.append(0.0 if denom == 0 else FP / denom)
    return float(np.mean(fprs)) if fprs else 0.0

def macro_prf(y_true, y_pred, labels: List) -> Tuple[float,float,float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return float(p), float(r), float(f1)

def latency_ms_per_sample(model, X) -> float:
    n = len(X)
    t0 = time.perf_counter()
    _ = model.predict(X)
    t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / max(n, 1)

def model_size_mb(model) -> float:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.joblib")
        joblib.dump(model, path)
        return os.path.getsize(path) / (1024 * 1024)

def compute_all_metrics(y_true, y_pred, labels: List, model, X_test) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1 = macro_prf(y_true, y_pred, labels)
    mfpr = macro_fpr(y_true, y_pred, labels)
    lat = latency_ms_per_sample(model, X_test)
    return {
        "accuracy": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "macro_fpr": mfpr,
        "latency_ms_per_sample": lat,
        "model_size_mb": model_size_mb(model),
    }
