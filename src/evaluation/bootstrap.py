import logging
from typing import Any, Dict

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


def fast_mann_whitney_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes exact Area Under the ROC Curve via Mann-Whitney U rank statistic.
    Handles ties with average midranks identically to scikit-learn's trapezoidal integration.
    """
    y_true_bool = y_true.astype(bool)
    n_pos = int(np.count_nonzero(y_true_bool))
    n_neg = len(y_true_bool) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = rankdata(y_pred, method="average")
    r_pos = float(np.sum(ranks[y_true_bool]))
    u_stat = r_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def bootstrap_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        if len(set(y_true[indices])) < 2:
            continue
        try:
            auc = fast_mann_whitney_auc(y_true[indices], y_pred[indices])
            aucs.append(auc)
        except Exception:
            continue

    if not aucs:
        return {"auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_bootstrap": 0}

    aucs = np.array(aucs)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(aucs, alpha / 2 * 100))
    upper = float(np.percentile(aucs, (1.0 - alpha / 2) * 100))

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "ci_level": ci_level,
        "n_bootstrap": len(aucs),
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        if len(set(y_true[indices])) < 2:
            continue
        try:
            auc_a = fast_mann_whitney_auc(y_true[indices], pred_a[indices])
            auc_b = fast_mann_whitney_auc(y_true[indices], pred_b[indices])
            diffs.append(auc_a - auc_b)
        except Exception:
            continue

    if not diffs:
        return {
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_bootstrap": 0,
            "significant": False,
        }

    diffs = np.array(diffs)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(diffs, alpha / 2 * 100))
    upper = float(np.percentile(diffs, (1.0 - alpha / 2) * 100))
    mean_diff = float(np.mean(diffs))
    significant = (lower > 0) or (upper < 0)

    return {
        "mean_diff": round(mean_diff, 6),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "ci_level": ci_level,
        "n_bootstrap": len(diffs),
        "significant": significant,
        "interpretation": "robust" if significant else "not robust under bootstrap",
    }


def student_clustered_bootstrap_auc(
    student_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Computes bootstrap confidence intervals by resampling at the student level (cluster bootstrap),
    preserving correlation structure among interactions of the same student.
    """
    unique_students = np.unique(student_ids)
    n_students = len(unique_students)
    if n_students == 0:
        return {"auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_bootstrap": 0}

    # Map each student to their row indices
    student_idx_map = {}
    for idx, sid in enumerate(student_ids):
        student_idx_map.setdefault(sid, []).append(idx)

    student_indices_list = [np.array(student_idx_map[s]) for s in unique_students]

    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstrap):
        sampled_stud_indices = rng.randint(0, n_students, size=n_students)
        sampled_rows = np.concatenate([student_indices_list[i] for i in sampled_stud_indices])

        sampled_true = y_true[sampled_rows]
        if len(set(sampled_true)) < 2:
            continue

        try:
            auc = fast_mann_whitney_auc(sampled_true, y_pred[sampled_rows])
            aucs.append(auc)
        except Exception:
            continue

    if not aucs:
        return {"auc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_bootstrap": 0}

    aucs = np.array(aucs)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(aucs, alpha / 2 * 100))
    upper = float(np.percentile(aucs, (1.0 - alpha / 2) * 100))

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "ci_level": ci_level,
        "n_bootstrap": len(aucs),
        "sampling_unit": "student",
    }

