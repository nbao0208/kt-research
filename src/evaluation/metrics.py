import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "auc": 0.0,
            "logloss": 0.0,
            "brier": 0.0,
            "rmse": 0.0,
            "accuracy": 0.0,
            "positive_rate": 0.0,
            "n": 0,
            "is_multi_class": False,
        }

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    n = len(y_true)
    positive_rate = float(y_true.mean())
    is_multi_class = len(set(y_true)) > 1

    try:
        auc = float(roc_auc_score(y_true, y_pred)) if is_multi_class else 0.5
    except ValueError:
        auc = 0.5

    try:
        logloss = float(log_loss(y_true, y_pred, labels=[0, 1]))
    except ValueError:
        logloss = 0.0
    brier = float(brier_score_loss(y_true, y_pred))
    rmse = float(np.sqrt(brier))
    accuracy = float(((y_pred > 0.5).astype(int) == y_true).mean())

    return {
        "auc": round(auc, 6),
        "logloss": round(logloss, 6),
        "brier": round(brier, 6),
        "rmse": round(rmse, 6),
        "accuracy": round(accuracy, 6),
        "positive_rate": round(positive_rate, 6),
        "n": n,
        "is_multi_class": is_multi_class,
    }


def compute_skill_metrics(
    df: pd.DataFrame,
    y_true_col: str = "correct",
    y_pred_col: str = "prediction",
    skill_col: str = "skill",
) -> Dict[str, Any]:
    if df.empty:
        return {
            "skill_macro_auc": 0.0,
            "skill_weighted_auc": 0.0,
            "n_evaluable_skills": 0,
            "n_single_class_skills": 0,
            "per_skill": {},
        }

    per_skill = {}
    for skill, group in df.groupby(skill_col):
        if len(group) < 2:
            continue
        metrics = compute_metrics(group[y_true_col].values, group[y_pred_col].values)
        metrics["n"] = len(group)
        per_skill[str(skill)] = metrics

    if not per_skill:
        return {
            "skill_macro_auc": 0.0,
            "skill_weighted_auc": 0.0,
            "n_evaluable_skills": 0,
            "n_single_class_skills": 0,
            "per_skill": {},
        }

    evaluable_aucs = [m["auc"] for m in per_skill.values() if m.get("is_multi_class", True)]
    evaluable_ns = [m["n"] for m in per_skill.values() if m.get("is_multi_class", True)]

    if evaluable_aucs:
        macro_auc = float(np.mean(evaluable_aucs))
        weighted_auc = float(np.average(evaluable_aucs, weights=evaluable_ns))
    else:
        macro_auc = 0.5
        weighted_auc = 0.5

    return {
        "skill_macro_auc": round(macro_auc, 6),
        "skill_weighted_auc": round(weighted_auc, 6),
        "n_evaluable_skills": len(evaluable_aucs),
        "n_single_class_skills": len(per_skill) - len(evaluable_aucs),
        "per_skill": per_skill,
    }


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_boundaries, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_pred[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.sum()

    return round(float(ece / len(y_true) if len(y_true) > 0 else 0.0), 6)


def compute_bucket_metrics(
    df: pd.DataFrame,
    y_true_col: str = "correct",
    y_pred_col: str = "prediction",
    skill_col: str = "skill",
    student_col: str = "studentId",
) -> Dict[str, Any]:
    buckets: Dict[str, Any] = {"by_skill_frequency": {}, "by_student_history": {}}

    if df.empty:
        return buckets

    # 1. Bucket by skill frequency
    skill_counts = df[skill_col].value_counts()
    freq_bins = {
        "<100": skill_counts[skill_counts < 100].index,
        "100-1000": skill_counts[(skill_counts >= 100) & (skill_counts <= 1000)].index,
        ">1000": skill_counts[skill_counts > 1000].index,
    }
    for bin_name, skills in freq_bins.items():
        subset = df[df[skill_col].isin(skills)]
        if not subset.empty:
            if len(np.unique(subset[y_true_col].values)) > 1:
                buckets["by_skill_frequency"][bin_name] = compute_metrics(
                    subset[y_true_col].values, subset[y_pred_col].values
                )
            else:
                buckets["by_skill_frequency"][bin_name] = {
                    "auc": 0.5,
                    "accuracy": float((subset[y_true_col].values == (subset[y_pred_col].values >= 0.5)).mean()),
                }

    # 2. Bucket by student history length (prior interactions per student)
    if student_col in df.columns:
        hist = df.groupby(student_col).cumcount()
    elif "attempts_before" in df.columns:
        hist = df["attempts_before"]
    else:
        hist = pd.Series(0, index=df.index)

    h_bins = {
        "<10": df[hist < 10],
        "10-50": df[(hist >= 10) & (hist <= 50)],
        ">50": df[hist > 50],
    }
    for bin_name, subset in h_bins.items():
        if not subset.empty:
            if len(np.unique(subset[y_true_col].values)) > 1:
                buckets["by_student_history"][bin_name] = compute_metrics(
                    subset[y_true_col].values, subset[y_pred_col].values
                )
            else:
                buckets["by_student_history"][bin_name] = {
                    "auc": 0.5,
                    "accuracy": float((subset[y_true_col].values == (subset[y_pred_col].values >= 0.5)).mean()),
                }

    return buckets


def compute_comprehensive_metrics(
    df: pd.DataFrame,
    y_true_col: str = "correct",
    y_pred_col: str = "prediction",
    skill_col: str = "skill",
    student_col: str = "studentId",
) -> Dict[str, Any]:
    if df.empty:
        return {}

    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values

    global_metrics = compute_metrics(y_true, y_pred)
    global_metrics["ece"] = compute_ece(y_true, y_pred)

    skill_res = compute_skill_metrics(df, y_true_col, y_pred_col, skill_col)
    global_metrics["skill_macro_auc"] = skill_res["skill_macro_auc"]
    global_metrics["skill_weighted_auc"] = skill_res["skill_weighted_auc"]

    bucket_res = compute_bucket_metrics(df, y_true_col, y_pred_col, skill_col, student_col)

    return {
        "global": global_metrics,
        "skill_metrics": skill_res,
        "buckets": bucket_res,
    }
