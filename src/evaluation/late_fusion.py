import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def compute_late_fusion_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    qidxs: np.ndarray,
    uids: Optional[np.ndarray] = None,
    fusion_type: str = "mean",
) -> Dict[str, float]:
    """
    Computes Question-Level evaluation metrics using Late Fusion according to pyKT standards.

    In pyKT benchmark datasets (e.g. XES3G5M), questions often map to multiple Knowledge Components (KCs).
    During KC-level sequence modeling, models produce predictions for each individual KC.
    Late Fusion aggregates KC-level predictions belonging to the same question interaction
    into a single question-level prediction before computing evaluation metrics.

    Args:
        preds: 1D array of predicted probabilities at KC level (in [0, 1]).
        targets: 1D array of binary ground-truth correctness labels (0 or 1).
        qidxs: 1D array of question indices identifying the question interaction.
        uids: Optional 1D array of user IDs. If provided, grouping key is (uid, qidx).
        fusion_type: Aggregation method ('mean', 'vote', 'all'). Default is 'mean'.

    Returns:
        Dictionary of question-level metrics:
            - auc: Area Under the ROC Curve
            - accuracy: Binary accuracy
            - logloss: Binary cross-entropy log loss
            - brier: Brier score
            - num_questions: Total number of aggregated questions evaluated
            - num_kcs: Total number of underlying KC interactions
            - fusion_type: Fusion method applied
    """
    if len(preds) == 0 or len(targets) == 0:
        return {
            "auc": 0.5,
            "accuracy": 0.0,
            "logloss": 0.0,
            "brier": 0.0,
            "num_questions": 0,
            "num_kcs": 0,
            "fusion_type": fusion_type,
        }

    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    qidxs = np.asarray(qidxs, dtype=np.int64)

    # Build DataFrame for stable, vectorized grouping
    df_data = {
        "pred": preds,
        "target": targets,
        "qidx": qidxs,
    }
    if uids is not None:
        df_data["uid"] = np.asarray(uids, dtype=np.int64)
        group_keys = ["uid", "qidx"]
    else:
        group_keys = ["qidx"]

    df = pd.DataFrame(df_data)

    # Late Fusion aggregations
    fusion_key = fusion_type.lower()
    if fusion_key in ["mean", "late_mean"]:
        grouped = df.groupby(group_keys).agg(
            q_pred=("pred", "mean"),
            q_target=("target", "first"),
        ).reset_index()
    elif fusion_key in ["vote", "late_vote"]:
        # Majority voting over binary KC thresholded decisions
        grouped = df.groupby(group_keys).agg(
            q_pred=("pred", lambda x: float(np.mean(np.array(x) >= 0.5) >= 0.5)),
            q_target=("target", "first"),
        ).reset_index()
    elif fusion_key in ["all", "late_all"]:
        # All KCs must be predicted correct (product of probabilities)
        grouped = df.groupby(group_keys).agg(
            q_pred=("pred", lambda x: float(np.prod(x))),
            q_target=("target", "first"),
        ).reset_index()
    else:
        logger.warning("Unknown fusion type '%s'. Defaulting to 'mean'.", fusion_type)
        grouped = df.groupby(group_keys).agg(
            q_pred=("pred", "mean"),
            q_target=("target", "first"),
        ).reset_index()

    q_targets = grouped["q_target"].values
    q_preds = grouped["q_pred"].values

    num_questions = len(q_targets)
    if num_questions == 0:
        return {
            "auc": 0.5,
            "accuracy": 0.0,
            "logloss": 0.0,
            "brier": 0.0,
            "num_questions": 0,
            "num_kcs": len(preds),
            "fusion_type": fusion_type,
        }

    # AUC calculation requires at least 2 distinct classes in target
    unique_classes = np.unique(q_targets)
    if len(unique_classes) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(q_targets, q_preds))

    # Binary accuracy with threshold 0.5
    q_preds_binary = (q_preds >= 0.5).astype(int)
    acc = float(accuracy_score(q_targets, q_preds_binary))

    # Clip probabilities for stable numerical log loss
    eps = 1e-15
    q_preds_clipped = np.clip(q_preds, eps, 1.0 - eps)
    ll = float(log_loss(q_targets, q_preds_clipped, labels=[0, 1]))
    brier = float(brier_score_loss(q_targets, q_preds_clipped))

    return {
        "auc": round(auc, 6),
        "accuracy": round(acc, 6),
        "logloss": round(ll, 6),
        "brier": round(brier, 6),
        "num_questions": int(num_questions),
        "num_kcs": int(len(preds)),
        "fusion_type": fusion_type,
    }


def compute_all_late_fusion_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    qidxs: np.ndarray,
    uids: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes all standard pyKT late fusion modes ('mean', 'vote', 'all')
    for comprehensive benchmarking reports.
    """
    return {
        "late_mean": compute_late_fusion_metrics(preds, targets, qidxs, uids=uids, fusion_type="mean"),
        "late_vote": compute_late_fusion_metrics(preds, targets, qidxs, uids=uids, fusion_type="vote"),
        "late_all": compute_late_fusion_metrics(preds, targets, qidxs, uids=uids, fusion_type="all"),
    }
