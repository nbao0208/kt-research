import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_bkt_sequences(
    df: pd.DataFrame,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    split_col: str = "split",
    split_name: str = "train",
    min_sequence_length: int = 3,
) -> List[Dict[str, Any]]:
    train_df = df[df[split_col] == split_name].copy()
    if len(train_df) == 0:
        logger.warning("No data found for split '%s'", split_name)
        return []

    sequences = []
    grouped = train_df.groupby([student_col, skill_col])
    for (student, skill), group in grouped:
        group = group.sort_values(["startTime", "action_num"] if "action_num" in group.columns else ["startTime"])
        seq = group[correct_col].tolist()
        if len(seq) >= min_sequence_length:
            sequences.append(
                {
                    "student_id": student,
                    "skill": skill,
                    "sequence": seq,
                    "length": len(seq),
                }
            )

    logger.info(
        "Built %d BKT sequences from split '%s' (min length=%d)", len(sequences), split_name, min_sequence_length
    )
    return sequences


def build_dbkt_sequences(
    df: pd.DataFrame,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    split_col: str = "split",
    split_name: str = "train",
    min_sequence_length: int = 3,
    skill_encoder: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    train_df = df[df[split_col] == split_name].copy()
    if len(train_df) == 0:
        logger.warning("No data found for split '%s'", split_name)
        return []

    sequences = []
    grouped = train_df.groupby([student_col, skill_col])
    for (student, skill), group in grouped:
        sort_cols = ["startTime"]
        if "action_num" in group.columns:
            sort_cols.append("action_num")
        group = group.sort_values(sort_cols).reset_index(drop=True)
        seq = group[correct_col].values.astype(np.float32)

        if len(seq) < min_sequence_length:
            continue

        if feature_cols is not None and all(col in group.columns for col in feature_cols):
            features = group[feature_cols].values.astype(np.float32)
        else:
            # Default leak-free DBKT features: log(1 + attempts_before), log(1 + failure_before)
            attempts_before = np.arange(len(group), dtype=np.float32)
            success_before = np.maximum(0.0, np.cumsum(seq) - seq)
            failure_before = np.maximum(0.0, attempts_before - success_before)
            features = np.column_stack([
                np.log1p(attempts_before),
                np.log1p(failure_before),
            ]).astype(np.float32)

        skill_id = int(skill_encoder.transform([skill])[0]) if skill_encoder is not None else 0

        sequences.append(
            {
                "student_id": student,
                "skill": str(skill),
                "skill_id": skill_id,
                "sequence": seq.tolist(),
                "features": features,
                "length": len(seq),
            }
        )

    logger.info(
        "Built %d DBKT sequences with dynamic features from split '%s' (min length=%d)",
        len(sequences),
        split_name,
        min_sequence_length,
    )
    return sequences


def compute_sequence_metadata(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sequences:
        return {
            "n_sequences": 0,
            "avg_length": 0.0,
            "n_skills": 0,
            "n_students": 0,
        }

    lengths = [s["length"] for s in sequences]
    skills = set(s["skill"] for s in sequences)
    students = set(s["student_id"] for s in sequences)

    return {
        "n_sequences": len(sequences),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "n_skills": len(skills),
        "n_students": len(students),
    }

