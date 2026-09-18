import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_online_next_step(
    df: pd.DataFrame,
    predict_fn: Callable,
    update_fn: Optional[Callable] = None,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    split_col: str = "split",
    split_name: str = "test",
) -> Dict[str, Any]:
    test_df = df[df[split_col] == split_name].copy()
    if len(test_df) == 0:
        logger.warning("No data found for split '%s'", split_name)
        return {"auc": 0.0, "logloss": 0.0, "n": 0}

    test_df = test_df.sort_values([student_col, "startTime", "action_num"]).reset_index(drop=True)
    y_true = []
    y_pred = []

    state: Dict = {}

    for _, row in test_df.iterrows():
        student = row[student_col]
        skill = row[skill_col]
        correct = int(row[correct_col])

        prob = predict_fn(student, skill, state, row)
        y_pred.append(prob)
        y_true.append(correct)

        if update_fn:
            update_fn(student, skill, correct, state, row)

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    return metrics


def evaluate_held_out_block(
    df: pd.DataFrame,
    predict_fn: Callable,
    student_col: str = "studentId",
    correct_col: str = "correct",
    block_frac: float = 0.1,
    min_history: int = 3,
) -> Dict[str, Any]:
    df = df.sort_values([student_col, "startTime", "action_num"]).reset_index(drop=True)
    y_true = []
    y_pred = []

    for student in df[student_col].unique():
        mask = df[student_col] == student
        student_df = df[mask].reset_index(drop=True)
        n = len(student_df)

        if n < min_history + 1:
            continue

        block_size = max(1, int(n * block_frac))
        history_df = student_df.iloc[:-block_size]
        test_df = student_df.iloc[-block_size:]

        state: Dict = {}
        for _, row in history_df.iterrows():
            if student not in state:
                state[student] = {}
            skill = row.get("skill", "default")
            predict_fn(student, skill, state, row)
            if skill in state.get(student, {}):
                pass

        for _, row in test_df.iterrows():
            skill = row.get("skill", "default")
            prob = predict_fn(student, skill, state, row, update=False)
            y_pred.append(prob)
            y_true.append(int(row[correct_col]))

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    return metrics
