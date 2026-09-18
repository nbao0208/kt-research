import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class MajorityBaseline:
    def __init__(self):
        self.global_rate: float = 0.0

    def fit(self, train_df: pd.DataFrame, correct_col: str = "correct") -> "MajorityBaseline":
        self.global_rate = float(train_df[correct_col].mean())
        logger.info("Majority baseline: global correct rate = %.4f", self.global_rate)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.global_rate)


class SkillAverageBaseline:
    def __init__(self, fallback_rate: Optional[float] = None):
        self.skill_rates: Dict[str, float] = {}
        self.fallback_rate: Optional[float] = None

    def fit(
        self, train_df: pd.DataFrame, skill_col: str = "skill", correct_col: str = "correct"
    ) -> "SkillAverageBaseline":
        self.skill_rates = {
            str(k): float(v)
            for k, v in train_df.groupby(skill_col)[correct_col].mean().to_dict().items()
        }
        self.fallback_rate = float(train_df[correct_col].mean())
        logger.info("Skill average baseline: %d skills, fallback=%.4f", len(self.skill_rates), self.fallback_rate)
        return self

    def predict(self, df: pd.DataFrame, skill_col: str = "skill") -> np.ndarray:
        return np.array([self.skill_rates.get(str(s), self.fallback_rate) for s in df[skill_col]])


class StudentAverageBaseline:
    def __init__(self, global_fallback: Optional[float] = None):
        self.student_rates: Dict[str, float] = {}
        self.global_fallback: Optional[float] = None

    def fit(
        self, train_df: pd.DataFrame, student_col: str = "studentId", correct_col: str = "correct"
    ) -> "StudentAverageBaseline":
        self.student_rates = {
            str(k): float(v)
            for k, v in train_df.groupby(student_col)[correct_col].mean().to_dict().items()
        }
        self.global_fallback = float(train_df[correct_col].mean())
        logger.info(
            "Student average baseline: %d students, fallback=%.4f",
            len(self.student_rates),
            self.global_fallback,
        )
        return self

    def predict(self, df: pd.DataFrame, student_col: str = "studentId") -> np.ndarray:
        return np.array([self.student_rates.get(str(s), self.global_fallback) for s in df[student_col]])


def evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    skill_col: str = "skill",
    student_col: str = "studentId",
    correct_col: str = "correct",
) -> Dict[str, Any]:
    results = {}

    majority = MajorityBaseline().fit(train_df)
    val_pred = majority.predict(val_df)
    test_pred = majority.predict(test_df)
    results["majority"] = {
        "val": compute_metrics(val_df[correct_col].values, val_pred),
        "test": compute_metrics(test_df[correct_col].values, test_pred),
        "global_rate": majority.global_rate,
    }

    skill_avg = SkillAverageBaseline().fit(train_df)
    val_pred = skill_avg.predict(val_df)
    test_pred = skill_avg.predict(test_df)
    results["skill_average"] = {
        "val": compute_metrics(val_df[correct_col].values, val_pred),
        "test": compute_metrics(test_df[correct_col].values, test_pred),
    }

    student_avg = StudentAverageBaseline().fit(train_df)
    val_pred = student_avg.predict(val_df)
    test_pred = student_avg.predict(test_df)
    results["student_average"] = {
        "val": compute_metrics(val_df[correct_col].values, val_pred),
        "test": compute_metrics(test_df[correct_col].values, test_pred),
    }

    return results
