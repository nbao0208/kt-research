import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.encoders import LabelEncoderWithUNK

logger = logging.getLogger(__name__)


def build_lfa_core_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    time_col: str = "startTime",
    action_col: str = "action_num",
) -> Tuple[pd.DataFrame, LabelEncoderWithUNK, LabelEncoderWithUNK]:
    df = df.copy()
    train_df = train_df.copy()

    sort_cols = [student_col, time_col]
    if action_col in df.columns:
        sort_cols.append(action_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    student_encoder = LabelEncoderWithUNK()
    skill_encoder = LabelEncoderWithUNK()

    student_encoder.fit(train_df[student_col].unique())
    skill_encoder.fit(train_df[skill_col].unique())

    df["student_id_encoded"] = student_encoder.transform(df[student_col].values)
    df["skill_id_encoded"] = skill_encoder.transform(df[skill_col].values)

    df["attempts_before"] = df.groupby([student_col, skill_col]).cumcount()
    df["success_cum"] = df.groupby([student_col, skill_col])[correct_col].cumsum()
    df["success_before"] = df["success_cum"] - df[correct_col]
    df["failure_before"] = df["attempts_before"] - df["success_before"]

    df = df.drop(columns=["success_cum"])

    return df, student_encoder, skill_encoder


def build_lfa_aux_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    aux_cols: List[str],
    student_col: str = "studentId",
    skill_col: str = "skill",
    time_col: str = "startTime",
    action_col: str = "action_num",
    missing_strategy: str = "zero",
    outlier_cap_std: float = 3.0,
) -> Tuple[pd.DataFrame, Optional[Dict[str, StandardScaler]]]:
    df = df.copy()

    # Ensure chronological order per student
    sort_cols = [student_col, time_col]
    if action_col in df.columns:
        sort_cols.append(action_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 1. Compute lagged hint features if raw 'hintCount' is available
    if "hintCount" in df.columns:
        hints = pd.to_numeric(df["hintCount"], errors="coerce").fillna(0).values
        df["_temp_hints"] = hints

        # Cumulative hints on the specific skill prior to interaction t
        cum_hints_skill = df.groupby([student_col, skill_col])["_temp_hints"].cumsum()
        df["cum_hints_skill_before"] = (cum_hints_skill - df["_temp_hints"]).clip(lower=0)

        # Cumulative hints across all skills prior to interaction t
        cum_hints_total = df.groupby(student_col)["_temp_hints"].cumsum()
        cum_hints_total_before = (cum_hints_total - df["_temp_hints"]).clip(lower=0)

        # Cumulative student total attempts prior to interaction t
        total_attempts_before = df.groupby(student_col).cumcount()
        df["hint_rate_before"] = cum_hints_total_before / (total_attempts_before + 1.0)

        df = df.drop(columns=["_temp_hints"])

    # 2. Compute lagged retry attempts if raw 'attemptCount' is available
    if "attemptCount" in df.columns:
        raw_attempts = pd.to_numeric(df["attemptCount"], errors="coerce").fillna(1).values
        # retries on question = max(0, attemptCount - 1)
        df["_temp_retries"] = np.clip(raw_attempts - 1, 0, None)

        cum_retries_skill = df.groupby([student_col, skill_col])["_temp_retries"].cumsum()
        df["cum_retries_skill_before"] = (cum_retries_skill - df["_temp_retries"]).clip(lower=0)

        df = df.drop(columns=["_temp_retries"])

    scalers: Dict[str, StandardScaler] = {}

    # Determine train mask for fitting scalers without leakage
    if "split" in df.columns:
        train_mask = df["split"] == "train"
    else:
        # Fallback if train_df is separate
        train_student_ids = train_df[student_col].unique() if student_col in train_df.columns else []
        train_mask = df[student_col].isin(train_student_ids)

    for col in aux_cols:
        if col not in df.columns:
            logger.warning("Aux column '%s' not found in dataframe", col)
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        train_series = df.loc[train_mask, col] if train_mask.any() else df[col]
        train_mean = float(train_series.mean()) if not train_series.empty and not pd.isna(train_series.mean()) else 0.0
        train_std = float(train_series.std()) if not train_series.empty and not pd.isna(train_series.std()) else 1.0

        if missing_strategy == "zero":
            df[col] = df[col].fillna(0)
        elif missing_strategy == "mean":
            df[col] = df[col].fillna(train_mean)
        else:
            raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

        if train_std > 0 and outlier_cap_std > 0:
            lower = train_mean - outlier_cap_std * train_std
            upper = train_mean + outlier_cap_std * train_std
            df[col] = df[col].clip(lower, upper)

        scaler = StandardScaler()
        fit_values = df.loc[train_mask, col].values.reshape(-1, 1) if train_mask.any() else df[col].values.reshape(-1, 1)
        scaler.fit(fit_values)
        df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
        scalers[col] = scaler

    return df, scalers if scalers else None


def build_lfa_feature_matrix(
    df: pd.DataFrame,
    student_encoder: LabelEncoderWithUNK,
    skill_encoder: LabelEncoderWithUNK,
    aux_config: Optional[Dict[str, Any]] = None,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    split_col: str = "split",
) -> pd.DataFrame:
    core_cols = ["student_id_encoded", "skill_id_encoded", "success_before", "failure_before", "attempts_before"]

    if not all(c in df.columns for c in core_cols):
        raise ValueError("Core LFA features not found. Run build_lfa_core_features first.")

    feature_df = df[core_cols + [correct_col, split_col]].copy()

    if aux_config and aux_config.get("enabled", False):
        aux_cols = [c for c in aux_config.get("columns", []) if c in df.columns]
        for col in aux_cols:
            feature_df[col] = pd.to_numeric(df[col], errors="coerce")

    return feature_df

