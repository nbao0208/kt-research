import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def temporal_per_student_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_sequence_length: int = 3,
    student_col: str = "studentId",
) -> pd.DataFrame:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    df = df.copy()
    df["split"] = "train"

    for student in df[student_col].unique():
        mask = df[student_col] == student
        indices = df[mask].index
        n = len(indices)

        if n < min_sequence_length:
            df.loc[indices, "split"] = "train"
            continue

        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        if train_end > 0:
            df.loc[indices[:train_end], "split"] = "train"
        if val_end > train_end:
            df.loc[indices[train_end:val_end], "split"] = "val"
        if val_end < n:
            df.loc[indices[val_end:], "split"] = "test"

    train_count = (df["split"] == "train").sum()
    val_count = (df["split"] == "val").sum()
    test_count = (df["split"] == "test").sum()
    logger.info("Split: train=%d val=%d test=%d (total=%d)", train_count, val_count, test_count, len(df))

    return df
