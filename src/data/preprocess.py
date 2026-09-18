import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.eda import clean_column_names
from src.data.splits import temporal_per_student_split
from src.utils.io import ensure_dir, get_git_commit, save_json, save_parquet

logger = logging.getLogger(__name__)


def _clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype("int8")
    return df


def _handle_multi_skill(df: pd.DataFrame, strategy: str = "first") -> pd.DataFrame:
    df = df.copy()
    df["skill"] = df["skill"].astype(str).str.strip()
    if strategy == "first":
        df["skill"] = df["skill"].str.split(",").str[0].str.strip()
    elif strategy == "explode":
        df["skill"] = df["skill"].str.split(",")
        df = df.explode("skill")
        df["skill"] = df["skill"].str.strip()
    else:
        raise ValueError(f"Unknown multi_skill_strategy: {strategy}")
    return df


def _handle_noskill(df: pd.DataFrame, strategy: str = "drop") -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    before = len(df)
    skill_str = df["skill"].astype(str).str.strip().str.lower()
    noskill_mask = skill_str == "noskill"

    if strategy == "drop":
        df = df[~noskill_mask].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Dropped %d 'noskill' rows", dropped)
        return df, dropped
    elif strategy == "problem_id":
        if "problemId" in df.columns:
            df.loc[noskill_mask, "skill"] = "problem_" + df.loc[noskill_mask, "problemId"].astype(str)
            logger.info("Replaced %d 'noskill' rows with problemId", noskill_mask.sum())
        return df, 0
    elif strategy == "keep":
        return df, 0
    else:
        raise ValueError(f"Unknown noskill_strategy: {strategy}")



def _sort_interactions(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["studentId", "startTime"]
    if "action_num" in df.columns:
        sort_cols.append("action_num")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    subset = ["studentId", "problemId", "startTime"]
    if "action_num" in df.columns:
        subset.append("action_num")
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(df)
    if before - after > 0:
        logger.info("Removed %d duplicate rows", before - after)
    return df


def _filter_rare_students(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    before = df["studentId"].nunique()
    counts = df["studentId"].value_counts()
    valid = counts[counts >= min_interactions].index
    df = df[df["studentId"].isin(valid)].reset_index(drop=True)
    after = df["studentId"].nunique()
    logger.info("Filtered students: %d -> %d (min interactions: %d)", before, after, min_interactions)
    return df


def _filter_rare_skills(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    before = df["skill"].nunique()
    counts = df["skill"].value_counts()
    valid = counts[counts >= min_interactions].index
    df = df[df["skill"].isin(valid)].reset_index(drop=True)
    after = df["skill"].nunique()
    logger.info("Filtered skills: %d -> %d (min interactions: %d)", before, after, min_interactions)
    return df


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def clean_assistments(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    recommended_cols: Optional[List[str]] = None,
    multi_skill_strategy: str = "first",
    noskill_strategy: str = "drop",
    min_student_interactions: int = 10,
    min_skill_interactions: int = 50,
    min_sequence_length: int = 3,
    split_config: Optional[Dict[str, Any]] = None,
    raw_path: Optional[str] = None,
) -> Dict[str, Any]:
    if required_cols is None:
        required_cols = ["studentId", "skill", "problemId", "correct", "startTime", "action_num"]
    if recommended_cols is None:
        recommended_cols = []

    all_target_cols = list(dict.fromkeys(required_cols + recommended_cols))

    tracking = {"n_rows_raw": len(df)}

    df = clean_column_names(df)
    tracking["n_cols_initial"] = len(df.columns)

    available = [c for c in all_target_cols if c in df.columns]
    df = df[available].copy()
    tracking["n_cols_selected"] = len(df.columns)

    tracking["n_students_raw"] = int(df["studentId"].nunique()) if "studentId" in df.columns else 0
    tracking["n_skills_raw"] = int(df["skill"].nunique()) if "skill" in df.columns else 0

    before = len(df)
    subset_drop = [c for c in ["studentId", "skill", "correct", "startTime"] if c in df.columns]
    df = df.dropna(subset=subset_drop)
    after = len(df)
    tracking["n_rows_missing_dropped"] = before - after

    df = _clean_target(df)
    tracking["n_rows_after_target_clean"] = len(df)

    df = _handle_multi_skill(df, strategy=multi_skill_strategy)
    tracking["n_rows_after_multi_skill"] = len(df)

    df, dropped_noskill = _handle_noskill(df, strategy=noskill_strategy)
    tracking["n_rows_dropped_noskill"] = dropped_noskill
    tracking["n_rows_after_noskill"] = len(df)

    df = _sort_interactions(df)

    df = _remove_duplicates(df)
    tracking["n_rows_after_dedup"] = len(df)

    df = _filter_rare_students(df, min_interactions=min_student_interactions)
    tracking["n_rows_after_student_filter"] = len(df)

    df = _filter_rare_skills(df, min_interactions=min_skill_interactions)
    tracking["n_rows_final"] = len(df)

    df = _downcast_dtypes(df)

    df = temporal_per_student_split(
        df,
        train_frac=split_config["train_frac"],
        val_frac=split_config["val_frac"],
        test_frac=split_config["test_frac"],
        min_sequence_length=min_sequence_length,
    )
    tracking["split_counts"] = df["split"].value_counts().to_dict()

    df["split"] = df["split"].astype("category")

    metadata = {
        "raw_path": str(raw_path) if raw_path else None,
        "n_rows_raw": tracking["n_rows_raw"],
        "n_rows_final": len(df),
        "n_students_final": int(df["studentId"].nunique()),
        "n_skills_final": int(df["skill"].nunique()),
        "multi_skill_strategy": multi_skill_strategy,
        "noskill_strategy": noskill_strategy,
        "min_student_interactions": min_student_interactions,
        "min_skill_interactions": min_skill_interactions,
        "min_sequence_length": min_sequence_length,
        "split_protocol": split_config["protocol"],
        "train_frac": split_config["train_frac"],
        "val_frac": split_config["val_frac"],
        "test_frac": split_config["test_frac"],
        "git_commit": get_git_commit(),
        "created_at": pd.Timestamp.now().isoformat(),
        "tracking": tracking,
    }

    return {"df": df, "metadata": metadata}


def save_processed(result: Dict[str, Any], output_dir: Path) -> None:
    out = ensure_dir(output_dir)
    df = result["df"]
    metadata = result["metadata"]

    save_parquet(df, out / "interactions.parquet")
    logger.info("Saved interactions.parquet to %s", out)

    save_json(metadata, out / "metadata.json")
    logger.info("Saved metadata.json to %s", out)

    split_info = {k: v for k, v in metadata.items() if k.startswith("split") or k.startswith("n_")}
    save_json(split_info, out / "split_info.json")
    logger.info("Saved split_info.json to %s", out)

    skill_map = dict(enumerate(df["skill"].astype("category").cat.categories))
    save_json(skill_map, out / "skill_map.json")
    logger.info("Saved skill_map.json to %s", out)

    student_map = dict(enumerate(df["studentId"].astype("category").cat.categories.astype(str)))
    save_json(student_map, out / "student_map.json")
    logger.info("Saved student_map.json to %s", out)
