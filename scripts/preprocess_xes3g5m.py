import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def preprocess_and_validate_xes3g5m(raw_dir: Path, output_dir: Path) -> dict:
    kc_level_dir = raw_dir / "kc_level"
    metadata_dir = raw_dir / "metadata"

    train_valid_file = kc_level_dir / "train_valid_sequences.csv"
    test_file = kc_level_dir / "test.csv"
    questions_file = metadata_dir / "questions.json"
    kc_map_file = metadata_dir / "kc_routes_map.json"

    logger.info("Checking XES3G5M dataset files in: %s", raw_dir)
    for p in [train_valid_file, test_file, questions_file, kc_map_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing essential XES3G5M file: {p}")
        logger.info("Found file: %s (%d bytes)", p.name, p.stat().st_size)

    # 1. Inspect questions metadata
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    n_questions = len(questions)

    # 2. Inspect KC routes map
    with open(kc_map_file, "r", encoding="utf-8") as f:
        kc_map = json.load(f)
    n_kcs = len(kc_map)

    # 3. Inspect train_valid sequences
    logger.info("Validating train_valid_sequences.csv...")
    df_train_valid = pd.read_csv(train_valid_file, usecols=["fold", "uid", "questions", "selectmasks"])
    n_train_valid_seqs = len(df_train_valid)
    fold_counts = df_train_valid["fold"].value_counts().sort_index().to_dict()

    # Verify zero leakage across folds
    uids_by_fold = [set(df_train_valid[df_train_valid["fold"] == f]["uid"]) for f in sorted(df_train_valid["fold"].unique())]
    leakage_count = 0
    for i in range(len(uids_by_fold)):
        for j in range(i + 1, len(uids_by_fold)):
            leakage_count += len(uids_by_fold[i].intersection(uids_by_fold[j]))

    if leakage_count != 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {leakage_count} UIDs overlap between folds!")
    logger.info("Verified: 0 UID overlap across 5 training folds (student-level split).")

    # 4. Inspect test sequences
    logger.info("Validating test.csv...")
    df_test = pd.read_csv(test_file, usecols=["fold", "uid"])
    n_test_seqs = len(df_test)
    train_uids = set(df_train_valid["uid"])
    test_uids = set(df_test["uid"])
    test_leakage = len(train_uids.intersection(test_uids))
    if test_leakage != 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {test_leakage} UIDs overlap between train and test!")
    logger.info("Verified: 0 UID overlap between train_valid and test.csv.")

    summary = {
        "dataset_name": "XES3G5M",
        "num_questions": n_questions,
        "num_kcs": n_kcs,
        "train_valid_sequences": n_train_valid_seqs,
        "test_sequences": n_test_seqs,
        "total_students": len(train_uids) + len(test_uids),
        "train_students": len(train_uids),
        "test_students": len(test_uids),
        "fold_distribution": fold_counts,
        "zero_uid_leakage_verified": True,
        "raw_dir": str(raw_dir),
    }

    ensure_dir(output_dir)
    save_json(summary, output_dir / "dataset_summary.json")
    logger.info("Saved dataset summary to: %s", output_dir / "dataset_summary.json")
    return summary


def main():
    raw_dir = Path("data/raw/XES3G5M")
    output_dir = Path("data/processed/xes3g5m")

    if not raw_dir.exists():
        logger.error("Raw directory %s does not exist. Please check dataset location.", raw_dir)
        sys.exit(1)

    summary = preprocess_and_validate_xes3g5m(raw_dir, output_dir)
    print("\n--- XES3G5M Preprocessing & Validation Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()
