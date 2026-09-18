import re
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir, save_json

matplotlib.use("Agg")

# Setup style chung cho đẹp và đồng bộ
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

REQUIRED_COLUMNS = [
    "studentId",
    "skill",
    "problemId",
    "correct",
    "startTime",
    "action_num",
    "timeTaken",
    "hintCount",
    "attemptCount",
    "scaffold",
    "hintTotal",
]


def clean_column_name(col: str) -> str:
    col = col.strip()
    col = re.sub(r"[^0-9a-zA-Z_]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col


def load_raw(path: Path, usecols: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    if usecols is None:
        usecols = REQUIRED_COLUMNS
    return pd.read_csv(path, usecols=usecols, low_memory=False, nrows=nrows)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def _check_time_ordering(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values(["studentId", "startTime", "action_num"])
    shifted = df_sorted.groupby("studentId").shift(1)
    negative = (df_sorted["startTime"] < shifted["startTime"]).sum()
    return {"negative_time_differences": int(negative)}


def compute_eda_report(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> dict:
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS

    df = clean_column_names(df.copy())
    n_rows = len(df)

    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_required = sum(missing.get(col, 0) for col in required_cols if col in df.columns)

    correct = pd.to_numeric(df["correct"], errors="coerce")
    valid = correct.isin([0, 1])
    n_valid = int(valid.sum())
    correct_rate = float(correct[valid].mean()) if n_valid else 0.0

    student_counts = df["studentId"].value_counts()
    skill_counts = df["skill"].astype(str).value_counts()

    multi_skill = int(df["skill"].astype(str).str.contains(",").sum())

    duplicates = df.duplicated(subset=["studentId", "problemId", "startTime", "action_num"])
    n_duplicates = int(duplicates.sum())

    time_ordering = _check_time_ordering(df)

    time_taken = pd.to_numeric(df["timeTaken"], errors="coerce")
    hints = pd.to_numeric(df["hintCount"], errors="coerce")

    report = {
        "n_rows": n_rows,
        "n_students": int(df["studentId"].nunique()),
        "n_skills": int(df["skill"].nunique()),
        "correct_rate": round(correct_rate, 6),
        "missing_required_rows": missing_required,
        "duplicate_rows": n_duplicates,
        "multi_skill_rows": multi_skill,
        "students_lt_5_interactions": int((student_counts < 5).sum()),
        "students_lt_10_interactions": int((student_counts < 10).sum()),
        "students_lt_20_interactions": int((student_counts < 20).sum()),
        "skills_lt_20_interactions": int((skill_counts < 20).sum()),
        "skills_lt_50_interactions": int((skill_counts < 50).sum()),
        "skills_lt_100_interactions": int((skill_counts < 100).sum()),
        **time_ordering,
        "student_distribution": {
            "min": float(student_counts.min()),
            "max": float(student_counts.max()),
            "mean": float(student_counts.mean()),
            "median": float(student_counts.median()),
        },
        "skill_distribution": {
            "min": float(skill_counts.min()),
            "max": float(skill_counts.max()),
            "mean": float(skill_counts.mean()),
            "median": float(skill_counts.median()),
        },
        "time_taken": {
            "mean": float(time_taken.mean()) if not time_taken.isna().all() else None,
            "median": float(time_taken.median()) if not time_taken.isna().all() else None,
            "max": float(time_taken.max()) if not time_taken.isna().all() else None,
            "missing": int(time_taken.isna().sum()),
        },
        "hint_count": {
            "mean": float(hints.mean()) if not hints.isna().all() else None,
            "max": float(hints.max()) if not hints.isna().all() else None,
            "missing": int(hints.isna().sum()),
        },
        "skill_correct_rate": (
            df.groupby("skill")["correct"].mean().round(6).to_dict() if "correct" in df.columns else {}
        ),
        "student_correct_rate": (
            df.groupby("studentId")["correct"].mean().round(6).to_dict() if "correct" in df.columns else {}
        ),
    }
    return report


def save_eda_report(report: dict, output_dir: Path) -> Path:
    out_path = ensure_dir(output_dir) / "eda_assistments.json"
    save_json(report, out_path)
    return out_path


def save_eda_figures(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    fig_dir = ensure_dir(output_dir)
    saved: List[Path] = []
    df = clean_column_names(df.copy())

    # ========================================================
    # Figure 1: Distribution of interactions per student
    # ========================================================
    student_counts = df["studentId"].value_counts()
    # [FIX] Chuyển sang numpy array 1D để tránh việc seaborn vẽ nhầm theo index của pandas Series
    counts_arr = student_counts.to_numpy()
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Dùng trực tiếp matplotlib hist cho numpy array để ổn định nhất với log scale
    ax.hist(counts_arr, bins=50, log=True, color="#4C72B0", alpha=0.7, edgecolor="white")
    
    median_val = float(np.median(counts_arr))
    mean_val = float(np.mean(counts_arr))
    ax.axvline(median_val, color="red", linestyle="--", linewidth=2, label=f"Median: {median_val:.0f}")
    ax.axvline(mean_val, color="orange", linestyle="-", linewidth=2, label=f"Mean: {mean_val:.1f}")
    
    ax.set_title("Distribution of Interactions per Student", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Interactions", fontsize=12)
    ax.set_ylabel("Number of Students (log scale)", fontsize=12)
    ax.legend(frameon=True, shadow=True)
    
    p1 = fig_dir / "01_student_interactions_dist.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(p1)

    # ========================================================
    # Figure 2: Top 50 skills (Horizontal Bar Chart)
    # ========================================================
    skill_counts = df["skill"].astype(str).value_counts().head(50)
    skill_counts_sorted = skill_counts.sort_values(ascending=True)
    
    # [FIX] Tách index và values thành List và Numpy array thuần túy
    y_labels = skill_counts_sorted.index.tolist()
    widths = skill_counts_sorted.to_numpy()
    y_pos = np.arange(len(y_labels))
    
    fig, ax = plt.subplots(figsize=(11, 12))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(widths)))
    bars = ax.barh(y_pos, widths, color=colors, edgecolor="white")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    
    max_width = float(np.max(widths)) if len(widths) > 0 else 0
    for bar, count in zip(bars, widths):
        ax.text(bar.get_width() + max_width * 0.01, bar.get_y() + bar.get_height() / 2, 
                f"{int(count):,}", va="center", fontsize=9, color="#333333")
    
    ax.set_title("Top 50 Skills by Interaction Count", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Interactions", fontsize=12)
    ax.set_ylabel("Skill Name", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.margins(x=0.15)
    
    p2 = fig_dir / "02_top_skills_dist.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2)

    # ========================================================
    # Figure 3: Distribution of time taken
    # ========================================================
    time_taken = pd.to_numeric(df["timeTaken"], errors="coerce").dropna()
    p99 = float(np.percentile(time_taken, 99))
    # [FIX] Convert sang numpy array
    time_clipped = time_taken.clip(upper=p99).to_numpy()
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Chỉ định rõ x=time_clipped để bypass lỗi type checker của Pylance
    sns.histplot(x=time_clipped, bins=60, kde=True, color="#55A868", ax=ax, 
                 stat="density", alpha=0.7, line_kws={"linewidth": 2})
    
    median_t = float(np.median(time_taken.to_numpy()))
    mean_t = float(np.mean(time_taken.to_numpy()))
    ax.axvline(median_t, color="red", linestyle="--", linewidth=2, label=f"Median: {median_t:.1f}s")
    ax.axvline(mean_t, color="darkorange", linestyle="-", linewidth=2, label=f"Mean: {mean_t:.1f}s")
    ax.axvline(p99, color="purple", linestyle=":", linewidth=2, label=f"99th percentile: {p99:.1f}s")
    
    ax.set_title("Distribution of Time Taken per Interaction", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds) - clipped at 99th percentile", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=True, shadow=True)
    
    p3 = fig_dir / "03_time_taken_dist.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(p3)

    # ========================================================
    # Figure 4: Skill difficulty vs volume (Scatter plot)
    # ========================================================
    skill_stats = df.groupby("skill").agg(
        volume=("correct", "size"),
        correct_rate=("correct", "mean")
    ).reset_index()
    
    # [FIX] Convert columns sang numpy array
    volumes = skill_stats["volume"].to_numpy()
    correct_rates = skill_stats["correct_rate"].to_numpy()
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    ax.scatter(
        volumes, 
        correct_rates,
        alpha=0.5, 
        s=40, 
        c="#4C72B0",
        edgecolor="white",
        linewidth=0.5
    )
    
    global_median = float(np.median(correct_rates))
    ax.axhline(global_median, color="red", linestyle="--", linewidth=1.5, 
               label=f"Global Median Correct Rate: {global_median:.3f}")
    
    ax.set_xscale("log")
    ax.set_title("Skill Difficulty vs. Volume", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Skill Volume (log scale)", fontsize=12)
    ax.set_ylabel("Correct Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    
    p4 = fig_dir / "04_skill_difficulty_scatter.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(p4)

    return saved