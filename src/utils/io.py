import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def resolve_data_version(processed_dir: Path) -> Path:
    if processed_dir.name.startswith("v") and processed_dir.name[1:].isdigit():
        return processed_dir

    if processed_dir.exists():
        existing = sorted(
            [
                d
                for d in processed_dir.iterdir()
                if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
            ],
            key=lambda x: int(x.name[1:]),
        )
        if existing:
            next_version = int(existing[-1].name[1:]) + 1
        else:
            next_version = 1
    else:
        next_version = 1
    return processed_dir / f"v{next_version}"


def get_latest_data_version(processed_dir: Path) -> Path:
    if processed_dir.name.startswith("v") and processed_dir.name[1:].isdigit():
        return processed_dir

    if processed_dir.exists():
        existing = sorted(
            [
                d
                for d in processed_dir.iterdir()
                if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
            ],
            key=lambda x: int(x.name[1:]),
        )
        if existing:
            return existing[-1]
    return processed_dir
