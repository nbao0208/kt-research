from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/baseline_skill_average")
def main(cfg: DictConfig) -> None:
    if "experiment" in cfg:
        cfg = cfg.experiment

    logger.info("Starting skill average baseline with config:\n%s", OmegaConf.to_yaml(cfg))

    from src.utils.io import get_latest_data_version, save_json
    raw_proc_dir = Path(cfg.data.processed_dir)
    processed_dir = get_latest_data_version(raw_proc_dir)
    data_path = processed_dir / "interactions.parquet"

    if not data_path.exists():
        logger.warning("Processed data not found at %s. Baseline will not run.", data_path)
        logger.info("Run 'python scripts/preprocess_assistments.py' first to create processed data.")
        return

    import pandas as pd

    from src.models.baselines import evaluate_baselines

    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    results = evaluate_baselines(train_df, val_df, test_df)

    outputs_dir = Path("outputs/metrics")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, outputs_dir / "baseline_metrics.json")

    logger.info("Baseline results:\n%s", OmegaConf.to_yaml(OmegaConf.create(results)))

    for name, metrics in results.items():
        logger.info(
            "%s: val_auc=%.4f test_auc=%.4f",
            name,
            metrics["val"]["auc"],
            metrics["test"]["auc"],
        )


if __name__ == "__main__":
    main()
