from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.eda import load_raw
from src.data.preprocess import clean_assistments, save_processed
from src.utils.io import resolve_data_version
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="data/assistments2017")
def main(cfg: DictConfig) -> None:
    logger.info("Starting preprocessing with config:\n%s", OmegaConf.to_yaml(cfg))

    data_cfg = cfg.data if "data" in cfg else cfg

    raw_path = Path(data_cfg.raw_path)
    if not raw_path.exists():
        logger.warning("Raw data not found at %s. Preprocessing will not run.", raw_path)
        return

    logger.info("Loading raw data from %s", raw_path)
    df = load_raw(
        raw_path,
        usecols=list(data_cfg.columns.required) + list(data_cfg.columns.recommended),
    )
    logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))

    logger.info("Running preprocessing pipeline...")
    recommended_cols = list(data_cfg.columns.get("recommended", []))
    result = clean_assistments(
        df,
        required_cols=list(data_cfg.columns.required),
        recommended_cols=recommended_cols,
        multi_skill_strategy=data_cfg.skill_handling.multi_skill_strategy,
        noskill_strategy=data_cfg.skill_handling.get("noskill_strategy", "drop"),
        min_student_interactions=data_cfg.student_handling.min_student_interactions,
        min_skill_interactions=data_cfg.skill_handling.min_skill_interactions,
        min_sequence_length=data_cfg.sequence_filter.min_sequence_length,
        split_config={
            "protocol": data_cfg.split.protocol,
            "train_frac": data_cfg.split.train_frac,
            "val_frac": data_cfg.split.val_frac,
            "test_frac": data_cfg.split.test_frac,
        },
        raw_path=str(raw_path),
    )

    processed_dir = resolve_data_version(Path(data_cfg.processed_dir))
    logger.info("Saving processed data to %s", processed_dir)

    save_processed(result, processed_dir)

    df_out = result["df"]
    train_n = (df_out["split"] == "train").sum()
    val_n = (df_out["split"] == "val").sum()
    test_n = (df_out["split"] == "test").sum()
    logger.info(
        "Preprocessing complete: train=%d val=%d test=%d (total=%d)",
        train_n,
        val_n,
        test_n,
        len(df_out),
    )


if __name__ == "__main__":
    main()
