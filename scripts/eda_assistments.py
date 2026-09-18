from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.eda import compute_eda_report, load_raw, save_eda_figures, save_eda_report
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="data/assistments2017")
def main(cfg: DictConfig) -> None:
    logger.info("Starting EDA with config:\n%s", OmegaConf.to_yaml(cfg))

    raw_path = Path(cfg.data.raw_path)
    if not raw_path.exists():
        logger.warning("Raw data not found at %s. EDA will not run.", raw_path)
        return

    logger.info("Loading raw data from %s", raw_path)
    df = load_raw(raw_path)

    logger.info("Computing EDA report...")
    report = compute_eda_report(df)

    outputs_dir = Path("outputs")
    report_path = save_eda_report(report, outputs_dir / "reports")
    logger.info("EDA report saved to %s", report_path)

    fig_paths = save_eda_figures(df, outputs_dir / "figures/eda")
    logger.info("Saved %d EDA figures", len(fig_paths))


if __name__ == "__main__":
    main()
