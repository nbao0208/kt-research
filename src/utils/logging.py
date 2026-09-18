import logging
from pathlib import Path
import sys
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class WandbLogger:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.run = None

    def init(self, config: dict, project: str = "kt-research", **kwargs: Any) -> None:
        if not self.enabled:
            return
        import wandb

        self.run = wandb.init(project=project, config=config, **kwargs)

    def log_config(self, cfg: DictConfig) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def log_summary(self, summary_dict: dict) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        if hasattr(wandb, "run") and wandb.run is not None:
            wandb.run.summary.update(summary_dict)

    def log_artifact(self, local_path: str, artifact_type: str, name: str) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        path_obj = Path(local_path)
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if path_obj.is_dir():
            artifact.add_dir(str(path_obj))
        else:
            artifact.add_file(str(path_obj))
        self.run.log_artifact(artifact)

    def log_artifacts_from_dir(self, directory: Path, artifact_type: str, name: str) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        if directory.exists():
            artifact.add_dir(str(directory))
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.finish()

