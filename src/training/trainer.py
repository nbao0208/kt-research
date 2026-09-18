import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.early_stopping import EarlyStopping
from src.utils.logging import WandbLogger

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device("cpu"),
        max_epochs: int = 50,
        patience: int = 5,
        wandb_logger: Optional[WandbLogger] = None,
        checkpoint_dir: Optional[Path] = None,
        log_every: int = 10,
        eval_every: int = 1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.wandb_logger = wandb_logger
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every
        self.eval_every = eval_every
        self.best_state_dict = None
        self.best_metric = 0.0

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(*inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def eval_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
                labels_np = labels.numpy()
                labels = labels.to(self.device)
                outputs = self.model(*inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_labels.extend(labels_np)
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        return {"loss": total_loss / len(val_loader), "auc": auc}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metric_name: str = "auc",
    ) -> Dict[str, Any]:
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)

            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_auc=%.4f",
                epoch,
                self.max_epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["auc"],
            )

            if self.wandb_logger:
                self.wandb_logger.log_metrics(
                    {
                        "train/loss": train_loss,
                        "val/loss": val_metrics["loss"],
                        "val/auc": val_metrics["auc"],
                        "epoch": epoch,
                    }
                )

            current_metric = val_metrics.get(metric_name, 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_state_dict = self.model.state_dict()
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch)

            if self.early_stopping(current_metric, epoch):
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {"best_val_metric": self.best_metric, "epochs_trained": epoch}

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(), "best_metric": self.best_metric},
            path,
        )
