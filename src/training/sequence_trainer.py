import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.sequence_builder import SequenceBatch
from src.evaluation.metrics import compute_metrics
from src.training.early_stopping import EarlyStopping
from src.utils.logging import WandbLogger

logger = logging.getLogger(__name__)


class SequenceTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        gradient_clip_norm: float = 1.0,
        seed: int = 42,
        wandb_logger: Optional[WandbLogger] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.gradient_clip_norm = gradient_clip_norm
        self.seed = seed
        self.wandb_logger = wandb_logger
        self.collate_fn = collate_fn
        self.best_val_auc = 0.0
        self.best_state_dict = None
        self.epochs_since_improvement = 0

    def _build_loader(self, sequences, shuffle: bool) -> DataLoader:
        dataset = SequenceDatasetWrapper(sequences)
        use_pin_memory = self.device.type in ["cuda", "mps"]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            pin_memory=use_pin_memory,
        )

    def _compute_loss(
        self,
        batch: SequenceBatch,
    ) -> torch.Tensor:
        batch = batch.to(self.device)

        logits, _, _ = self.model(
            skill_ids=batch.history_skill_ids,
            corrects=batch.history_corrects,
            target_skill_ids=batch.target_skill_ids,
            mask=batch.mask,
        )

        loss_per_position = self.criterion(logits, batch.target_corrects)

        train_mask = (batch.split_labels == 0) & batch.mask
        masked_loss = loss_per_position * train_mask.float()

        n_valid = train_mask.sum()
        if n_valid > 0:
            return masked_loss.sum() / n_valid
        return masked_loss.sum()

    def _train_epoch(self, loader: DataLoader, epoch: int = 0) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/{self.max_epochs} [Train]", leave=False, unit="batch")
        for batch in pbar:
            loss = self._compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total_loss / len(loader)

    def _eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        pbar = tqdm(loader, desc="  Evaluating [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)

                logits, _, _ = self.model(
                    skill_ids=batch.history_skill_ids,
                    corrects=batch.history_corrects,
                    target_skill_ids=batch.target_skill_ids,
                    mask=batch.mask,
                )

                loss_per_position = self.criterion(logits, batch.target_corrects)
                eval_mask = (batch.split_labels == 1) & batch.mask
                masked_loss = loss_per_position * eval_mask.float()
                n_valid = eval_mask.sum()
                if n_valid > 0:
                    total_loss += masked_loss.sum().item()

                valid_positions = eval_mask
                all_labels.extend(batch.target_corrects[valid_positions].cpu().numpy())
                all_preds.extend(torch.sigmoid(logits[valid_positions]).cpu().numpy())

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds)) if all_labels else {"auc": 0.0, "logloss": 0.0}
        return {"loss": avg_loss, "auc": metrics.get("auc", 0.0)}

    def fit(self, train_sequences, val_sequences) -> Dict[str, Any]:
        train_loader = self._build_loader(train_sequences, shuffle=True)
        val_loader = self._build_loader(val_sequences, shuffle=False)

        logger.info(
            "Starting sequence model training on device '%s' (%d train batches, %d val batches)...",
            self.device, len(train_loader), len(val_loader),
        )

        early_stopping = EarlyStopping(patience=self.patience, mode="max")

        pbar = tqdm(range(self.max_epochs), desc=f"Training [{self.device.type.upper()}]", unit="epoch")
        for epoch in pbar:
            train_loss = self._train_epoch(train_loader, epoch=epoch)
            val_metrics = self._eval_epoch(val_loader)
            val_auc = val_metrics["auc"]

            pbar.set_postfix({"loss": f"{train_loss:.4f}", "val_auc": f"{val_auc:.4f}"})

            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_auc=%.4f",
                epoch + 1, self.max_epochs, train_loss, val_metrics["loss"], val_auc,
            )

            if self.wandb_logger and self.wandb_logger.enabled:
                self.wandb_logger.log_metrics({
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/auc": val_auc,
                    "epoch": epoch + 1,
                }, step=epoch + 1)

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_state_dict = self.model.state_dict()
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

            if early_stopping(val_auc, epoch + 1):
                logger.info("Early stopping triggered at epoch %d/%d", epoch + 1, self.max_epochs)
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        if self.wandb_logger and self.wandb_logger.enabled:
            self.wandb_logger.log_summary({
                "best_val_auc": self.best_val_auc,
                "epochs_trained": epoch + 1,
            })

        return {"best_val_auc": self.best_val_auc, "epochs_trained": epoch + 1}

    def predict(self, sequences) -> np.ndarray:
        self.model.eval()
        loader = self._build_loader(sequences, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits, _, _ = self.model(
                    skill_ids=batch.history_skill_ids,
                    corrects=batch.history_corrects,
                    target_skill_ids=batch.target_skill_ids,
                    mask=batch.mask,
                )
                preds = torch.sigmoid(logits)
                for i in range(batch.mask.size(0)):
                    row_mask = batch.mask[i]
                    if row_mask.any():
                        all_preds.extend(preds[i][row_mask].cpu().numpy())

        return np.array(all_preds)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "num_skills": getattr(self.model, "num_skills", None),
            "embedding_dim": getattr(self.model, "embedding_dim", None),
        }
        if hasattr(self.model, "hidden_size"):
            config_dict["hidden_size"] = self.model.hidden_size
        if hasattr(self.model, "num_heads"):
            config_dict["num_heads"] = self.model.num_heads
        if hasattr(self.model, "num_layers"):
            config_dict["num_layers"] = self.model.num_layers

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_val_auc": self.best_val_auc,
            "config": config_dict,
        }, path)


class SequenceDatasetWrapper(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]