import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.logging import WandbLogger

logger = logging.getLogger(__name__)


class LFACore(nn.Module):
    def __init__(
        self,
        n_students: int,
        n_skills: int,
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_students = n_students
        self.n_skills = n_skills
        self.embedding_dim = embedding_dim

        self.student_embedding = nn.Embedding(n_students + 1, embedding_dim, padding_idx=0)
        self.skill_embedding = nn.Embedding(n_skills + 1, embedding_dim, padding_idx=0)
        self.student_bias = nn.Embedding(n_students + 1, 1, padding_idx=0)
        self.skill_bias = nn.Embedding(n_skills + 1, 1, padding_idx=0)

        self.history_fc = nn.Linear(3, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        student_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        success_before: torch.Tensor,
        failure_before: torch.Tensor,
        attempts_before: torch.Tensor,
    ) -> torch.Tensor:
        s_ids = student_ids.clamp(0, self.n_students)
        sk_ids = skill_ids.clamp(0, self.n_skills)

        s_emb = self.student_embedding(s_ids)
        sk_emb = self.skill_embedding(sk_ids)

        interaction = (s_emb * sk_emb).sum(dim=1, keepdim=True)
        s_bias = self.student_bias(s_ids)
        sk_bias = self.skill_bias(sk_ids)

        history = torch.stack([success_before, failure_before, attempts_before], dim=1)
        history_score = self.history_fc(self.dropout(history))

        logit = s_bias + sk_bias + interaction + history_score
        return logit.squeeze(-1)


class LFAAux(LFACore):
    def __init__(
        self,
        n_students: int,
        n_skills: int,
        n_aux_features: int = 0,
        embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__(n_students, n_skills, embedding_dim, dropout)
        self.n_aux_features = n_aux_features
        if n_aux_features > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(n_aux_features, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
            )

    def forward(
        self,
        student_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        success_before: torch.Tensor,
        failure_before: torch.Tensor,
        attempts_before: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        core_logit = super().forward(student_ids, skill_ids, success_before, failure_before, attempts_before)

        if aux_features is not None and self.n_aux_features > 0:
            aux_score = self.aux_mlp(aux_features).squeeze(-1)
            return core_logit + aux_score

        return core_logit


class LFATrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        max_epochs: int = 50,
        early_stopping_patience: int = 5,
        seed: int = 42,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        self.model = model.to(device)
        self.device = device
        logger.info("LFATrainer initialized on target device: %s", self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.best_val_auc = 0.0
        self.best_state_dict = None
        self.epochs_no_improve = 0
        self.seed = seed
        self.wandb_logger = wandb_logger

    def _build_loader(self, feature_df, shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
        bs = batch_size if batch_size is not None else self.batch_size
        student_ids = torch.clamp(torch.tensor(feature_df["student_id_encoded"].to_numpy(), dtype=torch.long), min=0)
        skill_ids = torch.clamp(torch.tensor(feature_df["skill_id_encoded"].to_numpy(), dtype=torch.long), min=0)
        success_before = torch.tensor(feature_df["success_before"].to_numpy(), dtype=torch.float32)
        failure_before = torch.tensor(feature_df["failure_before"].to_numpy(), dtype=torch.float32)
        attempts_before = torch.tensor(feature_df["attempts_before"].to_numpy(), dtype=torch.float32)
        labels = torch.tensor(feature_df["correct"].to_numpy(), dtype=torch.float32)

        aux_cols = [
            c
            for c in feature_df.columns
            if c
            not in [
                "student_id_encoded",
                "skill_id_encoded",
                "success_before",
                "failure_before",
                "attempts_before",
                "correct",
                "split",
            ]
        ]
        aux = None
        if aux_cols:
            aux = torch.tensor(feature_df[aux_cols].to_numpy(), dtype=torch.float32)

        dataset = TensorDataset(student_ids, skill_ids, success_before, failure_before, attempts_before, labels)
        if aux is not None:
            dataset = TensorDataset(
                student_ids,
                skill_ids,
                success_before,
                failure_before,
                attempts_before,
                labels,
                aux,
            )

        use_pin_memory = self.device.type == "cuda"
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=use_pin_memory)

    def _train_epoch(self, loader: DataLoader, epoch: int = 0) -> float:
        self.model.train()
        total_loss = 0.0
        batch_pbar = tqdm(
            loader,
            desc=f"  Epoch {epoch + 1}/{self.max_epochs} [Train]",
            leave=False,
            unit="batch",
        )
        for batch in batch_pbar:
            if len(batch) == 7:
                student_ids, skill_ids, sb, fb, ab, labels, aux = batch
            else:
                student_ids, skill_ids, sb, fb, ab, labels = batch
                aux = None

            student_ids = student_ids.to(self.device)
            skill_ids = skill_ids.to(self.device)
            sb = sb.to(self.device)
            fb = fb.to(self.device)
            ab = ab.to(self.device)
            labels = labels.to(self.device)
            aux = aux.to(self.device) if aux is not None else None

            self.optimizer.zero_grad()

            if isinstance(self.model, LFAAux):
                logits = self.model(student_ids, skill_ids, sb, fb, ab, aux)
            else:
                logits = self.model(student_ids, skill_ids, sb, fb, ab)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            current_avg_loss = total_loss / (batch_pbar.n + 1)
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{current_avg_loss:.4f}"})

        return total_loss / len(loader)

    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        val_pbar = tqdm(loader, desc="  Evaluating [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                if len(batch) == 7:
                    student_ids, skill_ids, sb, fb, ab, labels, aux = batch
                else:
                    student_ids, skill_ids, sb, fb, ab, labels = batch
                    aux = None

                student_ids = student_ids.to(self.device)
                skill_ids = skill_ids.to(self.device)
                sb = sb.to(self.device)
                fb = fb.to(self.device)
                ab = ab.to(self.device)
                labels_np = labels.numpy()
                labels = labels.to(self.device)
                aux = aux.to(self.device) if aux is not None else None

                if isinstance(self.model, LFAAux):
                    logits = self.model(student_ids, skill_ids, sb, fb, ab, aux)
                else:
                    logits = self.model(student_ids, skill_ids, sb, fb, ab)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                all_labels.extend(labels_np)
                all_preds.extend(preds)

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        return total_loss / len(loader), metrics.get("auc", 0.0)

    def fit(self, train_df, val_df) -> Dict[str, Any]:
        train_loader = self._build_loader(train_df, shuffle=True)
        val_loader = self._build_loader(val_df, shuffle=False)

        logger.info(
            "Starting LFA training on device '%s' (%d train batches, %d val batches)...",
            self.device,
            len(train_loader),
            len(val_loader),
        )

        pbar = tqdm(range(self.max_epochs), desc=f"Training LFA [{self.device.type.upper()}]", unit="epoch")
        for epoch in pbar:
            train_loss = self._train_epoch(train_loader, epoch=epoch)
            val_loss, val_auc = self._eval_epoch(val_loader)

            pbar.set_postfix({"loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}", "val_auc": f"{val_auc:.4f}"})

            logger.info(
                "Epoch %d/%d [%s]: train_loss=%.4f val_loss=%.4f val_auc=%.4f",
                epoch + 1,
                self.max_epochs,
                self.device,
                train_loss,
                val_loss,
                val_auc,
            )

            if self.wandb_logger and self.wandb_logger.enabled:
                self.wandb_logger.log_metrics(
                    {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/auc": val_auc,
                        "epoch": epoch + 1,
                    },
                    step=epoch + 1,
                )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_state_dict = self.model.state_dict()
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logger.info("Early stopping triggered at epoch %d/%d", epoch + 1, self.max_epochs)
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        if self.wandb_logger and self.wandb_logger.enabled:
            self.wandb_logger.log_summary(
                {
                    "best_val_auc": self.best_val_auc,
                    "epochs_trained": epoch + 1,
                }
            )

        return {"best_val_auc": self.best_val_auc, "epochs_trained": epoch + 1}

    def predict(self, feature_df, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        loader = self._build_loader(feature_df, shuffle=False, batch_size=batch_size)
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 7:
                    student_ids, skill_ids, sb, fb, ab, labels, aux = batch
                else:
                    student_ids, skill_ids, sb, fb, ab, labels = batch
                    aux = None

                student_ids = student_ids.to(self.device, non_blocking=True)
                skill_ids = skill_ids.to(self.device, non_blocking=True)
                sb = sb.to(self.device, non_blocking=True)
                fb = fb.to(self.device, non_blocking=True)
                ab = ab.to(self.device, non_blocking=True)
                aux = aux.to(self.device, non_blocking=True) if aux is not None else None

                if isinstance(self.model, LFAAux):
                    logits = self.model(student_ids, skill_ids, sb, fb, ab, aux)
                else:
                    logits = self.model(student_ids, skill_ids, sb, fb, ab)

                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds)

    def save_checkpoint(self, path: Path) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_val_auc": self.best_val_auc,
                "config": {
                    "embedding_dim": self.model.embedding_dim,
                    "n_students": self.model.n_students,
                    "n_skills": self.model.n_skills,
                },
            },
            path,
        )


class LFAOnlineEvaluator:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        if device is None:
            from src.utils.seed import resolve_device
            self.device = resolve_device("auto")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.trainer = LFATrainer(model=self.model, device=self.device)

    def predict(self, feature_df, batch_size: int = 4096) -> np.ndarray:
        return self.trainer.predict(feature_df, batch_size=batch_size)

    def evaluate(self, feature_df, batch_size: int = 4096) -> Dict[str, Any]:
        all_labels = feature_df["correct"].to_numpy()
        all_preds = self.predict(feature_df, batch_size=batch_size)
        return compute_metrics(all_labels, all_preds)
