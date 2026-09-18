import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

GLOBAL_SKILL_KEY = "__GLOBAL_DBKT__"


def dbkt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that dynamically pads sequences in a batch to max sequence length.
    Enables single-pass tensor operations on GPU/MPS with zero padding waste.
    """
    batch_size = len(batch)
    lengths = [len(it["sequence"]) for it in batch]
    max_len = max(lengths)

    skill_ids = torch.tensor([it.get("skill_id", 0) for it in batch], dtype=torch.long)
    corrects = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    has_features = ("features" in batch[0] and batch[0]["features"] is not None)
    if has_features:
        feat_dim = batch[0]["features"].shape[1]
        features = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    else:
        features = None

    for i, it in enumerate(batch):
        seq = it["sequence"]
        l = len(seq)
        corrects[i, :l] = torch.as_tensor(seq, dtype=torch.float32)
        mask[i, :l] = 1.0
        if has_features:
            features[i, :l, :] = torch.as_tensor(it["features"], dtype=torch.float32)

    return {
        "skill_ids": skill_ids,
        "corrects": corrects,
        "mask": mask,
        "features": features,
    }


class DynamicBayesianKnowledgeTracing(nn.Module):
    def __init__(
        self,
        num_skills: int,
        feature_dim: int = 2,
        use_forgetting: bool = True,
        constrain_slip_guess: bool = True,
        slip_guess_penalty: float = 1.0,
        pad_skill_id: int = 0,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.feature_dim = feature_dim
        self.use_forgetting = use_forgetting
        self.constrain_slip_guess = constrain_slip_guess
        self.slip_guess_penalty = slip_guess_penalty
        self.pad_skill_id = pad_skill_id

        # Parameters for each skill (indexed 0 to num_skills)
        self.init_logit = nn.Parameter(torch.zeros(num_skills + 1))
        self.slip_logit = nn.Parameter(torch.full((num_skills + 1,), -1.386))   # sigmoid(-1.386) ≈ 0.20
        self.guess_logit = nn.Parameter(torch.full((num_skills + 1,), -1.386))  # sigmoid(-1.386) ≈ 0.20
        self.learn_logit = nn.Parameter(torch.full((num_skills + 1,), -1.0))    # sigmoid(-1.0) ≈ 0.27

        # Shared dynamic transition weights for learning rate
        self.learn_w = nn.Parameter(torch.zeros(1, feature_dim))
        self.learn_b = nn.Parameter(torch.zeros(1))

        if use_forgetting:
            self.forget_logit = nn.Parameter(torch.full((num_skills + 1,), -2.94))  # sigmoid(-2.94) ≈ 0.05
            self.forget_w = nn.Parameter(torch.zeros(1, feature_dim))
            self.forget_b = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("forget_logit", torch.full((num_skills + 1,), -10.0))
            self.register_buffer("forget_w", torch.zeros(1, feature_dim))
            self.register_buffer("forget_b", torch.zeros(1))

    def get_p_init(self, skill_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skill_id is not None:
            return torch.sigmoid(self.init_logit[skill_id.clamp(0, self.num_skills)])
        return torch.sigmoid(self.init_logit)

    def get_p_slip(self, skill_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skill_id is not None:
            return torch.sigmoid(self.slip_logit[skill_id.clamp(0, self.num_skills)])
        return torch.sigmoid(self.slip_logit)

    def get_p_guess(self, skill_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skill_id is not None:
            return torch.sigmoid(self.guess_logit[skill_id.clamp(0, self.num_skills)])
        return torch.sigmoid(self.guess_logit)

    def get_learn_rate(self, skill_id: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.learn_logit[skill_id.clamp(0, self.num_skills)]
        if features is not None and self.feature_dim > 0:
            if features.dim() == 3:
                # features shape: (B, T, D)
                dynamic = (features * self.learn_w.unsqueeze(0)).sum(dim=-1) + self.learn_b
                return torch.sigmoid(base.unsqueeze(-1) + dynamic)
            elif features.dim() == 2:
                # features shape: (B, D) or (T, D)
                dynamic = (features * self.learn_w).sum(dim=-1, keepdim=True) + self.learn_b
                return torch.sigmoid(base.unsqueeze(-1) + dynamic).squeeze(-1)
            elif features.dim() == 1:
                dynamic = (features * self.learn_w).sum() + self.learn_b
                return torch.sigmoid(base + dynamic)
        return torch.sigmoid(base)

    def get_forget_rate(self, skill_id: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_forgetting:
            if features is not None and features.dim() >= 2:
                return torch.zeros(features.shape[:-1], device=skill_id.device)
            return torch.zeros_like(skill_id, dtype=torch.float32, device=skill_id.device)

        base = self.forget_logit[skill_id.clamp(0, self.num_skills)]
        if features is not None and self.feature_dim > 0:
            if features.dim() == 3:
                dynamic = (features * self.forget_w.unsqueeze(0)).sum(dim=-1) + self.forget_b
                return torch.sigmoid(base.unsqueeze(-1) + dynamic)
            elif features.dim() == 2:
                dynamic = (features * self.forget_w).sum(dim=-1, keepdim=True) + self.forget_b
                return torch.sigmoid(base.unsqueeze(-1) + dynamic).squeeze(-1)
            elif features.dim() == 1:
                dynamic = (features * self.forget_w).sum() + self.forget_b
                return torch.sigmoid(base + dynamic)
        return torch.sigmoid(base)

    def predict(self, p_known: torch.Tensor, skill_id: torch.Tensor) -> torch.Tensor:
        p_slip = self.get_p_slip(skill_id)
        p_guess = self.get_p_guess(skill_id)
        return p_known * (1.0 - p_slip) + (1.0 - p_known) * p_guess

    def update(
        self,
        p_known: torch.Tensor,
        correct: torch.Tensor,
        skill_id: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p_correct = self.predict(p_known, skill_id).clamp(1e-7, 1.0 - 1e-7)
        p_slip = self.get_p_slip(skill_id)

        p_known_given_obs = torch.where(
            correct == 1,
            (p_known * (1.0 - p_slip)) / p_correct,
            (p_known * p_slip) / (1.0 - p_correct),
        ).clamp(0.0, 1.0)

        learn_rate = self.get_learn_rate(skill_id, features)
        forget_rate = self.get_forget_rate(skill_id, features)

        p_known_next = p_known_given_obs * (1.0 - forget_rate) + (1.0 - p_known_given_obs) * learn_rate
        return p_known_next.clamp(0.0, 1.0)

    def forward_log_likelihood_and_preds(
        self,
        skill_ids: torch.Tensor,
        corrects: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sequence log-likelihood and emission predictions P(y_t = 1).
        """
        batch_size = skill_ids.size(0)
        seq_len = corrects.size(1)
        device = skill_ids.device

        init_skill = skill_ids[:, 0] if skill_ids.dim() == 2 else skill_ids
        p_init = self.get_p_init(init_skill)  # (B,)

        p_known = p_init.clone()
        total_ll = torch.zeros(batch_size, device=device)
        all_preds = torch.zeros((batch_size, seq_len), device=device)

        for t in range(seq_len):
            c = corrects[:, t]
            feat_t = features[:, t, :] if features is not None else None
            cur_skill = skill_ids[:, t] if skill_ids.dim() == 2 else skill_ids

            p_slip = self.get_p_slip(cur_skill)
            p_guess = self.get_p_guess(cur_skill)

            p_correct = (p_known * (1.0 - p_slip) + (1.0 - p_known) * p_guess).clamp(1e-7, 1.0 - 1e-7)
            all_preds[:, t] = p_correct

            ll = torch.where(c == 1, torch.log(p_correct), torch.log(1.0 - p_correct))
            if mask is not None:
                m_t = mask[:, t]
                ll = ll * m_t
            total_ll = total_ll + ll

            # Posterior update given observation at t
            p_known_given_obs = torch.where(
                c == 1,
                (p_known * (1.0 - p_slip)) / p_correct,
                (p_known * p_slip) / (1.0 - p_correct),
            ).clamp(0.0, 1.0)

            # Dynamic transition rates for step t
            lr = self.get_learn_rate(cur_skill, feat_t)
            fr = self.get_forget_rate(cur_skill, feat_t)

            p_known_next = (p_known_given_obs * (1.0 - fr) + (1.0 - p_known_given_obs) * lr).clamp(0.0, 1.0)

            if mask is not None:
                p_known = torch.where(m_t == 1.0, p_known_next, p_known)
            else:
                p_known = p_known_next

        return total_ll, all_preds

    def forward_log_likelihood(
        self,
        skill_ids: torch.Tensor,
        corrects: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sequence log-likelihood with optional sequence mask for padded batches.
        """
        ll, _ = self.forward_log_likelihood_and_preds(skill_ids, corrects, features, mask)
        return ll

    def compute_loss(
        self,
        skill_ids: torch.Tensor,
        corrects: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            total_valid = mask.sum().clamp(min=1.0)
            nll = -self.forward_log_likelihood(skill_ids, corrects, features, mask).sum() / total_valid
        else:
            nll = -self.forward_log_likelihood(skill_ids, corrects, features).mean()

        if self.constrain_slip_guess:
            p_slip = self.get_p_slip()
            p_guess = self.get_p_guess()
            violation = (p_slip + p_guess - 0.75).clamp(min=0)
            penalty = self.slip_guess_penalty * (violation ** 2).mean()
            return nll + penalty

        return nll

    def get_params(self, skill_id: int) -> Dict[str, float]:
        sid = torch.tensor([skill_id])
        with torch.no_grad():
            return {
                "p_init": float(self.get_p_init(sid).item()),
                "p_slip": float(self.get_p_slip(sid).item()),
                "p_guess": float(self.get_p_guess(sid).item()),
                "p_learn_base": float(torch.sigmoid(self.learn_logit[sid.clamp(0, self.num_skills)]).item()),
                "p_forget_base": float(torch.sigmoid(self.forget_logit[sid.clamp(0, self.num_skills)]).item()) if self.use_forgetting else 0.0,
            }


class DBKTTrainer:
    def __init__(
        self,
        model: DynamicBayesianKnowledgeTracing,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 0.005,
        max_epochs: int = 50,
        batch_size: int = 64,
        tolerance: float = 1e-5,
        patience: int = 8,
        gradient_clip_norm: float = 1.0,
        min_skill_sequences: int = 20,
        fallback_to_global: bool = True,
        seed: int = 42,
        wandb_logger: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        self.min_skill_sequences = min_skill_sequences
        self.fallback_to_global = fallback_to_global
        self.seed = seed
        self.wandb_logger = wandb_logger
        self.best_val_auc = 0.0

    def fit(
        self,
        train_sequences: List[Dict[str, Any]],
        val_sequences: Optional[List[Dict[str, Any]]] = None,
        val_df: Optional[pd.DataFrame] = None,
        skill_encoder: Optional[Any] = None,
        feature_cols: Optional[List[str]] = None,
        skill_col: str = "skill",
    ) -> Tuple[Dict[str, DynamicBayesianKnowledgeTracing], Optional[DynamicBayesianKnowledgeTracing]]:
        """
        Trains the DBKT model end-to-end across sequences with PyTorch DataLoader and masked vectorization.
        Evaluates on validation split each epoch for both loss and AUC if val_df or val_sequences is provided.
        When val_df is passed, evaluates on 100% of validation rows via DBKTOnlineEvaluator to ensure zero discrepancy.
        """
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if not train_sequences:
            logger.warning("No sequences provided to DBKTTrainer.fit")
            return {}, None

        # Build count statistics per skill
        skill_counts: Dict[str, int] = {}
        for s in train_sequences:
            skill = s.get("skill", "unknown")
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

        val_desc = f"{len(val_df)} val rows" if val_df is not None else (f"{len(val_sequences)} val sequences" if val_sequences else "no val")
        logger.info(
            "DBKTTrainer starting optimization: %d train sequences (%s) across %d skills",
            len(train_sequences),
            val_desc,
            len(skill_counts),
        )

        train_loader = DataLoader(
            train_sequences,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dbkt_collate_fn,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )

        val_loader = None
        if val_sequences and val_df is None:
            val_loader = DataLoader(
                val_sequences,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=dbkt_collate_fn,
                pin_memory=(self.device.type == "cuda"),
                num_workers=0,
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float("inf")
        best_val_auc = 0.0
        best_state = None
        no_improve_epochs = 0

        epoch_pbar = tqdm(range(1, self.max_epochs + 1), desc=f"Training DBKT [{self.device.type.upper()}]", unit="epoch")
        for epoch in epoch_pbar:
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            batch_pbar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch:02d}/{self.max_epochs} [Train]",
                leave=False,
                unit="batch",
            )
            for batch in batch_pbar:
                skill_ids = batch["skill_ids"].to(self.device, non_blocking=True)
                corrects = batch["corrects"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                features = batch["features"].to(self.device, non_blocking=True) if batch["features"] is not None else None

                optimizer.zero_grad()
                loss = self.model.compute_loss(skill_ids, corrects, features, mask)
                loss.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                num_batches += 1
                batch_pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            avg_epoch_loss = epoch_loss / max(num_batches, 1)

            # Evaluate on validation split if available
            avg_val_loss = None
            val_auc = None

            if val_df is not None:
                self.model.eval()
                evaluator = DBKTOnlineEvaluator(
                    model=self.model,
                    device=self.device,
                    skill_encoder=skill_encoder,
                    feature_cols=feature_cols,
                )
                y_true, y_pred, _ = evaluator.predict(val_df)
                if len(y_true) > 0 and len(np.unique(y_true)) > 1:
                    val_auc = float(roc_auc_score(y_true, y_pred))
                else:
                    val_auc = 0.5

                eps = 1e-7
                y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
                avg_val_loss = float(-np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)))
                metric_to_track = val_auc
            elif val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                all_val_labels = []
                all_val_preds = []
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_skill_ids = v_batch["skill_ids"].to(self.device, non_blocking=True)
                        v_corrects = v_batch["corrects"].to(self.device, non_blocking=True)
                        v_mask = v_batch["mask"].to(self.device, non_blocking=True)
                        v_features = v_batch["features"].to(self.device, non_blocking=True) if v_batch["features"] is not None else None
                        
                        v_loss = self.model.compute_loss(v_skill_ids, v_corrects, v_features, v_mask)
                        val_loss_sum += v_loss.item()
                        val_batches += 1

                        _, v_preds = self.model.forward_log_likelihood_and_preds(v_skill_ids, v_corrects, v_features, v_mask)
                        valid_pos = (v_mask == 1.0)
                        all_val_labels.extend(v_corrects[valid_pos].cpu().numpy())
                        all_val_preds.extend(v_preds[valid_pos].cpu().numpy())

                avg_val_loss = val_loss_sum / max(val_batches, 1)
                if len(all_val_labels) > 0 and len(set(all_val_labels)) > 1:
                    val_auc = float(roc_auc_score(all_val_labels, all_val_preds))
                else:
                    val_auc = 0.5
                metric_to_track = val_auc
            else:
                metric_to_track = -avg_epoch_loss

            is_improved = False
            if val_auc is not None:
                if val_auc > best_val_auc + self.tolerance:
                    best_val_auc = val_auc
                    best_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve_epochs = 0
                    is_improved = True
                else:
                    no_improve_epochs += 1
            else:
                if avg_epoch_loss < best_loss - self.tolerance:
                    best_loss = avg_epoch_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve_epochs = 0
                    is_improved = True
                else:
                    no_improve_epochs += 1

            postfix_dict = {"loss": f"{avg_epoch_loss:.4f}"}
            if avg_val_loss is not None:
                postfix_dict["val_loss"] = f"{avg_val_loss:.4f}"
            if val_auc is not None:
                postfix_dict["val_auc"] = f"{val_auc:.4f}"
                postfix_dict["best_auc"] = f"{best_val_auc:.4f}"
            else:
                postfix_dict["best"] = f"{best_loss:.4f}"
            epoch_pbar.set_postfix(postfix_dict)

            if self.wandb_logger and getattr(self.wandb_logger, "enabled", False):
                wandb_metrics = {
                    "train/loss": avg_epoch_loss,
                    "epoch": epoch,
                }
                if avg_val_loss is not None:
                    wandb_metrics["val/loss"] = avg_val_loss
                if val_auc is not None:
                    wandb_metrics["val/auc"] = val_auc
                    wandb_metrics["val/best_auc"] = best_val_auc
                self.wandb_logger.log_metrics(wandb_metrics, step=epoch)

            if epoch % 5 == 0 or epoch == 1 or epoch == self.max_epochs:
                if val_auc is not None:
                    logger.info("DBKT Epoch %02d/%02d - Train Loss: %.5f | Val Loss: %.5f | Val AUC: %.4f", epoch, self.max_epochs, avg_epoch_loss, avg_val_loss, val_auc)
                elif avg_val_loss is not None:
                    logger.info("DBKT Epoch %02d/%02d - Train Loss: %.5f | Val Loss: %.5f", epoch, self.max_epochs, avg_epoch_loss, avg_val_loss)
                else:
                    logger.info("DBKT Epoch %02d/%02d - Loss: %.5f", epoch, self.max_epochs, avg_epoch_loss)

            if no_improve_epochs >= self.patience:
                logger.info("Early stopping triggered at epoch %d (best tracked val AUC: %.5f)", epoch, best_val_auc if val_auc is not None else best_loss)
                break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.best_val_auc = best_val_auc

        if self.wandb_logger and getattr(self.wandb_logger, "enabled", False):
            self.wandb_logger.log_summary({
                "best_val_auc": best_val_auc,
                "epochs_trained": epoch,
            })

        # Return dict mapping skill -> model
        skill_models = {skill: self.model for skill in skill_counts.keys()}
        global_model = self.model if self.fallback_to_global else None

        return skill_models, global_model

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_val_auc": self.best_val_auc,
            "config": {
                "num_skills": self.model.num_skills,
                "feature_dim": self.model.feature_dim,
                "use_forgetting": self.model.use_forgetting,
            },
        }, path_obj)
        logger.info("Saved DBKT checkpoint to %s", path_obj)


class DBKTOnlineEvaluator:
    def __init__(
        self,
        model: DynamicBayesianKnowledgeTracing,
        device: Optional[torch.device] = None,
        skill_encoder: Optional[Any] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        if device is None:
            from src.utils.seed import resolve_device
            self.device = resolve_device("auto")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.skill_encoder = skill_encoder
        self.feature_cols = feature_cols
        self.model.eval()
        self.state: Dict[Tuple[int, str], float] = {}

    def reset_state(self) -> None:
        self.state = {}

    def _get_skill_id(self, skill_val: Any) -> int:
        if self.skill_encoder is not None:
            try:
                encoded = self.skill_encoder.transform([skill_val])[0]
                return int(encoded)
            except Exception:
                return 0
        return 0

    def predict(
        self,
        df: pd.DataFrame,
        student_col: str = "studentId",
        skill_col: str = "skill",
        correct_col: str = "correct",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        cols = self.feature_cols if feature_cols is None else feature_cols

        sort_cols = [student_col, "startTime"]
        if "action_num" in df.columns:
            sort_cols.append("action_num")
        df_sorted = df.sort_values(sort_cols).reset_index(drop=True).copy()

        n_rows = len(df_sorted)
        if n_rows == 0:
            return np.array([]), np.array([]), df_sorted

        y_true = df_sorted[correct_col].to_numpy(dtype=np.int64)
        y_pred = np.zeros(n_rows, dtype=np.float32)

        # Precompute attempts and failure history if not explicitly provided
        if cols is None or not all(c in df_sorted.columns for c in cols):
            attempts_before = df_sorted.groupby([student_col, skill_col]).cumcount().to_numpy(dtype=np.float32)
            success_before = (df_sorted.groupby([student_col, skill_col])[correct_col].cumsum() - df_sorted[correct_col]).to_numpy(dtype=np.float32)
            failure_before = np.maximum(0.0, attempts_before - success_before)
            feat_matrix = np.column_stack([
                np.log1p(attempts_before),
                np.log1p(failure_before),
            ]).astype(np.float32)
        else:
            feat_matrix = df_sorted[cols].to_numpy(dtype=np.float32)

        students = df_sorted[student_col].to_numpy()
        skills = df_sorted[skill_col].astype(str).to_numpy()

        unique_skills = np.unique(skills)
        skill_to_id: Dict[str, int] = {s: self._get_skill_id(s) for s in unique_skills}

        # Pre-extract model parameters as NumPy arrays for lightning-fast vectorized evaluation
        with torch.no_grad():
            p_init_arr = self.model.get_p_init().detach().cpu().numpy()
            p_slip_arr = self.model.get_p_slip().detach().cpu().numpy()
            p_guess_arr = self.model.get_p_guess().detach().cpu().numpy()

            learn_base = self.model.learn_logit.detach().cpu().numpy()
            has_features = self.model.feature_dim > 0 and feat_matrix is not None
            if has_features:
                learn_w = self.model.learn_w.detach().cpu().numpy().reshape(-1)
                learn_b = float(self.model.learn_b.detach().cpu().item())
            else:
                learn_w, learn_b = None, 0.0

            use_forgetting = self.model.use_forgetting
            if use_forgetting:
                forget_base = self.model.forget_logit.detach().cpu().numpy()
                if has_features:
                    forget_w = self.model.forget_w.detach().cpu().numpy().reshape(-1)
                    forget_b = float(self.model.forget_b.detach().cpu().item())
                else:
                    forget_w, forget_b = None, 0.0
            else:
                forget_base, forget_w, forget_b = None, None, 0.0

        for i in range(n_rows):
            student = int(students[i])
            skill_str = skills[i]
            correct = int(y_true[i])
            skill_id = min(skill_to_id.get(skill_str, 0), self.model.num_skills)

            key = (student, skill_str)

            if key not in self.state:
                p_known = float(p_init_arr[skill_id])
                self.state[key] = p_known
            else:
                p_known = self.state[key]

            p_slip = float(p_slip_arr[skill_id])
            p_guess = float(p_guess_arr[skill_id])

            p_correct = np.clip(p_known * (1.0 - p_slip) + (1.0 - p_known) * p_guess, 1e-7, 1.0 - 1e-7)
            y_pred[i] = p_correct

            if correct == 1:
                p_known_obs = (p_known * (1.0 - p_slip)) / p_correct
            else:
                p_known_obs = (p_known * p_slip) / (1.0 - p_correct)
            p_known_obs = np.clip(p_known_obs, 0.0, 1.0)

            # Dynamic learn rate
            if has_features and learn_w is not None:
                logit_lr = learn_base[skill_id] + float(np.dot(feat_matrix[i], learn_w)) + learn_b
            else:
                logit_lr = learn_base[skill_id]
            lr = 1.0 / (1.0 + np.exp(-logit_lr))

            # Dynamic forget rate
            if use_forgetting and forget_base is not None:
                if has_features and forget_w is not None:
                    logit_fr = forget_base[skill_id] + float(np.dot(feat_matrix[i], forget_w)) + forget_b
                else:
                    logit_fr = forget_base[skill_id]
                fr = 1.0 / (1.0 + np.exp(-logit_fr))
            else:
                fr = 0.0

            p_next = p_known_obs * (1.0 - fr) + (1.0 - p_known_obs) * lr
            self.state[key] = float(np.clip(p_next, 0.0, 1.0))

        df_sorted["prediction"] = y_pred
        return y_true, y_pred, df_sorted

    def evaluate(
        self,
        df: pd.DataFrame,
        student_col: str = "studentId",
        skill_col: str = "skill",
        correct_col: str = "correct",
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from src.evaluation.metrics import compute_comprehensive_metrics

        _, _, eval_df = self.predict(
            df=df,
            student_col=student_col,
            skill_col=skill_col,
            correct_col=correct_col,
            feature_cols=feature_cols,
        )
        return compute_comprehensive_metrics(
            eval_df,
            y_true_col=correct_col,
            y_pred_col="prediction",
            skill_col=skill_col,
            student_col=student_col,
        )


def save_dbkt_params(
    skill_models: Dict[str, DynamicBayesianKnowledgeTracing],
    global_model: Optional[DynamicBayesianKnowledgeTracing] = None,
    filepath: Union[str, Path] = "dbkt_params.csv",
    n_sequences: Optional[Dict[str, int]] = None,
    n_interactions: Optional[Dict[str, int]] = None,
    skill_encoder: Optional[Any] = None,
) -> None:
    params_list = []

    for skill, model in skill_models.items():
        sid = 0
        if skill_encoder is not None:
            try:
                sid = int(skill_encoder.transform([skill])[0])
            except Exception:
                sid = 0

        params = model.get_params(sid)
        params_list.append({
            "skill": skill,
            "skill_id": sid,
            "n_sequences": n_sequences.get(skill, 0) if n_sequences else 0,
            "n_interactions": n_interactions.get(skill, 0) if n_interactions else 0,
            "p_init": params["p_init"],
            "p_learn_base": params["p_learn_base"],
            "p_forget_base": params["p_forget_base"],
            "p_slip": params["p_slip"],
            "p_guess": params["p_guess"],
        })

    if global_model is not None:
        params = global_model.get_params(0)
        params_list.append({
            "skill": GLOBAL_SKILL_KEY,
            "skill_id": -1,
            "n_sequences": sum(n_sequences.values()) if n_sequences else 0,
            "n_interactions": sum(n_interactions.values()) if n_interactions else 0,
            "p_init": params["p_init"],
            "p_learn_base": params["p_learn_base"],
            "p_forget_base": params["p_forget_base"],
            "p_slip": params["p_slip"],
            "p_guess": params["p_guess"],
        })

    params_df = pd.DataFrame(params_list)
    params_df.to_csv(filepath, index=False)
    logger.info("Saved DBKT params (%d skills) to %s", len(params_list), filepath)