import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sequence_builder import SequenceBatch

logger = logging.getLogger(__name__)


class AttentiveContextualKT(nn.Module):
    def __init__(
        self,
        num_skills: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_seq_len: int = 200,
        use_position: bool = True,
        use_decay: bool = True,
        decay_type: str = "position",
        pad_skill_id: int = 0,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_position = use_position
        self.use_decay = use_decay
        self.decay_type = decay_type
        self.pad_skill_id = pad_skill_id

        self.skill_embedding = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=pad_skill_id)
        self.correct_embedding = nn.Embedding(2, embedding_dim)

        if use_position:
            self.position_embedding = nn.Embedding(max_seq_len + 1, embedding_dim)

        if use_decay:
            self.decay_param = nn.Parameter(torch.zeros(1))

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.attn_dropout = nn.Dropout(dropout)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_layers)
        ])

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        self.skill_bias = nn.Embedding(num_skills + 1, 1, padding_idx=pad_skill_id)
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def _build_position_ids(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return pos

    def _build_attention_mask(self, mask: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        batch_size = mask.size(0)
        causal_mask = self._build_causal_mask(seq_len, device)
        if self.use_decay:
            lam = F.softplus(self.decay_param)
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0.0)
            base_mask = causal_mask - lam * dist
        else:
            base_mask = causal_mask

        # Expand to (batch_size, seq_len, seq_len)
        attn_mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Mask padded key columns (positions where mask == False)
        key_padding = (~mask).unsqueeze(1)
        attn_mask = attn_mask.masked_fill(key_padding, float("-inf"))

        # For padded query rows, set position 0 to 0.0 to prevent all -inf rows (avoiding NaN in Softmax)
        query_padding = (~mask).unsqueeze(2)
        attn_mask = attn_mask.masked_fill(query_padding, float("-inf"))
        pad_q_col0 = query_padding.clone()
        pad_q_col0[:, :, 1:] = False
        attn_mask = attn_mask.masked_fill(pad_q_col0, 0.0)

        # Repeat for multi-head attention: (batch_size * num_heads, seq_len, seq_len)
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        return attn_mask

    def forward(
        self,
        skill_ids: torch.Tensor,
        corrects: torch.Tensor,
        target_skill_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        batch_size, seq_len = skill_ids.shape
        device = skill_ids.device

        skill_emb = self.skill_embedding(skill_ids.clamp(0, self.num_skills))
        correct_emb = self.correct_embedding(corrects.long())
        value_emb = skill_emb + correct_emb

        if self.use_position:
            pos_ids = self._build_position_ids(batch_size, seq_len, device)
            value_emb = value_emb + self.position_embedding(pos_ids)

        query_emb = self.skill_embedding(target_skill_ids.clamp(0, self.num_skills))
        if self.use_position:
            query_emb = query_emb + self.position_embedding(
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        q = self.query_proj(query_emb)
        k = self.key_proj(value_emb)
        v = self.value_proj(value_emb)

        attn_mask = self._build_attention_mask(mask, seq_len, device)

        for i in range(self.num_layers):
            attn_out, _ = self.attn_layers[i](
                q, k, v,
                attn_mask=attn_mask,
            )
            attn_out = self.attn_dropout(attn_out)
            q = self.layer_norms[i](q + attn_out)

            ffn_out = self.ffn(q)
            q = self.ffn_norm(q + ffn_out)

        context = q

        skill_bias = self.skill_bias(target_skill_ids.clamp(0, self.num_skills)).squeeze(-1)
        query_for_head = self.skill_embedding(target_skill_ids.clamp(0, self.num_skills))
        head_input = torch.cat([query_for_head, context], dim=-1)
        head_logit = self.prediction_head(head_input).squeeze(-1)

        logits = skill_bias + head_logit

        return logits, context, None

    def predict_cold_start(
        self,
        target_skill_id: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            target_emb = self.skill_embedding(target_skill_id.clamp(0, self.num_skills))
            context = torch.zeros_like(target_emb)
            skill_bias = self.skill_bias(target_skill_id.clamp(0, self.num_skills)).squeeze(-1)
            head_input = torch.cat([target_emb, context], dim=-1)
            head_logit = self.prediction_head(head_input).squeeze(-1)
            logit = skill_bias + head_logit
            return torch.sigmoid(logit)

    def predict_step(
        self,
        history_skill_ids: torch.Tensor,
        history_corrects: torch.Tensor,
        target_skill_id: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            batch_size = 1
            seq_len = history_skill_ids.size(1)

            device = history_skill_ids.device

            skill_emb = self.skill_embedding(history_skill_ids.clamp(0, self.num_skills))
            correct_emb = self.correct_embedding(history_corrects.long())
            value_emb = skill_emb + correct_emb

            if self.use_position:
                pos_ids = self._build_position_ids(batch_size, seq_len, device)
                value_emb = value_emb + self.position_embedding(pos_ids)

            target_emb = self.skill_embedding(target_skill_id.clamp(0, self.num_skills))
            if self.use_position:
                pos_idx = min(seq_len, self.max_seq_len)
                target_emb = target_emb + self.position_embedding(
                    torch.full((1, 1), pos_idx, device=device, dtype=torch.long)
                )

            q = self.query_proj(target_emb)
            k = self.key_proj(value_emb)
            v = self.value_proj(value_emb)

            if self.use_decay:
                lam = F.softplus(self.decay_param)
                pos_hist = torch.arange(seq_len, device=device, dtype=torch.float32)
                dist = (seq_len - pos_hist).unsqueeze(0)
                attn_mask = -lam * dist
            else:
                attn_mask = torch.zeros(1, seq_len, device=device)

            for i in range(self.num_layers):
                attn_out, _ = self.attn_layers[i](q, k, v, attn_mask=attn_mask)
                attn_out = self.attn_dropout(attn_out)
                q = self.layer_norms[i](q + attn_out)
                ffn_out = self.ffn(q)
                q = self.ffn_norm(q + ffn_out)

            context = q
            skill_bias = self.skill_bias(target_skill_id.clamp(0, self.num_skills)).squeeze(-1)
            query_for_head = self.skill_embedding(target_skill_id.clamp(0, self.num_skills))
            head_input = torch.cat([query_for_head, context], dim=-1)
            head_logit = self.prediction_head(head_input).squeeze(-1)

            logit = skill_bias + head_logit
            return torch.sigmoid(logit)


class AttentionOnlineEvaluator:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, max_seq_len: int = 200):
        if device is None:
            from src.utils.seed import resolve_device
            self.device = resolve_device("auto")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.max_seq_len = max_seq_len
        self.model.eval()
        self.history_buffers: Dict[Any, Dict[str, List]] = {}

    def reset_state(self) -> None:
        self.history_buffers = {}

    def predict(
        self,
        df: pd.DataFrame,
        student_col: str = "studentId",
        skill_col: str = "skill",
        correct_col: str = "correct",
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        sort_cols = [student_col, "startTime"]
        if "action_num" in df.columns:
            sort_cols.append("action_num")
        df_sorted = df.sort_values(sort_cols).reset_index(drop=True).copy()

        n_rows = len(df_sorted)
        if n_rows == 0:
            return np.array([]), np.array([]), df_sorted

        has_encoded = "skill_id_encoded" in df_sorted.columns
        if has_encoded:
            raw_skills = df_sorted["skill_id_encoded"].to_numpy(dtype=np.int64)
        else:
            raw_skills = np.array([
                hash(s) % self.model.num_skills if isinstance(s, str) else int(s) % self.model.num_skills
                for s in df_sorted[skill_col].to_numpy()
            ], dtype=np.int64)

        skills_arr = np.clip(raw_skills, 0, self.model.num_skills - 1)
        students_arr = df_sorted[student_col].to_numpy()
        corrects_arr = df_sorted[correct_col].to_numpy(dtype=np.int64)
        y_pred = np.zeros(n_rows, dtype=np.float32)

        student_indices: Dict[Any, List[int]] = {}
        for idx, sid in enumerate(students_arr):
            student_indices.setdefault(sid, []).append(idx)

        student_items = list(student_indices.items())
        batch_size = 128

        with torch.no_grad():
            for b_start in range(0, len(student_items), batch_size):
                b_items = student_items[b_start : b_start + batch_size]

                # Step 0: Cold start batch
                s0_list = [skills_arr[idx_list[0]] for _, idx_list in b_items]
                s0_tensor = torch.tensor(s0_list, dtype=torch.long, device=self.device).unsqueeze(1)
                p0_batch = self.model.predict_cold_start(s0_tensor).cpu().numpy().squeeze(1)
                for j, (_, idx_list) in enumerate(b_items):
                    y_pred[idx_list[0]] = float(p0_batch[j])

                # Sequences with T > 1
                multi_items = [(stud, idx_list) for stud, idx_list in b_items if len(idx_list) > 1]
                if multi_items:
                    short_items = [(stud, idx_list) for stud, idx_list in multi_items if len(idx_list) - 1 <= self.max_seq_len]
                    long_items = [(stud, idx_list) for stud, idx_list in multi_items if len(idx_list) - 1 > self.max_seq_len]

                    if short_items:
                        max_len = max(len(idx_list) - 1 for _, idx_list in short_items)
                        B_short = len(short_items)
                        in_skills = torch.zeros((B_short, max_len), dtype=torch.long, device=self.device)
                        in_corrects = torch.zeros((B_short, max_len), dtype=torch.float32, device=self.device)
                        tgt_skills = torch.zeros((B_short, max_len), dtype=torch.long, device=self.device)
                        mask = torch.zeros((B_short, max_len), dtype=torch.bool, device=self.device)

                        for j, (_, idx_list) in enumerate(short_items):
                            s_skills = skills_arr[idx_list]
                            s_corrects = corrects_arr[idx_list]
                            L_j = len(idx_list) - 1
                            in_skills[j, :L_j] = torch.from_numpy(s_skills[:-1])
                            in_corrects[j, :L_j] = torch.from_numpy(s_corrects[:-1])
                            tgt_skills[j, :L_j] = torch.from_numpy(s_skills[1:])
                            mask[j, :L_j] = True

                        tgt_logits, _, _ = self.model(
                            skill_ids=in_skills,
                            corrects=in_corrects,
                            target_skill_ids=tgt_skills,
                            mask=mask,
                        )
                        preds_batch = torch.sigmoid(tgt_logits).cpu().numpy()

                        for j, (stud, idx_list) in enumerate(short_items):
                            L_j = len(idx_list) - 1
                            y_pred[idx_list[1:]] = preds_batch[j, :L_j]
                            s_skills = skills_arr[idx_list]
                            s_corrects = corrects_arr[idx_list]
                            self.history_buffers[stud] = {
                                "skill_ids": [int(x) for x in s_skills[-self.max_seq_len:]],
                                "corrects": [float(x) for x in s_corrects[-self.max_seq_len:]],
                            }

                    for stud, idx_list in long_items:
                        s_skills = skills_arr[idx_list]
                        s_corrects = corrects_arr[idx_list]
                        T_j = len(idx_list)
                        hist_s = list(s_skills[:1])
                        hist_c = [float(s_corrects[0])]
                        for step_idx in range(1, T_j):
                            tgt_s = int(s_skills[step_idx])
                            h_s_t = torch.tensor([hist_s[-self.max_seq_len:]], dtype=torch.long, device=self.device)
                            h_c_t = torch.tensor([hist_c[-self.max_seq_len:]], dtype=torch.float32, device=self.device)
                            tgt_t = torch.tensor([[tgt_s]], dtype=torch.long, device=self.device)
                            y_pred[idx_list[step_idx]] = self.model.predict_step(h_s_t, h_c_t, tgt_t).item()
                            hist_s.append(tgt_s)
                            hist_c.append(float(s_corrects[step_idx]))

                        self.history_buffers[stud] = {
                            "skill_ids": hist_s[-self.max_seq_len:],
                            "corrects": hist_c[-self.max_seq_len:],
                        }
                else:
                    for stud, idx_list in b_items:
                        s_skills = skills_arr[idx_list]
                        s_corrects = corrects_arr[idx_list]
                        self.history_buffers[stud] = {
                            "skill_ids": [int(s_skills[0])],
                            "corrects": [float(s_corrects[0])],
                        }

        df_sorted["prediction"] = y_pred
        return corrects_arr, y_pred, df_sorted

    def evaluate(
        self,
        df,
        student_col: str = "studentId",
        skill_col: str = "skill",
        correct_col: str = "correct",
    ) -> Dict[str, Any]:
        from src.evaluation.metrics import compute_comprehensive_metrics

        _, _, eval_df = self.predict(df=df, student_col=student_col, skill_col=skill_col, correct_col=correct_col)
        return compute_comprehensive_metrics(eval_df, y_true_col=correct_col, y_pred_col="prediction", skill_col=skill_col, student_col=student_col)