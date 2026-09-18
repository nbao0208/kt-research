import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.sequence_builder import SequenceBatch

logger = logging.getLogger(__name__)


class DeepKnowledgeTracing(nn.Module):
    def __init__(
        self,
        num_skills: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
        input_mode: str = "embedding",
        pad_skill_id: int = 0,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_mode = input_mode
        self.pad_skill_id = pad_skill_id

        self.skill_embedding = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=pad_skill_id)
        self.correct_embedding = nn.Embedding(2, embedding_dim)

        if input_mode == "one_hot":
            self.input_proj = nn.Identity()
            input_size = 2 * num_skills
            rnn_input_size = input_size
        else:
            input_size = embedding_dim * 2
            self.input_proj = nn.Linear(input_size, embedding_dim)
            rnn_input_size = embedding_dim

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, num_skills)

    def forward(
        self,
        skill_ids: torch.Tensor,
        corrects: torch.Tensor,
        target_skill_ids: torch.Tensor,
        mask: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = skill_ids.shape

        if self.input_mode == "one_hot":
            batch_size, seq_len = skill_ids.shape
            one_hot_skills = torch.zeros(batch_size, seq_len, self.num_skills, device=skill_ids.device)
            one_hot_skills.scatter_(2, skill_ids.unsqueeze(-1).clamp(0, self.num_skills - 1), 1.0)
            correct_float = corrects.unsqueeze(-1)
            input_vec = one_hot_skills * correct_float + torch.roll(one_hot_skills, 1, dims=-1) * (1 - correct_float)
            indices = 2 * skill_ids.clamp(0, self.num_skills - 1) + corrects.long()
            indices = indices.clamp(0, 2 * self.num_skills - 1)
            input_vec = torch.zeros(batch_size, seq_len, 2 * self.num_skills, device=skill_ids.device)
            input_vec.scatter_(2, indices.unsqueeze(-1), 1.0)
        else:
            skill_emb = self.skill_embedding(skill_ids.clamp(0, self.num_skills))
            correct_emb = self.correct_embedding(corrects.long())
            input_vec = self.input_proj(torch.cat([skill_emb, correct_emb], dim=-1))

        input_vec = self.dropout(input_vec)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_vec,
            lengths=mask.sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        if hidden is not None:
            rnn_out, new_hidden = self.rnn(packed_input, hidden)
        else:
            rnn_out, new_hidden = self.rnn(packed_input)

        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=seq_len)

        logits = self.output_layer(self.dropout(rnn_out))
        target_logits = logits.gather(2, target_skill_ids.clamp(0, self.num_skills - 1).unsqueeze(-1)).squeeze(-1)

        return target_logits, rnn_out, new_hidden

    def predict_from_hidden(
        self,
        skill_id: torch.Tensor,
        hidden: Optional[Any],
    ) -> torch.Tensor:
        """
        Extracts prediction logits for target skill from current hidden state h_{t-1}.
        """
        if hidden is not None:
            if isinstance(hidden, tuple):
                h = hidden[0]  # (num_layers, batch_size, hidden_size)
            else:
                h = hidden
            h_top = h[-1]  # Top layer hidden state: (batch_size, hidden_size)
        else:
            h_top = torch.zeros(1, self.hidden_size, device=skill_id.device)

        logits = self.output_layer(h_top)
        return logits

    def step_hidden(
        self,
        skill_id: torch.Tensor,
        correct: torch.Tensor,
        hidden: Optional[Any],
    ) -> Any:
        """
        Updates RNN hidden state after observing interaction (skill_id, correct).
        """
        batch_size = 1
        seq_len = 1

        if self.input_mode == "one_hot":
            indices = 2 * skill_id.clamp(0, self.num_skills - 1) + correct.long()
            indices = indices.clamp(0, 2 * self.num_skills - 1)
            input_vec = torch.zeros(1, 1, 2 * self.num_skills, device=skill_id.device)
            input_vec.scatter_(2, indices.unsqueeze(-1), 1.0)
        else:
            skill_emb = self.skill_embedding(skill_id.clamp(0, self.num_skills))
            correct_emb = self.correct_embedding(correct.long().view(1, 1))
            input_vec = self.input_proj(torch.cat([skill_emb, correct_emb], dim=-1))

        if hidden is not None:
            _, new_hidden = self.rnn(input_vec, hidden)
        else:
            _, new_hidden = self.rnn(input_vec)

        return new_hidden

    def predict_step(
        self,
        skill_id: torch.Tensor,
        hidden: Optional[Any],
    ) -> Tuple[torch.Tensor, Any]:
        logits = self.predict_from_hidden(skill_id, hidden)
        return logits, hidden


class DKTOnlineEvaluator:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        if device is None:
            from src.utils.seed import resolve_device
            self.device = resolve_device("auto")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.hidden_states: Dict[int, Optional[Any]] = {}

    def reset_state(self) -> None:
        self.hidden_states = {}

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
            init_h = torch.zeros(1, self.model.hidden_size, device=self.device)
            init_logits = self.model.output_layer(init_h).squeeze(0)
            init_probs = torch.sigmoid(init_logits).cpu().numpy()

            for b_start in range(0, len(student_items), batch_size):
                b_items = student_items[b_start : b_start + batch_size]

                # Step 0: Set cold start predictions
                for _, idx_list in b_items:
                    s0 = skills_arr[idx_list[0]]
                    y_pred[idx_list[0]] = float(init_probs[s0]) if s0 < self.model.num_skills else 0.5

                # Sequences with T > 1
                multi_items = [(stud, idx_list) for stud, idx_list in b_items if len(idx_list) > 1]
                if multi_items:
                    max_len = max(len(idx_list) - 1 for _, idx_list in multi_items)
                    B_multi = len(multi_items)
                    in_skills = torch.zeros((B_multi, max_len), dtype=torch.long, device=self.device)
                    in_corrects = torch.zeros((B_multi, max_len), dtype=torch.float32, device=self.device)
                    tgt_skills = torch.zeros((B_multi, max_len), dtype=torch.long, device=self.device)
                    mask = torch.zeros((B_multi, max_len), dtype=torch.bool, device=self.device)

                    for j, (_, idx_list) in enumerate(multi_items):
                        s_skills = skills_arr[idx_list]
                        s_corrects = corrects_arr[idx_list]
                        L_j = len(idx_list) - 1
                        in_skills[j, :L_j] = torch.from_numpy(s_skills[:-1])
                        in_corrects[j, :L_j] = torch.from_numpy(s_corrects[:-1])
                        tgt_skills[j, :L_j] = torch.from_numpy(s_skills[1:])
                        mask[j, :L_j] = True

                    tgt_logits, _, new_hidden = self.model(
                        skill_ids=in_skills,
                        corrects=in_corrects,
                        target_skill_ids=tgt_skills,
                        mask=mask,
                    )
                    preds_batch = torch.sigmoid(tgt_logits).cpu().numpy()

                    for j, (stud, idx_list) in enumerate(multi_items):
                        L_j = len(idx_list) - 1
                        y_pred[idx_list[1:]] = preds_batch[j, :L_j]
                        last_s = torch.tensor([[skills_arr[idx_list[-1]]]], dtype=torch.long, device=self.device)
                        last_c = torch.tensor([[corrects_arr[idx_list[-1]]]], dtype=torch.long, device=self.device)
                        final_hidden = self.model.step_hidden(last_s, last_c, None)
                        if isinstance(self.model.rnn, nn.LSTM):
                            self.hidden_states[stud] = (final_hidden[0].detach(), final_hidden[1].detach())
                        else:
                            self.hidden_states[stud] = final_hidden.detach()
                else:
                    for stud, idx_list in b_items:
                        last_s = torch.tensor([[skills_arr[idx_list[0]]]], dtype=torch.long, device=self.device)
                        last_c = torch.tensor([[corrects_arr[idx_list[0]]]], dtype=torch.long, device=self.device)
                        final_hidden = self.model.step_hidden(last_s, last_c, None)
                        if isinstance(self.model.rnn, nn.LSTM):
                            self.hidden_states[stud] = (final_hidden[0].detach(), final_hidden[1].detach())
                        else:
                            self.hidden_states[stud] = final_hidden.detach()

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