import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class SequenceBatch:
    student_ids: torch.Tensor
    history_skill_ids: torch.Tensor
    history_corrects: torch.Tensor
    target_skill_ids: torch.Tensor
    target_corrects: torch.Tensor
    mask: torch.Tensor
    split_labels: torch.Tensor
    aux_features: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "SequenceBatch":
        return SequenceBatch(
            student_ids=self.student_ids.to(device),
            history_skill_ids=self.history_skill_ids.to(device),
            history_corrects=self.history_corrects.to(device),
            target_skill_ids=self.target_skill_ids.to(device),
            target_corrects=self.target_corrects.to(device),
            mask=self.mask.to(device),
            split_labels=self.split_labels.to(device),
            aux_features=self.aux_features.to(device) if self.aux_features is not None else None,
        )


def build_student_sequences(
    df: pd.DataFrame,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    split_col: str = "split",
    time_col: str = "startTime",
    action_col: str = "action_num",
    skill_encoded_col: str = "skill_id_encoded",
    student_encoded_col: str = "student_id_encoded",
    max_seq_len: int = 200,
    min_history: int = 1,
    stride: Optional[int] = 100,
) -> List[Dict[str, Any]]:
    sort_cols = [student_col, time_col]
    if action_col in df.columns:
        sort_cols.append(action_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    split_map = {"train": 0, "val": 1, "test": 2}

    sequences = []
    for student_id, group in df.groupby(student_col):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < min_history + 1:
            continue

        skill_ids = group[skill_encoded_col].values.astype(np.int64)
        corrects = group[correct_col].values.astype(np.float32)
        splits = group[split_col].map(split_map).fillna(0).values.astype(np.int64)
        encoded_student = group[student_encoded_col].iloc[0]

        aux_cols = [
            c for c in group.columns
            if c not in [
                student_col, skill_col, correct_col, split_col,
                time_col, action_col, skill_encoded_col, student_encoded_col,
            ]
        ]
        aux_values = group[aux_cols].values.astype(np.float32) if aux_cols else None

        if stride is not None and n > max_seq_len + 1:
            # Sliding window chunking
            start = 0
            chunk_id = 0
            while start < n - 1:
                end = min(start + max_seq_len + 1, n)
                chunk_len = end - start
                if chunk_len >= min_history + 1:
                    sequences.append({
                        "student_id": int(student_id),
                        "encoded_student_id": int(encoded_student),
                        "chunk_id": chunk_id,
                        "skill_ids": skill_ids[start:end],
                        "corrects": corrects[start:end],
                        "splits": splits[start:end],
                        "length": chunk_len,
                        "aux_values": aux_values[start:end] if aux_values is not None else None,
                        "truncated": True,
                    })
                    chunk_id += 1
                if end == n:
                    break
                start += stride
        else:
            truncated = n > max_seq_len + 1
            if truncated:
                skill_ids = skill_ids[-max_seq_len - 1:]
                corrects = corrects[-max_seq_len - 1:]
                splits = splits[-max_seq_len - 1:]
                if aux_values is not None:
                    aux_values = aux_values[-max_seq_len - 1:]
                n = len(skill_ids)

            sequences.append({
                "student_id": int(student_id),
                "encoded_student_id": int(encoded_student),
                "chunk_id": 0,
                "skill_ids": skill_ids,
                "corrects": corrects,
                "splits": splits,
                "length": n,
                "aux_values": aux_values,
                "truncated": truncated,
            })

    return sequences


def pad_sequences(
    sequences: List[Dict[str, Any]],
    max_seq_len: int,
    pad_skill_id: int = 0,
    pad_aux: float = 0.0,
) -> SequenceBatch:
    batch_size = len(sequences)
    target_len = max_seq_len + 1

    student_ids = np.zeros(batch_size, dtype=np.int64)
    history_skill_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    history_corrects = np.zeros((batch_size, max_seq_len), dtype=np.float32)
    target_skill_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    target_corrects = np.zeros((batch_size, max_seq_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_seq_len), dtype=bool)
    split_labels = np.zeros((batch_size, max_seq_len), dtype=np.int64)

    aux_dim = None
    aux_data = None
    if sequences[0].get("aux_values") is not None:
        aux_dim = sequences[0]["aux_values"].shape[1]
        aux_data = np.full((batch_size, max_seq_len, aux_dim), pad_aux, dtype=np.float32)

    for i, seq in enumerate(sequences):
        student_ids[i] = seq["encoded_student_id"]
        n = seq["length"]
        n_hist = min(n - 1, max_seq_len)

        if n_hist < 1:
            continue

        hist_skills = seq["skill_ids"][:n_hist]
        hist_corrects = seq["corrects"][:n_hist]
        tgt_skills = seq["skill_ids"][1:n_hist + 1]
        tgt_corrects = seq["corrects"][1:n_hist + 1]
        seq_splits = seq["splits"][1:n_hist + 1]

        history_skill_ids[i, :n_hist] = hist_skills
        history_corrects[i, :n_hist] = hist_corrects
        target_skill_ids[i, :n_hist] = tgt_skills
        target_corrects[i, :n_hist] = tgt_corrects
        mask[i, :n_hist] = True
        split_labels[i, :n_hist] = seq_splits

        if aux_data is not None and seq["aux_values"] is not None:
            aux_data[i, :n_hist, :] = seq["aux_values"][:n_hist]

    return SequenceBatch(
        student_ids=torch.from_numpy(student_ids),
        history_skill_ids=torch.from_numpy(history_skill_ids),
        history_corrects=torch.from_numpy(history_corrects),
        target_skill_ids=torch.from_numpy(target_skill_ids),
        target_corrects=torch.from_numpy(target_corrects),
        mask=torch.from_numpy(mask),
        split_labels=torch.from_numpy(split_labels),
        aux_features=torch.from_numpy(aux_data) if aux_data is not None else None,
    )