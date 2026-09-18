import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.data.sequence_builder import SequenceBatch, build_student_sequences, pad_sequences

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        max_seq_len: int = 200,
        pad_skill_id: int = 0,
    ):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.pad_skill_id = pad_skill_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.sequences[idx]


def collate_sequence_batch(
    batch: List[Dict[str, Any]],
    max_seq_len: int,
    pad_skill_id: int = 0,
) -> SequenceBatch:
    return pad_sequences(
        sequences=batch,
        max_seq_len=max_seq_len,
        pad_skill_id=pad_skill_id,
    )


def build_sequence_dataset(
    df,
    student_encoder,
    skill_encoder,
    student_col: str = "studentId",
    skill_col: str = "skill",
    correct_col: str = "correct",
    split_col: str = "split",
    time_col: str = "startTime",
    action_col: str = "action_num",
    max_seq_len: int = 200,
    min_history: int = 1,
    stride: Optional[int] = 100,
) -> Tuple[List[Dict[str, Any]], Callable]:
    temp_df = df.copy()
    temp_df["skill_id_encoded"] = skill_encoder.transform(temp_df[skill_col].values)
    temp_df["student_id_encoded"] = student_encoder.transform(temp_df[student_col].values)

    sequences = build_student_sequences(
        df=temp_df,
        student_col=student_col,
        skill_col=skill_col,
        correct_col=correct_col,
        split_col=split_col,
        time_col=time_col,
        action_col=action_col,
        skill_encoded_col="skill_id_encoded",
        student_encoded_col="student_id_encoded",
        max_seq_len=max_seq_len,
        min_history=min_history,
        stride=stride,
    )

    def collate_fn(batch_list: List[Dict[str, Any]]) -> SequenceBatch:
        return collate_sequence_batch(batch_list, max_seq_len)

    return sequences, collate_fn