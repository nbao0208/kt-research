import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Constants for XES3G5M
NUM_QUESTIONS_RAW = 7652
NUM_CONCEPTS_RAW = 1175
DEFAULT_SEQ_LEN = 200


class XES3G5MMetadata:
    """
    Manager for XES3G5M auxiliary metadata including question text,
    solutions, KC hierarchy, and pretrained RoBERTa embeddings.
    Provides safe accessors mapping both 0-based raw IDs and 1-based encoded dataset IDs.
    """

    def __init__(self, metadata_dir: Union[str, Path]):
        self.metadata_dir = Path(metadata_dir)
        self.questions_file = self.metadata_dir / "questions.json"
        self.kc_map_file = self.metadata_dir / "kc_routes_map.json"
        self.embeddings_dir = self.metadata_dir / "embeddings"

        self._questions: Optional[Dict[str, Any]] = None
        self._kc_map: Optional[Dict[str, str]] = None
        self._qid2content_emb: Optional[Dict[str, List[float]]] = None
        self._qid2analysis_emb: Optional[Dict[str, List[float]]] = None
        self._cid2content_emb: Optional[Dict[str, List[float]]] = None

    @property
    def questions(self) -> Dict[str, Any]:
        if self._questions is None:
            if self.questions_file.exists():
                logger.info("Loading questions metadata from %s", self.questions_file)
                with open(self.questions_file, "r", encoding="utf-8") as f:
                    self._questions = json.load(f)
            else:
                logger.warning("Questions metadata file not found at %s", self.questions_file)
                self._questions = {}
        return self._questions

    @property
    def kc_map(self) -> Dict[str, str]:
        if self._kc_map is None:
            if self.kc_map_file.exists():
                logger.info("Loading KC routes map from %s", self.kc_map_file)
                with open(self.kc_map_file, "r", encoding="utf-8") as f:
                    self._kc_map = json.load(f)
            else:
                logger.warning("KC routes map file not found at %s", self.kc_map_file)
                self._kc_map = {}
        return self._kc_map

    # ---------------- Raw 0-based ID Accessors ----------------
    def get_question_info(self, qid: Union[int, str]) -> Dict[str, Any]:
        """Retrieve question content, analysis, answer, options, and KC routes by raw 0-based ID."""
        str_qid = str(qid)
        return self.questions.get(str_qid, {
            "content": f"Question {qid}",
            "analysis": "No solution analysis available.",
            "answer": [],
            "options": {},
            "type": "unknown",
            "kc_routes": [],
        })

    def get_kc_name(self, cid: Union[int, str]) -> str:
        """Retrieve KC textual name/path from raw 0-based KC ID."""
        str_cid = str(cid)
        return self.kc_map.get(str_cid, f"Concept_{cid}")

    def get_qid_content_emb(self, qid: Union[int, str]) -> Optional[List[float]]:
        if self._qid2content_emb is None:
            emb_file = self.embeddings_dir / "qid2content_emb.json"
            if emb_file.exists():
                with open(emb_file, "r", encoding="utf-8") as f:
                    self._qid2content_emb = json.load(f)
            else:
                self._qid2content_emb = {}
        return self._qid2content_emb.get(str(qid))

    def get_qid_analysis_emb(self, qid: Union[int, str]) -> Optional[List[float]]:
        if self._qid2analysis_emb is None:
            emb_file = self.embeddings_dir / "qid2analysis_emb.json"
            if emb_file.exists():
                with open(emb_file, "r", encoding="utf-8") as f:
                    self._qid2analysis_emb = json.load(f)
            else:
                self._qid2analysis_emb = {}
        return self._qid2analysis_emb.get(str(qid))

    def get_cid_content_emb(self, cid: Union[int, str]) -> Optional[List[float]]:
        if self._cid2content_emb is None:
            emb_file = self.embeddings_dir / "cid2content_emb.json"
            if emb_file.exists():
                with open(emb_file, "r", encoding="utf-8") as f:
                    self._cid2content_emb = json.load(f)
            else:
                self._cid2content_emb = {}
        return self._cid2content_emb.get(str(cid))

    # ---------------- 1-based Encoded Dataset ID Accessors ----------------
    def get_question_info_from_encoded_id(self, encoded_qid: Union[int, str]) -> Dict[str, Any]:
        """
        Safely retrieve question metadata from 1-based encoded dataset ID.
        If encoded_qid <= 0 (PAD), returns safe default dictionary.
        """
        try:
            int_qid = int(encoded_qid)
        except (ValueError, TypeError):
            int_qid = 0
        if int_qid <= 0:
            return {
                "content": "Padding interaction",
                "analysis": "None",
                "answer": [],
                "options": {},
                "type": "pad",
                "kc_routes": [],
            }
        raw_qid = int_qid - 1
        return self.get_question_info(raw_qid)

    def get_kc_name_from_encoded_id(self, encoded_cid: Union[int, str]) -> str:
        """
        Safely retrieve concept name from 1-based encoded dataset ID.
        If encoded_cid <= 0 (PAD), returns 'PAD'.
        """
        try:
            int_cid = int(encoded_cid)
        except (ValueError, TypeError):
            int_cid = 0
        if int_cid <= 0:
            return "PAD"
        raw_cid = int_cid - 1
        return self.get_kc_name(raw_cid)

    def get_qid_content_emb_from_encoded_id(self, encoded_qid: Union[int, str]) -> Optional[List[float]]:
        try:
            int_qid = int(encoded_qid)
        except (ValueError, TypeError):
            int_qid = 0
        if int_qid <= 0:
            return None
        return self.get_qid_content_emb(int_qid - 1)

    def get_qid_analysis_emb_from_encoded_id(self, encoded_qid: Union[int, str]) -> Optional[List[float]]:
        try:
            int_qid = int(encoded_qid)
        except (ValueError, TypeError):
            int_qid = 0
        if int_qid <= 0:
            return None
        return self.get_qid_analysis_emb(int_qid - 1)

    def get_cid_content_emb_from_encoded_id(self, encoded_cid: Union[int, str]) -> Optional[List[float]]:
        try:
            int_cid = int(encoded_cid)
        except (ValueError, TypeError):
            int_cid = 0
        if int_cid <= 0:
            return None
        return self.get_cid_content_emb(int_cid - 1)


class XES3G5MDataset(Dataset):
    """
    PyTorch Dataset for XES3G5M sequence interactions with zero data leakage.
    Encodes padding as ID 0, shifting valid question/concept IDs by +1.
    Includes temporal time-gap calculation delta_tau_t = log(1 + max(0, t_t - t_{t-1})).
    """

    def __init__(
        self,
        data_file: Union[str, Path],
        folds: Optional[List[int]] = None,
        max_seq_len: int = DEFAULT_SEQ_LEN,
        num_questions: int = NUM_QUESTIONS_RAW,
        num_concepts: int = NUM_CONCEPTS_RAW,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_file = Path(data_file)
        self.folds = folds
        self.max_seq_len = max_seq_len
        self.num_questions = num_questions
        self.num_concepts = num_concepts

        if not self.data_file.exists():
            raise FileNotFoundError(f"XES3G5M data file not found at: {self.data_file}")

        self.sequences = self._load_and_filter(max_samples=max_samples)
        logger.info(
            "XES3G5MDataset loaded %d sequences from %s (folds=%s)",
            len(self.sequences),
            self.data_file.name,
            folds,
        )

    def _load_and_filter(self, max_samples: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        df = pd.read_csv(self.data_file)

        # Filter by fold if specified
        if self.folds is not None and "fold" in df.columns:
            df = df[df["fold"].isin(self.folds)].reset_index(drop=True)

        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples].reset_index(drop=True)

        parsed_records = []
        for idx, row in df.iterrows():
            q_arr = np.fromstring(row["questions"], sep=",", dtype=np.int64)
            c_arr = np.fromstring(row["concepts"], sep=",", dtype=np.int64)
            r_arr = np.fromstring(row["responses"], sep=",", dtype=np.int64)
            t_arr = np.fromstring(row["timestamps"], sep=",", dtype=np.int64) if "timestamps" in row and pd.notna(row["timestamps"]) else np.zeros_like(q_arr)
            s_arr = np.fromstring(row["selectmasks"], sep=",", dtype=np.int64) if "selectmasks" in row and pd.notna(row["selectmasks"]) else np.ones_like(q_arr)
            rep_arr = np.fromstring(row["is_repeat"], sep=",", dtype=np.int64) if "is_repeat" in row and pd.notna(row["is_repeat"]) else np.zeros_like(q_arr)

            # Slicing or padding to fixed max_seq_len
            length = len(q_arr)
            if length > self.max_seq_len:
                q_arr = q_arr[:self.max_seq_len]
                c_arr = c_arr[:self.max_seq_len]
                r_arr = r_arr[:self.max_seq_len]
                t_arr = t_arr[:self.max_seq_len]
                s_arr = s_arr[:self.max_seq_len]
                rep_arr = rep_arr[:self.max_seq_len]
            elif length < self.max_seq_len:
                pad_len = self.max_seq_len - length
                q_arr = np.pad(q_arr, (0, pad_len), constant_values=-1)
                c_arr = np.pad(c_arr, (0, pad_len), constant_values=-1)
                r_arr = np.pad(r_arr, (0, pad_len), constant_values=-1)
                t_arr = np.pad(t_arr, (0, pad_len), constant_values=-1)
                s_arr = np.pad(s_arr, (0, pad_len), constant_values=-1)
                rep_arr = np.pad(rep_arr, (0, pad_len), constant_values=-1)

            # Valid interaction mask: elements where question != -1
            mask = (q_arr != -1)

            # ID re-indexing: 0 is PAD, valid IDs shifted by +1
            q_encoded = np.where(q_arr != -1, q_arr + 1, 0)
            c_encoded = np.where(c_arr != -1, c_arr + 1, 0)
            r_clean = np.where(r_arr != -1, r_arr, 0)

            # Time gaps: log(1 + max(0, t_t - t_{t-1}))
            time_gaps = np.zeros_like(t_arr, dtype=np.float32)
            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 1:
                valid_timestamps = t_arr[valid_idx]
                diffs = np.diff(valid_timestamps, prepend=valid_timestamps[0])
                diffs = np.maximum(diffs, 0)
                time_gaps[valid_idx] = np.log1p(diffs.astype(np.float32))

            parsed_records.append({
                "uid": int(row.get("uid", idx)),
                "fold": int(row.get("fold", -1)),
                "questions": q_encoded,
                "concepts": c_encoded,
                "responses": r_clean,
                "raw_responses": r_arr,
                "selectmasks": s_arr,
                "timestamps": t_arr,
                "time_gaps": time_gaps,
                "is_repeat": rep_arr,
                "mask": mask,
            })

        return parsed_records

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.sequences[idx]
        return {
            "uid": torch.tensor(rec["uid"], dtype=torch.long),
            "fold": torch.tensor(rec["fold"], dtype=torch.long),
            "questions": torch.from_numpy(rec["questions"]).long(),
            "concepts": torch.from_numpy(rec["concepts"]).long(),
            "responses": torch.from_numpy(rec["responses"]).long(),
            "raw_responses": torch.from_numpy(rec["raw_responses"]).long(),
            "selectmasks": torch.from_numpy(rec["selectmasks"]).long(),
            "timestamps": torch.from_numpy(rec["timestamps"]).long(),
            "time_gaps": torch.from_numpy(rec["time_gaps"]).float(),
            "is_repeat": torch.from_numpy(rec["is_repeat"]).long(),
            "mask": torch.from_numpy(rec["mask"]).bool(),
        }


class XES3G5MQuestionLevelDataset(XES3G5MDataset):
    """
    Dataset variant specifically formatted for Question-Level evaluation.
    Supports reading files from `data/raw/XES3G5M/question_level/` where
    concepts can be composite strings (e.g. '62_65').
    """

    def _load_and_filter(self, max_samples: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        df = pd.read_csv(self.data_file)

        if self.folds is not None and "fold" in df.columns:
            df = df[df["fold"].isin(self.folds)].reset_index(drop=True)

        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples].reset_index(drop=True)

        parsed_records = []
        for idx, row in df.iterrows():
            q_arr = np.fromstring(row["questions"], sep=",", dtype=np.int64)
            r_arr = np.fromstring(row["responses"], sep=",", dtype=np.int64)
            t_arr = np.fromstring(row["timestamps"], sep=",", dtype=np.int64) if "timestamps" in row and pd.notna(row["timestamps"]) else np.zeros_like(q_arr)

            # In question_level CSVs, concepts can be multi-concept strings like "62_65"
            # Map first concept or primary concept as default concept representation
            c_str_list = str(row["concepts"]).split(",")
            c_list = []
            for item in c_str_list:
                item_clean = item.strip()
                if "_" in item_clean:
                    # Multi-concept: take first primary concept ID
                    primary_c = item_clean.split("_")[0]
                    c_list.append(int(primary_c) if primary_c.isdigit() else 0)
                elif item_clean.isdigit():
                    c_list.append(int(item_clean))
                else:
                    c_list.append(0)
            c_arr = np.array(c_list, dtype=np.int64)

            # Ensure equal lengths
            min_len = min(len(q_arr), len(c_arr), len(r_arr), len(t_arr))
            q_arr = q_arr[:min_len]
            c_arr = c_arr[:min_len]
            r_arr = r_arr[:min_len]
            t_arr = t_arr[:min_len]

            # In question level, each interaction represents a unique question step
            s_arr = np.ones_like(q_arr)
            rep_arr = np.zeros_like(q_arr)

            # Slicing or padding
            length = len(q_arr)
            if length > self.max_seq_len:
                q_arr = q_arr[:self.max_seq_len]
                c_arr = c_arr[:self.max_seq_len]
                r_arr = r_arr[:self.max_seq_len]
                t_arr = t_arr[:self.max_seq_len]
                s_arr = s_arr[:self.max_seq_len]
                rep_arr = rep_arr[:self.max_seq_len]
            elif length < self.max_seq_len:
                pad_len = self.max_seq_len - length
                q_arr = np.pad(q_arr, (0, pad_len), constant_values=-1)
                c_arr = np.pad(c_arr, (0, pad_len), constant_values=-1)
                r_arr = np.pad(r_arr, (0, pad_len), constant_values=-1)
                t_arr = np.pad(t_arr, (0, pad_len), constant_values=-1)
                s_arr = np.pad(s_arr, (0, pad_len), constant_values=-1)
                rep_arr = np.pad(rep_arr, (0, pad_len), constant_values=-1)

            mask = (q_arr != -1)
            q_encoded = np.where(q_arr != -1, q_arr + 1, 0)
            c_encoded = np.where(c_arr != -1, c_arr + 1, 0)
            r_clean = np.where(r_arr != -1, r_arr, 0)

            time_gaps = np.zeros_like(t_arr, dtype=np.float32)
            valid_idx = np.where(mask)[0]
            if len(valid_idx) > 1:
                valid_timestamps = t_arr[valid_idx]
                diffs = np.diff(valid_timestamps, prepend=valid_timestamps[0])
                diffs = np.maximum(diffs, 0)
                time_gaps[valid_idx] = np.log1p(diffs.astype(np.float32))

            parsed_records.append({
                "uid": int(row.get("uid", idx)),
                "fold": int(row.get("fold", -1)),
                "questions": q_encoded,
                "concepts": c_encoded,
                "responses": r_clean,
                "raw_responses": r_arr,
                "selectmasks": s_arr,
                "timestamps": t_arr,
                "time_gaps": time_gaps,
                "is_repeat": rep_arr,
                "mask": mask,
            })

        return parsed_records


def collate_xes3g5m_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate list of sequence items into batched tensors."""
    res = {
        "uid": torch.stack([item["uid"] for item in batch]),
        "fold": torch.stack([item["fold"] for item in batch]),
        "questions": torch.stack([item["questions"] for item in batch]),
        "concepts": torch.stack([item["concepts"] for item in batch]),
        "responses": torch.stack([item["responses"] for item in batch]),
        "raw_responses": torch.stack([item["raw_responses"] for item in batch]),
        "selectmasks": torch.stack([item["selectmasks"] for item in batch]),
        "timestamps": torch.stack([item["timestamps"] for item in batch]),
        "is_repeat": torch.stack([item["is_repeat"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }
    if "time_gaps" in batch[0]:
        res["time_gaps"] = torch.stack([item["time_gaps"] for item in batch])
    return res
