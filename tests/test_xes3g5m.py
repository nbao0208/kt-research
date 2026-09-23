from pathlib import Path
import pytest
import torch

from src.data.xes3g5m import (
    XES3G5MDataset,
    XES3G5MMetadata,
    collate_xes3g5m_batch,
    NUM_QUESTIONS_RAW,
    NUM_CONCEPTS_RAW,
)


@pytest.fixture
def xes_paths():
    raw_dir = Path("data/raw/XES3G5M")
    return {
        "train_valid_file": raw_dir / "kc_level" / "train_valid_sequences.csv",
        "metadata_dir": raw_dir / "metadata",
    }


class TestXES3G5MDataPipeline:
    def test_metadata_manager(self, xes_paths):
        metadata = XES3G5MMetadata(xes_paths["metadata_dir"])
        assert len(metadata.questions) == NUM_QUESTIONS_RAW
        assert len(metadata.kc_map) == NUM_CONCEPTS_RAW

        # Test question info retrieval
        q_info = metadata.get_question_info(0)
        assert "content" in q_info
        assert "analysis" in q_info
        assert "type" in q_info

        # Test KC name retrieval
        kc_name = metadata.get_kc_name(0)
        assert isinstance(kc_name, str)
        assert len(kc_name) > 0

    def test_dataset_loading_and_shapes(self, xes_paths):
        dataset = XES3G5MDataset(
            data_file=xes_paths["train_valid_file"],
            folds=[0],
            max_seq_len=200,
            max_samples=20,
        )

        assert len(dataset) == 20
        item = dataset[0]

        assert item["questions"].shape == (200,)
        assert item["concepts"].shape == (200,)
        assert item["responses"].shape == (200,)
        assert item["selectmasks"].shape == (200,)
        assert item["mask"].shape == (200,)
        assert item["uid"].ndim == 0

        # Valid IDs must be >= 0 (0 is PAD, >= 1 are valid IDs)
        assert (item["questions"] >= 0).all()
        assert (item["concepts"] >= 0).all()
        assert (item["responses"] >= 0).all()

    def test_batch_collator(self, xes_paths):
        dataset = XES3G5MDataset(
            data_file=xes_paths["train_valid_file"],
            folds=[0],
            max_seq_len=200,
            max_samples=10,
        )

        batch = collate_xes3g5m_batch([dataset[i] for i in range(4)])
        assert batch["questions"].shape == (4, 200)
        assert batch["concepts"].shape == (4, 200)
        assert batch["responses"].shape == (4, 200)
        assert batch["mask"].shape == (4, 200)
        assert batch["uid"].shape == (4,)

    def test_zero_uid_leakage_between_dataset_instances(self, xes_paths):
        train_ds = XES3G5MDataset(
            data_file=xes_paths["train_valid_file"],
            folds=[0, 1],
            max_samples=200,
        )
        val_ds = XES3G5MDataset(
            data_file=xes_paths["train_valid_file"],
            folds=[2],
            max_samples=100,
        )

        train_uids = set(seq["uid"] for seq in train_ds.sequences)
        val_uids = set(seq["uid"] for seq in val_ds.sequences)

        overlap = train_uids.intersection(val_uids)
        assert len(overlap) == 0, f"Detected UID overlap: {overlap}"

    def test_encoded_metadata_accessors_and_time_gaps(self, xes_paths):
        metadata = XES3G5MMetadata(xes_paths["metadata_dir"])
        # Encoded ID 1 should match raw ID 0
        raw_info_0 = metadata.get_question_info(0)
        enc_info_1 = metadata.get_question_info_from_encoded_id(1)
        assert enc_info_1["content"] == raw_info_0["content"]

        # Encoded ID 0 (PAD) should return safe pad dictionary
        pad_info = metadata.get_question_info_from_encoded_id(0)
        assert pad_info["type"] == "pad"

        # Encoded KC 1 should match raw KC 0
        raw_kc_0 = metadata.get_kc_name(0)
        enc_kc_1 = metadata.get_kc_name_from_encoded_id(1)
        assert enc_kc_1 == raw_kc_0
        assert metadata.get_kc_name_from_encoded_id(0) == "PAD"

        # Check time gaps in dataset
        dataset = XES3G5MDataset(
            data_file=xes_paths["train_valid_file"],
            folds=[0],
            max_seq_len=50,
            max_samples=5,
        )
        item = dataset[0]
        assert "time_gaps" in item
        assert item["time_gaps"].shape == (50,)
        assert (item["time_gaps"] >= 0.0).all()

        batch = collate_xes3g5m_batch([dataset[0], dataset[1]])
        assert "time_gaps" in batch
        assert batch["time_gaps"].shape == (2, 50)
