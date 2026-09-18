import numpy as np
import pandas as pd

from src.data.sequence_builder import build_student_sequences, pad_sequences


class TestSequenceBuilder:
    def test_sequences_are_sorted(self):
        df = pd.DataFrame({
            "studentId": [1, 1, 1],
            "skill": ["A", "A", "A"],
            "skill_id_encoded": [1, 1, 1],
            "student_id_encoded": [0, 0, 0],
            "correct": [1, 0, 1],
            "startTime": [3, 1, 2],
            "action_num": [1, 1, 1],
            "split": ["train", "train", "train"],
        })
        seqs = build_student_sequences(df, max_seq_len=10, min_history=1)
        assert len(seqs) == 1
        seq = seqs[0]
        assert seq["length"] == 3
        assert list(seq["corrects"]) == [0.0, 1.0, 1.0]
        assert list(seq["skill_ids"]) == [1, 1, 1]

    def test_train_val_test_split_preserved(self):
        df = pd.DataFrame({
            "studentId": [1, 1, 1, 1],
            "skill": ["A", "A", "A", "A"],
            "skill_id_encoded": [1, 1, 1, 1],
            "student_id_encoded": [0, 0, 0, 0],
            "correct": [1, 0, 1, 0],
            "startTime": [1, 2, 3, 4],
            "action_num": [1, 1, 1, 1],
            "split": ["train", "train", "val", "test"],
        })
        seqs = build_student_sequences(df, max_seq_len=10, min_history=1)
        assert len(seqs) == 1
        seq = seqs[0]
        assert list(seq["splits"]) == [0, 0, 1, 2]

    def test_target_shift_is_correct(self):
        df = pd.DataFrame({
            "studentId": [1, 1, 1],
            "skill": ["A", "A", "A"],
            "skill_id_encoded": [1, 1, 1],
            "student_id_encoded": [0, 0, 0],
            "correct": [1, 0, 1],
            "startTime": [1, 2, 3],
            "action_num": [1, 1, 1],
            "split": ["train", "train", "train"],
        })
        seqs = build_student_sequences(df, max_seq_len=10, min_history=1)
        batch = pad_sequences(seqs, max_seq_len=3)
        assert batch.history_corrects[0, :2].tolist() == [1.0, 0.0]
        assert batch.target_corrects[0, :2].tolist() == [0.0, 1.0]

    def test_no_target_row_in_input(self):
        df = pd.DataFrame({
            "studentId": [1, 1],
            "skill": ["A", "A"],
            "skill_id_encoded": [1, 1],
            "student_id_encoded": [0, 0],
            "correct": [1, 0],
            "startTime": [1, 2],
            "action_num": [1, 1],
            "split": ["train", "train"],
        })
        seqs = build_student_sequences(df, max_seq_len=5, min_history=1)
        batch = pad_sequences(seqs, max_seq_len=5)
        assert batch.history_corrects[0, 0] == 1.0
        assert batch.target_corrects[0, 0] == 0.0

    def test_padding_mask_works(self):
        df = pd.DataFrame({
            "studentId": [1, 1],
            "skill": ["A", "A"],
            "skill_id_encoded": [1, 1],
            "student_id_encoded": [0, 0],
            "correct": [1, 0],
            "startTime": [1, 2],
            "action_num": [1, 1],
            "split": ["train", "train"],
        })
        seqs = build_student_sequences(df, max_seq_len=5, min_history=1)
        batch = pad_sequences(seqs, max_seq_len=5)
        assert batch.mask[0, 0] == True
        assert batch.mask[0, 1:].sum() == 0
        assert batch.mask[0].sum() == 1

    def test_min_history_threshold(self):
        df = pd.DataFrame({
            "studentId": [1, 1],
            "skill": ["A", "A"],
            "skill_id_encoded": [1, 1],
            "student_id_encoded": [0, 0],
            "correct": [1, 0],
            "startTime": [1, 2],
            "action_num": [1, 1],
            "split": ["train", "train"],
        })
        seqs = build_student_sequences(df, max_seq_len=5, min_history=5)
        assert len(seqs) == 0

    def test_max_seq_truncation_keeps_recent(self):
        df = pd.DataFrame({
            "studentId": [1] * 10,
            "skill": ["A"] * 10,
            "skill_id_encoded": [1] * 10,
            "student_id_encoded": [0] * 10,
            "correct": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "startTime": list(range(10)),
            "action_num": [1] * 10,
            "split": ["train"] * 10,
        })
        seqs = build_student_sequences(df, max_seq_len=5, min_history=1)
        batch = pad_sequences(seqs, max_seq_len=5)
        mask = batch.mask[0]
        hist_corrects = batch.history_corrects[0][mask]
        assert len(hist_corrects) == 5

    def test_multiple_students(self):
        df = pd.DataFrame({
            "studentId": [1, 1, 2, 2],
            "skill": ["A", "A", "B", "B"],
            "skill_id_encoded": [1, 1, 2, 2],
            "student_id_encoded": [0, 0, 1, 1],
            "correct": [1, 0, 0, 1],
            "startTime": [1, 2, 1, 2],
            "action_num": [1, 1, 1, 1],
            "split": ["train", "train", "train", "train"],
        })
        seqs = build_student_sequences(df, max_seq_len=5, min_history=1)
        assert len(seqs) == 2

    def test_sliding_window_chunking_stride_100(self):
        N = 350
        df = pd.DataFrame({
            "studentId": [1] * N,
            "skill": ["A"] * N,
            "skill_id_encoded": [1] * N,
            "student_id_encoded": [0] * N,
            "correct": [1 if i % 2 == 0 else 0 for i in range(N)],
            "startTime": list(range(N)),
            "action_num": [1] * N,
            "split": ["train"] * N,
        })
        seqs = build_student_sequences(df, max_seq_len=200, min_history=1, stride=100)
        # N = 350:
        # Chunk 0: start=0, end=201 (len=201)
        # Chunk 1: start=100, end=301 (len=201)
        # Chunk 2: start=200, end=350 (len=150)
        assert len(seqs) == 3
        assert seqs[0]["chunk_id"] == 0
        assert seqs[1]["chunk_id"] == 1
        assert seqs[2]["chunk_id"] == 2
        assert seqs[0]["length"] == 201
        assert seqs[1]["length"] == 201
        assert seqs[2]["length"] == 150

    def test_sliding_window_covers_entire_history(self):
        N = 500
        corrects = [i % 2 for i in range(N)]
        df = pd.DataFrame({
            "studentId": [42] * N,
            "skill": ["Math"] * N,
            "skill_id_encoded": [10] * N,
            "student_id_encoded": [0] * N,
            "correct": corrects,
            "startTime": list(range(N)),
            "action_num": [1] * N,
            "split": ["train"] * N,
        })
        seqs = build_student_sequences(df, max_seq_len=200, min_history=1, stride=100)
        assert len(seqs) == 4 # 0..201, 100..301, 200..401, 300..500

        # Verify that every target token from index 1 to 499 is covered in at least one chunk
        covered_targets = set()
        for seq in seqs:
            # target items are from index 1 to end of chunk
            start_offset = seq["chunk_id"] * 100
            for local_idx in range(1, seq["length"]):
                global_idx = start_offset + local_idx
                covered_targets.add(global_idx)

        expected_targets = set(range(1, N))
        assert covered_targets == expected_targets

    def test_backward_compatibility_stride_none(self):
        N = 300
        df = pd.DataFrame({
            "studentId": [1] * N,
            "skill": ["A"] * N,
            "skill_id_encoded": [1] * N,
            "student_id_encoded": [0] * N,
            "correct": [1] * N,
            "startTime": list(range(N)),
            "action_num": [1] * N,
            "split": ["train"] * N,
        })
        seqs = build_student_sequences(df, max_seq_len=200, min_history=1, stride=None)
        assert len(seqs) == 1
        assert seqs[0]["length"] == 201
        assert seqs[0]["truncated"] == True