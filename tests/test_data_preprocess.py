import numpy as np
import pandas as pd
import pytest

from src.data.eda import clean_column_names, compute_eda_report
from src.data.preprocess import (
    _clean_target,
    _filter_rare_skills,
    _filter_rare_students,
    _handle_multi_skill,
    _handle_noskill,
    _remove_duplicates,
)
from src.data.splits import temporal_per_student_split


class TestDataPreprocess:
    @pytest.fixture
    def toy_df(self):
        return pd.DataFrame(
            {
                "studentId": [1, 1, 1, 1, 2, 2, 2, 2, 2, 3],
                "skill": ["A", "A", "B", "B", "A", "A", "B", "B", "C", "A"],
                "problemId": [101, 102, 103, 104, 201, 202, 203, 204, 205, 301],
                "correct": [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                "startTime": [1, 2, 3, 4, 1, 2, 3, 4, 5, 1],
                "action_num": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

    def test_handle_noskill_drop(self):
        df = pd.DataFrame({"skill": ["algebra", "noskill", "NOSKILL", "geometry"], "problemId": [1, 2, 3, 4]})
        result, dropped = _handle_noskill(df, strategy="drop")
        assert dropped == 2
        assert result["skill"].tolist() == ["algebra", "geometry"]

    def test_handle_noskill_problem_id(self):
        df = pd.DataFrame({"skill": ["algebra", "noskill", "geometry"], "problemId": [101, 102, 103]})
        result, dropped = _handle_noskill(df, strategy="problem_id")
        assert dropped == 0
        assert result["skill"].tolist() == ["algebra", "problem_102", "geometry"]

    def test_handle_noskill_keep(self):
        df = pd.DataFrame({"skill": ["algebra", "noskill"], "problemId": [1, 2]})
        result, dropped = _handle_noskill(df, strategy="keep")
        assert dropped == 0
        assert len(result) == 2

    def test_handle_noskill_invalid_strategy(self):
        df = pd.DataFrame({"skill": ["algebra"]})
        with pytest.raises(ValueError, match="Unknown noskill_strategy"):
            _handle_noskill(df, strategy="invalid")


    def test_clean_target_keeps_only_binary(self):
        df = pd.DataFrame({"correct": [1, 0, 1.0, 0.0, "1", "0", 2, -1, None]})
        result = _clean_target(df)
        assert result["correct"].dtype == np.int8
        assert len(result) == 6
        assert set(result["correct"].unique()) == {0, 1}

    def test_clean_target_no_invalid_values(self):
        df = pd.DataFrame({"correct": [1, 0, 1, 0]})
        result = _clean_target(df)
        assert len(result) == 4

    def test_handle_multi_skill_first_strategy(self):
        df = pd.DataFrame({"skill": ["algebra", "geometry, algebra", "  trig, calc  "]})
        result = _handle_multi_skill(df, strategy="first")
        expected = ["algebra", "geometry", "trig"]
        assert result["skill"].tolist() == expected

    def test_handle_multi_skill_explode_strategy(self):
        df = pd.DataFrame({"skill": ["A,B", "C"], "correct": [1, 0]})
        result = _handle_multi_skill(df, strategy="explode")
        assert len(result) == 3
        assert "A" in result["skill"].values
        assert "B" in result["skill"].values

    def test_handle_multi_skill_invalid_strategy(self):
        df = pd.DataFrame({"skill": ["A"]})
        with pytest.raises(ValueError, match="Unknown multi_skill_strategy"):
            _handle_multi_skill(df, strategy="invalid")

    def test_remove_duplicates(self):
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 1, 1],
                "problemId": [101, 101, 102, 102],
                "startTime": [1, 1, 2, 2],
                "action_num": [1, 1, 1, 2],
            }
        )
        result = _remove_duplicates(df)
        assert len(result) == 3

    def test_filter_rare_students(self):
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 2],
                "skill": ["A", "A", "B"],
                "correct": [1, 0, 1],
                "startTime": [1, 2, 3],
            }
        )
        result = _filter_rare_students(df, min_interactions=2)
        assert result["studentId"].nunique() == 1
        assert 1 in result["studentId"].values

    def test_filter_rare_skills(self):
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 1, 2],
                "skill": ["A", "A", "A", "B"],
                "correct": [1, 0, 1, 1],
                "startTime": [1, 2, 3, 4],
            }
        )
        result = _filter_rare_skills(df, min_interactions=3)
        assert result["skill"].nunique() == 1
        assert "A" in result["skill"].values

    def test_clean_column_names(self):
        df = pd.DataFrame({"student Id": [1], "skill(name)": ["A"], "correct!": [1]})
        result = clean_column_names(df)
        assert "student_Id" in result.columns or "studentId" in "".join(result.columns)

    def test_temporal_per_student_split_preserves_order(self):
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "skill": ["A"] * 10,
                "correct": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "startTime": list(range(10)),
                "action_num": [1] * 10,
            }
        )
        result = temporal_per_student_split(df)
        splits = result[result["studentId"] == 1]["split"].tolist()
        train_idx = [i for i, s in enumerate(splits) if s == "train"]
        test_idx = [i for i, s in enumerate(splits) if s == "test"]
        assert max(train_idx) < min(test_idx) if train_idx and test_idx else True

    def test_temporal_per_student_split_counts(self):
        df = pd.DataFrame(
            {
                "studentId": [1] * 100,
                "skill": ["A"] * 100,
                "correct": [1] * 100,
                "startTime": list(range(100)),
                "action_num": [1] * 100,
            }
        )
        result = temporal_per_student_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        counts = result["split"].value_counts()
        assert counts.get("train", 0) >= 75
        assert counts.get("test", 0) >= 5
        assert counts.get("val", 0) >= 5

    def test_temporal_split_min_sequence_filter(self):
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 2],
                "skill": ["A", "A", "B"],
                "correct": [1, 0, 1],
                "startTime": [1, 2, 3],
                "action_num": [1, 1, 1],
            }
        )
        result = temporal_per_student_split(df, min_sequence_length=3)
        student2 = result[result["studentId"] == 2]
        assert (student2["split"] == "train").all()

    def test_compute_eda_report(self):
        df = pd.DataFrame(
            {
                "studentId": [1, 1, 2],
                "skill": ["A", "B", "C"],
                "problemId": [101, 102, 103],
                "correct": [1, 0, 1],
                "startTime": [1, 2, 3],
                "action_num": [1, 1, 1],
                "timeTaken": [10.0, 20.0, 30.0],
                "hintCount": [0, 1, 0],
                "attemptCount": [1, 2, 1],
                "scaffold": [0, 0, 0],
                "hintTotal": [0, 1, 0],
            }
        )
        report = compute_eda_report(df)
        assert report["n_rows"] == 3
        assert report["n_students"] == 2
        assert report["n_skills"] == 3
