import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_ece, compute_metrics, compute_skill_metrics


class TestMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.99, 0.01, 0.99, 0.01])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["auc"] > 0.9
        assert metrics["accuracy"] > 0.9
        assert metrics["logloss"] < 0.1

    def test_random_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["auc"] == 0.5
        assert metrics["logloss"] > 0.69
        assert pytest.approx(metrics["brier"], rel=0.1) == 0.25

    def test_empty_input(self):
        metrics = compute_metrics(np.array([]), np.array([]))
        assert metrics["auc"] == 0.0
        assert metrics["n"] == 0

    def test_single_class(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.95])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["auc"] == 0.5

    def test_inverted_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.1, 0.9])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["auc"] < 0.5

    def test_brier_score(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2])
        metrics = compute_metrics(y_true, y_pred)
        assert pytest.approx(metrics["brier"], abs=0.02) == 0.04

    def test_logloss_penalizes_overconfidence(self):
        y_true = np.array([1, 0])
        y_pred_confident = np.array([0.999, 0.001])
        y_pred_cautious = np.array([0.7, 0.3])
        metrics_confident = compute_metrics(y_true, y_pred_confident)
        metrics_cautious = compute_metrics(y_true, y_pred_cautious)
        assert metrics_confident["logloss"] < metrics_cautious["logloss"]


class TestSkillMetrics:
    def test_per_skill_metrics(self):
        df = pd.DataFrame(
            {
                "correct": [1, 0, 1, 0, 1, 0],
                "prediction": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
                "skill": ["A", "A", "A", "A", "B", "B"],
            }
        )
        metrics = compute_skill_metrics(df)
        assert "per_skill" in metrics
        assert "A" in metrics["per_skill"]
        assert "B" in metrics["per_skill"]
        assert metrics["skill_macro_auc"] > 0.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["correct", "prediction", "skill"])
        metrics = compute_skill_metrics(df)
        assert metrics["skill_macro_auc"] == 0.0


class TestECE:
    def test_perfectly_calibrated(self):
        np.random.seed(42)
        n = 1000
        y_pred = np.random.uniform(0.1, 0.9, n)
        y_true = (np.random.random(n) < y_pred).astype(int)
        ece = compute_ece(y_true, y_pred, n_bins=10)
        assert ece < 0.05

    def test_badly_calibrated(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        ece = compute_ece(y_true, y_pred, n_bins=5)
        assert ece > 0.1

    def test_single_class_skill_excluded_from_macro_auc(self):
        # Skill A is multi-class with AUC = 1.0; Skill B has only 1s (single-class)
        df = pd.DataFrame({
            "correct": [1, 0, 1, 0, 1, 1, 1],
            "prediction": [0.9, 0.1, 0.9, 0.1, 0.8, 0.8, 0.8],
            "skill": ["A", "A", "A", "A", "B", "B", "B"],
        })
        metrics = compute_skill_metrics(df)
        assert metrics["n_evaluable_skills"] == 1
        assert metrics["n_single_class_skills"] == 1
        # Skill A has AUC = 1.0, so macro AUC should be 1.0 instead of being dragged down by Skill B (0.5)
        assert metrics["skill_macro_auc"] == 1.0

    def test_student_clustered_bootstrap(self):
        from src.evaluation.bootstrap import student_clustered_bootstrap_auc
        student_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        res = student_clustered_bootstrap_auc(student_ids, y_true, y_pred, n_bootstrap=100, seed=42)
        assert res["auc"] == 1.0
        assert res["sampling_unit"] == "student"
