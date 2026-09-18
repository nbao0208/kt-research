import numpy as np
import pandas as pd
import pytest
import torch

from src.models.attention_kt import AttentionOnlineEvaluator, AttentiveContextualKT
from src.models.bkt import BKT, BKTOnlineEvaluator
from src.models.dbkt import DBKTOnlineEvaluator, DynamicBayesianKnowledgeTracing
from src.models.dkt import DKTOnlineEvaluator, DeepKnowledgeTracing
from src.models.lfa import LFACore, LFAOnlineEvaluator, LFATrainer


class TestFastEvaluators:
    def test_bkt_fast_evaluator_consistency(self):
        bkt_model = BKT(p_init=0.4, p_learn=0.2, p_guess=0.25, p_slip=0.1)
        skill_models = {"skill_1": bkt_model}
        evaluator = BKTOnlineEvaluator(skill_models=skill_models, global_model=bkt_model)

        df = pd.DataFrame({
            "studentId": [101, 101, 102, 102],
            "skill": ["skill_1", "skill_1", "skill_1", "skill_1"],
            "correct": [1, 0, 1, 1],
            "startTime": [1, 2, 1, 2],
        })

        y_true, y_pred, eval_df = evaluator.predict(df)
        assert len(y_pred) == 4
        assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
        assert "prediction" in eval_df.columns
        # First interaction for student 101 uses p_init
        assert np.isclose(y_pred[0], bkt_model.predict(0.4))

    def test_dkt_fast_sequence_prediction_matches_forward(self):
        torch.manual_seed(42)
        model = DeepKnowledgeTracing(num_skills=4, embedding_dim=16, hidden_size=32)
        model.eval()

        evaluator = DKTOnlineEvaluator(model)
        df = pd.DataFrame({
            "studentId": [1, 1, 1, 2, 2],
            "skill": ["s1", "s2", "s1", "s2", "s3"],
            "skill_id_encoded": [0, 1, 0, 1, 2],
            "correct": [1, 0, 1, 0, 1],
            "startTime": [1, 2, 3, 1, 2],
        })

        y_true, y_pred, eval_df = evaluator.predict(df)
        assert len(y_pred) == 5
        assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
        assert 1 in evaluator.hidden_states
        assert 2 in evaluator.hidden_states

    def test_attention_fast_sequence_prediction(self):
        torch.manual_seed(42)
        model = AttentiveContextualKT(num_skills=4, embedding_dim=16, num_heads=2)
        model.eval()

        evaluator = AttentionOnlineEvaluator(model, max_seq_len=10)
        df = pd.DataFrame({
            "studentId": [10, 10, 10, 20],
            "skill": ["s0", "s1", "s2", "s0"],
            "skill_id_encoded": [0, 1, 2, 0],
            "correct": [1, 0, 1, 1],
            "startTime": [1, 2, 3, 1],
        })

        y_true, y_pred, eval_df = evaluator.predict(df)
        assert len(y_pred) == 4
        assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
        assert 10 in evaluator.history_buffers
        assert len(evaluator.history_buffers[10]["skill_ids"]) == 3

    def test_dbkt_numpy_fast_evaluation_consistency(self):
        torch.manual_seed(42)
        model = DynamicBayesianKnowledgeTracing(num_skills=3, feature_dim=2)
        model.eval()

        evaluator = DBKTOnlineEvaluator(model)
        df = pd.DataFrame({
            "studentId": [1, 1, 1],
            "skill": ["skill_A", "skill_A", "skill_A"],
            "correct": [1, 0, 1],
            "startTime": [1, 2, 3],
        })

        y_true, y_pred, eval_df = evaluator.predict(df)
        assert len(y_pred) == 3
        assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
        assert (1, "skill_A") in evaluator.state
        assert 0.0 <= evaluator.state[(1, "skill_A")] <= 1.0

    def test_lfa_batched_evaluation(self):
        model = LFACore(n_students=5, n_skills=5, embedding_dim=8)
        evaluator = LFAOnlineEvaluator(model)

        df = pd.DataFrame({
            "student_id_encoded": [0, 1, 2, 3],
            "skill_id_encoded": [0, 1, 2, 3],
            "success_before": [0.0, 1.0, 2.0, 3.0],
            "failure_before": [0.0, 0.0, 1.0, 0.0],
            "attempts_before": [0.0, 1.0, 3.0, 3.0],
            "correct": [1, 0, 1, 1],
            "split": ["test", "test", "test", "test"],
        })

        preds = evaluator.predict(df, batch_size=2)
        assert len(preds) == 4
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_cross_model_row_alignment_consistency(self):
        from sklearn.metrics import roc_auc_score
        from src.evaluation.plots import plot_roc_curves

        # Multi-skill student interaction dataframe
        df = pd.DataFrame({
            "studentId": [1, 1, 1, 1, 2, 2],
            "skill": ["Math", "English", "Math", "English", "Math", "English"],
            "skill_id_encoded": [0, 1, 0, 1, 0, 1],
            "correct": [1, 0, 1, 1, 0, 1],
            "startTime": [10, 20, 30, 40, 10, 20],
            "action_num": [1, 1, 1, 1, 1, 1],
        })

        # Test chronological sort order
        sort_cols = ["studentId", "startTime", "action_num"]
        df_sorted = df.sort_values(sort_cols).reset_index(drop=True)
        y_true = df_sorted["correct"].values

        bkt = BKT(p_init=0.5, p_learn=0.1, p_guess=0.25, p_slip=0.1)
        bkt_eval = BKTOnlineEvaluator(skill_models={"Math": bkt, "English": bkt})
        _, bkt_preds, df_bkt = bkt_eval.predict(df)

        dkt_model = DeepKnowledgeTracing(num_skills=2, embedding_dim=8, hidden_size=16)
        dkt_eval = DKTOnlineEvaluator(dkt_model)
        _, dkt_preds, df_dkt = dkt_eval.predict(df)

        # Ensure prediction vectors match chronological y_true length and row-order exactly
        assert len(bkt_preds) == len(y_true)
        assert len(dkt_preds) == len(y_true)
        assert np.array_equal(df_bkt["correct"].values, y_true)
        assert np.array_equal(df_dkt["correct"].values, y_true)

        # Ensure ROC-AUC calculated on y_true matches df_eval ROC-AUC exactly
        bkt_auc_direct = roc_auc_score(y_true, bkt_preds)
        bkt_auc_df = roc_auc_score(df_bkt["correct"].values, df_bkt["prediction"].values)
        assert np.isclose(bkt_auc_direct, bkt_auc_df)

    def test_student_average_baseline_predicts_correctly(self):
        from src.evaluation.metrics import compute_metrics
        from src.models.baselines import MajorityBaseline, StudentAverageBaseline
        train_df = pd.DataFrame({
            "studentId": [101, 101, 101, 102, 102, 102],
            "correct": [1, 1, 1, 0, 0, 0],
        })
        val_df = pd.DataFrame({
            "studentId": [101, 102],
            "correct": [1, 0],
        })

        maj = MajorityBaseline().fit(train_df)
        maj_preds = maj.predict(val_df)
        assert np.all(maj_preds == 0.5)

        student_avg = StudentAverageBaseline().fit(train_df)
        preds = student_avg.predict(val_df)
        # Student 101 has 1.0 train accuracy, Student 102 has 0.0 train accuracy
        assert preds[0] == 1.0
        assert preds[1] == 0.0
        # Perfect discrimination on val set
        metrics = compute_metrics(val_df["correct"].values, preds)
        assert metrics["auc"] == 1.0

