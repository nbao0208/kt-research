import numpy as np
import pandas as pd
import pytest
import torch

from src.data.sequence import build_dbkt_sequences
from src.features.encoders import LabelEncoderWithUNK
from src.models.dbkt import DBKTOnlineEvaluator, DBKTTrainer, DynamicBayesianKnowledgeTracing, save_dbkt_params


class TestDBKT:
    def test_probabilities_in_range(self):
        model = DynamicBayesianKnowledgeTracing(num_skills=3)
        for i in range(3):
            p_init = model.get_p_init(torch.LongTensor([i])).item()
            p_slip = model.get_p_slip(torch.LongTensor([i])).item()
            p_guess = model.get_p_guess(torch.LongTensor([i])).item()
            assert 0.0 <= p_init <= 1.0
            assert 0.0 <= p_slip <= 1.0
            assert 0.0 <= p_guess <= 1.0

    def test_predict_p_correct_in_range(self):
        model = DynamicBayesianKnowledgeTracing(num_skills=3)
        for p_known in [0.0, 0.5, 1.0]:
            for skill in range(3):
                p_correct = model.predict(torch.tensor([p_known]), torch.LongTensor([skill]))
                assert 0.0 <= p_correct.item() <= 1.0

    def test_update_increases_knowledge_after_correct(self):
        model = DynamicBayesianKnowledgeTracing(num_skills=3)
        p_known = torch.tensor([0.3])
        skill_id = torch.LongTensor([0])
        correct = torch.tensor([1.0])
        p_next = model.update(p_known, correct, skill_id)
        assert p_next.item() >= p_known.item()

    def test_sequence_log_likelihood_with_features(self):
        model = DynamicBayesianKnowledgeTracing(num_skills=3, feature_dim=2)
        skill_ids = torch.LongTensor([0, 1])
        corrects = torch.FloatTensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        features = torch.zeros((2, 3, 2), dtype=torch.float32)
        loss = model.compute_loss(skill_ids, corrects, features)
        assert loss.item() > 0

    def test_build_dbkt_sequences_and_trainer_fit(self, tmp_path):
        df = pd.DataFrame({
            "studentId": [1, 1, 1, 1, 2, 2, 2, 2],
            "skill": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "correct": [1, 0, 1, 1, 0, 1, 0, 1],
            "startTime": [1, 2, 3, 4, 1, 2, 3, 4],
            "action_num": [1, 1, 1, 1, 1, 1, 1, 1],
            "split": ["train"] * 8,
        })

        skill_encoder = LabelEncoderWithUNK()
        skill_encoder.fit(df["skill"].unique())

        seqs = build_dbkt_sequences(df, skill_encoder=skill_encoder, min_sequence_length=3)
        assert len(seqs) == 2
        assert "features" in seqs[0]
        assert seqs[0]["features"].shape == (4, 2)

        model = DynamicBayesianKnowledgeTracing(num_skills=len(skill_encoder.encoder.classes_), feature_dim=2)
        trainer = DBKTTrainer(model=model, max_epochs=10, learning_rate=0.01, seed=42)
        skill_models, global_model = trainer.fit(seqs)

        assert len(skill_models) == 2
        assert "A" in skill_models and "B" in skill_models

        # Test save params
        params_file = tmp_path / "dbkt_params.csv"
        save_dbkt_params(skill_models, global_model, params_file, skill_encoder=skill_encoder)
        assert params_file.exists()
        saved_df = pd.read_csv(params_file)
        assert len(saved_df) >= 2

    def test_online_evaluator_with_skill_encoder(self):
        df = pd.DataFrame({
            "studentId": [1, 1, 1, 1],
            "skill": ["A", "A", "B", "A"],
            "correct": [1, 0, 1, 1],
            "startTime": [1, 2, 3, 4],
            "action_num": [1, 1, 1, 1],
        })
        skill_encoder = LabelEncoderWithUNK()
        skill_encoder.fit(["A", "B"])

        model = DynamicBayesianKnowledgeTracing(num_skills=2, feature_dim=2)
        evaluator = DBKTOnlineEvaluator(model=model, skill_encoder=skill_encoder)
        metrics = evaluator.evaluate(df)

        assert "global" in metrics
        assert "auc" in metrics["global"]
        assert len(evaluator.state) == 2  # (1, "A") and (1, "B")

    def test_trainer_fit_with_validation_auc(self):
        df_train = pd.DataFrame({
            "studentId": [1, 1, 1, 1, 2, 2, 2, 2],
            "skill": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "correct": [1, 0, 1, 1, 0, 1, 0, 1],
            "startTime": [1, 2, 3, 4, 1, 2, 3, 4],
            "action_num": [1, 1, 1, 1, 1, 1, 1, 1],
            "split": ["train"] * 8,
        })
        df_val = pd.DataFrame({
            "studentId": [3, 3, 3, 3],
            "skill": ["A", "A", "B", "B"],
            "correct": [1, 1, 0, 1],
            "startTime": [1, 2, 3, 4],
            "action_num": [1, 1, 1, 1],
            "split": ["val"] * 4,
        })

        skill_encoder = LabelEncoderWithUNK()
        skill_encoder.fit(["A", "B"])

        train_seqs = build_dbkt_sequences(df_train, skill_encoder=skill_encoder, min_sequence_length=3)
        val_seqs = build_dbkt_sequences(df_val, skill_encoder=skill_encoder, min_sequence_length=3)

        model = DynamicBayesianKnowledgeTracing(num_skills=2, feature_dim=2)
        trainer = DBKTTrainer(model=model, max_epochs=5, learning_rate=0.01, seed=42)
        trainer.fit(train_seqs, val_sequences=val_seqs)

        assert trainer.best_val_auc >= 0.0

    def test_trainer_fit_with_val_df(self):
        df_train = pd.DataFrame({
            "studentId": [1, 1, 1, 1, 2, 2, 2, 2],
            "skill": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "correct": [1, 0, 1, 1, 0, 1, 0, 1],
            "startTime": [1, 2, 3, 4, 1, 2, 3, 4],
            "action_num": [1, 1, 1, 1, 1, 1, 1, 1],
            "split": ["train"] * 8,
        })
        df_val = pd.DataFrame({
            "studentId": [3, 3, 3, 3],
            "skill": ["A", "A", "B", "B"],
            "correct": [1, 1, 0, 1],
            "startTime": [1, 2, 3, 4],
            "action_num": [1, 1, 1, 1],
            "split": ["val"] * 4,
        })

        skill_encoder = LabelEncoderWithUNK()
        skill_encoder.fit(["A", "B"])

        train_seqs = build_dbkt_sequences(df_train, skill_encoder=skill_encoder, min_sequence_length=3)

        model = DynamicBayesianKnowledgeTracing(num_skills=2, feature_dim=2)
        trainer = DBKTTrainer(model=model, max_epochs=5, learning_rate=0.01, seed=42)
        trainer.fit(train_seqs, val_df=df_val, skill_encoder=skill_encoder)

        assert trainer.best_val_auc >= 0.0