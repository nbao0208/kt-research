import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lfa import LFAAux, LFACore, LFATrainer


class TestLFACore:
    def test_output_shape(self):
        model = LFACore(n_students=10, n_skills=5, embedding_dim=8)
        batch_size = 4
        student_ids = torch.LongTensor([0, 1, 2, 3])
        skill_ids = torch.LongTensor([0, 0, 1, 2])
        sb = torch.FloatTensor([0, 1, 1, 0])
        fb = torch.FloatTensor([0, 0, 1, 1])
        ab = torch.FloatTensor([0, 1, 2, 1])
        output = model(student_ids, skill_ids, sb, fb, ab)
        assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"

    def test_output_range(self):
        model = LFACore(n_students=10, n_skills=5, embedding_dim=8)
        student_ids = torch.LongTensor([1, 2])
        skill_ids = torch.LongTensor([0, 1])
        sb = torch.FloatTensor([0, 1])
        fb = torch.FloatTensor([0, 0])
        ab = torch.FloatTensor([0, 1])
        output = model(student_ids, skill_ids, sb, fb, ab)
        probs = torch.sigmoid(output)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Output probabilities should be in [0, 1]"

    def test_different_students_give_different_predictions(self):
        model = LFACore(n_students=10, n_skills=5, embedding_dim=8)
        student_ids = torch.LongTensor([0, 1])
        skill_ids = torch.LongTensor([0, 0])
        sb = torch.FloatTensor([0, 0])
        fb = torch.FloatTensor([0, 0])
        ab = torch.FloatTensor([0, 0])
        output = model(student_ids, skill_ids, sb, fb, ab)
        assert not torch.allclose(output[0], output[1]), "Different students should give different predictions"

    def test_gradient_flows(self):
        model = LFACore(n_students=10, n_skills=5, embedding_dim=8)
        student_ids = torch.LongTensor([1, 2])
        skill_ids = torch.LongTensor([0, 1])
        sb = torch.FloatTensor([0, 1])
        fb = torch.FloatTensor([0, 0])
        ab = torch.FloatTensor([0, 1])
        labels = torch.FloatTensor([1, 0])
        criterion = torch.nn.BCEWithLogitsLoss()
        logits = model(student_ids, skill_ids, sb, fb, ab)
        loss = criterion(logits, labels)
        loss.backward()
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "At least one parameter should have non-zero gradient"


class TestLFAAux:
    def test_aux_output_shape(self):
        model = LFAAux(n_students=10, n_skills=5, n_aux_features=3, embedding_dim=8)
        batch_size = 4
        student_ids = torch.LongTensor([0, 1, 2, 3])
        skill_ids = torch.LongTensor([0, 0, 1, 2])
        sb = torch.FloatTensor([0, 1, 1, 0])
        fb = torch.FloatTensor([0, 0, 1, 1])
        ab = torch.FloatTensor([0, 1, 2, 1])
        aux = torch.FloatTensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        output = model(student_ids, skill_ids, sb, fb, ab, aux)
        assert output.shape == (batch_size,)

    def test_aux_output_range(self):
        model = LFAAux(n_students=10, n_skills=5, n_aux_features=2, embedding_dim=8)
        student_ids = torch.LongTensor([1])
        skill_ids = torch.LongTensor([0])
        sb = torch.FloatTensor([0])
        fb = torch.FloatTensor([0])
        ab = torch.FloatTensor([0])
        aux = torch.FloatTensor([[1.0, 2.0]])
        output = model(student_ids, skill_ids, sb, fb, ab, aux)
        probs = torch.sigmoid(output)
        assert 0.0 <= probs.item() <= 1.0

    def test_aux_adds_additional_parameters(self):
        model_core = LFACore(n_students=10, n_skills=5, embedding_dim=8)
        model_aux = LFAAux(n_students=10, n_skills=5, n_aux_features=3, embedding_dim=8)
        n_params_core = sum(p.numel() for p in model_core.parameters())
        n_params_aux = sum(p.numel() for p in model_aux.parameters())
        assert n_params_aux > n_params_core


class TestLFATrainer:
    @pytest.fixture
    def toy_data(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "student_id_encoded": np.random.randint(0, 5, n),
                "skill_id_encoded": np.random.randint(0, 3, n),
                "success_before": np.random.randint(0, 5, n).astype(float),
                "failure_before": np.random.randint(0, 3, n).astype(float),
                "attempts_before": np.random.randint(0, 8, n).astype(float),
                "correct": np.random.randint(0, 2, n),
            }
        )
        train = df[:80].reset_index(drop=True)
        val = df[80:].reset_index(drop=True)
        return train, val

    def test_trainer_fit_completes(self, toy_data):
        train, val = toy_data
        model = LFACore(n_students=5, n_skills=3, embedding_dim=4)
        trainer = LFATrainer(
            model=model,
            device=torch.device("cpu"),
            learning_rate=0.01,
            batch_size=32,
            max_epochs=3,
            early_stopping_patience=5,
        )
        result = trainer.fit(train, val)
        assert "best_val_auc" in result
        assert result["best_val_auc"] >= 0.0

    def test_trainer_predict_shape(self, toy_data):
        train, val = toy_data
        model = LFACore(n_students=5, n_skills=3, embedding_dim=4)
        trainer = LFATrainer(
            model=model,
            device=torch.device("cpu"),
            batch_size=32,
            max_epochs=2,
        )
        trainer.fit(train, val)
        preds = trainer.predict(val)
        assert len(preds) == len(val)
        assert all(0.0 <= p <= 1.0 for p in preds)

    def test_early_stopping_triggers(self, toy_data):
        train, val = toy_data
        model = LFACore(n_students=5, n_skills=3, embedding_dim=4)
        trainer = LFATrainer(
            model=model,
            device=torch.device("cpu"),
            max_epochs=50,
            early_stopping_patience=2,
        )
        result = trainer.fit(train, val)
        assert result["epochs_trained"] < 50

    def test_trainer_with_wandb_logger(self, toy_data):
        from unittest.mock import MagicMock
        from src.utils.logging import WandbLogger

        train, val = toy_data
        model = LFACore(n_students=5, n_skills=3, embedding_dim=4)

        mock_wandb_logger = MagicMock(spec=WandbLogger)
        mock_wandb_logger.enabled = True

        trainer = LFATrainer(
            model=model,
            device=torch.device("cpu"),
            learning_rate=0.01,
            batch_size=32,
            max_epochs=3,
            wandb_logger=mock_wandb_logger,
        )
        result = trainer.fit(train, val)
        assert "best_val_auc" in result
        assert mock_wandb_logger.log_metrics.called
        assert mock_wandb_logger.log_summary.called

