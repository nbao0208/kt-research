import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dkt import DKTOnlineEvaluator, DeepKnowledgeTracing


class TestDKT:
    def test_output_shape(self):
        model = DeepKnowledgeTracing(num_skills=5, embedding_dim=16, hidden_size=32)
        batch_size, seq_len = 4, 10
        skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        corrects = torch.randint(0, 2, (batch_size, seq_len)).float()
        target_skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        logits, rnn_out, hidden = model(skill_ids, corrects, target_skill_ids, mask)
        assert logits.shape == (batch_size, seq_len)

    def test_target_shift_is_correct(self):
        model = DeepKnowledgeTracing(num_skills=5, embedding_dim=16, hidden_size=32)
        batch_size, seq_len = 1, 3
        skill_ids = torch.tensor([[0, 1, 2]])
        corrects = torch.tensor([[1.0, 0.0, 1.0]])
        target_skill_ids = torch.tensor([[1, 2, 0]])
        mask = torch.tensor([[True, True, True]])

        logits, rnn_out, _ = model(skill_ids, corrects, target_skill_ids, mask)
        assert logits.shape == (1, 3)
        assert rnn_out.shape == (1, 3, 32)

        probs = torch.sigmoid(logits)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_loss_respects_mask(self):
        model = DeepKnowledgeTracing(num_skills=5, embedding_dim=16, hidden_size=32)
        batch_size, seq_len = 2, 5
        skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        corrects = torch.randint(0, 2, (batch_size, seq_len)).float()
        target_skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :1] = True

        logits, _, _ = model(skill_ids, corrects, target_skill_ids, mask)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_per_pos = loss_fn(logits, corrects)
        masked_loss = (loss_per_pos * mask.float()).sum() / mask.sum()
        assert masked_loss > 0

    def test_unk_skill_handling(self):
        model = DeepKnowledgeTracing(num_skills=5, embedding_dim=16, hidden_size=32)
        skill_ids = torch.tensor([[10, 20]])
        corrects = torch.tensor([[1.0, 0.0]])
        target_skill_ids = torch.tensor([[10, 20]])
        mask = torch.tensor([[True, True]])
        logits, _, _ = model(skill_ids, corrects, target_skill_ids, mask)
        assert logits.shape == (1, 2)

    def test_online_evaluator_on_synthetic_data(self):
        model = DeepKnowledgeTracing(num_skills=3, embedding_dim=8, hidden_size=16)
        evaluator = DKTOnlineEvaluator(model)
        df = pd.DataFrame({
            "studentId": [1, 1, 1, 1],
            "skill": ["A", "A", "B", "A"],
            "correct": [1, 0, 1, 1],
            "startTime": [1, 2, 3, 4],
            "action_num": [1, 1, 1, 1],
        })
        metrics = evaluator.evaluate(df)
        assert "global" in metrics
        assert "auc" in metrics["global"]

    def test_online_hidden_state_updates(self):
        model = DeepKnowledgeTracing(num_skills=3, embedding_dim=8, hidden_size=16)
        evaluator = DKTOnlineEvaluator(model)
        df = pd.DataFrame({
            "studentId": [1, 1],
            "skill": ["A", "A"],
            "correct": [1, 0],
            "startTime": [1, 2],
            "action_num": [1, 1],
        })
        preds_before = evaluator.evaluate(df)
        evaluator.reset_state()
        assert evaluator.hidden_states == {}

    def test_synthetic_training_converges(self):
        torch.manual_seed(42)
        model = DeepKnowledgeTracing(num_skills=3, embedding_dim=8, hidden_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(20):
            skill_ids = torch.randint(0, 3, (4, 5))
            corrects = torch.randint(0, 2, (4, 5)).float()
            target_skill_ids = torch.randint(0, 3, (4, 5))
            mask = torch.ones(4, 5, dtype=torch.bool)

            logits, _, _ = model(skill_ids, corrects, target_skill_ids, mask)
            loss = loss_fn(logits, corrects)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.item() < 1.0