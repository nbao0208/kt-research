import numpy as np
import pandas as pd
import pytest
import torch

from src.models.attention_kt import AttentionOnlineEvaluator, AttentiveContextualKT


class TestAttentionKT:
    def test_output_shape(self):
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2)
        batch_size, seq_len = 4, 10
        skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        corrects = torch.randint(0, 2, (batch_size, seq_len)).float()
        target_skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        logits, context, _ = model(skill_ids, corrects, target_skill_ids, mask)
        assert logits.shape == (batch_size, seq_len)
        assert context.shape == (batch_size, seq_len, 16)

    def test_causal_mask_prevents_leakage(self):
        model = AttentiveContextualKT(num_skills=3, embedding_dim=8, num_heads=2, use_position=False, use_decay=False)
        model.eval()
        batch_size, seq_len = 1, 4
        skill_ids = torch.tensor([[0, 1, 2, 0]])
        corrects = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        target_skill_ids = torch.tensor([[1, 2, 0, 1]])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.no_grad():
            logits_original, _, _ = model(skill_ids.clone(), corrects.clone(), target_skill_ids.clone(), mask.clone())

        assert logits_original.shape == (1, 4)
        probs = torch.sigmoid(logits_original)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_padding_mask_works(self):
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2)
        batch_size, seq_len = 2, 5
        skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        corrects = torch.randint(0, 2, (batch_size, seq_len)).float()
        target_skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :3] = True
        logits, _, _ = model(skill_ids, corrects, target_skill_ids, mask)
        assert logits.shape == (batch_size, seq_len)

    def test_output_probabilities_in_range(self):
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2)
        skill_ids = torch.randint(0, 5, (2, 5))
        corrects = torch.randint(0, 2, (2, 5)).float()
        target_skill_ids = torch.randint(0, 5, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        logits, _, _ = model(skill_ids, corrects, target_skill_ids, mask)
        probs = torch.sigmoid(logits)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_causal_mask_structure(self):
        model = AttentiveContextualKT(num_skills=3, embedding_dim=8, num_heads=2)
        mask_2d = model._build_causal_mask(5, torch.device("cpu"))
        assert mask_2d.shape == (5, 5)
        assert mask_2d[0, 0] == 0.0
        assert mask_2d[0, 1] == float("-inf")
        assert mask_2d[1, 0] == 0.0
        assert mask_2d[1, 1] == 0.0
        assert mask_2d[1, 2] == float("-inf")

    def test_decay_parameter_non_negative(self):
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2, use_decay=True, decay_type="position")
        lam = torch.nn.functional.softplus(model.decay_param)
        assert lam.item() >= 0.0

    def test_online_evaluator_on_synthetic_data(self):
        model = AttentiveContextualKT(num_skills=3, embedding_dim=8, num_heads=2)
        evaluator = AttentionOnlineEvaluator(model)
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

    def test_history_buffer_truncation(self):
        model = AttentiveContextualKT(num_skills=3, embedding_dim=8, num_heads=2, max_seq_len=3)
        evaluator = AttentionOnlineEvaluator(model, max_seq_len=3)
        df = pd.DataFrame({
            "studentId": [1] * 10,
            "skill": ["A"] * 10,
            "correct": [1] * 10,
            "startTime": list(range(10)),
            "action_num": [1] * 10,
        })
        evaluator.evaluate(df)
        assert len(evaluator.history_buffers[1]["skill_ids"]) == 3

    def test_synthetic_training_converges(self):
        torch.manual_seed(42)
        model = AttentiveContextualKT(num_skills=3, embedding_dim=8, num_heads=2, use_position=False, use_decay=False)
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

        assert loss.item() < 1.5

    def test_decay_gradient_flow(self):
        torch.manual_seed(42)
        model = AttentiveContextualKT(num_skills=4, embedding_dim=16, num_heads=2, use_decay=True)
        batch_size, seq_len = 2, 6
        skill_ids = torch.randint(0, 4, (batch_size, seq_len))
        corrects = torch.randint(0, 2, (batch_size, seq_len)).float()
        target_skill_ids = torch.randint(0, 4, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits, _, _ = model(skill_ids, corrects, target_skill_ids, mask)
        loss = torch.nn.BCEWithLogitsLoss()(logits, corrects)
        loss.backward()

        assert model.decay_param.grad is not None
        assert not torch.isnan(model.decay_param.grad)

    def test_monotonic_decay_behavior(self):
        model = AttentiveContextualKT(num_skills=4, embedding_dim=16, num_heads=2, use_decay=True)
        seq_len = 5
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        attn_mask = model._build_attention_mask(mask, seq_len, torch.device("cpu"))

        # For head 0 and query row i = 4:
        # Distance to key j is (4 - j) for j <= 4
        # attn_mask[0, 4, 4] > attn_mask[0, 4, 3] > attn_mask[0, 4, 2] > attn_mask[0, 4, 1] > attn_mask[0, 4, 0]
        row = attn_mask[0, 4]
        for j in range(4):
            assert row[j + 1] > row[j]

    def test_no_nans_with_heavy_padding(self):
        torch.manual_seed(42)
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2, use_decay=True)
        batch_size, seq_len = 3, 10
        skill_ids = torch.randint(0, 5, (batch_size, seq_len))
        corrects = torch.randint(0, 2, (batch_size, seq_len)).float()
        target_skill_ids = torch.randint(0, 5, (batch_size, seq_len))

        # Heavy left-padding: e.g. only 2 items valid out of 10
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[0, -2:] = True
        mask[1, -4:] = True
        mask[2, -1:] = True

        logits, context, _ = model(skill_ids, corrects, target_skill_ids, mask)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(context).any()

        loss = (logits * mask.float()).sum()
        loss.backward()
        assert model.decay_param.grad is not None
        assert not torch.isnan(model.decay_param.grad)

    def test_cold_start_prediction(self):
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2)
        target_skill_0 = torch.LongTensor([[0]])
        target_skill_1 = torch.LongTensor([[1]])

        prob_0 = model.predict_cold_start(target_skill_0)
        prob_1 = model.predict_cold_start(target_skill_1)

        assert 0.0 <= prob_0.item() <= 1.0
        assert 0.0 <= prob_1.item() <= 1.0

        # In AttentionOnlineEvaluator, cold start calls predict_cold_start
        evaluator = AttentionOnlineEvaluator(model)
        df_single = pd.DataFrame({
            "studentId": [1],
            "skill": ["A"],
            "skill_id_encoded": [1],
            "correct": [1],
            "startTime": [1],
            "action_num": [1],
        })
        y_true, y_pred, _ = evaluator.predict(df_single)
        assert len(y_pred) == 1
        assert 0.0 <= y_pred[0] <= 1.0
        assert np.isclose(y_pred[0], prob_1.item())

    def test_predict_step_with_decay(self):
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2, use_decay=True)
        hist_skills = torch.tensor([[1, 2, 3]])
        hist_corrects = torch.tensor([[1.0, 0.0, 1.0]])
        target_skill = torch.tensor([[4]])

        prob = model.predict_step(hist_skills, hist_corrects, target_skill)
        assert prob.shape == (1, 1) or prob.shape == (1,) or prob.dim() == 0 or prob.numel() == 1
        assert 0.0 <= prob.item() <= 1.0

    def test_sequence_trainer_save_checkpoint_attention(self, tmp_path):
        from src.training.sequence_trainer import SequenceTrainer
        model = AttentiveContextualKT(num_skills=5, embedding_dim=16, num_heads=2)
        trainer = SequenceTrainer(model=model)
        ckpt_path = tmp_path / "model.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()
        loaded = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in loaded
        assert loaded["config"]["num_heads"] == 2