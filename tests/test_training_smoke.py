import pandas as pd
import pytest
import torch

from src.models.attention_kt import AttentiveContextualKT
from src.models.dbkt import DynamicBayesianKnowledgeTracing
from src.models.dkt import DeepKnowledgeTracing


class TestTrainingSmoke:
    @pytest.fixture
    def synthetic_df(self):
        rows = []
        for student in range(3):
            for t in range(10):
                rows.append({
                    "studentId": student + 1,
                    "skill": f"SKILL_{t % 3}",
                    "skill_id_encoded": t % 3,
                    "student_id_encoded": student,
                    "correct": 1 if (t + student) % 2 == 0 else 0,
                    "startTime": t,
                    "action_num": 1,
                    "split": "train" if t < 7 else "val" if t < 9 else "test",
                })
        return pd.DataFrame(rows)

    def test_dkt_debug_training(self, synthetic_df):
        model = DeepKnowledgeTracing(num_skills=3, embedding_dim=8, hidden_size=16)
        train_df = synthetic_df[synthetic_df["split"] == "train"]
        skill_ids = torch.LongTensor(train_df["skill_id_encoded"].values).unsqueeze(0)
        corrects = torch.FloatTensor(train_df["correct"].values).unsqueeze(0)
        tgt_skills = torch.LongTensor(train_df["skill_id_encoded"].values).unsqueeze(0)
        mask = torch.ones(1, len(train_df), dtype=torch.bool)

        logits, _, _ = model(skill_ids, corrects, tgt_skills, mask)
        loss = torch.nn.BCEWithLogitsLoss()(logits, corrects)
        assert loss.item() > 0

    def test_attention_debug_training(self, synthetic_df):
        model = AttentiveContextualKT(num_skills=3, embedding_dim=8, num_heads=2)
        train_df = synthetic_df[synthetic_df["split"] == "train"]
        skill_ids = torch.LongTensor(train_df["skill_id_encoded"].values).unsqueeze(0)
        corrects = torch.FloatTensor(train_df["correct"].values).unsqueeze(0)
        tgt_skills = torch.LongTensor(train_df["skill_id_encoded"].values).unsqueeze(0)
        mask = torch.ones(1, len(train_df), dtype=torch.bool)

        logits, _, _ = model(skill_ids, corrects, tgt_skills, mask)
        loss = torch.nn.BCEWithLogitsLoss()(logits, corrects)
        assert loss.item() > 0

    def test_dbkt_debug_training(self, synthetic_df):
        model = DynamicBayesianKnowledgeTracing(num_skills=3)
        train_df = synthetic_df[synthetic_df["split"] == "train"]
        skill_ids = torch.LongTensor(train_df["skill_id_encoded"].values).unsqueeze(0)
        corrects = torch.FloatTensor(train_df["correct"].values).unsqueeze(0)

        loss = model.compute_loss(skill_ids, corrects)
        assert loss.item() > 0

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(5):
            loss = model.compute_loss(skill_ids, corrects)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert loss.item() > 0