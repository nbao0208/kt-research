import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.llm_reasoner import MockReasoner
from src.models.sfn_kt import CognitiveAnomalyRegulator, SFNKTModel
from src.training.sfn_kt_trainer import SFNKTTrainer


class SyntheticXESDataset(Dataset):
    def __init__(self, num_samples: int = 16, seq_len: int = 20, num_q: int = 50, num_c: int = 20):
        super().__init__()
        torch.manual_seed(42)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.questions = torch.randint(1, num_q, (num_samples, seq_len))
        self.concepts = torch.randint(1, num_c, (num_samples, seq_len))
        self.responses = torch.randint(0, 2, (num_samples, seq_len))
        self.selectmasks = torch.ones(num_samples, seq_len, dtype=torch.long)
        self.mask = torch.ones(num_samples, seq_len, dtype=torch.bool)
        self.uids = torch.arange(num_samples, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "uid": self.uids[idx],
            "questions": self.questions[idx],
            "concepts": self.concepts[idx],
            "responses": self.responses[idx],
            "selectmasks": self.selectmasks[idx],
            "mask": self.mask[idx],
        }


def collate_fn(batch):
    return {
        "uid": torch.stack([item["uid"] for item in batch]),
        "questions": torch.stack([item["questions"] for item in batch]),
        "concepts": torch.stack([item["concepts"] for item in batch]),
        "responses": torch.stack([item["responses"] for item in batch]),
        "selectmasks": torch.stack([item["selectmasks"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }


class TestSFNKTTrainingPipeline:
    def test_full_three_stage_pipeline_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_cache_file = tmp_path / "test_cache.h5"

            num_q, num_c, d_model = 50, 20, 32
            model = SFNKTModel(
                num_questions=num_q,
                num_concepts=num_c,
                d_model=d_model,
                num_queries=4,
                nheads=2,
                nlayers=1,
                dim_feedforward=64,
                max_seq_len=25,
                d_llm=64,
            )

            llm_reasoner = MockReasoner(model_name="mock-test", d_llm=64, max_tokens=16)
            scdt = CognitiveAnomalyRegulator(lambda_entropy=0.3, target_trigger_rate=0.20)

            trainer = SFNKTTrainer(
                model=model,
                llm_reasoner=llm_reasoner,
                scdt_regulator=scdt,
                device=torch.device("cpu"),
                checkpoint_dir=tmp_path,
                h5_cache_path=h5_cache_file,
                stage1_epochs=1,
                stage2_epochs=1,
                stage3_epochs=2,
                warmup_epochs=1,
            )

            train_ds = SyntheticXESDataset(num_samples=16, seq_len=20, num_q=num_q, num_c=num_c)
            val_ds = SyntheticXESDataset(num_samples=8, seq_len=20, num_q=num_q, num_c=num_c)

            train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
            test_ds = SyntheticXESDataset(num_samples=8, seq_len=20, num_q=num_q, num_c=num_c)
            test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)

            # Stage 1: Fast Backbone
            s1_res = trainer.train_stage1(train_loader, val_loader)
            assert "best_val_auc" in s1_res
            assert (tmp_path / "fast_backbone_best.pt").exists()

            # Stage 2: SCDT scan & cache with train, val, and test splits
            s2_res = trainer.run_stage2_scan_and_cache(
                val_loader, train_loader, test_loader=test_loader, max_cache_samples=10
            )
            assert "tau_star" in s2_res
            assert "total_cached_tensors" in s2_res
            assert h5_cache_file.exists()

            # Stage 3: Adapter & Soft-ECE
            s3_res = trainer.train_stage3(train_loader, val_loader)
            assert "best_sfn_kt_auc" in s3_res
            assert (tmp_path / "sfn_kt_best.pt").exists()

            # Evaluation on test set with pre-cached representations
            eval_res = trainer.evaluate(test_loader)
            assert "base" in eval_res
            assert "calibrated" in eval_res
            assert "gain_auc" in eval_res
            assert "auc" in eval_res["calibrated"]
            assert "active_sample_ratio" in eval_res
