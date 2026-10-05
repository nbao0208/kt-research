import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.xes3g5m import XES3G5MDataset, collate_xes3g5m_batch
from src.evaluation.late_fusion import (
    compute_all_late_fusion_metrics,
    compute_late_fusion_metrics,
)
from src.models.llm_reasoner import MockReasoner
from src.models.sfn_kt import CognitiveAnomalyRegulator, SFNKTModel
from src.training.sfn_kt_trainer import SFNKTTrainer


class TestPyKTLateFusion:
    """Tests pyKT Late Fusion Question-Level evaluation standards."""

    def test_late_fusion_mean(self):
        # 2 questions:
        # Question 0 has 2 KCs: preds [0.8, 0.6] -> mean = 0.7, target = 1
        # Question 1 has 2 KCs: preds [0.2, 0.4] -> mean = 0.3, target = 0
        preds = np.array([0.8, 0.6, 0.2, 0.4])
        targets = np.array([1, 1, 0, 0])
        qidxs = np.array([0, 0, 1, 1])

        metrics = compute_late_fusion_metrics(preds, targets, qidxs, fusion_type="mean")
        assert metrics["num_questions"] == 2
        assert metrics["num_kcs"] == 4
        assert metrics["auc"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["fusion_type"] == "mean"

    def test_late_fusion_vote_and_all(self):
        # Question 0: preds [0.9, 0.8] -> vote=1.0, all=0.72, target=1
        # Question 1: preds [0.7, 0.3] -> vote=1.0 (mean >= 0.5 is True), all=0.21, target=0
        preds = np.array([0.9, 0.8, 0.7, 0.3])
        targets = np.array([1, 1, 0, 0])
        qidxs = np.array([0, 0, 1, 1])

        metrics_vote = compute_late_fusion_metrics(preds, targets, qidxs, fusion_type="vote")
        metrics_all = compute_late_fusion_metrics(preds, targets, qidxs, fusion_type="all")
        assert metrics_vote["num_questions"] == 2
        assert metrics_all["num_questions"] == 2

        all_variants = compute_all_late_fusion_metrics(preds, targets, qidxs)
        assert "late_mean" in all_variants
        assert "late_vote" in all_variants
        assert "late_all" in all_variants

    def test_late_fusion_with_uids(self):
        # Same qidx=0 for two distinct students (uid=101, uid=102)
        # uid=101: target=1, preds=[0.9, 0.8]
        # uid=102: target=0, preds=[0.1, 0.2]
        preds = np.array([0.9, 0.8, 0.1, 0.2])
        targets = np.array([1, 1, 0, 0])
        qidxs = np.array([0, 0, 0, 0])
        uids = np.array([101, 101, 102, 102])

        metrics = compute_late_fusion_metrics(preds, targets, qidxs, uids=uids, fusion_type="mean")
        assert metrics["num_questions"] == 2
        assert metrics["auc"] == 1.0

    def test_late_fusion_edge_cases(self):
        empty_res = compute_late_fusion_metrics(np.array([]), np.array([]), np.array([]))
        assert empty_res["num_questions"] == 0
        assert empty_res["auc"] == 0.5

        # Single class
        single_class_res = compute_late_fusion_metrics(
            np.array([0.8, 0.9]), np.array([1, 1]), np.array([0, 1])
        )
        assert single_class_res["auc"] == 0.5


class TestPyKTDataLoading:
    """Tests loading and collating pyKT-format sequences with qidxs, rest, orirow."""

    def test_dataset_parsing_with_qidxs(self, tmp_path):
        csv_file = tmp_path / "test_window.csv"
        df = pd.DataFrame({
            "fold": [-1, -1],
            "uid": [1001, 1002],
            "questions": ["10,11,12", "20,21"],
            "concepts": ["1,2,3", "4,5"],
            "responses": ["1,0,1", "0,1"],
            "timestamps": ["1000,1010,1020", "2000,2010"],
            "selectmasks": ["1,1,1", "1,1"],
            "is_repeat": ["0,0,0", "0,0"],
            "qidxs": ["0,1,2", "0,1"],
            "rest": ["0,0,1", "0,1"],
            "orirow": ["0,0,0", "1,1"],
        })
        df.to_csv(csv_file, index=False)

        dataset = XES3G5MDataset(
            data_file=csv_file,
            max_seq_len=5,
            num_questions=100,
            num_concepts=100,
        )
        assert len(dataset) == 2
        item = dataset[0]
        assert "qidxs" in item
        assert "rest" in item
        assert "orirow" in item
        assert item["qidxs"][0] == 0
        assert item["qidxs"][1] == 1
        assert item["qidxs"][2] == 2

    def test_collate_fn_with_qidxs(self, tmp_path):
        csv_file = tmp_path / "test_window.csv"
        df = pd.DataFrame({
            "fold": [-1, -1],
            "uid": [1001, 1002],
            "questions": ["10,11,12", "20,21"],
            "concepts": ["1,2,3", "4,5"],
            "responses": ["1,0,1", "0,1"],
            "timestamps": ["1000,1010,1020", "2000,2010"],
            "selectmasks": ["1,1,1", "1,1"],
            "is_repeat": ["0,0,0", "0,0"],
            "qidxs": ["0,1,2", "0,1"],
            "rest": ["0,0,1", "0,1"],
            "orirow": ["0,0,0", "1,1"],
        })
        df.to_csv(csv_file, index=False)

        dataset = XES3G5MDataset(
            data_file=csv_file,
            max_seq_len=5,
            num_questions=100,
            num_concepts=100,
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_xes3g5m_batch)
        batch = next(iter(loader))

        assert "qidxs" in batch
        assert batch["qidxs"].shape == (2, 5)
        assert "rest" in batch
        assert "orirow" in batch


class TestPyKTEndToEndEvaluation:
    """Tests SFNKTTrainer.evaluate producing pyKT late fusion metrics."""

    def test_trainer_evaluate_produces_question_level_metrics(self, tmp_path):
        csv_file = tmp_path / "test_window.csv"
        df = pd.DataFrame({
            "fold": [-1, -1],
            "uid": [1001, 1002],
            "questions": ["10,11,12,13", "20,21,22,23"],
            "concepts": ["1,2,3,4", "4,5,6,7"],
            "responses": ["1,0,1,0", "0,1,0,1"],
            "timestamps": ["100,110,120,130", "200,210,220,230"],
            "selectmasks": ["1,1,1,1", "1,1,1,1"],
            "is_repeat": ["0,0,0,0", "0,0,0,0"],
            "qidxs": ["0,0,1,1", "0,1,1,2"],
            "rest": ["0,1,0,1", "1,0,1,1"],
            "orirow": ["0,0,0,0", "1,1,1,1"],
        })
        df.to_csv(csv_file, index=False)

        dataset = XES3G5MDataset(
            data_file=csv_file,
            max_seq_len=5,
            num_questions=100,
            num_concepts=100,
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_xes3g5m_batch)

        model = SFNKTModel(
            num_questions=100,
            num_concepts=100,
            d_model=32,
            num_queries=4,
            nheads=2,
            nlayers=1,
            dim_feedforward=64,
            max_seq_len=5,
            d_llm=64,
        )
        llm = MockReasoner(d_llm=64)
        scdt = CognitiveAnomalyRegulator()

        trainer = SFNKTTrainer(
            model=model,
            llm_reasoner=llm,
            scdt_regulator=scdt,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
            h5_cache_path=tmp_path / "cache.h5",
        )

        preds_parquet = tmp_path / "preds.parquet"
        results = trainer.evaluate(loader, save_predictions_path=preds_parquet, fusion_type="mean")

        # Verify Base and Calibrated KC-level metrics
        assert "base" in results
        assert "calibrated" in results
        assert "gain_auc" in results

        # Verify pyKT Question-Level metrics
        assert "question_level" in results
        ql = results["question_level"]
        assert "base" in ql
        assert "calibrated" in ql
        assert "gain_auc" in ql
        assert "fusion_variants" in ql
        assert "late_mean" in ql["fusion_variants"]
        assert "late_vote" in ql["fusion_variants"]
        assert "late_all" in ql["fusion_variants"]

        # Verify parquet saved with uid and qidx columns
        assert preds_parquet.exists()
        df_saved = pd.read_parquet(preds_parquet)
        assert "uid" in df_saved.columns
        assert "qidx" in df_saved.columns
        assert "y_true" in df_saved.columns
        assert "p_base" in df_saved.columns
        assert "p_calibrated" in df_saved.columns
