import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.xes3g5m import XES3G5MDataset, XES3G5MMetadata, collate_xes3g5m_batch
from src.models.llm_reasoner import build_llm_reasoner
from src.models.sfn_kt import CognitiveAnomalyRegulator, SFNKTModel
from src.training.sfn_kt_trainer import SFNKTTrainer
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger
from src.utils.seed import resolve_device, set_seed

logger = setup_logger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Evaluate SFN-KT model on XES3G5M test data.")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/artifacts/sfn_kt_xes3g5m_default", help="Checkpoint directory.")
    parser.add_argument("--test-file", type=str, default="data/raw/XES3G5M/kc_level/train_valid_sequences.csv", help="Evaluation sequences file.")
    parser.add_argument("--test-fold", type=int, default=4, help="Fold index to evaluate on.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate.")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test.")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    device = resolve_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)

    config_path = ckpt_dir / "resolved_config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        cfg = OmegaConf.create({
            "data": OmegaConf.load(config_dir / "data" / "xes3g5m.yaml"),
            "model": OmegaConf.load(config_dir / "model" / "sfn_kt.yaml"),
            "trainer": {"checkpoint_dir": str(ckpt_dir)},
        })


    set_seed(42)

    data_file = Path(args.test_file)
    max_samples = 50 if args.smoke_test else args.max_samples

    logger.info("Evaluating on %s (fold %d)...", data_file.name, args.test_fold)
    test_dataset = XES3G5MDataset(
        data_file=data_file,
        folds=[args.test_fold] if "train_valid" in data_file.name else None,
        max_seq_len=cfg.data.max_seq_len,
        num_questions=cfg.data.num_questions,
        num_concepts=cfg.data.num_concepts,
        max_samples=max_samples,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(16 if args.smoke_test else args.batch_size, len(test_dataset)),
        shuffle=False,
        collate_fn=collate_xes3g5m_batch,
    )

    model = SFNKTModel(
        num_questions=cfg.data.num_questions,
        num_concepts=cfg.data.num_concepts,
        d_model=cfg.model.d_model,
        num_queries=cfg.model.num_queries,
        nheads=cfg.model.nheads,
        nlayers=cfg.model.nlayers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.data.max_seq_len,
        d_llm=cfg.model.get("d_llm", cfg.model.llm.get("d_llm", 2048)),
    )


    sfn_ckpt = ckpt_dir / "sfn_kt_best.pt"
    fast_ckpt = ckpt_dir / "fast_backbone_best.pt"

    if sfn_ckpt.exists():
        logger.info("Loading SFN-KT weights from %s", sfn_ckpt)
        ckpt = torch.load(sfn_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif fast_ckpt.exists():
        logger.info("Loading Fast Backbone weights from %s", fast_ckpt)
        ckpt = torch.load(fast_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        logger.warning("No checkpoint found in %s! Running with initialized weights.", ckpt_dir)

    metadata_mgr = XES3G5MMetadata(cfg.data.metadata_dir)
    llm_reasoner = build_llm_reasoner(
        OmegaConf.to_container(cfg.model.llm, resolve=True),
        metadata_manager=metadata_mgr,
    )
    scdt = CognitiveAnomalyRegulator()
    h5_cache = ckpt_dir / "cognitive_qformer_cache.h5"

    trainer = SFNKTTrainer(
        model=model,
        llm_reasoner=llm_reasoner,
        scdt_regulator=scdt,
        device=device,
        checkpoint_dir=ckpt_dir,
        h5_cache_path=h5_cache,
    )

    output_preds_file = ckpt_dir / "test_eval_predictions.parquet"
    results = trainer.evaluate(test_loader, save_predictions_path=output_preds_file)

    print("\n" + "=" * 60)
    print("             SFN-KT EVALUATION REPORT")
    print("=" * 60)
    print(f"  Fast Backbone AUC   : {results['base']['auc']:.4f}")
    print(f"  Fast Backbone ACC   : {results['base']['accuracy']:.4f}")
    print(f"  Fast Backbone Loss  : {results['base']['logloss']:.4f}")
    print("-" * 60)
    print(f"  SFN-KT Calibrated AUC: {results['calibrated']['auc']:.4f}")
    print(f"  SFN-KT Calibrated ACC: {results['calibrated']['accuracy']:.4f}")
    print(f"  SFN-KT Calibrated Loss: {results['calibrated']['logloss']:.4f}")
    print("-" * 60)
    print(f"  AUC Gain (SFN vs Base): {results['gain_auc']:+.4f}")
    print(f"  Active Interactions %%: {results['active_sample_ratio']*100:.2f}%")
    if results.get("active_region"):
        print(f"  Active Region AUC   : {results['active_region'].get('auc', 0.0):.4f}")
    if results.get("inactive_region"):
        print(f"  Inactive Region AUC : {results['inactive_region'].get('auc', 0.0):.4f}")
    print("=" * 60 + "\n")

    eval_file = ckpt_dir / "evaluation_report.json"
    save_json(results, eval_file)
    logger.info("Saved evaluation report to: %s", eval_file)


if __name__ == "__main__":
    main()
