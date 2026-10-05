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
    parser.add_argument("--experiment-name", "--exp-name", type=str, default=None, help="Experiment name to evaluate (automatically routes to outputs/artifacts/<experiment_name>).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Explicit checkpoint directory override.")
    parser.add_argument("--test-file", type=str, default=None, help="Evaluation sequences file override.")
    parser.add_argument("--test-mode", type=str, default=None, choices=["question_window", "test_fold", "full_test"], help="Evaluation split mode: question_window (pyKT standard), test_fold, full_test.")
    parser.add_argument("--fusion-type", type=str, default="mean", choices=["mean", "vote", "all"], help="pyKT Late Fusion method for question-level evaluation.")
    parser.add_argument("--test-fold", type=int, default=4, help="Fold index to evaluate on if using train_valid file.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate.")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test.")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    device = resolve_device(args.device)

    if args.experiment_name is not None:
        ckpt_dir = Path("outputs/artifacts") / args.experiment_name
    elif args.checkpoint_dir is not None:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = Path("outputs/artifacts/sfn_kt_xes3g5m_default")

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

    test_mode = args.test_mode or getattr(cfg.data, "test_mode", "question_window")
    fusion_type = args.fusion_type or getattr(cfg.data, "fusion_type", "mean")

    if args.test_file:
        data_file = Path(args.test_file)
        eval_folds = [args.test_fold] if "train_valid" in data_file.name else None
    elif test_mode == "question_window":
        data_file = Path(getattr(cfg.data, "test_window_file", "data/raw/XES3G5M/kc_level/test_question_window_sequences.csv"))
        eval_folds = None
    elif test_mode == "full_test":
        data_file = Path(getattr(cfg.data, "test_file", "data/raw/XES3G5M/kc_level/test.csv"))
        eval_folds = None
    else:  # test_fold
        data_file = Path(cfg.data.kc_level_file)
        eval_folds = [args.test_fold]

    max_samples = 50 if args.smoke_test else args.max_samples

    logger.info("Evaluating on %s (mode=%s, folds=%s, max_samples=%s)...", data_file.name, test_mode, eval_folds, max_samples)
    test_dataset = XES3G5MDataset(
        data_file=data_file,
        folds=eval_folds,
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
    results = trainer.evaluate(
        test_loader,
        save_predictions_path=output_preds_file,
        fusion_type=fusion_type,
    )

    print("\n" + "=" * 60)
    print("             SFN-KT EVALUATION REPORT")
    print("=" * 60)
    print(f"  Fast Backbone AUC     : {results['base']['auc']:.4f}")
    print(f"  Fast Backbone ACC     : {results['base']['accuracy']:.4f}")
    print(f"  Fast Backbone Loss    : {results['base']['logloss']:.4f}")
    print("-" * 60)
    print(f"  SFN-KT Calibrated AUC : {results['calibrated']['auc']:.4f}")
    print(f"  SFN-KT Calibrated ACC : {results['calibrated']['accuracy']:.4f}")
    print(f"  SFN-KT Calibrated Loss: {results['calibrated']['logloss']:.4f}")
    print("-" * 60)
    print(f"  AUC Gain (SFN vs Base): {results['gain_auc']:+.4f}")
    print(f"  Active Interactions %%: {results['active_sample_ratio']*100:.2f}%")
    if results.get("active_region"):
        print(f"  Active Region AUC     : {results['active_region'].get('auc', 0.0):.4f}")
    if results.get("inactive_region"):
        print(f"  Inactive Region AUC   : {results['inactive_region'].get('auc', 0.0):.4f}")

    if "question_level" in results:
        ql = results["question_level"]
        print("-" * 60)
        print(f"     pyKT QUESTION-LEVEL (LATE FUSION: {ql['calibrated'].get('fusion_type', 'mean').upper()}) METRICS")
        print("-" * 60)
        print(f"  Base Question AUC     : {ql['base']['auc']:.4f} | ACC: {ql['base']['accuracy']:.4f}")
        print(f"  SFN-KT Question AUC   : {ql['calibrated']['auc']:.4f} | ACC: {ql['calibrated']['accuracy']:.4f}")
        print(f"  Question AUC Gain     : {ql['gain_auc']:+.4f}")
        print(f"  Questions Evaluated   : {ql['calibrated'].get('num_questions', 0)} (over {ql['calibrated'].get('num_kcs', 0)} KCs)")
        if "fusion_variants" in ql:
            print("  Fusion Variants (AUC / ACC):")
            for f_name, f_metrics in ql["fusion_variants"].items():
                print(f"    - {f_name:12s}: AUC = {f_metrics.get('auc', 0.0):.4f} | ACC = {f_metrics.get('accuracy', 0.0):.4f}")
    print("=" * 60 + "\n")

    eval_file = ckpt_dir / "evaluation_report.json"
    save_json(results, eval_file)
    logger.info("Saved evaluation report to: %s", eval_file)


if __name__ == "__main__":
    main()
