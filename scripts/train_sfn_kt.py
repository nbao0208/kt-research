import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.xes3g5m import XES3G5MDataset, XES3G5MMetadata, collate_xes3g5m_batch
from src.models.llm_reasoner import build_llm_reasoner
from src.models.sfn_kt import CognitiveAnomalyRegulator, SFNKTModel
from src.training.sfn_kt_trainer import SFNKTTrainer
from src.utils.io import ensure_dir, save_json
from src.utils.logging import WandbLogger, setup_logger
from src.utils.seed import resolve_device, set_seed

logger = setup_logger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train SFN-KT model on XES3G5M benchmark.")
    parser.add_argument("--config-name", type=str, default="sfn_kt_xes3g5m", help="Experiment config file name.")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "all"], help="Training stage to execute.")
    parser.add_argument("--llm", type=str, default=None, help="LLM model name/path (e.g. Qwen/Qwen2.5-Math-7B, mock).")
    parser.add_argument("--llm-backend", type=str, default=None, choices=["mock", "huggingface", "hf", "vllm", "api"], help="LLM execution backend.")
    parser.add_argument("--llm-d-llm", type=int, default=None, help="LLM hidden state dimension (e.g. 2048, 3584, 768).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for API backend (e.g. OpenAI/DeepSeek).")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible endpoint.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (auto, cuda, mps, cpu).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override for all stages.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max sequence samples to load (useful for debugging).")
    parser.add_argument("--max-triggers", type=int, default=None, help="Max anomaly triggers to extract/cache in Stage 2 (useful for debugging/resource constraints).")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test (1 epoch, small sample).")
    return parser.parse_known_args()


def run_training():
    cli_args, remaining_hydra_args = parse_cli_args()

    # Load base configurations
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    exp_config_file = config_dir / "experiment" / f"{cli_args.config_name}.yaml"
    if not exp_config_file.exists():
        exp_config_file = config_dir / "experiment" / "sfn_kt_xes3g5m.yaml"

    exp_cfg = OmegaConf.load(exp_config_file)
    data_cfg = OmegaConf.load(config_dir / "data" / "xes3g5m.yaml")
    model_cfg = OmegaConf.load(config_dir / "model" / "sfn_kt.yaml")

    cfg = OmegaConf.create({
        "data": data_cfg,
        "model": model_cfg,
        "experiment_name": exp_cfg.get("experiment_name", "sfn_kt_xes3g5m_default"),
        "seed": exp_cfg.get("seed", 42),
        "trainer": exp_cfg.get("trainer", {}),
    })

    # Apply remaining command-line overrides
    if remaining_hydra_args:
        cli_conf = OmegaConf.from_dotlist(remaining_hydra_args)
        cfg = OmegaConf.merge(cfg, cli_conf)


    # Apply explicit CLI flags
    if cli_args.llm is not None:
        cfg.model.llm.model_name = cli_args.llm
    if cli_args.llm_backend is not None:
        cfg.model.llm.backend = cli_args.llm_backend
    if cli_args.llm_d_llm is not None:
        cfg.model.llm.d_llm = cli_args.llm_d_llm
        cfg.model.d_llm = cli_args.llm_d_llm
    if cli_args.api_key is not None:
        cfg.model.llm.api_key = cli_args.api_key
    if cli_args.base_url is not None:
        cfg.model.llm.base_url = cli_args.base_url
    if cli_args.device is not None:
        cfg.trainer.device = cli_args.device
    if cli_args.batch_size is not None:
        cfg.trainer.batch_size = cli_args.batch_size

    if cli_args.smoke_test:
        cfg.trainer.stage1_epochs = 1
        cfg.trainer.stage2_epochs = 1
        cfg.trainer.stage3_epochs = 1
        cfg.trainer.warmup_epochs = 1
        cfg.trainer.batch_size = min(16, cfg.trainer.batch_size)
        cli_args.max_samples = 100
    elif cli_args.epochs is not None:
        cfg.trainer.stage1_epochs = cli_args.epochs
        cfg.trainer.stage2_epochs = max(1, cli_args.epochs // 3)
        cfg.trainer.stage3_epochs = cli_args.epochs

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.trainer.device)
    logger.info("Compute device resolved: %s", device)
    logger.info(
        "SFN-KT Configuration:\n  Stage: %s\n  LLM Model: %s\n  LLM Backend: %s\n  Hidden Dim: %d",
        cli_args.stage,
        cfg.model.llm.model_name,
        cfg.model.llm.backend,
        cfg.model.llm.d_llm,
    )

    # Paths & directories
    experiment_name = cfg.experiment_name
    artifacts_dir = Path(cfg.trainer.checkpoint_dir) / experiment_name
    metrics_dir = Path("outputs/metrics") / experiment_name
    ensure_dir(artifacts_dir)
    ensure_dir(metrics_dir)

    OmegaConf.save(config=cfg, f=artifacts_dir / "resolved_config.yaml")

    # Initialize WandB if enabled
    wandb_cfg = cfg.trainer.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)
    wandb_logger = WandbLogger(enabled=wandb_enabled)
    if wandb_enabled:
        wandb_logger.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=wandb_cfg.get("project", "kt-research"),
            name=experiment_name,
            tags=list(wandb_cfg.get("tags", ["sfn_kt", "xes3g5m"])),
        )

    # 1. Load Data
    data_cfg = cfg.data
    logger.info("Loading XES3G5M dataset from: %s", data_cfg.kc_level_file)

    train_dataset = XES3G5MDataset(
        data_file=data_cfg.kc_level_file,
        folds=list(data_cfg.train_folds),
        max_seq_len=data_cfg.max_seq_len,
        num_questions=data_cfg.num_questions,
        num_concepts=data_cfg.num_concepts,
        max_samples=cli_args.max_samples,
    )
    val_dataset = XES3G5MDataset(
        data_file=data_cfg.kc_level_file,
        folds=[data_cfg.val_fold],
        max_seq_len=data_cfg.max_seq_len,
        num_questions=data_cfg.num_questions,
        num_concepts=data_cfg.num_concepts,
        max_samples=cli_args.max_samples // 4 if cli_args.max_samples else None,
    )
    test_dataset = XES3G5MDataset(
        data_file=data_cfg.kc_level_file,
        folds=[data_cfg.test_fold],
        max_seq_len=data_cfg.max_seq_len,
        num_questions=data_cfg.num_questions,
        num_concepts=data_cfg.num_concepts,
        max_samples=cli_args.max_samples // 4 if cli_args.max_samples else None,
    )

    batch_size = cfg.trainer.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_xes3g5m_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_xes3g5m_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_xes3g5m_batch,
    )

    logger.info(
        "DataLoaders ready: Train batches=%d, Val batches=%d, Test batches=%d",
        len(train_loader), len(val_loader), len(test_loader)
    )

    # 2. Initialize Model Components
    metadata_mgr = XES3G5MMetadata(data_cfg.metadata_dir)
    llm_reasoner = build_llm_reasoner(
        OmegaConf.to_container(cfg.model.llm, resolve=True),
        metadata_manager=metadata_mgr,
    )

    scdt = CognitiveAnomalyRegulator(
        lambda_entropy=cfg.model.scdt.lambda_entropy,
        target_trigger_rate=cfg.model.scdt.target_trigger_rate,
        lambda_momentum=cfg.model.scdt.get("lambda_momentum", 0.0),
        lambda_conflict=cfg.model.scdt.get("lambda_conflict", 0.0),
    )

    model = SFNKTModel(
        num_questions=data_cfg.num_questions,
        num_concepts=data_cfg.num_concepts,
        d_model=cfg.model.d_model,
        num_queries=cfg.model.num_queries,
        nheads=cfg.model.nheads,
        nlayers=cfg.model.nlayers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_seq_len=data_cfg.max_seq_len,
        d_llm=cfg.model.get("d_llm", cfg.model.llm.get("d_llm", 2048)),
    )


    h5_cache_path = artifacts_dir / "cognitive_qformer_cache.h5"

    trainer = SFNKTTrainer(
        model=model,
        llm_reasoner=llm_reasoner,
        scdt_regulator=scdt,
        device=device,
        checkpoint_dir=artifacts_dir,
        h5_cache_path=h5_cache_path,
        metadata_manager=metadata_mgr,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        stage1_epochs=cfg.trainer.stage1_epochs,
        stage2_epochs=cfg.trainer.stage2_epochs,
        stage3_epochs=cfg.trainer.stage3_epochs,
        warmup_epochs=cfg.trainer.warmup_epochs,
        stage2_max_alignment_samples=cfg.trainer.get("stage2_max_alignment_samples", 2000),
        alpha=cfg.model.loss.alpha,
        beta=cfg.model.loss.beta,
        tau_s=cfg.model.loss.tau_s,
        gradient_clip_norm=cfg.trainer.gradient_clip_norm,
        wandb_logger=wandb_logger,
    )

    # 3. Execution based on Stage
    stage = cli_args.stage.lower()

    if stage in ["1", "all"]:
        stage1_res = trainer.train_stage1(train_loader, val_loader)
        logger.info("Stage 1 completed. Best Val AUC: %.4f", stage1_res["best_val_auc"])

    if stage in ["2", "all"]:
        # Ensure fast backbone weights are loaded
        fast_ckpt = artifacts_dir / "fast_backbone_best.pt"
        if fast_ckpt.exists():
            ckpt = torch.load(fast_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        stage2_res = trainer.run_stage2_scan_and_cache(
            val_loader=val_loader,
            train_loader=train_loader,
            test_loader=test_loader,
            max_cache_samples=cli_args.max_triggers if cli_args.max_triggers is not None else (100 if cli_args.smoke_test else None),
        )
        logger.info(
            "Stage 2 completed. Anomaly threshold tau*=%.4f, total cached tensors=%d",
            stage2_res["tau_star"],
            stage2_res.get("total_cached_tensors", stage2_res.get("train_triggers", 0)),
        )

    if stage in ["3", "all"]:
        fast_ckpt = artifacts_dir / "fast_backbone_best.pt"
        if fast_ckpt.exists():
            ckpt = torch.load(fast_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        stage3_res = trainer.train_stage3(train_loader, val_loader)
        logger.info("Stage 3 completed. Best SFN-KT AUC: %.4f", stage3_res["best_sfn_kt_auc"])

    # 4. Evaluation on Test Set
    logger.info("Running final evaluation on Test Set (fold %d)...", data_cfg.test_fold)
    best_sfn_ckpt = artifacts_dir / "sfn_kt_best.pt"
    if best_sfn_ckpt.exists():
        ckpt = torch.load(best_sfn_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("Loaded best SFN-KT checkpoint for test evaluation.")

    test_preds_path = metrics_dir / "test_predictions.parquet"
    test_metrics = trainer.evaluate(
        test_loader,
        save_predictions_path=test_preds_path,
    )

    logger.info("\n================ SFN-KT EVALUATION REPORT ================")
    logger.info("  Fast Backbone (Base) AUC: %.4f | ACC: %.4f", test_metrics["base"]["auc"], test_metrics["base"]["accuracy"])
    logger.info("  SFN-KT Calibrated AUC   : %.4f | ACC: %.4f", test_metrics["calibrated"]["auc"], test_metrics["calibrated"]["accuracy"])
    logger.info("  Overall AUC Gain        : %+.4f", test_metrics["gain_auc"])
    logger.info("  Active Region Sample %%  : %.2f%%", test_metrics["active_sample_ratio"] * 100)
    if test_metrics.get("active_region"):
        logger.info("  Active Region Calibrated AUC: %.4f", test_metrics["active_region"].get("auc", 0.0))
    if test_metrics.get("inactive_region"):
        logger.info("  Inactive Region Calibrated AUC: %.4f", test_metrics["inactive_region"].get("auc", 0.0))
    logger.info("=========================================================\n")

    # Save metrics summary
    metrics_summary_file = metrics_dir / "evaluation_metrics.json"
    save_json(test_metrics, metrics_summary_file)
    save_json(test_metrics, artifacts_dir / "evaluation_metrics.json")
    logger.info("Saved evaluation metrics to: %s", metrics_summary_file)

    if wandb_logger and wandb_enabled:
        wandb_logger.log_summary({
            "test/base_auc": test_metrics["base"]["auc"],
            "test/sfn_kt_auc": test_metrics["calibrated"]["auc"],
            "test/gain_auc": test_metrics["gain_auc"],
        })
        wandb_logger.finish()

    logger.info("Training and evaluation finished successfully.")


if __name__ == "__main__":
    run_training()
