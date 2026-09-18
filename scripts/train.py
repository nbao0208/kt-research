import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.sequence import build_bkt_sequences, compute_sequence_metadata
from src.evaluation.metrics import compute_comprehensive_metrics, compute_metrics
from src.features.encoders import LabelEncoderWithUNK, save_encoder
from src.features.lfa_features import build_lfa_aux_features, build_lfa_core_features, build_lfa_feature_matrix
from src.models.bkt import BKTOnlineEvaluator, BKTTrainer, save_bkt_params
from src.models.lfa import LFAAux, LFACore, LFATrainer
from src.utils.io import ensure_dir, get_latest_data_version, save_json
from src.utils.logging import WandbLogger, setup_logger
from src.utils.seed import resolve_device, set_seed

logger = setup_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "experiment" in cfg:
        cfg = cfg.experiment

    logger.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))

    input_path = Path(cfg.data.processed_dir)
    if input_path.is_file() or str(input_path).endswith(".parquet"):
        data_path = input_path
    else:
        processed_dir = get_latest_data_version(input_path)
        data_path = processed_dir / "interactions.parquet"

    if not data_path.exists():
        logger.warning("Processed data not found at %s. Training will not run.", data_path)
        logger.info("Run 'python scripts/preprocess_assistments.py' first.")
        return

    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.trainer.device)
    logger.info("Target compute device resolved: %s", device)

    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    sort_cols = ["studentId", "startTime"]
    if "action_num" in df.columns:
        sort_cols.append("action_num")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["attempts_before"] = df.groupby(["studentId", "skill"]).cumcount()

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    experiment_name = cfg.experiment_name
    artifacts_dir = Path(cfg.trainer.checkpoint_dir) / experiment_name
    ensure_dir(artifacts_dir)
    OmegaConf.save(config=cfg, f=artifacts_dir / "resolved_config.yaml")

    wandb_cfg = cfg.trainer.get("wandb", {}) if hasattr(cfg, "trainer") else {}
    wandb_enabled = wandb_cfg.get("enabled", False)
    wandb_logger = WandbLogger(enabled=wandb_enabled)
    if wandb_enabled:
        wandb_logger.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=wandb_cfg.get("project", "kt-research"),
            entity=wandb_cfg.get("entity", None),
            name=experiment_name,
            tags=list(wandb_cfg.get("tags", ["kt"])),
        )

    try:
        # Determine model type from config
        has_bkt = "model" in cfg and hasattr(cfg.model, "p_init")
        has_lfa = "model" in cfg and hasattr(cfg.model, "lfa_variant")
        has_dkt = "model" in cfg and hasattr(cfg.model, "rnn_type")
        has_attention = "model" in cfg and hasattr(cfg.model, "num_heads")
        has_dbkt = "model" in cfg and hasattr(cfg.model, "dynamic_transition")

        if has_bkt:
            logger.info("Training BKT model on device: %s (Scipy CPU optimization)...", device)
            sequences = build_bkt_sequences(
                df,
                min_sequence_length=cfg.data.sequence_filter.min_sequence_length,
            )

            seq_meta = compute_sequence_metadata(sequences)
            logger.info("BKT sequences: %s", seq_meta)

            trainer = BKTTrainer(OmegaConf.to_container(cfg.model, resolve=True))
            skill_models, global_model = trainer.fit(sequences)

            logger.info("Trained %d skill-specific BKT models", len(skill_models))

            params_path = artifacts_dir / "bkt_params.csv"
            save_bkt_params(skill_models, global_model, params_path)

            eval_results = {}
            for split_name, split_df in [("val", val_df), ("test", test_df)]:
                evaluator = BKTOnlineEvaluator(skill_models, global_model)
                y_true, y_pred, eval_df = evaluator.predict(split_df)
                metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
                eval_results[split_name] = metrics
                logger.info("BKT %s global: %s", split_name, metrics.get("global", {}))
                if split_name == "test":
                    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
                if wandb_logger and wandb_enabled:
                    wandb_logger.log_metrics({f"{split_name}/{k}": v for k, v in metrics.get("global", {}).items()})
                    if split_name == "test":
                        wandb_logger.log_summary({f"test/{k}": v for k, v in metrics.get("global", {}).items()})

            metrics_dir = Path("outputs/metrics") / experiment_name
            ensure_dir(metrics_dir)
            save_json({"bkt": eval_results}, metrics_dir / "evaluation_metrics.json")
            save_json({"bkt": eval_results}, artifacts_dir / "evaluation_metrics.json")
            logger.info("Saved evaluation metrics to %s", metrics_dir / "evaluation_metrics.json")

            if wandb_logger and wandb_enabled and wandb_cfg.get("log_artifacts", False):
                wandb_logger.log_artifacts_from_dir(
                    directory=artifacts_dir,
                    artifact_type="model",
                    name=f"{experiment_name}_artifacts",
                )

            logger.info("BKT training complete.")

        elif has_lfa:
            logger.info("Training LFA model...")
            variant = cfg.model.lfa_variant
            feature_df, student_encoder, skill_encoder = build_lfa_core_features(
                df,
                train_df,
            )

            aux_config = OmegaConf.to_container(cfg.model.aux_features, resolve=True)
            if aux_config.get("enabled", False):
                aux_cols = aux_config.get("columns", [])
                feature_df, scalers = build_lfa_aux_features(
                    feature_df,
                    train_df,
                    aux_cols,
                    missing_strategy=aux_config.get("missing_strategy", "zero"),
                    outlier_cap_std=aux_config.get("outlier_cap_std", 3.0),
                )

            save_encoder(student_encoder, artifacts_dir / "encoder_student.json")
            save_encoder(skill_encoder, artifacts_dir / "encoder_skill.json")

            feature_matrix = build_lfa_feature_matrix(
                feature_df,
                student_encoder,
                skill_encoder,
                aux_config=aux_config,
            )

            train_features = feature_matrix[feature_matrix["split"] == "train"].reset_index(drop=True)
            val_features = feature_matrix[feature_matrix["split"] == "val"].reset_index(drop=True)
            test_features = feature_matrix[feature_matrix["split"] == "test"].reset_index(drop=True)

            n_students = len(student_encoder.encoder.classes_)
            n_skills = len(skill_encoder.encoder.classes_)
            aux_cols_in_matrix = [
                c
                for c in test_features.columns
                if c
                not in [
                    "student_id_encoded",
                    "skill_id_encoded",
                    "success_before",
                    "failure_before",
                    "attempts_before",
                    "correct",
                    "split",
                ]
            ]
            n_aux = len(aux_cols_in_matrix)
            logger.info("LFA variant: %s, auxiliary features count: %d (%s)", variant, n_aux, aux_cols_in_matrix)

            if variant == "aux" and n_aux > 0:
                model = LFAAux(
                    n_students,
                    n_skills,
                    n_aux_features=n_aux,
                    embedding_dim=cfg.model.embedding_dim,
                    dropout=cfg.model.dropout,
                )
            else:
                model = LFACore(n_students, n_skills, embedding_dim=cfg.model.embedding_dim, dropout=cfg.model.dropout)

            trainer = LFATrainer(
                model=model,
                device=device,
                learning_rate=cfg.model.learning_rate,
                weight_decay=cfg.model.weight_decay,
                batch_size=cfg.model.batch_size,
                max_epochs=cfg.model.max_epochs,
                early_stopping_patience=cfg.model.early_stopping_patience,
                seed=cfg.get("seed", 42),
                wandb_logger=wandb_logger,
            )

            result = trainer.fit(train_features, val_features)
            logger.info("LFA training result: %s", result)

            trainer.save_checkpoint(artifacts_dir / "best_model.pt")

            eval_results = {}
            for split_name, split_features, split_raw_df in [("val", val_features, val_df), ("test", test_features, test_df)]:
                preds = trainer.predict(split_features, batch_size=4096)
                eval_df = split_features.copy()
                eval_df["prediction"] = preds
                eval_df["skill"] = split_raw_df["skill"].values if len(split_raw_df) == len(split_features) else eval_df["skill_id_encoded"].astype(str)
                eval_df["studentId"] = split_raw_df["studentId"].values if len(split_raw_df) == len(split_features) else eval_df["student_id_encoded"].astype(str)
                
                metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
                eval_results[split_name] = metrics
                logger.info("LFA %s global: %s", split_name, metrics.get("global", {}))
                if split_name == "test":
                    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
                if wandb_logger and wandb_enabled:
                    wandb_logger.log_metrics({f"{split_name}/{k}": v for k, v in metrics.get("global", {}).items()})
                    if split_name == "test":
                        wandb_logger.log_summary({f"test/{k}": v for k, v in metrics.get("global", {}).items()})

            metrics_dir = Path("outputs/metrics") / experiment_name
            ensure_dir(metrics_dir)
            save_json({"lfa": eval_results}, metrics_dir / "evaluation_metrics.json")
            save_json({"lfa": eval_results}, artifacts_dir / "evaluation_metrics.json")
            logger.info("Saved evaluation metrics to %s", metrics_dir / "evaluation_metrics.json")

            if wandb_logger and wandb_enabled and wandb_cfg.get("log_artifacts", False):
                wandb_logger.log_artifacts_from_dir(
                    directory=artifacts_dir,
                    artifact_type="model",
                    name=f"{experiment_name}_artifacts",
                )

            logger.info("LFA training complete.")

        elif has_dkt or has_attention:
            model_name = "dkt" if has_dkt else "attention_kt"
            logger.info("Training %s model on device %s...", model_name.upper(), device)

            skill_encoder = LabelEncoderWithUNK()
            skill_encoder.fit(train_df["skill"].unique())
            n_skills = len(skill_encoder.encoder.classes_)
            save_encoder(skill_encoder, artifacts_dir / "encoder_skill.json")

            student_encoder = LabelEncoderWithUNK()
            student_encoder.fit(train_df["studentId"].unique())
            save_encoder(student_encoder, artifacts_dir / "encoder_student.json")

            from src.data.sequence_dataset import build_sequence_dataset
            from src.training.sequence_trainer import SequenceTrainer

            sequences, collate_fn = build_sequence_dataset(
                df=df,
                student_encoder=student_encoder,
                skill_encoder=skill_encoder,
                max_seq_len=cfg.model.get("max_seq_len", 200),
                min_history=cfg.model.get("min_history", 1),
                stride=cfg.model.get("stride", 100),
            )

            train_seqs = [s for s in sequences]
            val_seqs = [s for s in sequences]

            if has_dkt:
                from src.models.dkt import DeepKnowledgeTracing

                model = DeepKnowledgeTracing(
                    num_skills=n_skills,
                    embedding_dim=cfg.model.get("embedding_dim", 64),
                    hidden_size=cfg.model.get("hidden_size", 128),
                    num_layers=cfg.model.get("num_layers", 1),
                    dropout=cfg.model.get("dropout", 0.2),
                    rnn_type=cfg.model.get("rnn_type", "lstm"),
                    input_mode=cfg.model.get("input_mode", "embedding"),
                )
            else:
                from src.models.attention_kt import AttentiveContextualKT

                model = AttentiveContextualKT(
                    num_skills=n_skills,
                    embedding_dim=cfg.model.get("embedding_dim", 64),
                    num_heads=cfg.model.get("num_heads", 2),
                    num_layers=cfg.model.get("num_layers", 1),
                    dropout=cfg.model.get("dropout", 0.2),
                    max_seq_len=cfg.model.get("max_seq_len", 200),
                    use_position=cfg.model.get("use_position", True),
                    use_decay=cfg.model.get("use_decay", True),
                    decay_type=cfg.model.get("decay_type", "position"),
                )

            trainer = SequenceTrainer(
                model=model,
                device=device,
                learning_rate=cfg.model.get("optimizer", {}).get("learning_rate", 0.001) if hasattr(cfg.model, "optimizer") else cfg.model.get("learning_rate", 0.001),
                weight_decay=cfg.model.get("optimizer", {}).get("weight_decay", 1e-5) if hasattr(cfg.model, "optimizer") else 1e-5,
                batch_size=cfg.model.get("trainer", {}).get("batch_size", 256) if hasattr(cfg.model, "trainer") else cfg.get("batch_size", 256),
                max_epochs=cfg.model.get("trainer", {}).get("max_epochs", 50) if hasattr(cfg.model, "trainer") else 50,
                early_stopping_patience=cfg.model.get("trainer", {}).get("early_stopping_patience", 5) if hasattr(cfg.model, "trainer") else 5,
                gradient_clip_norm=cfg.model.get("trainer", {}).get("gradient_clip_norm", 1.0) if hasattr(cfg.model, "trainer") else 1.0,
                seed=cfg.get("seed", 42),
                wandb_logger=wandb_logger,
                collate_fn=collate_fn,
            )

            result = trainer.fit(train_seqs, val_seqs)
            logger.info("%s training result: %s", model_name.upper(), result)

            trainer.save_checkpoint(artifacts_dir / "best_model.pt")

            eval_results = {}
            for split_name in ["val", "test"]:
                split_df = df[df["split"] == split_name].copy()
                temp_split = split_df.copy()
                temp_split["skill_id_encoded"] = skill_encoder.transform(temp_split["skill"].values)

                if has_dkt:
                    from src.models.dkt import DKTOnlineEvaluator
                    evaluator = DKTOnlineEvaluator(model, device=device)
                else:
                    from src.models.attention_kt import AttentionOnlineEvaluator
                    evaluator = AttentionOnlineEvaluator(model, device=device, max_seq_len=cfg.model.get("max_seq_len", 200))

                _, _, eval_df = evaluator.predict(temp_split)
                metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
                eval_results[split_name] = metrics
                logger.info("%s %s global: %s", model_name.upper(), split_name, metrics.get("global", {}))
                if split_name == "test":
                    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
                if wandb_logger and wandb_enabled:
                    wandb_logger.log_metrics({f"{split_name}/{k}": v for k, v in metrics.get("global", {}).items()})
                    if split_name == "test":
                        wandb_logger.log_summary({f"test/{k}": v for k, v in metrics.get("global", {}).items()})

            metrics_dir = Path("outputs/metrics") / experiment_name
            ensure_dir(metrics_dir)
            save_json({model_name: eval_results}, metrics_dir / "evaluation_metrics.json")
            save_json({model_name: eval_results}, artifacts_dir / "evaluation_metrics.json")
            logger.info("Saved evaluation metrics to %s", metrics_dir / "evaluation_metrics.json")

            if wandb_logger and wandb_enabled and wandb_cfg.get("log_artifacts", False):
                wandb_logger.log_artifacts_from_dir(
                    directory=artifacts_dir,
                    artifact_type="model",
                    name=f"{experiment_name}_artifacts",
                )

            logger.info("%s training complete.", model_name.upper())

        elif has_dbkt:
            logger.info("Training DBKT model on device %s...", device)

            from src.models.dbkt import DBKTOnlineEvaluator, DBKTTrainer, DynamicBayesianKnowledgeTracing, save_dbkt_params

            skill_encoder = LabelEncoderWithUNK()
            skill_encoder.fit(train_df["skill"].unique())
            n_skills = len(skill_encoder.encoder.classes_)
            save_encoder(skill_encoder, artifacts_dir / "encoder_skill.json")

            feature_cols = cfg.model.get("dynamic_transition", {}).get("features", ["attempts_before", "failure_before"])
            feature_dim = len(feature_cols) if cfg.model.get("dynamic_transition", {}).get("enabled", True) else 0

            model = DynamicBayesianKnowledgeTracing(
                num_skills=n_skills,
                feature_dim=feature_dim,
                use_forgetting=cfg.model.get("dynamic_transition", {}).get("use_forgetting", True),
                constrain_slip_guess=cfg.model.get("parameters", {}).get("constrain_slip_guess", True),
                slip_guess_penalty=cfg.model.get("parameters", {}).get("slip_guess_penalty", 1.0),
            )

            opt_cfg = cfg.model.get("optimizer", {})
            trainer = DBKTTrainer(
                model=model,
                device=device,
                learning_rate=opt_cfg.get("learning_rate", 0.001),
                max_epochs=opt_cfg.get("max_epochs", 100),
                batch_size=opt_cfg.get("batch_size", 256),
                patience=opt_cfg.get("patience", 8),
                gradient_clip_norm=opt_cfg.get("gradient_clip_norm", 1.0),
                tolerance=opt_cfg.get("tolerance", 1e-5),
                min_skill_sequences=cfg.model.get("skill_handling", {}).get("min_skill_sequences", 20),
                fallback_to_global=cfg.model.get("skill_handling", {}).get("fallback_to_global", True),
                seed=cfg.get("seed", 42),
                wandb_logger=wandb_logger,
            )

            from src.data.sequence import build_dbkt_sequences

            feat_cols_to_use = feature_cols if all(c in df.columns for c in feature_cols) else None
            train_sequences = build_dbkt_sequences(
                df,
                split_name="train",
                min_sequence_length=cfg.data.sequence_filter.min_sequence_length,
                skill_encoder=skill_encoder,
                feature_cols=feat_cols_to_use,
            )
            val_sequences = build_dbkt_sequences(
                df,
                split_name="val",
                min_sequence_length=cfg.data.sequence_filter.min_sequence_length,
                skill_encoder=skill_encoder,
                feature_cols=feat_cols_to_use,
            )

            skill_models, global_model = trainer.fit(
                train_sequences,
                val_sequences=val_sequences,
                val_df=val_df,
                skill_encoder=skill_encoder,
                feature_cols=feat_cols_to_use,
            )
            logger.info("Trained DBKT model: %d skill models", len(skill_models))

            params_path = artifacts_dir / "dbkt_params.csv"
            save_dbkt_params(skill_models, global_model, params_path, skill_encoder=skill_encoder)

            trainer.save_checkpoint(artifacts_dir / "best_model.pt")

            eval_results = {}
            for split_name, split_df in [("val", val_df), ("test", test_df)]:
                evaluator = DBKTOnlineEvaluator(
                    model=model,
                    device=device,
                    skill_encoder=skill_encoder,
                    feature_cols=feature_cols if all(c in split_df.columns for c in feature_cols) else None,
                )
                _, _, eval_df = evaluator.predict(split_df)
                metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
                eval_results[split_name] = metrics
                logger.info("DBKT %s global: %s", split_name, metrics.get("global", {}))
                if split_name == "test":
                    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
                if wandb_logger and wandb_enabled:
                    wandb_logger.log_metrics({f"{split_name}/{k}": v for k, v in metrics.get("global", {}).items()})
                    if split_name == "test":
                        wandb_logger.log_summary({f"test/{k}": v for k, v in metrics.get("global", {}).items()})

            metrics_dir = Path("outputs/metrics") / experiment_name
            ensure_dir(metrics_dir)
            save_json({"dbkt": eval_results}, metrics_dir / "evaluation_metrics.json")
            save_json({"dbkt": eval_results}, artifacts_dir / "evaluation_metrics.json")
            logger.info("Saved evaluation metrics to %s", metrics_dir / "evaluation_metrics.json")

            if wandb_logger and wandb_enabled and wandb_cfg.get("log_artifacts", False):
                wandb_logger.log_artifacts_from_dir(
                    directory=artifacts_dir,
                    artifact_type="model",
                    name=f"{experiment_name}_artifacts",
                )

            logger.info("DBKT training complete.")

        else:
            logger.warning("Unknown model type in config")

    finally:
        if wandb_logger and wandb_enabled:
            wandb_logger.finish()


if __name__ == "__main__":
    main()

