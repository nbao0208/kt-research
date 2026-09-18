import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.evaluation.metrics import compute_comprehensive_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger
from src.utils.seed import resolve_device

logger = setup_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "experiment" in cfg:
        cfg = cfg.experiment

    logger.info("Starting evaluation with config:\n%s", OmegaConf.to_yaml(cfg))

    device_str = cfg.trainer.get("device", "auto") if hasattr(cfg, "trainer") else "auto"
    device = resolve_device(device_str)
    logger.info("Resolved evaluation compute device: %s", device)

    from src.utils.io import get_latest_data_version
    processed_dir = get_latest_data_version(Path(cfg.data.processed_dir))
    data_path = processed_dir / "interactions.parquet"
    if not data_path.exists():
        logger.warning("Processed data not found at %s.", data_path)
        return

    experiment_name = cfg.experiment_name
    artifacts_dir = Path(cfg.trainer.checkpoint_dir) / experiment_name

    df = pd.read_parquet(data_path)
    sort_cols = ["studentId", "startTime"]
    if "action_num" in df.columns:
        sort_cols.append("action_num")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["attempts_before"] = df.groupby(["studentId", "skill"]).cumcount()

    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    has_bkt = "model" in cfg and hasattr(cfg.model, "p_init")
    has_lfa = "model" in cfg and hasattr(cfg.model, "lfa_variant")
    has_dkt = "model" in cfg and hasattr(cfg.model, "rnn_type")
    has_attention = "model" in cfg and hasattr(cfg.model, "num_heads")
    has_dbkt = "model" in cfg and hasattr(cfg.model, "dynamic_transition")

    results = {}
    test_eval_df = None

    if has_bkt:
        params_path = artifacts_dir / "bkt_params.csv"
        if not params_path.exists():
            logger.warning("BKT params not found at %s", params_path)
            return

        from src.models.bkt import BKTOnlineEvaluator, load_bkt_params

        skill_models, global_model = load_bkt_params(params_path)
        evaluator = BKTOnlineEvaluator(skill_models, global_model)

        _, _, val_eval_df = evaluator.predict(val_df)
        val_metrics = compute_comprehensive_metrics(val_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

        evaluator.reset_state()
        _, _, test_eval_df = evaluator.predict(test_df)
        test_metrics = compute_comprehensive_metrics(test_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

        results["bkt"] = {"val": val_metrics, "test": test_metrics}
        logger.info("BKT val metrics: %s", val_metrics.get("global", val_metrics))
        logger.info("BKT test metrics: %s", test_metrics.get("global", test_metrics))

    elif has_lfa:
        checkpoint_path = artifacts_dir / "best_model.pt"
        if not checkpoint_path.exists():
            logger.warning("LFA checkpoint not found at %s", checkpoint_path)
            return

        import json
        from src.features.encoders import load_encoder
        from src.features.lfa_features import build_lfa_aux_features, build_lfa_core_features, build_lfa_feature_matrix
        from src.models.lfa import LFAAux, LFACore, LFATrainer

        student_encoder = load_encoder(artifacts_dir / "encoder_student.json")
        skill_encoder = load_encoder(artifacts_dir / "encoder_skill.json")

        n_students = len(student_encoder.encoder.classes_)
        n_skills = len(skill_encoder.encoder.classes_)

        train_df = df[df["split"] == "train"]
        feature_df, _, _ = build_lfa_core_features(df, train_df)

        variant = cfg.model.get("lfa_variant", "core")
        aux_cols = [c for c in ["timeTaken", "hintCount", "attemptCount", "scaffold", "hintTotal"] if c in feature_df.columns]

        if variant == "aux" and aux_cols:
            feature_df, _ = build_lfa_aux_features(feature_df, train_df, aux_cols)
            aux_config = {"enabled": True, "columns": aux_cols}
            model = LFAAux(n_students, n_skills, n_aux_features=len(aux_cols), embedding_dim=cfg.model.embedding_dim)
        else:
            aux_config = None
            model = LFACore(n_students, n_skills, embedding_dim=cfg.model.embedding_dim)

        feature_matrix = build_lfa_feature_matrix(feature_df, student_encoder, skill_encoder, aux_config=aux_config)
        test_features = feature_matrix[feature_matrix["split"] == "test"].reset_index(drop=True)
        val_features = feature_matrix[feature_matrix["split"] == "val"].reset_index(drop=True)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        trainer = LFATrainer(model=model, device=device)
        test_preds = trainer.predict(test_features, batch_size=4096)
        val_preds = trainer.predict(val_features, batch_size=4096)

        test_eval_df = test_features.copy()
        test_eval_df["prediction"] = test_preds
        test_eval_df["skill"] = test_df["skill"].values if len(test_df) == len(test_features) else test_eval_df["skill_id_encoded"].astype(str)
        test_eval_df["studentId"] = test_df["studentId"].values if len(test_df) == len(test_features) else test_eval_df["student_id_encoded"].astype(str)

        val_eval_df = val_features.copy()
        val_eval_df["prediction"] = val_preds
        val_eval_df["skill"] = val_df["skill"].values if len(val_df) == len(val_features) else val_eval_df["skill_id_encoded"].astype(str)
        val_eval_df["studentId"] = val_df["studentId"].values if len(val_df) == len(val_features) else val_eval_df["student_id_encoded"].astype(str)

        test_metrics = compute_comprehensive_metrics(test_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
        val_metrics = compute_comprehensive_metrics(val_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
        results["lfa"] = {"val": val_metrics, "test": test_metrics}
        logger.info("LFA test metrics: %s", test_metrics.get("global", test_metrics))

    elif has_dkt or has_attention:
        checkpoint_path = artifacts_dir / "best_model.pt"
        if not checkpoint_path.exists():
            logger.warning("Neural model checkpoint not found at %s", checkpoint_path)
            return

        from src.features.encoders import load_encoder

        skill_encoder = load_encoder(artifacts_dir / "encoder_skill.json")
        n_skills = len(skill_encoder.encoder.classes_)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        if has_dkt:
            from src.models.dkt import DKTOnlineEvaluator, DeepKnowledgeTracing
            model = DeepKnowledgeTracing(
                num_skills=n_skills,
                embedding_dim=saved_cfg.get("embedding_dim", 64),
                hidden_size=saved_cfg.get("hidden_size", 128),
            )
            model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
            model.eval()

            val_temp = val_df.copy()
            val_temp["skill_id_encoded"] = skill_encoder.transform(val_temp["skill"].values)
            evaluator = DKTOnlineEvaluator(model, device=device)
            _, _, val_eval_df = evaluator.predict(val_temp)
            val_metrics = compute_comprehensive_metrics(val_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

            evaluator.reset_state()
            test_temp = test_df.copy()
            test_temp["skill_id_encoded"] = skill_encoder.transform(test_temp["skill"].values)
            _, _, test_eval_df = evaluator.predict(test_temp)
            test_metrics = compute_comprehensive_metrics(test_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

            results["dkt"] = {"val": val_metrics, "test": test_metrics}
            logger.info("DKT val metrics: %s", val_metrics.get("global", val_metrics))
            logger.info("DKT test metrics: %s", test_metrics.get("global", test_metrics))
        else:
            from src.models.attention_kt import AttentionOnlineEvaluator, AttentiveContextualKT
            model = AttentiveContextualKT(
                num_skills=n_skills,
                embedding_dim=saved_cfg.get("embedding_dim", 64),
                num_heads=saved_cfg.get("num_heads", 2),
                num_layers=saved_cfg.get("num_layers", 1),
            )
            model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
            model.eval()

            val_temp = val_df.copy()
            val_temp["skill_id_encoded"] = skill_encoder.transform(val_temp["skill"].values)
            evaluator = AttentionOnlineEvaluator(model, device=device)
            _, _, val_eval_df = evaluator.predict(val_temp)
            val_metrics = compute_comprehensive_metrics(val_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

            evaluator.reset_state()
            test_temp = test_df.copy()
            test_temp["skill_id_encoded"] = skill_encoder.transform(test_temp["skill"].values)
            _, _, test_eval_df = evaluator.predict(test_temp)
            test_metrics = compute_comprehensive_metrics(test_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

            results["attention"] = {"val": val_metrics, "test": test_metrics}
            logger.info("Attention val metrics: %s", val_metrics.get("global", val_metrics))
            logger.info("Attention test metrics: %s", test_metrics.get("global", test_metrics))

    elif has_dbkt:
        params_path = artifacts_dir / "dbkt_params.csv"
        if not params_path.exists():
            logger.warning("DBKT params not found at %s", params_path)
            return

        from src.features.encoders import load_encoder
        from src.models.dbkt import DBKTOnlineEvaluator, DynamicBayesianKnowledgeTracing

        skill_encoder = load_encoder(artifacts_dir / "encoder_skill.json") if (artifacts_dir / "encoder_skill.json").exists() else None
        n_skills = len(skill_encoder.encoder.classes_) if skill_encoder else len(df["skill"].unique())
        feature_cols = cfg.model.get("dynamic_transition", {}).get("features", ["attempts_before", "failure_before"]) if hasattr(cfg.model, "dynamic_transition") else None
        feature_dim = len(feature_cols) if feature_cols else 0

        model = DynamicBayesianKnowledgeTracing(num_skills=n_skills, feature_dim=feature_dim)
        checkpoint_path = artifacts_dir / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)

        evaluator = DBKTOnlineEvaluator(
            model=model,
            device=device,
            skill_encoder=skill_encoder,
            feature_cols=feature_cols if all(c in df.columns for c in feature_cols) else None,
        )
        _, _, val_eval_df = evaluator.predict(val_df)
        val_metrics = compute_comprehensive_metrics(val_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

        evaluator.reset_state()
        _, _, test_eval_df = evaluator.predict(test_df)
        test_metrics = compute_comprehensive_metrics(test_eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

        results["dbkt"] = {"val": val_metrics, "test": test_metrics}
        logger.info("DBKT test metrics: %s", test_metrics.get("global", test_metrics))

    metrics_dir = Path("outputs/metrics") / experiment_name
    ensure_dir(metrics_dir)
    save_json(results, metrics_dir / "evaluation_metrics.json")
    save_json(results, artifacts_dir / "evaluation_metrics.json")

    if test_eval_df is not None:
        save_cols = [c for c in ["studentId", "skill", "correct", "prediction", "split"] if c in test_eval_df.columns]
        test_eval_df[save_cols].to_parquet(artifacts_dir / "test_predictions.parquet", index=False)
        test_eval_df[save_cols].to_parquet(metrics_dir / "test_predictions.parquet", index=False)

        # Generate evaluation figures for this run
        figures_dir = Path("outputs/figures") / experiment_name
        ensure_dir(figures_dir)
        from src.evaluation.plots import plot_calibration_curves, plot_roc_curves

        y_true = test_eval_df["correct"].to_numpy()
        y_pred = test_eval_df["prediction"].to_numpy()
        pred_dict = {experiment_name: y_pred}
        plot_roc_curves(y_true, pred_dict, figures_dir / "roc_curve.png")
        plot_calibration_curves(y_true, pred_dict, figures_dir / "calibration_curve.png")
        logger.info("Evaluation plots saved to %s", figures_dir)

    logger.info("Evaluation results saved to %s and %s", metrics_dir, artifacts_dir)


if __name__ == "__main__":
    main()
