import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.evaluation.bootstrap import paired_bootstrap_difference
from src.evaluation.metrics import compute_comprehensive_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger
from src.utils.seed import resolve_device

logger = setup_logger(__name__)


def _load_or_eval_bkt(
    bkt_exp_name: str,
    test_df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    bkt_artifacts = Path("outputs/artifacts") / bkt_exp_name
    cache_file = bkt_artifacts / "test_predictions.parquet"
    if cache_file.exists():
        eval_df = pd.read_parquet(cache_file)
        if len(eval_df) == len(test_df) and np.array_equal(eval_df["correct"].values, test_df["correct"].values):
            logger.info("Loading cached BKT test predictions from %s", cache_file)
            preds = eval_df["prediction"].to_numpy()
            metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
            return preds, metrics
        else:
            logger.warning("Cached BKT predictions alignment does not match test_df. Re-evaluating...")

    bkt_params_path = bkt_artifacts / "bkt_params.csv"
    if not bkt_params_path.exists():
        return None, None

    logger.info("Evaluating BKT fallback: %s", bkt_exp_name)
    from src.models.bkt import BKTOnlineEvaluator, load_bkt_params

    skill_models, global_model = load_bkt_params(bkt_params_path)
    evaluator = BKTOnlineEvaluator(skill_models, global_model)
    _, bkt_preds, df_eval = evaluator.predict(test_df)
    bkt_eval = compute_comprehensive_metrics(df_eval, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

    df_eval[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(cache_file, index=False)
    return bkt_preds, bkt_eval


def _load_or_eval_lfa(
    lfa_exp_name: str,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    is_aux: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    lfa_artifacts = Path("outputs/artifacts") / lfa_exp_name
    cache_file = lfa_artifacts / "test_predictions.parquet"
    if cache_file.exists():
        eval_df = pd.read_parquet(cache_file)
        if len(eval_df) == len(test_df) and np.array_equal(eval_df["correct"].values, test_df["correct"].values):
            logger.info("Loading cached LFA (%s) predictions from %s", lfa_exp_name, cache_file)
            preds = eval_df["prediction"].to_numpy()
            metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
            return preds, metrics
        else:
            logger.warning("Cached LFA (%s) predictions alignment does not match test_df. Re-evaluating...", lfa_exp_name)

    lfa_checkpoint = lfa_artifacts / "best_model.pt"
    if not lfa_checkpoint.exists():
        return None, None

    logger.info("Evaluating LFA (%s) fallback on device %s...", lfa_exp_name, device)
    from src.features.encoders import load_encoder
    from src.features.lfa_features import build_lfa_aux_features, build_lfa_core_features, build_lfa_feature_matrix
    from src.models.lfa import LFAAux, LFACore, LFATrainer

    student_encoder = load_encoder(lfa_artifacts / "encoder_student.json")
    skill_encoder = load_encoder(lfa_artifacts / "encoder_skill.json")

    n_students = len(student_encoder.encoder.classes_)
    n_skills = len(skill_encoder.encoder.classes_)

    train_df = df[df["split"] == "train"]
    feature_df, _, _ = build_lfa_core_features(df, train_df)

    checkpoint = torch.load(lfa_checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    n_aux = 0
    if is_aux and "aux_mlp.0.weight" in state_dict:
        n_aux = state_dict["aux_mlp.0.weight"].shape[1]

    if n_aux > 0:
        candidate_cols = [c for c in ["timeTaken", "hintCount", "attemptCount", "scaffold", "hintTotal"] if c in feature_df.columns]
        aux_cols = candidate_cols[:n_aux]
        if aux_cols:
            feature_df, _ = build_lfa_aux_features(feature_df, train_df, aux_cols)
        aux_config = {"enabled": True, "columns": aux_cols} if aux_cols else None
        model = LFAAux(n_students, n_skills, n_aux_features=n_aux, embedding_dim=16)
    else:
        aux_config = None
        model = LFACore(n_students, n_skills, embedding_dim=16)

    feature_matrix = build_lfa_feature_matrix(feature_df, student_encoder, skill_encoder, aux_config=aux_config)
    test_features = feature_matrix[feature_matrix["split"] == "test"].reset_index(drop=True)

    model.load_state_dict(state_dict)
    model.eval()

    trainer = LFATrainer(model=model, device=device)
    lfa_preds = trainer.predict(test_features, batch_size=4096)

    eval_df = test_features.copy()
    eval_df["prediction"] = lfa_preds
    eval_df["skill"] = test_df["skill"].values if len(test_df) == len(test_features) else eval_df["skill_id_encoded"].astype(str)
    eval_df["studentId"] = test_df["studentId"].values if len(test_df) == len(test_features) else eval_df["student_id_encoded"].astype(str)

    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(cache_file, index=False)
    metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
    return lfa_preds, metrics


def _load_or_eval_dbkt(
    dbkt_exp_name: str,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    dbkt_artifacts = Path("outputs/artifacts") / dbkt_exp_name
    cache_file = dbkt_artifacts / "test_predictions.parquet"
    if cache_file.exists():
        eval_df = pd.read_parquet(cache_file)
        if len(eval_df) == len(test_df) and np.array_equal(eval_df["correct"].values, test_df["correct"].values):
            logger.info("Loading cached DBKT predictions from %s", cache_file)
            preds = eval_df["prediction"].to_numpy()
            metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
            return preds, metrics
        else:
            logger.warning("Cached DBKT predictions alignment does not match test_df. Re-evaluating...")

    dbkt_params_path = dbkt_artifacts / "dbkt_params.csv"
    if not dbkt_params_path.exists():
        return None, None

    logger.info("Evaluating DBKT fallback: %s", dbkt_exp_name)
    from src.features.encoders import load_encoder
    from src.models.dbkt import DBKTOnlineEvaluator, DynamicBayesianKnowledgeTracing

    skill_encoder = load_encoder(dbkt_artifacts / "encoder_skill.json") if (dbkt_artifacts / "encoder_skill.json").exists() else None
    n_skills = len(skill_encoder.encoder.classes_) if skill_encoder else len(df["skill"].unique())
    model = DynamicBayesianKnowledgeTracing(num_skills=n_skills)

    checkpoint_path = dbkt_artifacts / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    evaluator = DBKTOnlineEvaluator(model=model, device=device, skill_encoder=skill_encoder)
    _, dbkt_preds, eval_df = evaluator.predict(test_df)
    dbkt_eval = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(cache_file, index=False)
    return dbkt_preds, dbkt_eval


def _load_or_eval_dkt(
    dkt_exp_name: str,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    dkt_artifacts = Path("outputs/artifacts") / dkt_exp_name
    cache_file = dkt_artifacts / "test_predictions.parquet"
    if cache_file.exists():
        eval_df = pd.read_parquet(cache_file)
        if len(eval_df) == len(test_df) and np.array_equal(eval_df["correct"].values, test_df["correct"].values):
            logger.info("Loading cached DKT predictions from %s", cache_file)
            preds = eval_df["prediction"].to_numpy()
            metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
            return preds, metrics
        else:
            logger.warning("Cached DKT predictions alignment does not match test_df. Re-evaluating...")

    dkt_checkpoint = dkt_artifacts / "best_model.pt"
    if not dkt_checkpoint.exists():
        return None, None

    logger.info("Evaluating DKT fallback: %s on device %s", dkt_exp_name, device)
    from src.features.encoders import load_encoder
    from src.models.dkt import DKTOnlineEvaluator, DeepKnowledgeTracing

    skill_encoder = load_encoder(dkt_artifacts / "encoder_skill.json")
    n_skills = len(skill_encoder.encoder.classes_)

    checkpoint = torch.load(dkt_checkpoint, map_location=device)
    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = DeepKnowledgeTracing(
        num_skills=n_skills,
        embedding_dim=saved_cfg.get("embedding_dim", 64),
        hidden_size=saved_cfg.get("hidden_size", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    test_temp = test_df.copy()
    test_temp["skill_id_encoded"] = skill_encoder.transform(test_temp["skill"].values)

    evaluator = DKTOnlineEvaluator(model, device=device)
    _, dkt_preds, eval_df = evaluator.predict(test_temp)
    dkt_eval = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(cache_file, index=False)
    return dkt_preds, dkt_eval


def _load_or_eval_attention(
    attention_exp_name: str,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    attention_artifacts = Path("outputs/artifacts") / attention_exp_name
    cache_file = attention_artifacts / "test_predictions.parquet"
    if cache_file.exists():
        eval_df = pd.read_parquet(cache_file)
        if len(eval_df) == len(test_df) and np.array_equal(eval_df["correct"].values, test_df["correct"].values):
            logger.info("Loading cached Attention-KT predictions from %s", cache_file)
            preds = eval_df["prediction"].to_numpy()
            metrics = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")
            return preds, metrics
        else:
            logger.warning("Cached Attention-KT predictions alignment does not match test_df. Re-evaluating...")

    attention_checkpoint = attention_artifacts / "best_model.pt"
    if not attention_checkpoint.exists():
        return None, None

    logger.info("Evaluating Attention-KT fallback: %s on device %s", attention_exp_name, device)
    from src.features.encoders import load_encoder
    from src.models.attention_kt import AttentionOnlineEvaluator, AttentiveContextualKT

    skill_encoder = load_encoder(attention_artifacts / "encoder_skill.json")
    n_skills = len(skill_encoder.encoder.classes_)

    checkpoint = torch.load(attention_checkpoint, map_location=device)
    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = AttentiveContextualKT(
        num_skills=n_skills,
        embedding_dim=saved_cfg.get("embedding_dim", 64),
        num_heads=saved_cfg.get("num_heads", 2),
        num_layers=saved_cfg.get("num_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    test_temp = test_df.copy()
    test_temp["skill_id_encoded"] = skill_encoder.transform(test_temp["skill"].values)

    evaluator = AttentionOnlineEvaluator(model, device=device)
    _, attention_preds, eval_df = evaluator.predict(test_temp)
    attention_eval = compute_comprehensive_metrics(eval_df, y_true_col="correct", y_pred_col="prediction", skill_col="skill", student_col="studentId")

    eval_df[["studentId", "skill", "correct", "prediction", "split"]].to_parquet(cache_file, index=False)
    return attention_preds, attention_eval


@hydra.main(version_base=None, config_path="../configs", config_name="compare")
def main(cfg: DictConfig) -> None:
    logger.info("Starting model comparison...")

    version = str(cfg.get("version", "")) if cfg.get("version") is not None else ""
    if version and version.lower() not in ["none", "null", ""]:
        comparison_dir = Path("outputs/comparison") / version
        figures_dir = Path("outputs/figures/comparison") / version
        rel_fig_path = f"../../figures/comparison/{version}"
        default_suffix = f"_{version}"
    else:
        comparison_dir = Path("outputs/comparison")
        figures_dir = Path("outputs/figures/comparison")
        rel_fig_path = "../figures/comparison"
        default_suffix = "_v1"

    ensure_dir(comparison_dir)
    ensure_dir(figures_dir)

    results_dir = Path("outputs/metrics")

    device = resolve_device(cfg.get("device", "auto"))
    logger.info("Model comparison target device: %s | Output dir: %s", device, comparison_dir)

    model_results: Dict[str, Any] = {}
    prediction_map: Dict[str, np.ndarray] = {}

    # 1. Check processed dataset
    from src.utils.io import get_latest_data_version
    raw_proc_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed/assistments2017"))
    processed_dir = get_latest_data_version(raw_proc_dir)
    data_path = processed_dir / "interactions.parquet"
    if not data_path.exists():
        logger.warning("Processed data not found at %s", data_path)
        return

    df = pd.read_parquet(data_path)
    sort_cols = ["studentId", "startTime"]
    if "action_num" in df.columns:
        sort_cols.append("action_num")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["attempts_before"] = df.groupby(["studentId", "skill"]).cumcount()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # 2. Compute baseline metrics dynamically
    from src.models.baselines import evaluate_baselines
    baseline_path = results_dir / "baseline_metrics.json"
    logger.info("Computing fresh baseline models (Majority, Skill-Average, Student-Average)...")
    baseline_results = evaluate_baselines(train_df, val_df, test_df)
    save_json(baseline_results, baseline_path)

    for name, metrics in baseline_results.items():
        if name != "global_rate" and isinstance(metrics, dict) and "test" in metrics:
            model_results[f"baseline_{name}"] = metrics

    def get_exp(key: str, default: str) -> str:
        val = cfg.get(key)
        if val is None or str(val).strip().lower() in ["none", "null", ""]:
            return default
        return str(val).strip()

    # Extract target experiment names from config or CLI overrides
    bkt_exp_name = get_exp("bkt_exp", f"bkt_assistments2017{default_suffix}")
    lfa_core_exp_name = get_exp("lfa_core_exp", f"lfa_core_assistments2017{default_suffix}")
    lfa_aux_exp_name = get_exp("lfa_aux_exp", f"lfa_aux_assistments2017{default_suffix}")
    dbkt_core_exp_name = get_exp("dbkt_core_exp", f"dbkt_core_assistments2017{default_suffix}")
    dkt_core_exp_name = get_exp("dkt_core_exp", f"dkt_core_assistments2017{default_suffix}")
    attention_core_exp_name = get_exp("attention_core_exp", f"attention_core_assistments2017{default_suffix}")

    logger.info("Target experiments for comparison (version suffix: '%s'):", default_suffix)
    logger.info("  BKT: %s", bkt_exp_name)
    logger.info("  LFA-core: %s", lfa_core_exp_name)
    logger.info("  LFA-aux: %s", lfa_aux_exp_name)
    logger.info("  DBKT-core: %s", dbkt_core_exp_name)
    logger.info("  DKT-core: %s", dkt_core_exp_name)
    logger.info("  Attention-core: %s", attention_core_exp_name)

    # Load / Evaluate BKT
    bkt_preds, bkt_eval = _load_or_eval_bkt(bkt_exp_name, test_df)
    if bkt_eval is not None:
        model_results[f"bkt ({bkt_exp_name})"] = bkt_eval
        prediction_map["bkt"] = bkt_preds

    # Load / Evaluate LFA-core
    lfa_core_preds, lfa_core_eval = _load_or_eval_lfa(lfa_core_exp_name, df, test_df, device, is_aux=False)
    if lfa_core_eval is not None:
        model_results[f"lfa_core ({lfa_core_exp_name})"] = lfa_core_eval
        prediction_map["lfa_core"] = lfa_core_preds

    # Load / Evaluate LFA-aux
    lfa_aux_preds, lfa_aux_eval = _load_or_eval_lfa(lfa_aux_exp_name, df, test_df, device, is_aux=True)
    if lfa_aux_eval is not None:
        model_results[f"lfa_aux ({lfa_aux_exp_name})"] = lfa_aux_eval
        prediction_map["lfa_aux"] = lfa_aux_preds

    # Load / Evaluate DBKT-core
    dbkt_core_preds, dbkt_core_eval = _load_or_eval_dbkt(dbkt_core_exp_name, df, test_df, device)
    if dbkt_core_eval is not None:
        model_results[f"dbkt_core ({dbkt_core_exp_name})"] = dbkt_core_eval
        prediction_map["dbkt_core"] = dbkt_core_preds

    # Load / Evaluate DKT-core
    dkt_core_preds, dkt_core_eval = _load_or_eval_dkt(dkt_core_exp_name, df, test_df, device)
    if dkt_core_eval is not None:
        model_results[f"dkt_core ({dkt_core_exp_name})"] = dkt_core_eval
        prediction_map["dkt_core"] = dkt_core_preds

    # Load / Evaluate Attention-core
    attention_core_preds, attention_core_eval = _load_or_eval_attention(attention_core_exp_name, df, test_df, device)
    if attention_core_eval is not None:
        model_results[f"attention_core ({attention_core_exp_name})"] = attention_core_eval
        prediction_map["attention_core"] = attention_core_preds

    # 3. Bootstrap Hypothesis Testing
    comparisons: Dict[str, Any] = {}
    y_test = test_df["correct"].values

    model_pairs = [
        ("bkt", "lfa_core"),
        ("bkt", "dbkt_core"),
        ("bkt", "dkt_core"),
        ("bkt", "attention_core"),
        ("lfa_core", "dbkt_core"),
        ("lfa_core", "dkt_core"),
        ("lfa_core", "attention_core"),
        ("dbkt_core", "dkt_core"),
        ("dbkt_core", "attention_core"),
        ("dkt_core", "attention_core"),
        ("lfa_core", "lfa_aux"),
    ]

    from joblib import Parallel, delayed

    def _eval_pair_bootstrap(model_a: str, model_b: str):
        pred_a = prediction_map.get(model_a)
        pred_b = prediction_map.get(model_b)
        if pred_a is not None and pred_b is not None and len(pred_a) == len(pred_b) == len(y_test):
            comp = paired_bootstrap_difference(y_test, pred_a, pred_b)
            return f"{model_a}_vs_{model_b}", {
                "model_a": model_a,
                "model_b": model_b,
                **comp,
            }
        return None

    logger.info("Running paired bootstrap hypothesis tests across 4 CPU cores (n_jobs=4)...")
    bootstrap_results = Parallel(n_jobs=4)(
        delayed(_eval_pair_bootstrap)(model_a, model_b)
        for model_a, model_b in model_pairs
    )

    for item in bootstrap_results:
        if item is not None:
            comp_key, comp_dict = item
            comparisons[comp_key] = comp_dict

    model_results["comparisons"] = comparisons

    save_json(model_results, comparison_dir / "model_comparison.json")
    logger.info("Comparison results saved to %s", comparison_dir / "model_comparison.json")

    # 4. Generate Comprehensive CSV and Markdown Reports
    summary_rows = []
    md_lines = [
        "# KT Model Comparison & Statistical Evaluation Report",
        f"\n**Generated At:** {pd.Timestamp.now().isoformat()}",
        f"**Dataset:** ASSISTments 2017 (Released Full Dataset, Strict Temporal Split)",
        f"**Test Instances:** {len(test_df):,} interactions\n",
        "## 1. Overall Performance Metrics Table\n",
        "| Model | AUC | Accuracy | LogLoss | Brier Score | RMSE | ECE | Macro-AUC | Weighted-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, data in model_results.items():
        if model_name == "comparisons":
            continue
        g = data.get("global", {}) if "global" in data else data.get("test", {})
        auc_val = g.get("auc", 0.0)
        acc_val = g.get("accuracy", 0.0)
        ll_val = g.get("logloss", 0.0)
        brier_val = g.get("brier", 0.0)
        rmse_val = g.get("rmse", 0.0)
        ece_val = g.get("ece", 0.0)
        macro_auc_val = g.get("skill_macro_auc", 0.0)
        weighted_auc_val = g.get("skill_weighted_auc", 0.0)

        auc = f"{auc_val:.4f}" if "auc" in g else "N/A"
        acc = f"{acc_val*100:.2f}%" if "accuracy" in g else "N/A"
        ll = f"{ll_val:.4f}" if "logloss" in g else "N/A"
        brier = f"{brier_val:.4f}" if "brier" in g else "N/A"
        rmse = f"{rmse_val:.4f}" if "rmse" in g else "N/A"
        ece = f"{ece_val:.4f}" if "ece" in g else "N/A"
        macro_auc = f"{macro_auc_val:.4f}" if "skill_macro_auc" in g else "N/A"
        weighted_auc = f"{weighted_auc_val:.4f}" if "skill_weighted_auc" in g else "N/A"

        summary_rows.append({
            "model": model_name,
            "auc": auc_val,
            "accuracy": acc_val,
            "logloss": ll_val,
            "brier": brier_val,
            "rmse": rmse_val,
            "ece": ece_val,
            "skill_macro_auc": macro_auc_val,
            "skill_weighted_auc": weighted_auc_val,
        })

        display_name = {
            "baseline_majority": "Majority Baseline",
            "baseline_skill_average": "Skill Average Baseline",
            "baseline_student_average": "Student Average Baseline",
            "bkt": "**BKT (Bayesian KT)**",
            "lfa_core": "**LFA-core**",
            "lfa_aux": "**LFA-aux**",
        }.get(model_name, model_name)

        md_lines.append(f"| {display_name} | {auc} | {acc} | {ll} | {brier} | {rmse} | {ece} | {macro_auc} | {weighted_auc} |")

    # 4. Generate Visual Plots
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    
    # Generate all comparison figures
    auc_ranking_path = figures_dir / "01_model_auc_ranking.png"
    bar_path = figures_dir / "02_skill_auc_breakdown.png"
    roc_path = figures_dir / "03_roc_curves.png"
    calib_path = figures_dir / "04_calibration_curves.png"
    bucket_path = figures_dir / "05_bucket_analysis.png"
    forest_path = figures_dir / "06_bootstrap_forest_plot.png"

    from src.evaluation.plots import (
        generate_mermaid_comparison_chart,
        plot_auc_comparison_bar,
        plot_bootstrap_forest,
        plot_bucket_analysis,
        plot_calibration_curves,
        plot_model_comparison_bars,
        plot_roc_curves,
    )

    if not summary_df.empty:
        plot_auc_comparison_bar(summary_df, auc_ranking_path)
        plot_model_comparison_bars(summary_df, bar_path)
    
    if prediction_map:
        calib_strategy = str(cfg.get("calibration_strategy", "uniform")).lower()
        calib_min_samples = int(cfg.get("calibration_min_samples", 20))
        calib_n_bins = int(cfg.get("calibration_n_bins", 10))

        plot_roc_curves(y_test, prediction_map, roc_path)
        plot_calibration_curves(
            y_test,
            prediction_map,
            calib_path,
            n_bins=calib_n_bins,
            strategy=calib_strategy,
            min_bin_samples=calib_min_samples,
        )

    plot_bucket_analysis(model_results, bucket_path)

    if comparisons:
        plot_bootstrap_forest(comparisons, forest_path)

    mermaid_chart = generate_mermaid_comparison_chart(summary_df)

    # 5. Generate Comprehensive CSV and Markdown Reports
    summary_df.to_csv(comparison_dir / "model_comparison.csv", index=False)
    logger.info("Summary CSV saved to %s", comparison_dir / "model_comparison.csv")

    md_lines.append("\n### 1.1. Visual Ranking (Interactive Architecture Diagram)")
    if mermaid_chart:
        md_lines.append(mermaid_chart)
    
    md_lines.append("\n### 1.2. Model ROC-AUC Ranking Chart")
    md_lines.append(f"![Model AUC Ranking]({rel_fig_path}/01_model_auc_ranking.png)")

    md_lines.append("\n### 1.3. Skill-Level (Macro vs Weighted) AUC Breakdown")
    md_lines.append(f"![Skill AUC Breakdown]({rel_fig_path}/02_skill_auc_breakdown.png)")

    md_lines.append("\n## 2. Discrimination & Probability Calibration Curves\n")
    md_lines.append("### 2.1. Receiver Operating Characteristic (ROC) Curves")
    md_lines.append(f"![ROC Curves]({rel_fig_path}/03_roc_curves.png)\n")
    md_lines.append("### 2.2. Reliability Diagrams (Calibration Curves)")
    md_lines.append(f"![Calibration Curves]({rel_fig_path}/04_calibration_curves.png)\n")

    md_lines.append("\n## 3. Statistical Significance & Paired Bootstrap Test (95% CI)\n")
    md_lines.append(f"![Bootstrap Forest Plot]({rel_fig_path}/06_bootstrap_forest_plot.png)\n")
    if comparisons:
        for comp_name, comp_data in comparisons.items():
            md_lines.append(f"### Comparison: `{comp_data['model_a']}` vs `{comp_data['model_b']}`")
            md_lines.append(f"- **Mean $\\Delta$AUC:** `{comp_data['mean_diff']:+.6f}`")
            md_lines.append(f"- **95% Bootstrap Confidence Interval:** `[{comp_data['ci_lower']:+.6f}, {comp_data['ci_upper']:+.6f}]`")
            md_lines.append(f"- **Interpretation:** {comp_data['interpretation']}")
            md_lines.append("")
    else:
        md_lines.append("No paired bootstrap comparisons available.")

    # 6. Bucket Analysis Tables
    md_lines.append("\n## 4. Fine-Grained Bucket Analysis (AUC Breakdown)\n")
    md_lines.append(f"![Bucket Analysis]({rel_fig_path}/05_bucket_analysis.png)\n")
    md_lines.append("### 4.1. By Skill Frequency")
    md_lines.append("| Model | Rare (<100) | Medium (100-1000) | Frequent (>1000) |")
    md_lines.append("| :--- | :---: | :---: | :---: |")

    for model_name, data in model_results.items():
        if model_name == "comparisons":
            continue
        b = data.get("buckets", {}).get("by_skill_frequency", {})
        r_auc = f"{b.get('<100', {}).get('auc', 0.0):.4f}" if "<100" in b else "N/A"
        m_auc = f"{b.get('100-1000', {}).get('auc', 0.0):.4f}" if "100-1000" in b else "N/A"
        f_auc = f"{b.get('>1000', {}).get('auc', 0.0):.4f}" if ">1000" in b else "N/A"
        md_lines.append(f"| {model_name} | {r_auc} | {m_auc} | {f_auc} |")

    md_lines.append("\n### 4.2. By Student History Length")
    md_lines.append("| Model | Short (<10) | Medium (10-50) | Long (>50) |")
    md_lines.append("| :--- | :---: | :---: | :---: |")

    for model_name, data in model_results.items():
        if model_name == "comparisons":
            continue
        b = data.get("buckets", {}).get("by_student_history", {})
        s_auc = f"{b.get('<10', {}).get('auc', 0.0):.4f}" if "<10" in b else "N/A"
        m_auc = f"{b.get('10-50', {}).get('auc', 0.0):.4f}" if "10-50" in b else "N/A"
        l_auc = f"{b.get('>50', {}).get('auc', 0.0):.4f}" if ">50" in b else "N/A"
        md_lines.append(f"| {model_name} | {s_auc} | {m_auc} | {l_auc} |")

    md_path = comparison_dir / "model_comparison.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Markdown report with embedded plots saved to %s", md_path)


if __name__ == "__main__":
    main()

