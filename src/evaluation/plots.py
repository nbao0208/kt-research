import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc as calc_auc

from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)

# Professional academic style palette
PALETTE = {
    "majority": "#95a5a6",
    "skill_average": "#7f8c8d",
    "student_average": "#bdc3c7",
    "baseline_majority": "#95a5a6",
    "baseline_skill_average": "#7f8c8d",
    "baseline_student_average": "#bdc3c7",
    "bkt": "#2980b9",
    "lfa_core": "#e67e22",
    "lfa_aux": "#d35400",
    "dbkt_core": "#27ae60",
    "dkt_core": "#8e44ad",
    "attention_core": "#c0392b",
}

DISPLAY_NAMES = {
    "baseline_majority": "Majority Baseline",
    "baseline_skill_average": "Skill Avg Baseline",
    "baseline_student_average": "Student Avg Baseline",
    "majority": "Majority Baseline",
    "skill_average": "Skill Avg Baseline",
    "student_average": "Student Avg Baseline",
    "bkt": "BKT",
    "lfa_core": "LFA-core",
    "lfa_aux": "LFA-aux",
    "dbkt_core": "DBKT-core",
    "dkt_core": "DKT-core",
    "attention_core": "Attention-core",
}


def _get_color(model_key: str, idx: int = 0) -> str:
    cleaned_key = model_key.lower().split("(")[0].strip()
    for key, color in PALETTE.items():
        if key == cleaned_key or key in cleaned_key:
            return color
    default_colors = sns.color_palette("tab10")
    return default_colors[idx % len(default_colors)]


def _get_display_name(model_key: str) -> str:
    cleaned_key = model_key.lower().split("(")[0].strip()
    for key, name in DISPLAY_NAMES.items():
        if key == cleaned_key or cleaned_key.startswith(key):
            return name
    return model_key.split("(")[0].strip()


def plot_auc_comparison_bar(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> Optional[Path]:
    """
    Plots a dedicated, publication-quality Bar Chart comparing Test AUC across all models,
    sorted by performance with distinct model category colors and exact data labels.
    """
    if summary_df.empty or "auc" not in summary_df.columns:
        logger.warning("Empty summary DataFrame provided to plot_auc_comparison_bar")
        return None

    ensure_dir(output_path.parent)

    # Sort models by AUC ascending so highest is on top (or on the right)
    df_sorted = summary_df.sort_values(by="auc", ascending=True).reset_index(drop=True)

    models = df_sorted["model"].tolist()
    labels = [_get_display_name(m) for m in models]
    auc_vals = np.array(df_sorted["auc"], dtype=np.float64, copy=True)
    colors = [_get_color(m, i) for i, m in enumerate(models)]

    fig, ax = plt.subplots(figsize=(10, max(5.5, len(models) * 0.65)), dpi=300)

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, auc_vals, height=0.6, color=colors, edgecolor="white", linewidth=1.2)

    # Reference line for random guessing
    ax.axvline(0.5, color="#e74c3c", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("ROC-AUC Score", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Knowledge Tracing Models: Test ROC-AUC Comparison", fontsize=14, fontweight="bold", pad=15)
    
    # Set proper x-axis limits with room for text labels
    max_auc = max(float(np.max(auc_vals)), 0.8)
    ax.set_xlim(0.0, min(1.0, max_auc + 0.12))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate AUC value on each bar
    for bar, val in zip(bars, auc_vals):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved dedicated AUC comparison bar chart to %s", output_path)
    return output_path


def plot_model_comparison_bars(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> Optional[Path]:
    """
    Plots a multi-metric grouped bar chart comparing overall AUC, Macro-AUC, and Weighted-AUC
    across all Knowledge Tracing models.
    """
    if summary_df.empty:
        logger.warning("Empty summary DataFrame provided to plot_model_comparison_bars")
        return None

    ensure_dir(output_path.parent)

    models = summary_df["model"].tolist()
    labels = [_get_display_name(m) for m in models]

    auc_vals = np.array(summary_df.get("auc", pd.Series(0.0, index=summary_df.index)), dtype=np.float64, copy=True)
    macro_auc = np.array(summary_df.get("skill_macro_auc", pd.Series(0.0, index=summary_df.index)), dtype=np.float64, copy=True)
    weighted_auc = np.array(summary_df.get("skill_weighted_auc", pd.Series(0.0, index=summary_df.index)), dtype=np.float64, copy=True)

    # For baselines that only have global AUC, fill macro/weighted with global AUC for consistent visual display
    for i in range(len(models)):
        if macro_auc[i] == 0.0 and auc_vals[i] > 0.0:
            macro_auc[i] = auc_vals[i]
        if weighted_auc[i] == 0.0 and auc_vals[i] > 0.0:
            weighted_auc[i] = auc_vals[i]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.35), 6), dpi=300)

    rects1 = ax.bar(x - width, auc_vals, width, label="Overall AUC", color="#2980b9", edgecolor="white", linewidth=1)
    rects2 = ax.bar(x, macro_auc, width, label="Skill Macro-AUC", color="#3498db", edgecolor="white", linewidth=1)
    rects3 = ax.bar(x + width, weighted_auc, width, label="Skill Weighted-AUC", color="#5dade2", edgecolor="white", linewidth=1)

    ax.set_ylabel("AUC Score", fontsize=12, fontweight="bold")
    ax.set_title("Overall vs. Skill-Level (Macro/Weighted) AUC Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="#e74c3c", linestyle="--", alpha=0.7, label="Random Guess (0.50)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, shadow=True, loc="lower right", fontsize=10)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0.1:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    rotation=90,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved model comparison bar chart to %s", output_path)
    return output_path


def plot_roc_curves(
    y_true: np.ndarray,
    prediction_map: Dict[str, np.ndarray],
    output_path: Path,
) -> Optional[Path]:
    """
    Plots multi-model ROC Curves on the same graph with AUC scores in the legend.
    """
    if len(y_true) == 0 or not prediction_map:
        return None

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Random (AUC = 0.5000)")

    # Sort curves by AUC descending for clean legend
    scored_models = []
    for model_name, y_pred in prediction_map.items():
        if y_pred is None or len(y_pred) != len(y_true):
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            score = calc_auc(fpr, tpr)
            scored_models.append((score, model_name, fpr, tpr))
        except Exception as e:
            logger.warning("Failed to compute ROC for %s: %s", model_name, e)

    scored_models.sort(key=lambda x: x[0], reverse=True)

    for idx, (score, model_name, fpr, tpr) in enumerate(scored_models):
        display = _get_display_name(model_name)
        color = _get_color(model_name, idx)
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{display} (AUC = {score:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved ROC curves to %s", output_path)
    return output_path


def plot_calibration_curves(
    y_true: np.ndarray,
    prediction_map: Dict[str, np.ndarray],
    output_path: Path,
    n_bins: int = 10,
    strategy: str = "uniform",
    min_bin_samples: int = 20,
) -> Optional[Path]:
    """
    Plots Reliability Diagrams / Calibration Curves comparing predicted confidence
    against empirical accuracy along with perfect calibration reference.
    
    Supports:
    - strategy="uniform": Equal-width bins with minimum sample filtering (filters out sparse tail bins N < min_bin_samples).
    - strategy="quantile": Equal-frequency (quantile) bins where each bin contains an equal number of samples.
    """
    if len(y_true) == 0 or not prediction_map:
        return None

    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.5, label="Perfect Calibration")

    effective_min_samples = min(min_bin_samples, max(1, len(y_true) // (2 * n_bins)))

    for idx, (model_name, y_pred) in enumerate(prediction_map.items()):
        if y_pred is None or len(y_pred) != len(y_true):
            continue
        try:
            if strategy == "quantile":
                try:
                    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
                except Exception:
                    # Fallback to uniform if quantile splitting fails (e.g. duplicate quantiles)
                    strategy_fallback = "uniform"
                else:
                    strategy_fallback = "quantile"
            else:
                strategy_fallback = "uniform"

            if strategy_fallback == "uniform":
                bins = np.linspace(0.0, 1.0, n_bins + 1)
                bin_indices = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
                prob_pred_list, prob_true_list = [], []
                for b in range(n_bins):
                    mask = (bin_indices == b)
                    if np.sum(mask) >= effective_min_samples:
                        prob_pred_list.append(float(np.mean(y_pred[mask])))
                        prob_true_list.append(float(np.mean(y_true[mask])))
                prob_true = np.array(prob_true_list)
                prob_pred = np.array(prob_pred_list)

            if len(prob_pred) > 0:
                display = _get_display_name(model_name)
                color = _get_color(model_name, idx)
                ax.plot(prob_pred, prob_true, marker="o", lw=2, color=color, label=f"{display}")
        except Exception as e:
            logger.warning("Failed to compute calibration for %s: %s", model_name, e)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Probability (Fraction of Positives)", fontsize=12, fontweight="bold")
    ax.set_title("Reliability Diagrams / Calibration Curves", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved calibration curves to %s", output_path)
    return output_path


def plot_bucket_analysis(
    model_results: Dict[str, Any],
    output_path: Path,
) -> Optional[Path]:
    """
    Plots a 2-panel figure showing AUC breakdown by:
    1. Skill Frequency Buckets (<100, 100-1000, >1000)
    2. Student History Length Buckets (<10, 10-50, >50)
    """
    if not model_results:
        return None

    ensure_dir(output_path.parent)

    models = [m for m in model_results.keys() if m != "comparisons"]
    if not models:
        return None

    labels = [_get_display_name(m) for m in models]
    x = np.arange(len(models))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Skill Frequency Panel
    rare_aucs = []
    med_aucs = []
    freq_aucs = []
    for m in models:
        b = model_results[m].get("buckets", {}).get("by_skill_frequency", {})
        rare_aucs.append(b.get("<100", {}).get("auc", 0.5))
        med_aucs.append(b.get("100-1000", {}).get("auc", 0.5))
        freq_aucs.append(b.get(">1000", {}).get("auc", 0.5))

    ax1.bar(x - width, rare_aucs, width, label="Rare (<100)", color="#e74c3c", edgecolor="white")
    ax1.bar(x, med_aucs, width, label="Medium (100-1000)", color="#f39c12", edgecolor="white")
    ax1.bar(x + width, freq_aucs, width, label="Frequent (>1000)", color="#27ae60", edgecolor="white")

    ax1.set_ylabel("AUC Score", fontsize=11, fontweight="bold")
    ax1.set_title("AUC by Skill Frequency", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim(0.0, 1.05)
    ax1.axhline(0.5, color="#e74c3c", linestyle="--", alpha=0.6)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="lower right", fontsize=9)

    # 2. Student History Panel
    short_aucs = []
    mid_aucs = []
    long_aucs = []
    for m in models:
        b = model_results[m].get("buckets", {}).get("by_student_history", {})
        short_aucs.append(b.get("<10", {}).get("auc", 0.5))
        mid_aucs.append(b.get("10-50", {}).get("auc", 0.5))
        long_aucs.append(b.get(">50", {}).get("auc", 0.5))

    ax2.bar(x - width, short_aucs, width, label="Short (<10)", color="#9b59b6", edgecolor="white")
    ax2.bar(x, mid_aucs, width, label="Medium (10-50)", color="#3498db", edgecolor="white")
    ax2.bar(x + width, long_aucs, width, label="Long (>50)", color="#1abc9c", edgecolor="white")

    ax2.set_ylabel("AUC Score", fontsize=11, fontweight="bold")
    ax2.set_title("AUC by Student History Length", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.axhline(0.5, color="#e74c3c", linestyle="--", alpha=0.6)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved bucket analysis chart to %s", output_path)
    return output_path


def plot_bootstrap_forest(
    comparisons: Dict[str, Any],
    output_path: Path,
) -> Optional[Path]:
    """
    Plots a Forest Plot for Paired Bootstrap Hypothesis Tests showing:
    - Delta AUC (mean difference)
    - 95% Confidence Interval error bars
    - Zero line (statistical significance indicator)
    """
    if not comparisons:
        return None

    ensure_dir(output_path.parent)

    pair_names = []
    mean_diffs = []
    ci_lowers = []
    ci_uppers = []
    significant_flags = []

    for comp_key, comp_data in comparisons.items():
        m_a = _get_display_name(comp_data.get("model_a", ""))
        m_b = _get_display_name(comp_data.get("model_b", ""))
        label = f"{m_a} vs {m_b}"
        pair_names.append(label)

        mean_diff = comp_data.get("mean_diff", 0.0)
        ci_l = comp_data.get("ci_lower", 0.0)
        ci_u = comp_data.get("ci_upper", 0.0)
        sig = comp_data.get("significant", False)

        mean_diffs.append(mean_diff)
        ci_lowers.append(ci_l)
        ci_uppers.append(ci_u)
        significant_flags.append(sig)

    y_pos = np.arange(len(pair_names))
    x_err_low = np.array(mean_diffs) - np.array(ci_lowers)
    x_err_high = np.array(ci_uppers) - np.array(mean_diffs)
    x_err = [np.maximum(0, x_err_low), np.maximum(0, x_err_high)]

    fig, ax = plt.subplots(figsize=(10, max(5, len(pair_names) * 0.45)), dpi=300)

    for idx in range(len(pair_names)):
        ecolor = "#27ae60" if significant_flags[idx] else "#7f8c8d"
        xerr_i = [[x_err[0][idx]], [x_err[1][idx]]]
        ax.errorbar(
            mean_diffs[idx],
            y_pos[idx],
            xerr=xerr_i,
            fmt="o",
            color="#2c3e50",
            ecolor=ecolor,
            elinewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=6,
        )

    ax.axvline(0.0, color="#e74c3c", linestyle="--", lw=1.5, label="Null Effect (ΔAUC = 0)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean ΔAUC with 95% Bootstrap Confidence Interval", fontsize=11, fontweight="bold")
    ax.set_title("Paired Bootstrap Hypothesis Testing (Forest Plot)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#27ae60", lw=2, label="Statistically Significant (p < 0.05)"),
        Line2D([0], [0], color="#7f8c8d", lw=2, label="Not Significant (CI contains 0)"),
        Line2D([0], [0], color="#e74c3c", linestyle="--", lw=1.5, label="Zero Line"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, shadow=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved bootstrap forest plot to %s", output_path)
    return output_path


def generate_mermaid_comparison_chart(summary_df: pd.DataFrame) -> str:
    """
    Generates Mermaid Flowchart ranking diagram for reliable rendering across all Markdown viewers.
    """
    if summary_df.empty or "auc" not in summary_df.columns:
        return ""

    df_sorted = summary_df.sort_values(by="auc", ascending=False).reset_index(drop=True)
    
    lines = [
        "```mermaid",
        "flowchart TD",
        "    classDef topRank fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff,font-weight:bold;",
        "    classDef midRank fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff,font-weight:bold;",
        "    classDef baseRank fill:#7f8c8d,stroke:#95a5a6,stroke-width:1px,color:#fff;",
    ]

    nodes = []
    for rank, row in enumerate(df_sorted.itertuples(), start=1):
        m_name = _get_display_name(getattr(row, "model"))
        auc_val = float(getattr(row, "auc"))
        acc_val = float(getattr(row, "accuracy", 0.0)) * 100
        node_id = f"M{rank}"
        style_class = "topRank" if rank <= 2 else ("baseRank" if "baseline" in str(getattr(row, "model")).lower() else "midRank")
        nodes.append(f'    {node_id}["Rank {rank}: {m_name} <br/> AUC: {auc_val:.4f} | Acc: {acc_val:.2f}%"]:::{style_class}')

    lines.extend(nodes)
    if len(nodes) > 1:
        flow = " --> ".join([f"M{i}" for i in range(1, len(nodes) + 1)])
        lines.append(f"    {flow}")

    lines.append("```")
    return "\n".join(lines)
