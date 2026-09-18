from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plots import (
    generate_mermaid_comparison_chart,
    plot_auc_comparison_bar,
    plot_bootstrap_forest,
    plot_bucket_analysis,
    plot_calibration_curves,
    plot_model_comparison_bars,
    plot_roc_curves,
)


class TestPlots:
    def test_plot_auc_comparison_bar(self, tmp_path: Path):
        summary_df = pd.DataFrame({
            "model": ["majority", "skill_average", "bkt", "lfa_core", "dkt_core", "attention_core"],
            "auc": [0.50, 0.65, 0.712, 0.725, 0.748, 0.756],
        })
        out_file = tmp_path / "auc_bars.png"
        res = plot_auc_comparison_bar(summary_df, out_file)
        assert res is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_plot_model_comparison_bars(self, tmp_path: Path):
        summary_df = pd.DataFrame({
            "model": ["baseline_majority", "baseline_skill_average", "bkt", "lfa_core", "dkt_core", "attention_core"],
            "auc": [0.500, 0.650, 0.712, 0.725, 0.748, 0.756],
            "accuracy": [0.65, 0.68, 0.68, 0.69, 0.71, 0.72],
            "skill_macro_auc": [0.0, 0.0, 0.701, 0.715, 0.735, 0.742],
            "skill_weighted_auc": [0.0, 0.0, 0.710, 0.722, 0.745, 0.753],
        })
        out_file = tmp_path / "bars.png"
        res = plot_model_comparison_bars(summary_df, out_file)
        assert res is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_plot_model_comparison_bars_empty(self, tmp_path: Path):
        summary_df = pd.DataFrame()
        out_file = tmp_path / "empty_bars.png"
        res = plot_model_comparison_bars(summary_df, out_file)
        assert res is None
        assert not out_file.exists()

    def test_plot_roc_curves(self, tmp_path: Path):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        pred_map = {
            "model_a": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.85, 0.3, 0.15, 0.95, 0.05]),
            "model_b": np.array([0.6, 0.4, 0.65, 0.35, 0.7, 0.6, 0.4, 0.45, 0.8, 0.2]),
        }
        out_file = tmp_path / "roc.png"
        res = plot_roc_curves(y_true, pred_map, out_file)
        assert res is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_plot_calibration_curves(self, tmp_path: Path):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 5)
        pred_map = {
            "model_a": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.85, 0.3, 0.15, 0.95, 0.05] * 5),
            "model_b": np.array([0.6, 0.4, 0.65, 0.35, 0.7, 0.6, 0.4, 0.45, 0.8, 0.2] * 5),
        }
        out_file = tmp_path / "calib_uniform.png"
        res = plot_calibration_curves(y_true, pred_map, out_file, n_bins=5, strategy="uniform", min_bin_samples=2)
        assert res is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0

        out_quantile = tmp_path / "calib_quantile.png"
        res_q = plot_calibration_curves(y_true, pred_map, out_quantile, n_bins=5, strategy="quantile")
        assert res_q is not None
        assert out_quantile.exists()
        assert out_quantile.stat().st_size > 0

    def test_plot_bucket_analysis(self, tmp_path: Path):
        model_results = {
            "bkt": {
                "buckets": {
                    "by_skill_frequency": {"<100": {"auc": 0.65}, "100-1000": {"auc": 0.70}, ">1000": {"auc": 0.73}},
                    "by_student_history": {"<10": {"auc": 0.62}, "10-50": {"auc": 0.69}, ">50": {"auc": 0.74}},
                }
            },
            "dkt_core": {
                "buckets": {
                    "by_skill_frequency": {"<100": {"auc": 0.68}, "100-1000": {"auc": 0.74}, ">1000": {"auc": 0.77}},
                    "by_student_history": {"<10": {"auc": 0.66}, "10-50": {"auc": 0.73}, ">50": {"auc": 0.78}},
                }
            },
        }
        out_file = tmp_path / "buckets.png"
        res = plot_bucket_analysis(model_results, out_file)
        assert res is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_plot_bootstrap_forest(self, tmp_path: Path):
        comparisons = {
            "bkt_vs_dkt": {
                "model_a": "bkt",
                "model_b": "dkt_core",
                "mean_diff": -0.035,
                "ci_lower": -0.042,
                "ci_upper": -0.028,
                "significant": True,
            },
            "dkt_vs_attention": {
                "model_a": "dkt_core",
                "model_b": "attention_core",
                "mean_diff": 0.005,
                "ci_lower": -0.003,
                "ci_upper": 0.012,
                "significant": False,
            },
        }
        out_file = tmp_path / "forest.png"
        res = plot_bootstrap_forest(comparisons, out_file)
        assert res is not None
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_generate_mermaid_comparison_chart(self):
        summary_df = pd.DataFrame({
            "model": ["bkt", "lfa_core", "dkt_core"],
            "auc": [0.71, 0.73, 0.76],
        })
        chart_str = generate_mermaid_comparison_chart(summary_df)
        assert "```mermaid" in chart_str
        assert "flowchart" in chart_str
        assert "Rank 1" in chart_str
        assert "DKT-core" in chart_str or "dkt_core" in chart_str
        assert "0.7600" in chart_str
