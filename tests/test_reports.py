import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.reports.report_generator import ReportGenerator


@pytest.fixture
def reports(tmp_path):
    return ReportGenerator(output_dir=str(tmp_path / "reports"))


class TestReportGenerator:
    def test_generate_training_report(self, reports):
        results = pd.DataFrame({
            "model": ["CatBoost", "XGBoost"],
            "test_r2": [0.85, 0.80],
            "test_rmse": [15000, 18000],
            "test_mae": [10000, 12000],
            "test_mape": [10.5, 12.3],
            "cv_mean": [0.84, 0.78],
            "cv_std": [0.01, 0.02],
        })
        path = reports.generate_training_report(
            results, {"total_samples": 5000}, best_model="CatBoost"
        )
        assert Path(path).exists()
        assert Path(path).suffix == ".json"

    def test_generate_eda_report(self, reports):
        insights = {
            "basic_stats": {"total_rows": 5000},
            "top_skills": {"Python": 4500, "SQL": 4000},
            "salary_stats": {"mean": 120000, "median": 115000},
        }
        path = reports.generate_eda_report(insights)
        assert Path(path).exists()

    def test_generate_forecast_report(self, reports):
        dates = pd.date_range("2025-01-01", periods=24, freq="ME")
        forecast = pd.DataFrame({
            "ds": dates,
            "yhat": np.linspace(70, 90, 24),
            "yhat_lower": np.linspace(65, 85, 24),
            "yhat_upper": np.linspace(75, 95, 24),
            "skill_name": "Python",
        })
        path = reports.generate_forecast_report({"Python": forecast})
        assert Path(path).exists()

    def test_generate_market_report(self, reports):
        df = pd.DataFrame({
            "title": ["DS", "DE", "ML"],
            "salary": [120000, 110000, 130000],
        })
        path = reports.generate_market_intelligence_report(df)
        assert Path(path).exists()
