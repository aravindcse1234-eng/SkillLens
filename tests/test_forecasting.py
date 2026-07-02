import pytest
import pandas as pd
import numpy as np
from src.forecasting.xgboost_ts import XGBoostTSForecaster
from src.forecasting.model_comparison import ForecastModelComparison


@pytest.fixture
def sample_ts_data():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=36, freq="ME")
    trend = np.linspace(50, 80, 36) + np.random.normal(0, 3, 36)
    return pd.DataFrame({"ds": dates, "y": trend})


class TestXGBoostTSForecaster:
    def test_fit_returns_metrics(self, sample_ts_data):
        model = XGBoostTSForecaster()
        metrics = model.fit(sample_ts_data)
        assert "train" in metrics
        assert "val" in metrics
        assert "rmse" in metrics["train"]

    def test_predict_returns_array(self, sample_ts_data):
        model = XGBoostTSForecaster()
        model.fit(sample_ts_data)
        preds = model.predict(sample_ts_data, steps=6)
        assert len(preds) == 6
        assert np.all(np.isfinite(preds))

    def test_predict_without_fit_raises(self):
        model = XGBoostTSForecaster()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(pd.DataFrame({"ds": ["2022-01-01"], "y": [50]}))

    def test_feature_importance(self, sample_ts_data):
        model = XGBoostTSForecaster()
        model.fit(sample_ts_data)
        importance = model.feature_importance()
        assert not importance.empty
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_feature_importance_before_fit(self):
        model = XGBoostTSForecaster()
        assert model.feature_importance().empty

    def test_save_load_roundtrip(self, sample_ts_data, tmp_path):
        model = XGBoostTSForecaster()
        model.fit(sample_ts_data)
        path = str(tmp_path / "xgb_model.pkl")
        model.save(path)
        loaded = XGBoostTSForecaster()
        loaded.load(path)
        preds_orig = model.predict(sample_ts_data, steps=3)
        preds_loaded = loaded.predict(sample_ts_data, steps=3)
        np.testing.assert_array_almost_equal(preds_orig, preds_loaded)


class TestForecastModelComparison:
    def test_compare_returns_dataframe(self, sample_ts_data):
        comparison = ForecastModelComparison()
        series = sample_ts_data.set_index("ds")["y"]
        result = comparison.compare(series, "TestSkill", test_size=3)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_get_best_model(self, sample_ts_data):
        comparison = ForecastModelComparison()
        series = sample_ts_data.set_index("ds")["y"]
        comparison.compare(series, "TestSkill", test_size=3)
        best = comparison.get_best_model()
        assert best is not None

    def test_compare_short_series(self):
        comparison = ForecastModelComparison()
        short = pd.Series(np.random.randn(5))
        result = comparison.compare(short, "ShortSkill")
        assert isinstance(result, pd.DataFrame)
