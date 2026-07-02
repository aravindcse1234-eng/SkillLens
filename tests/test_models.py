import pytest
import numpy as np
import pandas as pd
from src.salary_prediction.models import LinearRegressionModel, RandomForestModel
from src.forecasting.prophet_model import ProphetForecaster


class TestSalaryModels:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(100) * 0.1
        return X, y

    def test_linear_regression(self, sample_data):
        X, y = sample_data
        model = LinearRegressionModel()
        metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
        assert "r2_score" in metrics
        assert metrics["r2_score"] > -1

    def test_random_forest(self, sample_data):
        X, y = sample_data
        model = RandomForestModel({"n_estimators": 10, "max_depth": 5, "n_jobs": 1})
        metrics = model.fit(X[:80], y[:80], X[80:], y[80:])
        assert "r2_score" in metrics

    def test_prediction_shape(self, sample_data):
        X, y = sample_data
        model = LinearRegressionModel()
        model.fit(X[:80], y[:80])
        preds = model.predict(X[80:])
        assert len(preds) == 20


class TestForecasting:
    @pytest.fixture
    def time_series(self):
        dates = pd.date_range(start="2020-01-01", periods=24, freq="ME")
        values = np.sin(np.linspace(0, 4 * np.pi, 24)) * 10 + 50 + np.random.normal(0, 2, 24)
        return pd.DataFrame({"ds": dates, "y": values})

    def test_prophet_fit(self, time_series):
        model = ProphetForecaster()
        model.fit(time_series)
        assert model.fitted

    def test_prophet_predict(self, time_series):
        model = ProphetForecaster()
        model.fit(time_series)
        forecast = model.predict(periods=6)
        assert len(forecast) == 30
        assert "yhat" in forecast.columns
