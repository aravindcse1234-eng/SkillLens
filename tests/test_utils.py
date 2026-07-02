import pytest
import numpy as np
import json
from pathlib import Path
from src.utils.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_forecast_metrics,
)
from src.utils.helpers import ensure_dir, load_json, save_json, set_seed, get_device, chunk_list


class TestMetrics:
    def test_classification_metrics(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_regression_metrics(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 490])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics
        assert "mape" in metrics

    def test_regression_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["r2_score"] == 1.0
        assert metrics["mse"] == 0.0

    def test_forecast_metrics(self):
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([105, 108, 125, 128, 138])
        metrics = compute_forecast_metrics(y_true, y_pred)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics

    def test_forecast_zero_protection(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        metrics = compute_forecast_metrics(y_true, y_pred)
        assert np.isfinite(metrics["mape"])

    def test_perfect_forecast(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        metrics = compute_forecast_metrics(y_true, y_pred)
        assert metrics["mape"] == 0.0
        assert metrics["rmse"] == 0.0


class TestHelpers:
    def test_ensure_dir_creates(self, tmp_path):
        p = ensure_dir(str(tmp_path / "new_dir" / "sub"))
        assert p.exists()
        assert p.is_dir()

    def test_save_load_json_roundtrip(self, tmp_path):
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_set_seed(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_almost_equal(a, b)

    def test_get_device_returns_device(self):
        device = get_device()
        assert str(device) in ("cpu", "cuda", "mps")

    def test_chunk_list(self):
        result = chunk_list([1, 2, 3, 4, 5, 6, 7], chunk_size=3)
        assert result == [[1, 2, 3], [4, 5, 6], [7]]

    def test_chunk_list_empty(self):
        assert chunk_list([], 3) == []

    def test_load_json_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path.json")
