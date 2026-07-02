import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.forecasting.prophet_model import ProphetForecaster
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.xgboost_ts import XGBoostTSForecaster
from src.utils.metrics import compute_forecast_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ForecastModelComparison:
    def __init__(self):
        self.models = {
            "Prophet": ProphetForecaster(),
            "LSTM": LSTMForecaster(),
            "XGBoost": XGBoostTSForecaster(),
        }
        self.results: Dict = {}

    def compare(self, data: pd.Series, skill_name: str,
                test_size: int = 3) -> pd.DataFrame:
        logger.info(f"Comparing forecast models for {skill_name}")

        train = data[:-test_size] if test_size > 0 else data
        test = data[-test_size:] if test_size > 0 else pd.Series(dtype=float)

        results = {}
        train_df = pd.DataFrame({
            "ds": pd.date_range(start="2020-01-01", periods=len(train), freq="ME"),
            "y": train.values
        })

        for name, model in self.models.items():
            try:
                if name == "Prophet":
                    m = ProphetForecaster()
                    m.fit(train_df)
                    forecast = m.predict(periods=test_size if test_size > 0 else 12)
                    preds = forecast["yhat"].values[-max(test_size, 1):]

                elif name == "LSTM":
                    m = LSTMForecaster()
                    if test_size > 0:
                        full_data = pd.concat([train, test])
                        history = m.fit(train)
                        preds = m.predict(train, steps=test_size)
                    else:
                        preds = np.array([])

                elif name == "XGBoost":
                    m = XGBoostTSForecaster()
                    m.fit(train_df)
                    preds = m.predict(train_df, steps=test_size if test_size > 0 else 12)

                if len(preds) > 0 and len(test) > 0:
                    metrics = compute_forecast_metrics(
                        test.values[:len(preds)], preds[:len(test)]
                    )
                else:
                    metrics = {"rmse": 0, "mae": 0, "mape": 0}

                results[name] = metrics
                logger.info(f"{name}: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")

            except Exception as e:
                logger.error(f"{name} failed: {e}")
                results[name] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

        self.results[skill_name] = results
        return pd.DataFrame(results).T

    def get_best_model(self, metric: str = "mape") -> str:
        best_model = None
        best_score = float("inf")
        for skill_name, results in self.results.items():
            for model_name, metrics in results.items():
                if metrics.get(metric, float("inf")) < best_score:
                    best_score = metrics[metric]
                    best_model = model_name
        return best_model
