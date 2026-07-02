import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from src.utils.metrics import compute_forecast_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProphetForecaster:
    def __init__(self, seasonality_mode: str = "multiplicative",
                 changepoint_prior_scale: float = 0.05,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False):
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95
        )
        self.fitted = False

    def fit(self, df: pd.DataFrame, date_col: str = "ds", value_col: str = "y") -> None:
        df = df.rename(columns={date_col: "ds", value_col: "y"})
        self.model.fit(df)
        self.fitted = True
        logger.info("Prophet model fitted successfully")

    def predict(self, periods: int = 12, freq: str = "ME") -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        logger.info(f"Generated forecast for {periods} periods")
        return forecast

    def cross_validate(self, df: pd.DataFrame, initial: str = "365 days",
                       period: str = "180 days", horizon: str = "90 days") -> Dict:
        df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
        cv_results = cross_validation(
            self.model, initial=initial, period=period, horizon=horizon
        )
        perf = performance_metrics(cv_results)
        metrics = {
            "mse": float(perf["mse"].mean()),
            "rmse": float(perf["rmse"].mean()),
            "mae": float(perf["mae"].mean()),
            "mape": float(perf["mape"].mean()),
            "mdape": float(perf["mdape"].mean()),
        }
        logger.info(f"Cross-validation metrics: {metrics}")
        return metrics

    def forecast_skill(self, skill_data: pd.DataFrame, skill_name: str,
                       periods: int = 12) -> Tuple[pd.DataFrame, Dict]:
        df = skill_data[skill_data["skill_name"] == skill_name].copy()
        if df.empty:
            logger.warning(f"No data for skill: {skill_name}")
            return pd.DataFrame(), {}

        df = df.rename(columns={"date": "ds", "demand_count": "y"})
        df = df[["ds", "y"]].dropna()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        self.fit(df)
        forecast = self.predict(periods)

        metrics = compute_forecast_metrics(df["y"].values, forecast["yhat"].values[:len(df)])
        return forecast, metrics

    def get_components(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        return self.model.plot_components(self.model.predict(
            self.model.make_future_dataframe(periods=12, freq="ME")
        ))

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"Prophet model saved to {path}")

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)
        self.fitted = True
        logger.info(f"Prophet model loaded from {path}")
        return self
