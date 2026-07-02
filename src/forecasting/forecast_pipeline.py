import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.forecasting.prophet_model import ProphetForecaster
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.xgboost_ts import XGBoostTSForecaster
from src.forecasting.model_comparison import ForecastModelComparison
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ForecastPipeline:
    def __init__(self):
        self.comparison = ForecastModelComparison()
        self.best_models: Dict[str, object] = {}

    def run(self, df: pd.DataFrame, skill_col: str = "skill_name",
            date_col: str = "date", value_col: str = "demand_count",
            top_n: int = 10) -> Dict[str, pd.DataFrame]:
        logger.info("Running forecast pipeline...")
        results = {}

        if date_col not in df.columns:
            df[date_col] = pd.date_range(end=pd.Timestamp.today(),
                                           periods=len(df), freq="ME")

        top_skills = df.groupby(skill_col)[value_col].sum().nlargest(top_n).index

        for skill in top_skills:
            skill_data = df[df[skill_col] == skill].sort_values(date_col)
            if len(skill_data) < 12:
                continue

            series = skill_data.set_index(date_col)[value_col]
            comparison_df = self.comparison.compare(series, skill, test_size=3)

            best_model_name = self.comparison.get_best_model()
            logger.info(f"Best model for {skill}: {best_model_name}")

            forecast = self._generate_forecast(skill_data, skill, date_col, value_col)
            results[skill] = forecast

        return results

    def _generate_forecast(self, df: pd.DataFrame, skill: str,
                           date_col: str, value_col: str) -> pd.DataFrame:
        prophet = ProphetForecaster()
        prophet.fit(df, date_col, value_col)
        forecast = prophet.predict(periods=12)
        forecast["skill_name"] = skill
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "skill_name"]]

    def forecast_multiple_skills(self, df: pd.DataFrame, skills: List[str],
                                  periods: int = 12) -> Dict[str, pd.DataFrame]:
        results = {}
        for skill in skills:
            skill_df = df[df["skill_name"] == skill].copy()
            if skill_df.empty:
                continue
            prophet = ProphetForecaster()
            prophet.fit(skill_df)
            forecast = prophet.predict(periods)
            forecast["skill_name"] = skill
            results[skill] = forecast
        return results
