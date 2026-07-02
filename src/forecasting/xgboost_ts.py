import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from src.utils.metrics import compute_forecast_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class XGBoostTSForecaster:
    def __init__(self, params: Optional[Dict] = None):
        default_tree = "hist"
        try:
            import xgboost as _xgb
            if _xgb.__version__ >= "2.0.0":
                default_tree = "hist"
        except ImportError:
            pass
        self.params = params or {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "random_state": 42,
            "tree_method": default_tree,
            "early_stopping_rounds": 50,
        }
        self.model = None

    def _create_features(self, df: pd.DataFrame, date_col: str = "ds",
                         value_col: str = "y") -> pd.DataFrame:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter
        df["dayofyear"] = df["ds"].dt.dayofyear
        df["dayofmonth"] = df["ds"].dt.day
        df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        for lag in [1, 2, 3, 6, 12]:
            df[f"lag_{lag}"] = df[value_col].shift(lag)
        for window in [3, 6, 12]:
            df[f"rolling_mean_{window}"] = df[value_col].rolling(window=window).mean()
            df[f"rolling_std_{window}"] = df[value_col].rolling(window=window).std()

        df = df.dropna()
        return df

    def fit(self, df: pd.DataFrame, date_col: str = "ds", value_col: str = "y",
            val_split: float = 0.2) -> Dict:
        feature_df = self._create_features(df, date_col, value_col)
        feature_cols = [c for c in feature_df.columns if c not in ["ds", "y", value_col]]

        X = feature_df[feature_cols]
        y = feature_df[value_col]

        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        logger.info("XGBoost time series model fitted")

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train": compute_forecast_metrics(y_train, train_pred),
            "val": compute_forecast_metrics(y_val, val_pred)
        }
        return metrics

    def predict(self, df: pd.DataFrame, date_col: str = "ds",
                value_col: str = "y", steps: int = 12) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted yet")

        future_df = df.copy()
        predictions = []
        for _ in range(steps):
            last_date = pd.to_datetime(future_df["ds"].iloc[-1])
            next_date = last_date + pd.DateOffset(months=1)
            new_row = pd.DataFrame({"ds": [next_date], value_col: [future_df[value_col].iloc[-1]]})
            future_df = pd.concat([future_df, new_row], ignore_index=True)
            feature_df = self._create_features(future_df, "ds", value_col)
            if len(feature_df) > 0:
                latest_features = feature_df.iloc[-1:]
                feature_cols = [c for c in feature_df.columns if c not in ["ds", "y", value_col]]
                pred = self.model.predict(latest_features[feature_cols])[0]
                future_df.iloc[-1, future_df.columns.get_loc(value_col)] = type(future_df[value_col].iloc[-1])(pred)
                predictions.append(pred)

        return np.array(predictions)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = pd.DataFrame({
            "feature": self.model.feature_names_in_,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        return importance

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)
        logger.info(f"XGBoost model loaded from {path}")
        return self
