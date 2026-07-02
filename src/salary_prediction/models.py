import numpy as np
from typing import Dict, Optional, List, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from src.utils.metrics import compute_regression_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseSalaryModel:
    def __init__(self):
        self.model = None
        self.name = "base"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X, y) -> Dict:
        preds = self.predict(X)
        return compute_regression_metrics(y, preds)

    def _use_val(self, X_val, X_train, y_val, y_train):
        if X_val is not None:
            return X_val, y_val
        return X_train, y_train

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)
        return self


class LinearRegressionModel(BaseSalaryModel):
    def __init__(self):
        super().__init__()
        self.name = "LinearRegression"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        X_eval, y_eval = self._use_val(X_val, X_train, y_val, y_train)
        metrics = self.evaluate(X_eval, y_eval)
        logger.info(f"Linear Regression: R²={metrics['r2_score']:.4f}")
        return metrics


class RandomForestModel(BaseSalaryModel):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.name = "RandomForest"
        self.params = params or {
            "n_estimators": 300, "max_depth": 20,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "n_jobs": -1, "random_state": 42
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        X_eval, y_eval = self._use_val(X_val, X_train, y_val, y_train)
        metrics = self.evaluate(X_eval, y_eval)
        logger.info(f"Random Forest: R²={metrics['r2_score']:.4f}")
        return metrics


class XGBoostModel(BaseSalaryModel):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.name = "XGBoost"
        self.params = params or {
            "n_estimators": 500, "learning_rate": 0.05,
            "max_depth": 8, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": 42,
            "tree_method": "hist",
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        X_eval, y_eval = self._use_val(X_val, X_train, y_val, y_train)
        metrics = self.evaluate(X_eval, y_eval)
        logger.info(f"XGBoost: R²={metrics['r2_score']:.4f}")
        return metrics


class LightGBMModel(BaseSalaryModel):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.name = "LightGBM"
        self.params = params or {
            "n_estimators": 500, "learning_rate": 0.05,
            "max_depth": -1, "num_leaves": 31,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "random_state": 42, "verbosity": -1
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)
        X_eval, y_eval = self._use_val(X_val, X_train, y_val, y_train)
        metrics = self.evaluate(X_eval, y_eval)
        logger.info(f"LightGBM: R²={metrics['r2_score']:.4f}")
        return metrics


class CatBoostModel(BaseSalaryModel):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.name = "CatBoost"
        self.params = params or {
            "iterations": 500, "learning_rate": 0.05,
            "depth": 8, "l2_leaf_reg": 3,
            "random_seed": 42, "verbose": False,
            "task_type": "CPU",
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(X_train, y_train, verbose=False)
        X_eval, y_eval = self._use_val(X_val, X_train, y_val, y_train)
        metrics = self.evaluate(X_eval, y_eval)
        logger.info(f"CatBoost: R²={metrics['r2_score']:.4f}")
        return metrics

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = CatBoostRegressor()
        self.model.load_model(path)
        return self


class SalaryEnsemble(BaseSalaryModel):
    def __init__(self, models: Optional[List[BaseSalaryModel]] = None):
        super().__init__()
        self.name = "Ensemble"
        self.models = models or [
            RandomForestModel(),
            XGBoostModel(),
            LightGBMModel(),
            CatBoostModel()
        ]
        self.model = self

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        for m in self.models:
            m.fit(X_train, y_train, X_val, y_val)
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        return self.evaluate(eval_X, eval_y)

    def predict(self, X) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.mean(preds, axis=1)
