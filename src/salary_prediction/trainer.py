import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import r2_score
import lightgbm as lgb
from src.salary_prediction.models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel,
    LightGBMModel, CatBoostModel, SalaryEnsemble, BaseSalaryModel
)
from src.salary_prediction.feature_engineering import SalaryFeatureEngineer
from src.utils.metrics import compute_regression_metrics
from src.utils.logger import setup_logger
import optuna

logger = setup_logger(__name__)


class SalaryTrainer:
    def __init__(self):
        self.feature_engineer = SalaryFeatureEngineer()
        self.models: Dict[str, BaseSalaryModel] = {
            "LinearRegression": LinearRegressionModel(),
            "RandomForest": RandomForestModel(),
            "XGBoost": XGBoostModel(),
            "LightGBM": LightGBMModel(),
            "CatBoost": CatBoostModel(),
            "Ensemble": SalaryEnsemble(),
        }
        self.trained_models: Dict[str, BaseSalaryModel] = {}
        self.best_model: Optional[BaseSalaryModel] = None
        self.results: Dict = {}

    def train_all(self, df: pd.DataFrame, target_col: str = "salary",
                   test_size: float = 0.2, val_size: float = 0.1) -> pd.DataFrame:
        logger.info("Training all salary prediction models...")

        use_log_target = "salary_log" in df.columns
        if use_log_target:
            y = df["salary_log"].values
            logger.info("Using log-transformed salary target")
        else:
            y = df[target_col].values

        X, feature_cols = self.feature_engineer.prepare_features(df)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )

        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                train_metrics = model.fit(X_train, y_train, X_val, y_val)
                test_metrics = model.evaluate(X_test, y_test)
                cv_scores = self._cross_validate(model, X_train, y_train)

                self.trained_models[name] = model
                self.results[name] = {
                    "train": train_metrics,
                    "test": test_metrics,
                    "cv_mean": float(np.mean(cv_scores)),
                    "cv_std": float(np.std(cv_scores)),
                }
                log_info = f" (log target)" if use_log_target else ""
                logger.info(f"{name} - Test R²: {test_metrics['r2_score']:.4f}{log_info}")
            except Exception as e:
                logger.error(f"{name} failed: {e}")

        if use_log_target:
            for name in self.results:
                self.results[name]["test_raw"] = self._evaluate_on_raw(
                    self.trained_models[name], X_test, df[target_col].values[y_test.index if hasattr(y_test, 'index') else slice(None)]
                ) if False else None

        best_name = max(self.results, key=lambda k: self.results[k]["test"]["r2_score"])
        self.best_model = self.trained_models[best_name]
        logger.info(f"Best model: {best_name} with R²={self.results[best_name]['test']['r2_score']:.4f}")
        return self._results_dataframe()

    def _evaluate_on_raw(self, model, X, y_true):
        preds_log = model.predict(X)
        preds = np.expm1(preds_log)
        return compute_regression_metrics(y_true, preds)

    def _cross_validate(self, model: BaseSalaryModel, X, y, n_folds: int = 5) -> np.ndarray:
        try:
            if model.name == "Ensemble":
                scores = []
                fold_size = len(y) // n_folds
                for i in range(n_folds):
                    val_idx = slice(i * fold_size, (i + 1) * fold_size)
                    train_idx = list(set(range(len(y))) - set(range(*val_idx.indices(len(y)))))
                    X_fold_train = X[train_idx] if hasattr(X, 'iloc') else X[list(train_idx)]
                    y_fold_train = y[train_idx]
                    X_fold_val = X[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                    y_fold_val = y[val_idx]
                    model.fit(X_fold_train, y_fold_train)
                    preds = model.predict(X_fold_val)
                    from sklearn.metrics import r2_score
                    scores.append(r2_score(y_fold_val, preds))
                return np.array(scores)
            scores = cross_val_score(model.model, X, y, cv=n_folds, scoring="r2")
            return scores
        except:
            return np.array([0.0])

    def _results_dataframe(self) -> pd.DataFrame:
        records = []
        for name, metrics in self.results.items():
            records.append({
                "model": name,
                "test_r2": metrics["test"]["r2_score"],
                "test_rmse": metrics["test"]["rmse"],
                "test_mae": metrics["test"]["mae"],
                "test_mape": metrics["test"]["mape"],
                "cv_mean": metrics["cv_mean"],
                "cv_std": metrics["cv_std"],
            })
        return pd.DataFrame(records).sort_values("test_r2", ascending=False)

    def hyperparameter_tuning(self, model_name: str, X_train, y_train,
                               X_val, y_val, n_trials: int = 50) -> Dict:
        logger.info(f"Hyperparameter tuning for {model_name} with {n_trials} trials")

        def objective_lgb(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "verbosity": -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return -r2_score(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective_lgb, n_trials=n_trials)
        logger.info(f"Best params: {study.best_params}")
        return study.best_params

    def predict(self, X) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("No trained model available")
        return self.best_model.predict(X)

    def save_all_models(self, path_prefix: str = "models/salary/"):
        prefix = path_prefix.rstrip("/\\") + "/"
        for name, model in self.trained_models.items():
            model.save(f"{prefix}{name}.pkl")
        import joblib
        joblib.dump({
            "transformer": self.feature_engineer.transformer,
            "feature_columns": self.feature_engineer.feature_columns,
        }, f"{prefix}feature_engineer.pkl")
        logger.info(f"All models + feature engineer saved to {prefix}")
