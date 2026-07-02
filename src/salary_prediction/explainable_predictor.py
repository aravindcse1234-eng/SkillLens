from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExplainableSalaryPredictor:
    def __init__(self, models_dir: str = "models/salary"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.transformer = None
        self.feature_columns = []
        self.shap_explainer = None
        self._load_resources()

    def _load_resources(self):
        catboost_path = self.models_dir / "CatBoost.pkl"
        feat_path = self.models_dir / "feature_engineer.pkl"
        if catboost_path.exists():
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor()
                self.model.load_model(str(catboost_path))
            except Exception as e:
                logger.warning(f"Could not load CatBoost: {e}")
        if feat_path.exists():
            try:
                fe_data = joblib.load(feat_path)
                self.transformer = fe_data.get("transformer")
                self.feature_columns = fe_data.get("feature_columns", [])
            except Exception as e:
                logger.warning(f"Could not load feature engineer: {e}")

    def predict(self, input_df: pd.DataFrame) -> Tuple[float, Dict]:
        from src.salary_prediction.feature_engineering import SalaryFeatureEngineer
        sfe = SalaryFeatureEngineer()
        input_df = sfe.create_features(input_df)
        X = self.transformer.transform(input_df)
        pred_log = self.model.predict(X)[0]
        predicted = float(np.expm1(pred_log))

        explanations = self._explain_prediction(X[0], input_df)
        return predicted, explanations

    def _explain_prediction(self, X_instance: np.ndarray,
                             input_df: pd.DataFrame) -> Dict:
        try:
            import shap
            if self.shap_explainer is None:
                self.shap_explainer = shap.TreeExplainer(self.model)
            shap_values = self.shap_explainer.shap_values(X_instance.reshape(1, -1))
            values = shap_values[0] if shap_values.ndim == 2 else shap_values

            feature_names = self.feature_columns if self.feature_columns else [f"f{i}" for i in range(len(values))]
            if len(feature_names) > len(values):
                feature_names = feature_names[:len(values)]
            elif len(feature_names) < len(values):
                feature_names = feature_names + [f"f{i}" for i in range(len(feature_names), len(values))]

            contributions = [
                {"feature": feature_names[i], "shap_value": float(values[i]),
                 "impact": "positive" if values[i] > 0 else "negative",
                 "magnitude": abs(float(values[i]))}
                for i in range(len(values))
            ]
            contributions.sort(key=lambda x: x["magnitude"], reverse=True)

            top_factors = contributions[:5]
            feature_desc = {
                "composite_log": "Composite Skill-Salary Score",
                "exp_score": "Experience Level Score",
                "edu_score": "Education Level Score",
                "loc_mult": "Location Salary Multiplier",
                "ind_mult": "Industry Salary Multiplier",
                "years_experience": "Years of Experience",
                "num_skills": "Number of Skills",
            }
            factors_human = []
            for f in top_factors:
                name = feature_desc.get(f["feature"], f["feature"].replace("_", " ").title())
                direction = "increases" if f["impact"] == "positive" else "decreases"
                factors_human.append(f"{name} {direction} salary")

            base_value = float(self.shap_explainer.expected_value)
            return {
                "shap_values": contributions[:10],
                "top_factors": factors_human,
                "base_value": base_value,
                "predicted_log_value": float(np.log1p(
                    float(np.expm1(base_value + float(sum(c["shap_value"] for c in contributions))))))
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"top_factors": ["SHAP explanation unavailable"], "shap_values": []}

    def predict_with_factors(self, input_df: pd.DataFrame) -> Dict:
        predicted, explanations = self.predict(input_df)
        input_row = input_df.iloc[0]
        return {
            "predicted_salary": round(predicted),
            "confidence_interval": {
                "lower": round(predicted * 0.9),
                "upper": round(predicted * 1.1),
            },
            "factors": explanations.get("top_factors", []),
            "shap_details": explanations.get("shap_values", []),
            "input_features": {
                "role": input_row.get("title", input_row.get("job_title", "")),
                "experience": input_row.get("years_experience", 0),
                "education": input_row.get("education_level", ""),
                "location": input_row.get("location", ""),
            },
        }
