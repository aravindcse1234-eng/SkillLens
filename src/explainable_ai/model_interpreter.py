import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from src.explainable_ai.shap_explainer import SHAPExplainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelInterpreter:
    def __init__(self):
        self.explainers: Dict[str, SHAPExplainer] = {}

    def add_model(self, name: str, model, feature_names: List[str],
                  model_type: str = "tree") -> None:
        self.explainers[name] = SHAPExplainer(model, feature_names, model_type)
        logger.info(f"Added interpreter for {name}")

    def explain_all(self, X: np.ndarray) -> Dict[str, Any]:
        results = {}
        for name, explainer in self.explainers.items():
            try:
                shap_values = explainer.explain(X)
                importance = explainer.feature_importance_global()
                results[name] = {
                    "shap_values": shap_values,
                    "feature_importance": importance.to_dict("records") if not importance.empty else [],
                    "top_features": importance.head(10).to_dict("records") if not importance.empty else [],
                }
                logger.info(f"Explained {name}")
            except Exception as e:
                logger.error(f"Failed to explain {name}: {e}")
                results[name] = {"error": str(e)}
        return results

    def generate_report(self, X: np.ndarray, y_true: Optional[np.ndarray] = None,
                        y_pred: Optional[np.ndarray] = None) -> Dict:
        explanations = self.explain_all(X)
        report = {
            "model_explanations": {},
            "global_insights": {},
        }

        for name, exp in explanations.items():
            if "error" in exp:
                report["model_explanations"][name] = {"error": exp["error"]}
                continue

            top_features = exp.get("top_features", [])
            report["model_explanations"][name] = {
                "top_5_features": top_features[:5],
                "feature_importance_summary": {
                    f["feature"]: round(f["importance"], 4)
                    for f in top_features[:10]
                }
            }

            if top_features:
                report["global_insights"][name] = {
                    "most_important_feature": top_features[0]["feature"],
                    "top_3_features": [f["feature"] for f in top_features[:3]],
                }

        return report

    def compare_models(self, X: np.ndarray) -> pd.DataFrame:
        comparison = []
        for name, explainer in self.explainers.items():
            try:
                shap_values = explainer.explain(X)
                importance = explainer.feature_importance_global()
                if not importance.empty:
                    comparison.append({
                        "model": name,
                        "top_feature": importance.iloc[0]["feature"],
                        "top_feature_importance": round(importance.iloc[0]["importance"], 4),
                        "num_important_features": len(importance[importance["importance"] > 0.01]),
                    })
            except Exception as e:
                logger.warning(f"Comparison failed for {name}: {e}")

        return pd.DataFrame(comparison) if comparison else pd.DataFrame()

    def save_reports(self, output_dir: str, X: np.ndarray) -> None:
        from src.utils.helpers import ensure_dir, save_json
        output_path = ensure_dir(output_dir)

        for name, explainer in self.explainers.items():
            explainer.save_plots(f"{output_path}/{name}", X[:100])

        report = self.generate_report(X)
        save_json(report, f"{output_path}/explainability_report.json")
        logger.info(f"Explainability reports saved to {output_dir}")
