import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import shap
import matplotlib.pyplot as plt
from src.utils.helpers import ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SHAPExplainer:
    def __init__(self, model, feature_names: List[str], model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        try:
            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
            else:
                self.explainer = shap.Explainer(self.model, np.zeros((1, len(self.feature_names))))
            logger.info(f"{self.model_type} SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"SHAP explainer init failed: {e}, using default")
            self.explainer = shap.Explainer(self.model)

    def explain(self, X: np.ndarray) -> np.ndarray:
        try:
            self.shap_values = self.explainer.shap_values(X)
            if isinstance(self.shap_values, list):
                self.shap_values = np.array(self.shap_values)
            logger.info(f"SHAP values computed: {self.shap_values.shape}")
            return self.shap_values
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return np.zeros((X.shape[0], X.shape[1]))

    def feature_importance_global(self) -> pd.DataFrame:
        if self.shap_values is None:
            return pd.DataFrame()
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs_shap)],
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        return importance

    def feature_importance_plot(self, top_n: int = 20, figsize: Tuple = (10, 8)):
        if self.shap_values is None:
            return None
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, feature_names=self.feature_names,
                          max_display=top_n, show=False)
        plt.tight_layout()
        return plt.gcf()

    def summary_plot(self, figsize: Tuple = (12, 6)):
        if self.shap_values is None:
            return None
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, feature_names=self.feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        return plt.gcf()

    def dependence_plot(self, feature_idx: int, interaction_idx: Optional[str] = "auto"):
        if self.shap_values is None:
            return None
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, self.shap_values,
                              feature_names=self.feature_names,
                              interaction_index=interaction_idx, show=False)
        plt.tight_layout()
        return plt.gcf()

    def force_plot(self, instance_idx: int, X: np.ndarray, figsize: Tuple = (20, 3)):
        if self.shap_values is None:
            return None
        shap.initjs()
        return shap.force_plot(
            self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            self.shap_values[instance_idx],
            X[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )

    def waterfall_plot(self, instance_idx: int, X: np.ndarray, figsize: Tuple = (10, 8)):
        if self.shap_values is None:
            return None
        plt.figure(figsize=figsize)
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        return plt.gcf()

    def save_plots(self, output_dir: str, X: np.ndarray) -> None:
        output_path = ensure_dir(output_dir)
        self.explain(X)

        fig1 = self.feature_importance_plot()
        if fig1:
            fig1.savefig(f"{output_path}/shap_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close()

        fig2 = self.summary_plot()
        if fig2:
            fig2.savefig(f"{output_path}/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()

        logger.info(f"SHAP plots saved to {output_dir}")
