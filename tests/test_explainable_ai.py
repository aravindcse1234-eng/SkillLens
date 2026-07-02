import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.explainable_ai.shap_explainer import SHAPExplainer
from src.explainable_ai.model_interpreter import ModelInterpreter


@pytest.fixture
def trained_model_and_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "years_exp": np.random.rand(n) * 15,
        "num_skills": np.random.randint(1, 10, n),
        "education_score": np.random.randint(1, 5, n),
    })
    y = X["years_exp"] * 5000 + X["num_skills"] * 2000 + np.random.randn(n) * 5000
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X


class TestSHAPExplainer:
    def test_init_tree_explainer(self, trained_model_and_data):
        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns), model_type="tree")
        assert explainer.explainer is not None

    def test_explain_returns_shap_values(self, trained_model_and_data):
        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns), model_type="tree")
        shap_values = explainer.explain(X.values[:5])
        assert shap_values.shape == (5, 3)

    def test_feature_importance_global(self, trained_model_and_data):
        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns), model_type="tree")
        explainer.explain(X.values[:10])
        importance = explainer.feature_importance_global()
        assert not importance.empty
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_feature_importance_before_explain(self, trained_model_and_data):
        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns), model_type="tree")
        result = explainer.feature_importance_global()
        assert result.empty

    def test_save_plots(self, trained_model_and_data, tmp_path):
        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns), model_type="tree")
        explainer.save_plots(str(tmp_path), X.values[:5])
        assert (tmp_path / "shap_feature_importance.png").exists()


class TestModelInterpreter:
    def test_add_model(self, trained_model_and_data):
        model, X = trained_model_and_data
        interpreter = ModelInterpreter()
        interpreter.add_model("rf", model, list(X.columns), model_type="tree")
        assert "rf" in interpreter.explainers

    def test_explain_all(self, trained_model_and_data):
        model, X = trained_model_and_data
        interpreter = ModelInterpreter()
        interpreter.add_model("rf", model, list(X.columns), model_type="tree")
        results = interpreter.explain_all(X.values[:10])
        assert "rf" in results
        assert "feature_importance" in results["rf"]

    def test_generate_report(self, trained_model_and_data):
        model, X = trained_model_and_data
        interpreter = ModelInterpreter()
        interpreter.add_model("rf", model, list(X.columns), model_type="tree")
        report = interpreter.generate_report(X.values[:10])
        assert "model_explanations" in report
        assert "rf" in report["model_explanations"]

    def test_compare_models(self, trained_model_and_data):
        model, X = trained_model_and_data
        interpreter = ModelInterpreter()
        interpreter.add_model("rf", model, list(X.columns), model_type="tree")
        comparison = interpreter.compare_models(X.values[:10])
        assert isinstance(comparison, pd.DataFrame)
