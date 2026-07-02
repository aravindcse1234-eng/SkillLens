import os
from pathlib import Path

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "SkillLens_AI"
MLFLOW_ARTIFACT_LOCATION = str(Path("models/mlflow_artifacts").resolve())

os.makedirs(MLFLOW_ARTIFACT_LOCATION, exist_ok=True)
