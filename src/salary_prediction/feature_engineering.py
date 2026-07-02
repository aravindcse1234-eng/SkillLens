import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SalaryFeatureEngineer:
    def __init__(self):
        self.label_encoders: Dict = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.transformer = None
        self.fitted = False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        _LOC_MULTS = {
            "San Francisco": 1.25, "New York": 1.20, "Seattle": 1.15,
            "Austin": 0.98, "Boston": 1.10, "Chicago": 1.05,
            "Los Angeles": 1.08, "Denver": 1.02, "Remote": 1.00,
            "Bangalore": 0.45, "Mumbai": 0.48, "Delhi": 0.42,
            "Hyderabad": 0.40, "Pune": 0.38, "Chennai": 0.37,
            "Kolkata": 0.35, "Ahmedabad": 0.34, "Jaipur": 0.33,
        }
        _IND_MULTS = {
            "Technology": 1.10, "Finance": 1.15, "Healthcare": 1.05,
            "E-commerce": 1.08, "Consulting": 1.02, "Education": 0.85,
            "Manufacturing": 0.90, "Media": 0.95,
        }
        _EDU_SCORES = {"Bachelor": 1.0, "Master": 1.12, "PhD": 1.22}
        _EXP_SCORES = {"Entry": 1.0, "Mid": 1.20, "Senior": 1.40, "Lead": 1.60, "Principal": 1.80}

        if "years_experience" in df.columns:
            df["exp_squared"] = df["years_experience"] ** 2
            df["exp_log"] = np.log1p(df["years_experience"])
            df["experience_bin"] = pd.cut(
                df["years_experience"],
                bins=[0, 1, 3, 5, 10, 15, 20, 50],
                labels=["entry", "junior", "mid", "senior", "lead", "principal", "executive"]
            )
        if "num_skills" in df.columns:
            df["skill_density"] = df["num_skills"] / (df["years_experience"] + 1)

        if "education_level" in df.columns:
            df["edu_score"] = df["education_level"].map(_EDU_SCORES).fillna(1.0)

        if "experience_level" in df.columns:
            df["exp_score"] = df["experience_level"].map(_EXP_SCORES).fillna(1.0)

        if "location" in df.columns:
            df["loc_mult"] = df["location"].map(_LOC_MULTS).fillna(1.0)

        if "industry" in df.columns:
            df["ind_mult"] = df["industry"].map(_IND_MULTS).fillna(1.0)

        if "company_size" in df.columns:
            size_map = {"S": 1, "M": 2, "L": 3}
            df["company_size_score"] = df["company_size"].map(size_map).fillna(2)

        title_col = "job_title" if "job_title" in df.columns else ("title" if "title" in df.columns else None)
        if title_col:
            title_freq = df[title_col].value_counts().to_dict()
            df["title_popularity"] = df[title_col].map(title_freq)
            title_base_map = {
                "Data Analyst": 85000, "Software Engineer": 115000,
                "Data Engineer": 120000, "ML Engineer": 135000,
                "Data Scientist": 125000, "AI Engineer": 140000,
            }
            df["title_base_salary"] = df[title_col].map(title_base_map).fillna(100000)

        required = ["title_base_salary", "exp_score", "edu_score", "loc_mult", "ind_mult"]
        if all(c in df.columns for c in required):
            skills_bonus = 1.0
            if "num_skills" in df.columns:
                skills_bonus = 1.0 + (df["num_skills"] / 12.0 * 0.08)
            raw_estimate = (
                df["title_base_salary"]
                * df["exp_score"]
                * df["edu_score"]
                * df["loc_mult"]
                * df["ind_mult"]
                * skills_bonus
            )
            df["composite_estimate"] = raw_estimate
            df["composite_log"] = np.log(raw_estimate)

        if "exp_score" in df.columns and "edu_score" in df.columns:
            df["exp_edu_interaction"] = df["exp_score"] * df["edu_score"]
        if "exp_score" in df.columns and "title_popularity" in df.columns:
            df["exp_title_interaction"] = df["exp_score"] * df["title_popularity"]
        if "num_skills" in df.columns and "years_experience" in df.columns:
            df["skills_exp_interaction"] = df["num_skills"] * np.log1p(df["years_experience"])
        logger.info(f"Created {len(df.columns)} features from salary data")
        return df

    def prepare_features(self, df: pd.DataFrame,
                          categorical_cols: Optional[List[str]] = None,
                          numerical_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        df = self.create_features(df)
        if categorical_cols is None:
            categorical_cols = ["job_title", "location", "industry", "employment_type"]
        if numerical_cols is None:
            numerical_cols = ["years_experience", "num_skills", "exp_score",
                              "edu_score", "loc_mult", "ind_mult",
                              "company_size_score", "exp_squared",
                              "composite_estimate", "composite_log",
                              "title_base_salary",
                              "exp_edu_interaction", "exp_title_interaction",
                              "skills_exp_interaction"]

        categorical_cols = [c for c in categorical_cols if c in df.columns]
        numerical_cols = [c for c in numerical_cols if c in df.columns]

        self.transformer = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ])

        X = self.transformer.fit_transform(df)
        self.feature_columns = numerical_cols + list(
            self.transformer.named_transformers_["cat"].get_feature_names_out(categorical_cols)
        )
        self.fitted = True
        logger.info(f"Prepared {X.shape[1]} features for {len(df)} samples")
        return X, self.feature_columns

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Feature engineer not fitted yet")
        df = self.create_features(df)
        return self.transformer.transform(df)
