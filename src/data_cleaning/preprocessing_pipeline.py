import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from src.data_cleaning.cleaner import JobPostingCleaner, SalaryDataCleaner
from src.data_cleaning.skill_standardizer import SkillStandardizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PreprocessingPipeline:
    def __init__(self):
        self.job_cleaner = JobPostingCleaner()
        self.salary_cleaner = SalaryDataCleaner()
        self.skill_std = SkillStandardizer()
        self.pipeline_report: Dict = {}

    def run(self, df: pd.DataFrame, dataset_type: str = "job_postings",
            remove_outliers: bool = False) -> pd.DataFrame:
        logger.info(f"Running preprocessing pipeline for {dataset_type}...")
        if dataset_type == "job_postings":
            df = self.job_cleaner.clean_job_postings(df)
            if remove_outliers and "salary" in df.columns:
                df = self.job_cleaner.remove_outliers_iqr(df, ["salary"])
            if "description" in df.columns:
                df["extracted_skills"] = df["description"].apply(
                    self.skill_std.extract_skills_from_text
                )
            if "skills" in df.columns:
                df["skills"] = df["skills"].apply(
                    lambda x: self.skill_std.standardize_list(x) if isinstance(x, list) else x
                )
        elif dataset_type == "salary":
            df = self.salary_cleaner.clean_salary_data(df)
            if remove_outliers and "salary" in df.columns:
                df = self.salary_cleaner.remove_outliers_iqr(df, ["salary", "years_experience"])
        df = self._feature_engineering(df)
        self.pipeline_report = {
            "rows_after": len(df),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "missing_after": df.isna().sum().to_dict(),
        }
        logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "description" in df.columns:
            df["description_length"] = df["description"].astype(str).str.len()
        if "extracted_skills" in df.columns:
            df["num_skills"] = df["extracted_skills"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        if "salary" in df.columns:
            df["salary_log"] = df["salary"].apply(
                lambda x: np.log(x) if x > 0 else 0
            )
        if "title" in df.columns and "skills" in df.columns:
            df["title_encoded"] = df["title"].astype("category").cat.codes
        return df

    def get_report(self) -> Dict:
        return {
            "job_cleaner": self.job_cleaner.generate_report(),
            "salary_cleaner": self.salary_cleaner.generate_report(),
            "pipeline": self.pipeline_report,
        }
