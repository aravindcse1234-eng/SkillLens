import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List, Tuple
from src.utils.helpers import save_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCleaner:
    def __init__(self):
        self.cleaning_report: Dict = {}

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        self.cleaning_report["duplicates_removed"] = before - len(df)
        logger.info(f"Removed {before - len(df)} duplicates")
        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        report = {}
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            if strategy == "auto":
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna("Unknown")
            elif strategy == "drop":
                df = df.dropna(subset=[col])
            report[col] = int(n_missing)
        self.cleaning_report["missing_values"] = report
        logger.info(f"Handled missing values: {report}")
        return df

    def remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str],
                            multiplier: float = 1.5) -> pd.DataFrame:
        before = len(df)
        for col in columns:
            if col in df.columns and df[col].dtype in ["float64", "int64"]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        self.cleaning_report["outliers_removed"] = before - len(df)
        logger.info(f"Removed {before - len(df)} outliers from {columns}")
        return df

    def standardize_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        mapper = {}
        for col in df.columns:
            new = col.lower().strip().replace(" ", "_")
            new = re.sub(r"[^a-z0-9_]", "", new)
            mapper[col] = new
        df = df.rename(columns=mapper)
        logger.info(f"Standardized column names: {mapper}")
        return df

    def generate_report(self) -> Dict:
        return self.cleaning_report


class JobPostingCleaner(DataCleaner):
    def __init__(self):
        super().__init__()

    def clean_job_titles(self, df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
        title_mappings = {
            r"data scientist.*": "Data Scientist",
            r"data engineer.*": "Data Engineer",
            r"ml engineer.*": "Machine Learning Engineer",
            r"machine learning engineer.*": "Machine Learning Engineer",
            r"data analyst.*": "Data Analyst",
            r"software engineer.*": "Software Engineer",
            r"ai engineer.*": "AI Engineer",
            r"deep learning.*": "Deep Learning Engineer",
            r"business analyst.*": "Business Analyst",
            r"devops.*": "DevOps Engineer",
            r"research scientist.*": "Research Scientist",
        }
        df[col] = df[col].astype(str).str.lower().str.strip()
        for pattern, replacement in title_mappings.items():
            df.loc[df[col].str.match(pattern, na=False), col] = replacement
        logger.info("Standardized job titles")
        return df

    def parse_salary(self, df: pd.DataFrame) -> pd.DataFrame:
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
        if "salary_min" in df.columns:
            df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
        if "salary_max" in df.columns:
            df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")
        if "salary" in df.columns and "salary_min" in df.columns and "salary_max" in df.columns:
            df.loc[df["salary"].isna() & df["salary_min"].notna() & df["salary_max"].notna(),
                   "salary"] = (df["salary_min"] + df["salary_max"]) / 2
        logger.info("Parsed salary columns")
        return df

    def clean_job_postings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.standardize_column_names(df)
        df = self.remove_duplicates(df, subset=["job_id"] if "job_id" in df.columns else None)
        df = self.handle_missing_values(df)
        if "title" in df.columns:
            df = self.clean_job_titles(df)
        if any(c in df.columns for c in ["salary", "salary_min", "salary_max"]):
            df = self.parse_salary(df)
        if "description" in df.columns:
            df["description"] = df["description"].fillna("").apply(self.standardize_text)
        if "location" in df.columns:
            df["location"] = df["location"].astype(str).str.strip()
        return df


class SalaryDataCleaner(DataCleaner):
    def __init__(self):
        super().__init__()

    def clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.standardize_column_names(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
            df = df[df["salary"] > 0]
            df = df[df["salary"] < 500000]
        if "years_experience" in df.columns:
            df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce")
            df = df[df["years_experience"].between(0, 50)]
        logger.info(f"Cleaned salary data: {len(df)} rows")
        return df
