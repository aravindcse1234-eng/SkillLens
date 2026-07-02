import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from src.utils.helpers import save_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EDAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights: Dict = {}

    def basic_stats(self) -> Dict:
        df_dedup = self.df.copy()
        for col in df_dedup.columns:
            if df_dedup[col].apply(lambda x: isinstance(x, list)).any():
                df_dedup[col] = df_dedup[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        stats = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_values": self.df.isna().sum().to_dict(),
            "duplicates": int(df_dedup.duplicated().sum()),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1e6,
        }
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats["numeric_summary"] = self.df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
        self.insights["basic_stats"] = stats
        return stats

    def analyze_job_titles(self, col: str = "title", top_n: int = 20) -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame()
        titles = self.df[col].value_counts().head(top_n).reset_index()
        titles.columns = ["title", "count"]
        self.insights["top_titles"] = titles.to_dict("records")
        return titles

    def analyze_locations(self, col: str = "location", top_n: int = 20) -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame()
        locations = self.df[col].value_counts().head(top_n).reset_index()
        locations.columns = ["location", "count"]
        self.insights["top_locations"] = locations.to_dict("records")
        return locations

    def analyze_skills_demand(self, skill_col: str = "skills", top_n: int = 30) -> Dict[str, int]:
        all_skills = []
        for skills_list in self.df[skill_col].dropna():
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
            elif isinstance(skills_list, str):
                all_skills.append(skills_list)
        skill_counts = Counter(all_skills).most_common(top_n)
        result = dict(skill_counts)
        self.insights["top_skills"] = result
        return result

    def analyze_salary_distribution(self, col: str = "salary") -> Dict:
        if col not in self.df.columns:
            return {}
        salary = self.df[col].dropna()
        stats = {
            "mean": float(salary.mean()),
            "median": float(salary.median()),
            "std": float(salary.std()),
            "min": float(salary.min()),
            "max": float(salary.max()),
            "q25": float(salary.quantile(0.25)),
            "q75": float(salary.quantile(0.75)),
            "skewness": float(salary.skew()),
            "kurtosis": float(salary.kurtosis()),
        }
        self.insights["salary_stats"] = stats
        return stats

    def analyze_experience_levels(self, col: str = "experience_level") -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame()
        exp = self.df[col].value_counts().reset_index()
        exp.columns = ["level", "count"]
        self.insights["experience_levels"] = exp.to_dict("records")
        return exp

    def analyze_industry_trends(self, col: str = "industry") -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame()
        industry = self.df[col].value_counts().head(20).reset_index()
        industry.columns = ["industry", "count"]
        self.insights["industry_trends"] = industry.to_dict("records")
        return industry

    def analyze_education_requirements(self, col: str = "education_required") -> pd.DataFrame:
        if col not in self.df.columns:
            return pd.DataFrame()
        edu = self.df[col].value_counts().reset_index()
        edu.columns = ["education", "count"]
        self.insights["education"] = edu.to_dict("records")
        return edu

    def correlation_analysis(self) -> pd.DataFrame:
        numeric = self.df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return pd.DataFrame()
        corr = numeric.corr()
        self.insights["correlations"] = corr.to_dict()
        return corr

    def comprehensive_analysis(self) -> Dict:
        logger.info("Running comprehensive EDA...")
        self.basic_stats()
        self.analyze_job_titles()
        self.analyze_locations()
        self.analyze_skills_demand()
        self.analyze_salary_distribution()
        self.analyze_experience_levels()
        self.analyze_industry_trends()
        self.analyze_education_requirements()
        self.correlation_analysis()
        logger.info("EDA complete")
        return self.insights
