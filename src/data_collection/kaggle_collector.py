import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from src.utils.helpers import ensure_dir, save_json
from src.utils.database import SkillLensDB
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class KaggleDataCollector:
    DATASETS = {
        "data_science_salaries": "ruchi798/data-science-job-salaries",
        "linkedin_jobs": "arshkon/linkedin-job-postings",
        "glassdoor_reviews": "danilopeixoto/glassdoor-job-reviews-and-salaries",
        "data_analyst_jobs": "andrewmvd/data-analyst-job-postings",
        "ai_job_market": "promptcloud/ai-job-market-dataset-2024",
    }

    def __init__(self, db: Optional[SkillLensDB] = None, data_dir: str = "data/raw"):
        self.db = db
        self.data_dir = ensure_dir(data_dir)

    def download_dataset(self, dataset_name: str, force: bool = False) -> Optional[Path]:
        import kagglehub
        logger.info(f"Downloading dataset: {dataset_name}")
        try:
            path = kagglehub.dataset_download(self.DATASETS[dataset_name])
            dest = self.data_dir / dataset_name
            if not dest.exists() or force:
                import shutil
                shutil.copytree(path, dest, dirs_exist_ok=True)
                logger.info(f"Dataset saved to {dest}")
            return dest
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return None

    def download_all_datasets(self, force: bool = False) -> Dict[str, Optional[Path]]:
        results = {}
        for name in self.DATASETS:
            results[name] = self.download_dataset(name, force)
        return results

    def load_csv(self, dataset_name: str, filename: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / dataset_name / filename
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            return df
        logger.warning(f"File not found: {path}")
        return None

    def process_salary_dataset(self) -> pd.DataFrame:
        df = self.load_csv("data_science_salaries", "ds_salaries.csv")
        if df is None:
            return pd.DataFrame()
        df = df.rename(columns={
            "job_title": "title",
            "salary_in_usd": "salary",
            "company_location": "location",
            "experience_level": "experience_level",
            "employment_type": "employment_type",
            "company_size": "company_size"
        })
        df["source"] = "kaggle_ds_salaries"
        df["job_id"] = df.index.astype(str) + "_kaggle_ds"
        return df

    def process_linkedin_dataset(self) -> pd.DataFrame:
        df = self.load_csv("linkedin_jobs", "postings.csv")
        if df is None:
            return pd.DataFrame()
        df = df.rename(columns={
            "job_title": "title",
            "company_name": "company",
            "location": "location",
            "job_description": "description",
            "max_salary": "salary_max",
            "min_salary": "salary_min",
            "med_salary": "salary",
            "formatted_experience_level": "experience_level",
            "industry": "industry",
            "employment_type": "employment_type",
            "original_listed_time": "posted_date"
        })
        df["source"] = "kaggle_linkedin"
        df["job_id"] = df.index.astype(str) + "_linkedin"
        if "description" in df.columns:
            df["description"] = df["description"].fillna("")
        return df

    def to_database(self, df: pd.DataFrame, table: str) -> None:
        if self.db and not df.empty:
            self.db.to_sql(df, table, if_exists="append")
            logger.info(f"Stored {len(df)} rows in {table}")
