import pandas as pd
from typing import Dict, Optional
from src.data_collection.kaggle_collector import KaggleDataCollector
from src.data_collection.onet_collector import ONetDataCollector
from src.data_collection.wef_collector import WEFDataCollector
from src.data_collection.bls_collector import BLSDataCollector
from src.utils.database import SkillLensDB
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataCollectionPipeline:
    def __init__(self, db: Optional[SkillLensDB] = None):
        self.db = db
        self.kaggle = KaggleDataCollector(db)
        self.onet = ONetDataCollector()
        self.wef = WEFDataCollector()
        self.bls = BLSDataCollector()

    def run_all(self) -> Dict[str, pd.DataFrame]:
        logger.info("Starting data collection pipeline...")
        results = {}

        logger.info("Collecting O*NET data...")
        results["onet"] = self.onet.load_skills_taxonomy()

        logger.info("Collecting WEF data...")
        results["wef"] = self.wef.generate_wef_data()

        logger.info("Collecting BLS data...")
        results["bls"] = self.bls.generate_employment_projections()

        logger.info("Processing Kaggle datasets...")
        results["salaries"] = self.kaggle.process_salary_dataset()
        results["linkedin"] = self.kaggle.process_linkedin_dataset()

        logger.info("Merging all datasets...")
        combined = self._merge_datasets(results)
        results["combined"] = combined

        if self.db:
            logger.info("Storing data in database...")
            self._store_in_db(results)

        logger.info("Data collection pipeline complete")
        return results

    def _merge_datasets(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for key in ["salaries", "linkedin"]:
            if not results.get(key, pd.DataFrame()).empty:
                frames.append(results[key])
        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["job_id"], keep="first")
            return combined
        return pd.DataFrame()

    def _store_in_db(self, results: Dict[str, pd.DataFrame]) -> None:
        if self.db is None:
            return
        try:
            self.db.connect()
            self.db.init_schema()
            self.db.create_tables()

            if not results.get("combined", pd.DataFrame()).empty:
                self.db.to_sql(results["combined"], "job_postings", if_exists="replace")

            if not results.get("onet", pd.DataFrame()).empty:
                self.db.to_sql(results["onet"], "skills_taxonomy", if_exists="replace")

            logger.info("Data stored in database successfully")
        except Exception as e:
            logger.error(f"Failed to store data in database: {e}")
        finally:
            self.db.disconnect()
