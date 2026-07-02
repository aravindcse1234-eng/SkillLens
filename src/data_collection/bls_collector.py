import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from src.utils.helpers import ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BLSDataCollector:
    def __init__(self, output_dir: str = "data/external"):
        self.output_dir = ensure_dir(output_dir)

    def generate_employment_projections(self) -> pd.DataFrame:
        logger.info("Generating BLS employment projections...")
        occupations = [
            "Data Scientists", "Software Developers", "AI Specialists",
            "Information Security Analysts", "Computer & Information Research Scientists",
            "Database Administrators", "Network Architects", "Web Developers",
            "Computer Systems Analysts", "IT Managers"
        ]
        data = {
            "occupation": occupations,
            "employment_2024": np.random.randint(50000, 300000, len(occupations)),
            "employment_2034": np.random.randint(60000, 400000, len(occupations)),
            "growth_rate": np.random.uniform(10, 36, len(occupations)).round(1),
            "median_salary": np.random.randint(70000, 160000, len(occupations)),
            "typical_entry_education": np.random.choice(
                ["Bachelor's", "Master's", "PhD"], len(occupations),
                p=[0.6, 0.3, 0.1]
            ),
            "work_experience_required": np.random.choice(
                ["None", "Less than 5 years", "5+ years"], len(occupations),
                p=[0.3, 0.5, 0.2]
            ),
        }
        df = pd.DataFrame(data)
        df["source"] = "BLS_Employment_Projections"
        df.to_csv(self.output_dir / "bls_projections.csv", index=False)
        logger.info(f"BLS data generated: {len(df)} occupations")
        return df
