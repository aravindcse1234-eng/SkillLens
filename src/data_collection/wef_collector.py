import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from src.utils.helpers import ensure_dir, save_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WEFDataCollector:
    def __init__(self, output_dir: str = "data/external"):
        self.output_dir = ensure_dir(output_dir)

    def generate_wef_data(self) -> pd.DataFrame:
        logger.info("Generating WEF Future of Jobs data...")
        data = {
            "skill_name": [
                "Analytical Thinking", "Creative Thinking", "AI and ML",
                "Python", "Data Analysis", "Digital Literacy",
                "Critical Thinking", "Problem Solving", "Leadership",
                "Resilience", "Emotional Intelligence", "Cybersecurity",
                "Cloud Computing", "Agile Methodology", "SQL",
                "Deep Learning", "NLP", "Computer Vision",
                "Robotics", "IoT", "Blockchain", "Edge Computing",
                "Quantum Computing", "Sustainability", "Project Management"
            ],
            "category": [
                "Cognitive", "Cognitive", "Technology",
                "Technology", "Technology", "Technology",
                "Cognitive", "Cognitive", "Management",
                "Self-Efficacy", "Self-Efficacy", "Technology",
                "Technology", "Management", "Technology",
                "Technology", "Technology", "Technology",
                "Technology", "Technology", "Technology",
                "Technology", "Technology", "Cross-functional", "Management"
            ],
            "demand_score_2024": np.random.uniform(60, 100, 25).round(1),
            "demand_score_2025": np.random.uniform(65, 100, 25).round(1),
            "demand_score_2026": np.random.uniform(70, 100, 25).round(1),
            "demand_score_2027": np.random.uniform(75, 100, 25).round(1),
            "growth_rate": np.random.uniform(5, 35, 25).round(1),
            "is_emerging": np.random.choice([True, False], 25, p=[0.4, 0.6]),
        }
        df = pd.DataFrame(data)
        df["source"] = "WEF_Future_of_Jobs_2024"
        df.to_csv(self.output_dir / "wef_future_of_jobs.csv", index=False)
        logger.info(f"WEF data generated: {len(df)} skills")
        return df
