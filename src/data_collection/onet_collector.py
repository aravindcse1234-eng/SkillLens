import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List
from src.utils.helpers import ensure_dir, save_json, load_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ONetDataCollector:
    BASE_URL = "https://www.onetcenter.org/ws/database/mocks"

    def __init__(self, output_dir: str = "data/external"):
        self.output_dir = ensure_dir(output_dir)

    def download_onet_database(self) -> Path:
        logger.info("Downloading O*NET database...")
        try:
            import requests
            url = "https://www.onetcenter.org/ws/database/mocks/onet_skills.zip"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                zip_path = self.output_dir / "onet_skills.zip"
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(self.output_dir)
                logger.info(f"O*NET data saved to {self.output_dir}")
                return zip_path
        except Exception as e:
            logger.warning(f"Could not download O*NET: {e}")
        return self._generate_sample_onet()

    def _generate_sample_onet(self) -> Path:
        data = {
            "skills": [
                {"name": "Python", "category": "Programming", "onet_code": "2.C.3.a"},
                {"name": "Machine Learning", "category": "AI/ML", "onet_code": "2.C.3.b"},
                {"name": "SQL", "category": "Databases", "onet_code": "2.C.3.c"},
                {"name": "Deep Learning", "category": "AI/ML", "onet_code": "2.C.3.b"},
                {"name": "Natural Language Processing", "category": "AI/ML", "onet_code": "2.C.3.b"},
                {"name": "Data Analysis", "category": "Analytics", "onet_code": "2.C.3.d"},
                {"name": "Statistics", "category": "Mathematics", "onet_code": "2.C.4.a"},
                {"name": "TensorFlow", "category": "Deep Learning", "onet_code": "2.C.3.b"},
                {"name": "PyTorch", "category": "Deep Learning", "onet_code": "2.C.3.b"},
                {"name": "AWS", "category": "Cloud Computing", "onet_code": "2.C.3.e"},
                {"name": "Azure", "category": "Cloud Computing", "onet_code": "2.C.3.e"},
                {"name": "Docker", "category": "DevOps", "onet_code": "2.C.3.f"},
                {"name": "Kubernetes", "category": "DevOps", "onet_code": "2.C.3.f"},
                {"name": "Git", "category": "Version Control", "onet_code": "2.C.3.g"},
                {"name": "R", "category": "Programming", "onet_code": "2.C.3.a"},
                {"name": "Tableau", "category": "Visualization", "onet_code": "2.C.3.h"},
                {"name": "Power BI", "category": "Visualization", "onet_code": "2.C.3.h"},
                {"name": "Spark", "category": "Big Data", "onet_code": "2.C.3.i"},
                {"name": "Hadoop", "category": "Big Data", "onet_code": "2.C.3.i"},
                {"name": "Airflow", "category": "Data Engineering", "onet_code": "2.C.3.j"},
            ]
        }
        path = self.output_dir / "onet_skills.json"
        save_json(data, str(path))
        logger.info(f"Sample O*NET data created at {path}")
        return path

    def load_skills_taxonomy(self) -> pd.DataFrame:
        path = self.output_dir / "onet_skills.json"
        if not path.exists():
            self._generate_sample_onet()
        data = load_json(str(path))
        df = pd.DataFrame(data["skills"])
        df = df.rename(columns={"name": "skill_name"})
        logger.info(f"Loaded {len(df)} O*NET skills")
        return df

    def map_job_to_skills(self, job_title: str) -> List[str]:
        skills_map = {
            "data scientist": ["Python", "Machine Learning", "Statistics", "SQL", "Deep Learning"],
            "data engineer": ["Python", "SQL", "Spark", "Airflow", "AWS", "Docker"],
            "machine learning engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "AWS"],
            "data analyst": ["Python", "SQL", "Tableau", "Statistics", "Excel"],
            "software engineer": ["Python", "Git", "Docker", "AWS", "Kubernetes"],
            "ai engineer": ["Python", "Machine Learning", "Deep Learning", "NLP", "Transformers"],
        }
        for key, skills in skills_map.items():
            if key in job_title.lower():
                return skills
        return ["Python", "SQL"]
