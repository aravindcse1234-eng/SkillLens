import pandas as pd
import re
from typing import Dict, List, Optional, Set
from src.utils.helpers import save_json, load_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SkillStandardizer:
    def __init__(self):
        self.skill_mappings = self._load_default_mappings()
        self.reverse_mappings = {v.lower(): v for v in self.skill_mappings.values()}
        self._extend_reverse_mappings()

    def _load_default_mappings(self) -> Dict[str, str]:
        return {
            "python": "Python",
            "python3": "Python",
            "python 3": "Python",
            "py": "Python",
            "pytorch": "PyTorch",
            "pytorch lightning": "PyTorch",
            "tensorflow": "TensorFlow",
            "tf": "TensorFlow",
            "keras": "TensorFlow",
            "jax": "JAX",
            "scikit-learn": "Scikit-Learn",
            "sklearn": "Scikit-Learn",
            "scikit learn": "Scikit-Learn",
            "numpy": "NumPy",
            "pandas": "Pandas",
            "data science": "Data Science",
            "machine learning": "Machine Learning",
            "ml": "Machine Learning",
            "artificial intelligence": "Artificial Intelligence",
            "ai": "Artificial Intelligence",
            "deep learning": "Deep Learning",
            "dl": "Deep Learning",
            "nlp": "Natural Language Processing",
            "natural language processing": "Natural Language Processing",
            "computer vision": "Computer Vision",
            "cv": "Computer Vision",
            "gen ai": "Generative AI",
            "generative ai": "Generative AI",
            "genai": "Generative AI",
            "transformers": "Transformers",
            "hugging face": "Hugging Face",
            "huggingface": "Hugging Face",
            "aws": "AWS",
            "amazon web services": "AWS",
            "azure": "Azure",
            "gcp": "GCP",
            "google cloud": "GCP",
            "google cloud platform": "GCP",
            "docker": "Docker",
            "kubernetes": "Kubernetes",
            "k8s": "Kubernetes",
            "sql": "SQL",
            "mysql": "SQL",
            "postgresql": "PostgreSQL",
            "postgres": "PostgreSQL",
            "mongodb": "MongoDB",
            "nosql": "NoSQL",
            "spark": "Apache Spark",
            "apache spark": "Apache Spark",
            "pyspark": "Apache Spark",
            "hadoop": "Hadoop",
            "airflow": "Apache Airflow",
            "tableau": "Tableau",
            "power bi": "Power BI",
            "powerbi": "Power BI",
            "git": "Git",
            "github": "Git",
            "gitlab": "Git",
            "flask": "Flask",
            "django": "Django",
            "fastapi": "FastAPI",
            "r": "R",
            "rstudio": "R",
            "java": "Java",
            "scala": "Scala",
            "c++": "C++",
            "c#": "C#",
            "ruby": "Ruby",
            "react": "React",
            "node.js": "Node.js",
            "nodejs": "Node.js",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "html": "HTML",
            "css": "CSS",
            "linux": "Linux",
            "unix": "Linux",
        }

    def _extend_reverse_mappings(self) -> None:
        for k, v in self.skill_mappings.items():
            self.reverse_mappings[k] = v

    def standardize(self, skill: str) -> str:
        if pd.isna(skill):
            return ""
        skill_clean = skill.strip().lower()
        if skill_clean in self.skill_mappings:
            return self.skill_mappings[skill_clean]
        if skill_clean in self.reverse_mappings:
            return self.reverse_mappings[skill_clean]
        return skill.strip().title()

    def standardize_list(self, skills: List[str]) -> List[str]:
        return [self.standardize(s) for s in skills if s.strip()]

    def standardize_dataframe(self, df: pd.DataFrame, col: str = "skills") -> pd.DataFrame:
        if col not in df.columns:
            return df
        df[col] = df[col].apply(
            lambda x: self.standardize_list(x) if isinstance(x, list)
            else self.standardize(str(x))
        )
        return df

    def extract_skills_from_text(self, text: str) -> List[str]:
        if pd.isna(text) or not text:
            return []
        text_lower = text.lower()
        found = set()
        for pattern, skill in self.skill_mappings.items():
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                found.add(skill)
        return sorted(found)

    def add_custom_mapping(self, variations: List[str], standard: str) -> None:
        for var in variations:
            self.skill_mappings[var.lower().strip()] = standard
            self.reverse_mappings[var.lower().strip()] = standard

    def save_mappings(self, path: str) -> None:
        save_json(self.skill_mappings, path)
        logger.info(f"Skill mappings saved to {path}")

    def load_mappings(self, path: str) -> None:
        mappings = load_json(path)
        self.skill_mappings.update(mappings)
        self._extend_reverse_mappings()
        logger.info(f"Loaded {len(mappings)} skill mappings from {path}")
