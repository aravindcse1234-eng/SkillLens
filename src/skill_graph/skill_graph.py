from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


SKILL_RELATIONSHIPS = {
    "Python": {"related": ["NumPy", "Pandas", "Scikit-Learn", "TensorFlow", "PyTorch", "Flask", "FastAPI", "Airflow", "Spark"], "category": "Programming"},
    "SQL": {"related": ["PostgreSQL", "MySQL", "BigQuery", "Snowflake", "Tableau", "Spark"], "category": "Data Management"},
    "Machine Learning": {"related": ["Scikit-Learn", "TensorFlow", "PyTorch", "XGBoost", "MLflow", "Statistics", "NLP", "Computer Vision"], "category": "AI/ML"},
    "Deep Learning": {"related": ["TensorFlow", "PyTorch", "Keras", "NLP", "Computer Vision", "Generative AI"], "category": "AI/ML"},
    "NLP": {"related": ["Transformers", "Hugging Face", "spaCy", "NLTK", "BERT", "Generative AI"], "category": "AI/ML"},
    "Generative AI": {"related": ["LangChain", "LLaMA", "GPT", "Hugging Face", "RAG", "Vector Databases"], "category": "AI/ML"},
    "TensorFlow": {"related": ["Keras", "Deep Learning", "Python", "Computer Vision", "NLP", "MLflow"], "category": "AI/ML"},
    "PyTorch": {"related": ["Hugging Face", "Deep Learning", "Python", "Computer Vision", "NLP", "Generative AI"], "category": "AI/ML"},
    "Statistics": {"related": ["Python", "Machine Learning", "R", "Data Analysis", "Probability"], "category": "Mathematics"},
    "AWS": {"related": ["Docker", "Kubernetes", "Terraform", "CI/CD", "Cloud Computing", "Linux"], "category": "DevOps/Cloud"},
    "Azure": {"related": ["Docker", "Kubernetes", "Terraform", "CI/CD", "Cloud Computing"], "category": "DevOps/Cloud"},
    "GCP": {"related": ["Docker", "Kubernetes", "Terraform", "CI/CD", "Cloud Computing", "BigQuery"], "category": "DevOps/Cloud"},
    "Docker": {"related": ["Kubernetes", "AWS", "Azure", "CI/CD", "Linux", "MLOps"], "category": "DevOps/Cloud"},
    "Kubernetes": {"related": ["Docker", "AWS", "Azure", "Terraform", "CI/CD", "MLOps"], "category": "DevOps/Cloud"},
    "Spark": {"related": ["SQL", "Python", "Scala", "Hadoop", "Data Engineering", "Airflow"], "category": "Data Engineering"},
    "Airflow": {"related": ["Python", "SQL", "Docker", "Spark", "Data Engineering", "MLOps"], "category": "Data Engineering"},
    "Tableau": {"related": ["SQL", "Power BI", "Data Analysis", "Python", "Dashboarding"], "category": "Visualization"},
    "Power BI": {"related": ["SQL", "Tableau", "Excel", "Data Analysis", "Dashboarding"], "category": "Visualization"},
    "MLflow": {"related": ["Python", "Machine Learning", "Docker", "AWS", "CI/CD"], "category": "MLOps"},
    "FastAPI": {"related": ["Python", "Docker", "REST API", "MLflow", "Deployment"], "category": "Engineering"},
    "Flask": {"related": ["Python", "Docker", "REST API", "Deployment"], "category": "Engineering"},
    "R": {"related": ["Statistics", "Data Analysis", "Python", "Shiny", "Biostatistics"], "category": "Programming"},
    "Git": {"related": ["GitHub", "CI/CD", "Docker", "Linux", "DevOps"], "category": "Engineering"},
    "Linux": {"related": ["Docker", "Git", "AWS", "Bash", "DevOps"], "category": "Engineering"},
    "CI/CD": {"related": ["Docker", "Kubernetes", "Git", "AWS", "MLflow", "DevOps"], "category": "DevOps/Cloud"},
    "MongoDB": {"related": ["SQL", "Python", "Node.js", "NoSQL", "Data Engineering"], "category": "Data Management"},
    "Scikit-Learn": {"related": ["Python", "Machine Learning", "NumPy", "Pandas", "XGBoost"], "category": "AI/ML"},
    "XGBoost": {"related": ["Python", "Machine Learning", "Scikit-Learn", "NumPy", "LightGBM"], "category": "AI/ML"},
    "LightGBM": {"related": ["Python", "Machine Learning", "Scikit-Learn", "NumPy", "XGBoost"], "category": "AI/ML"},
    "CatBoost": {"related": ["Python", "Machine Learning", "Scikit-Learn", "NumPy", "XGBoost"], "category": "AI/ML"},
    "NumPy": {"related": ["Python", "Pandas", "Scikit-Learn", "Data Science"], "category": "Programming"},
    "Pandas": {"related": ["Python", "NumPy", "Scikit-Learn", "Data Analysis", "Data Science"], "category": "Programming"},
    "Excel": {"related": ["SQL", "Power BI", "Tableau", "Data Analysis", "VBA"], "category": "Productivity"},
    "LangChain": {"related": ["Python", "Generative AI", "LLaMA", "RAG", "Vector Databases", "Hugging Face"], "category": "AI/ML"},
    "Hugging Face": {"related": ["Python", "Transformers", "PyTorch", "Generative AI", "NLP", "RAG"], "category": "AI/ML"},
    "Transformers": {"related": ["Hugging Face", "PyTorch", "NLP", "BERT", "Generative AI"], "category": "AI/ML"},
    "Computer Vision": {"related": ["Python", "Deep Learning", "TensorFlow", "PyTorch", "OpenCV"], "category": "AI/ML"},
    "RAG": {"related": ["LangChain", "Vector Databases", "Hugging Face", "Generative AI", "NLP"], "category": "AI/ML"},
    "Data Engineering": {"related": ["Python", "SQL", "Spark", "Airflow", "AWS", "Docker"], "category": "Data Engineering"},
    "Data Science": {"related": ["Python", "Machine Learning", "Statistics", "SQL", "Deep Learning"], "category": "Data Science"},
}

ROLE_SKILL_MAP = {
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "Scikit-Learn", "NumPy", "Pandas"],
    "Data Engineer": ["Python", "SQL", "Spark", "Airflow", "Docker", "AWS", "Kubernetes", "MongoDB", "Linux", "Git"],
    "ML Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "AWS", "Docker", "MLflow", "CI/CD", "FastAPI"],
    "Data Analyst": ["Python", "SQL", "Statistics", "Tableau", "Power BI", "Excel", "NumPy", "Pandas"],
    "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "PyTorch", "AWS", "Generative AI", "LangChain", "RAG"],
    "Software Engineer": ["Python", "Java", "JavaScript", "Docker", "Git", "AWS", "React", "SQL", "Linux", "FastAPI"],
    "DevOps Engineer": ["Python", "Docker", "Kubernetes", "AWS", "Azure", "Terraform", "CI/CD", "Linux", "Git"],
    "Research Scientist": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "Statistics", "NLP", "Computer Vision", "Generative AI"],
    "Business Analyst": ["SQL", "Excel", "Tableau", "Power BI", "Statistics", "Python"],
    "Cloud Architect": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Linux", "CI/CD"],
}

SKILL_CATEGORIES = list(set(v["category"] for v in SKILL_RELATIONSHIPS.values()))


class SkillGraph:
    def __init__(self):
        self.relationships = SKILL_RELATIONSHIPS
        self.graph = self._build_graph()

    def _build_graph(self) -> Dict[str, List[str]]:
        graph = defaultdict(list)
        for skill, info in self.relationships.items():
            for related in info["related"]:
                graph[skill].append(related)
                graph[related].append(skill)
        return dict(graph)

    def get_related_skills(self, skill: str, max_depth: int = 1) -> List[str]:
        if max_depth == 1:
            return self.graph.get(skill, [])
        visited = set()
        queue = [(skill, 0)]
        result = []
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if depth > 0:
                result.append(current)
            if depth < max_depth:
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        return result

    def get_skill_category(self, skill: str) -> Optional[str]:
        info = self.relationships.get(skill)
        return info["category"] if info else None

    def get_skills_by_category(self, category: str) -> List[str]:
        return [s for s, info in self.relationships.items() if info["category"] == category]

    def get_categories(self) -> List[str]:
        return SKILL_CATEGORIES

    def get_path(self, from_skill: str, to_skill: str) -> List[str]:
        if from_skill == to_skill:
            return [from_skill]
        visited = set()
        queue = [[from_skill]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.graph.get(node, []):
                new_path = list(path) + [neighbor]
                if neighbor == to_skill:
                    return new_path
                queue.append(new_path)
        return []

    def get_prerequisite_chain(self, skill: str) -> List[str]:
        difficulty_order = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        ROADMAP_PREREQS = {
            "Machine Learning": ["Python", "Statistics"],
            "Deep Learning": ["Python", "Machine Learning"],
            "TensorFlow": ["Python", "Machine Learning"],
            "PyTorch": ["Python", "Deep Learning"],
            "NLP": ["Python", "Machine Learning", "Deep Learning"],
            "Generative AI": ["Python", "Deep Learning", "NLP"],
            "Spark": ["Python", "SQL"],
            "Kubernetes": ["Docker"],
            "Tableau": ["SQL"],
            "Power BI": ["SQL"],
            "AWS": ["Linux"],
            "Azure": ["Linux"],
            "LangChain": ["Python", "Generative AI"],
            "RAG": ["Python", "NLP", "Generative AI"],
        }
        chain = []
        visited = set()
        def walk(s):
            if s in visited:
                return
            visited.add(s)
            for prereq in ROADMAP_PREREQS.get(s, []):
                walk(prereq)
            if s != skill:
                chain.append(s)
        walk(skill)
        chain.append(skill)
        return chain

    def get_network_data(self, center_skill: Optional[str] = None,
                          max_nodes: int = 50) -> Dict:
        if center_skill:
            related = self.get_related_skills(center_skill, max_depth=2)
            nodes_set = {center_skill} | set(related)
        else:
            nodes_set = set(list(self.relationships.keys())[:max_nodes])

        nodes_map = {}
        for i, node in enumerate(sorted(nodes_set)):
            cat = self.get_skill_category(node) or "Other"
            size = 20 if node == center_skill else 10
            if node == center_skill:
                color = "#e94560"
            else:
                color_map = {
                    "AI/ML": "#00CC96", "Programming": "#636efa", "Data Engineering": "#EF553B",
                    "DevOps/Cloud": "#FFA15A", "Visualization": "#AB63FA", "Data Management": "#19D3F3",
                    "Mathematics": "#FF6692", "Data Science": "#B6E880", "Engineering": "#FF97FF",
                    "Productivity": "#FECB52", "MLOps": "#8DD3C7",
                }
                color = color_map.get(cat, "#636efa")
            nodes_map[node] = {"id": node, "category": cat, "size": size, "color": color}

        edges = []
        seen_edges = set()
        for node in sorted(nodes_set):
            for neighbor in self.graph.get(node, []):
                if neighbor in nodes_set:
                    edge_key = tuple(sorted([node, neighbor]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({"source": node, "target": neighbor})

        return {"nodes": list(nodes_map.values()), "edges": edges}

    def get_role_skills(self, role: str) -> List[str]:
        return ROLE_SKILL_MAP.get(role, [])

    def get_all_roles(self) -> List[str]:
        return list(ROLE_SKILL_MAP.keys())

    def compute_similarity(self, skill_a: str, skill_b: str) -> float:
        neighbors_a = set(self.graph.get(skill_a, []))
        neighbors_b = set(self.graph.get(skill_b, []))
        if not neighbors_a or not neighbors_b:
            return 0.0
        intersection = neighbors_a & neighbors_b
        union = neighbors_a | neighbors_b
        return len(intersection) / len(union) if union else 0.0

    def recommend_skills(self, known_skills: List[str], top_k: int = 5) -> List[Dict]:
        scores = defaultdict(float)
        for skill in known_skills:
            related = self.graph.get(skill, [])
            for r in related:
                if r not in known_skills:
                    scores[r] += 1.0 / len(related)
        sorted_recs = sorted(scores.items(), key=lambda x: -x[1])
        return [{"skill": s, "score": round(sc, 3)} for s, sc in sorted_recs[:top_k]]

    def export_to_json(self, output_path: str) -> None:
        data = self.get_network_data()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Skill graph exported to {output_path}")
