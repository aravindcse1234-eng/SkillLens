from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LearningRoadmapGenerator:
    def __init__(self):
        self.learning_paths = self._init_learning_paths()
        self.skill_graph = None
        try:
            from src.skill_graph.skill_graph import SkillGraph
            self.skill_graph = SkillGraph()
        except ImportError:
            pass

    def _init_learning_paths(self) -> Dict:
        return {
            "Python": {
                "resources": ["Python for Everybody (Coursera)", "Automate the Boring Stuff with Python",
                              "LeetCode Python Problems", "Real Python Tutorials"],
                "time_estimate": "2-4 weeks", "difficulty": "Beginner", "prerequisites": [],
                "learning_hours": 40, "projects": ["Build a CLI calculator", "Create a data analysis script"]
            },
            "Machine Learning": {
                "resources": ["Andrew Ng Machine Learning (Coursera)", "Hands-On ML with Scikit-Learn & TensorFlow",
                              "Kaggle ML Courses", "Fast.ai Practical Deep Learning"],
                "time_estimate": "4-8 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python", "Statistics"], "learning_hours": 80,
                "projects": ["House price prediction model", "Customer churn classifier"]
            },
            "Deep Learning": {
                "resources": ["Deep Learning Specialization (Coursera)", "Fast.ai Practical Deep Learning",
                              "d2l.ai Book", "PyTorch Official Tutorials"],
                "time_estimate": "8-12 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "Machine Learning"], "learning_hours": 120,
                "projects": ["Image classifier with CNN", "Text generator with RNN/LSTM"]
            },
            "SQL": {
                "resources": ["SQL for Data Analysis (Mode Analytics)", "LeetCode SQL Problems",
                              "HackerRank SQL", "SQLZoo"],
                "time_estimate": "1-2 weeks", "difficulty": "Beginner", "prerequisites": [],
                "learning_hours": 20, "projects": ["Build a sales dashboard query", "Customer segmentation analysis"]
            },
            "AWS": {
                "resources": ["AWS Certified Solutions Architect (Udemy)", "AWS Hands-On Tutorials",
                              "A Cloud Guru", "AWS Documentation"],
                "time_estimate": "6-10 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Linux"], "learning_hours": 100,
                "projects": ["Deploy a web app on EC2", "Serverless data pipeline with Lambda"]
            },
            "Docker": {
                "resources": ["Docker Mastery (Udemy)", "Docker Documentation", "Play with Docker"],
                "time_estimate": "2-3 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Linux"], "learning_hours": 25,
                "projects": ["Containerize a ML model", "Multi-container app with docker-compose"]
            },
            "Kubernetes": {
                "resources": ["CKA Certification Course (Udemy)", "Kubernetes Documentation",
                              "Killercoda Scenarios", "KodeKloud Kubernetes"],
                "time_estimate": "4-6 weeks", "difficulty": "Advanced",
                "prerequisites": ["Docker"], "learning_hours": 60,
                "projects": ["Deploy a microservice on K8s", "Set up auto-scaling cluster"]
            },
            "TensorFlow": {
                "resources": ["TensorFlow Developer Certificate (Coursera)", "TensorFlow Documentation",
                              "TF Official Tutorials"],
                "time_estimate": "4-6 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python", "Machine Learning"], "learning_hours": 60,
                "projects": ["Image recognition model", "Transfer learning pipeline"]
            },
            "PyTorch": {
                "resources": ["PyTorch for Deep Learning (Udemy)", "PyTorch Official Tutorials",
                              "Learn PyTorch"],
                "time_estimate": "4-6 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python", "Deep Learning"], "learning_hours": 60,
                "projects": ["Neural style transfer", "Custom dataset training pipeline"]
            },
            "Apache Spark": {
                "resources": ["Spark with Python (Udemy)", "Databricks Spark Academy",
                              "Spark: The Definitive Guide"],
                "time_estimate": "4-8 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "SQL"], "learning_hours": 80,
                "projects": ["ETL pipeline for big data", "Real-time streaming analysis"]
            },
            "NLP": {
                "resources": ["NLP Specialization (Coursera)", "Hugging Face NLP Course",
                              "Speech and Language Processing"],
                "time_estimate": "6-10 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "Machine Learning", "Deep Learning"], "learning_hours": 100,
                "projects": ["Sentiment analysis API", "Named entity recognition system"]
            },
            "Generative AI": {
                "resources": ["LangChain for LLM Apps (DeepLearning.AI)", "Hugging Face LLM Course",
                              "Full Stack Deep Learning", "LLM University by Cohere"],
                "time_estimate": "6-10 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "Deep Learning", "NLP"], "learning_hours": 100,
                "projects": ["RAG chatbot", "LLM fine-tuning pipeline", "AI content generator"]
            },
            "Statistics": {
                "resources": ["Statistics with Python (Coursera)", "Think Stats by Allen Downey",
                              "Khan Academy Statistics"],
                "time_estimate": "3-5 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python"], "learning_hours": 50,
                "projects": ["A/B testing analysis", "Statistical hypothesis testing"]
            },
            "Tableau": {
                "resources": ["Tableau Desktop Specialist (Udemy)", "Tableau Public Gallery",
                              "DataCamp Tableau Track"],
                "time_estimate": "2-4 weeks", "difficulty": "Beginner",
                "prerequisites": ["SQL"], "learning_hours": 35,
                "projects": ["Interactive sales dashboard", "Executive KPI tracker"]
            },
            "Power BI": {
                "resources": ["Power BI Desktop (Microsoft Learn)", "SQLBI.com", "Power BI Community"],
                "time_estimate": "2-4 weeks", "difficulty": "Beginner",
                "prerequisites": ["SQL"], "learning_hours": 35,
                "projects": ["Financial reporting dashboard", "HR analytics dashboard"]
            },
            "Airflow": {
                "resources": ["Apache Airflow Documentation", "Data Pipelines with Airflow (O'Reilly)",
                              "Astronomer Tutorials"],
                "time_estimate": "3-5 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python", "SQL"], "learning_hours": 45,
                "projects": ["Automated ETL pipeline", "Data quality monitoring DAG"]
            },
            "MLflow": {
                "resources": ["MLflow Documentation", "MLflow for MLOps (Databricks)",
                              "MLflow Official Tutorials"],
                "time_estimate": "1-2 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python", "Machine Learning"], "learning_hours": 20,
                "projects": ["Experiment tracking pipeline", "Model registry deployment"]
            },
            "FastAPI": {
                "resources": ["FastAPI Official Documentation", "FastAPI Course (Udemy)",
                              "TestDriven.io FastAPI"],
                "time_estimate": "1-2 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Python"], "learning_hours": 20,
                "projects": ["REST API for ML model", "Async data processing API"]
            },
            "LangChain": {
                "resources": ["LangChain Documentation", "LangChain for LLM Apps (DeepLearning.AI)",
                              "Building LLM Apps (O'Reilly)"],
                "time_estimate": "3-5 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "Generative AI"], "learning_hours": 50,
                "projects": ["Document Q&A system", "Agent-based research assistant"]
            },
            "Data Engineering": {
                "resources": ["Data Engineering Zoomcamp", "Fundamentals of Data Engineering (O'Reilly)",
                              "Databricks Academy"],
                "time_estimate": "8-16 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "SQL"], "learning_hours": 160,
                "projects": ["End-to-end data pipeline", "Data warehouse implementation"]
            },
            "Git": {
                "resources": ["Git & GitHub (FreeCodeCamp)", "Pro Git Book", "Learn Git Branching"],
                "time_estimate": "1 week", "difficulty": "Beginner", "prerequisites": [],
                "learning_hours": 10, "projects": ["Contribute to open source", "Set up CI/CD pipeline"]
            },
            "Linux": {
                "resources": ["Linux Command Line Basics (Udacity)", "Linux Journey",
                              "The Linux Command Line (Book)"],
                "time_estimate": "2-4 weeks", "difficulty": "Beginner", "prerequisites": [],
                "learning_hours": 30, "projects": ["Shell scripting automation", "Server setup and configuration"]
            },
            "CI/CD": {
                "resources": ["GitHub Actions Documentation", "CI/CD with Jenkins (Udemy)",
                              "GitLab CI/CD Tutorials"],
                "time_estimate": "2-3 weeks", "difficulty": "Intermediate",
                "prerequisites": ["Git", "Docker"], "learning_hours": 25,
                "projects": ["Automated testing pipeline", "Deployment pipeline with Docker"]
            },
            "Computer Vision": {
                "resources": ["Computer Vision (OpenCV University)", "CS231n Stanford",
                              "PyImageSearch Tutorials"],
                "time_estimate": "6-10 weeks", "difficulty": "Advanced",
                "prerequisites": ["Python", "Deep Learning", "TensorFlow/PyTorch"], "learning_hours": 100,
                "projects": ["Object detection system", "Face recognition app"]
            },
        }

    def generate(self, missing_skills: List[Dict], time_available: str = "medium",
                 priority: str = "all") -> Dict:
        if priority == "critical":
            filtered = [s for s in missing_skills if s.get("priority") == "critical"]
        elif priority == "high":
            filtered = [s for s in missing_skills if s.get("priority") in ("critical", "high")]
        else:
            filtered = missing_skills

        roadmap = []
        for i, skill_gap in enumerate(filtered):
            skill_name = skill_gap["skill"]
            path_info = self.learning_paths.get(skill_name, {
                "resources": [f"Search for {skill_name} courses on Coursera/Udemy"],
                "time_estimate": "2-4 weeks", "difficulty": "Intermediate",
                "prerequisites": [], "learning_hours": 40, "projects": []
            })

            chain = self._get_prerequisite_chain(skill_name, path_info)
            roadmap.append({
                "step": i + 1,
                "skill": skill_name,
                "priority": skill_gap.get("priority", "medium"),
                "market_demand": skill_gap.get("market_demand", 50),
                "gap_score": skill_gap.get("gap_score", 50),
                "difficulty": path_info["difficulty"],
                "time_estimate": path_info["time_estimate"],
                "learning_hours": path_info["learning_hours"],
                "prerequisites": chain,
                "resources": path_info["resources"],
                "projects": path_info.get("projects", []),
            })

        duration_map = {"short": 4, "medium": 8, "long": 16}
        total_weeks = duration_map.get(time_available, 8)
        weekly_plan = self._create_weekly_plan(roadmap, total_weeks)
        total_hours = sum(r["learning_hours"] for r in roadmap)
        timeline = self._estimate_timeline(roadmap)

        return {
            "roadmap": roadmap,
            "total_skills_to_learn": len(roadmap),
            "estimated_total_hours": total_hours,
            "estimated_duration_weeks": total_weeks,
            "weekly_plan": weekly_plan,
            "timeline": timeline,
            "recommended_courses": list(set(
                r for step in roadmap for r in step["resources"]
            ))[:15],
        }

    def _get_prerequisite_chain(self, skill: str, path_info: Dict) -> List[str]:
        if self.skill_graph:
            chain = self.skill_graph.get_prerequisite_chain(skill)
            if len(chain) > 1:
                if chain[-1] == skill:
                    return chain[:-1]
                return chain
        return path_info.get("prerequisites", [])

    def _estimate_timeline(self, roadmap: List[Dict]) -> List[Dict]:
        timeline = []
        current_week = 1
        for step in roadmap:
            hours = step.get("learning_hours", 40)
            weeks = max(1, round(hours / 10))
            timeline.append({
                "skill": step["skill"],
                "start_week": current_week,
                "end_week": current_week + weeks - 1,
                "total_weeks": weeks,
                "difficulty": step["difficulty"],
            })
            current_week += weeks
        return timeline

    def _create_weekly_plan(self, roadmap: List[Dict], weeks: int) -> List[Dict]:
        if not roadmap or weeks == 0:
            return []
        skills_per_week = max(1, len(roadmap) // weeks)
        plan = []
        for w in range(min(weeks, len(roadmap))):
            start = w * skills_per_week
            end = start + skills_per_week
            week_skills = roadmap[start:end]
            if week_skills:
                plan.append({
                    "week": w + 1,
                    "focus_skills": [s["skill"] for s in week_skills],
                    "resources": list(set(r for s in week_skills for r in s["resources"])),
                })
        return plan
