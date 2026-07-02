from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
from src.utils.logger import setup_logger
from src.data_collection.live_collector import LiveDataCollector, SKILL_TRENDING_QUERIES

logger = setup_logger(__name__)


class JobTrendTracker:
    def __init__(self, data_dir: str = "data/external"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trends_file = self.data_dir / "job_trends.json"
        self.history_file = self.data_dir / "trend_history.json"
        self._load_data()

    def _load_data(self):
        if self.trends_file.exists():
            with open(self.trends_file) as f:
                self.current_trends = json.load(f)
        else:
            self.current_trends = self._generate_initial_trends()
            self._save_trends()

        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)
        else:
            self.history = {"weekly": [], "monthly": []}
            self._save_history()

    def __init__(self, data_dir: str = "data/external"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trends_file = self.data_dir / "job_trends.json"
        self.history_file = self.data_dir / "trend_history.json"
        self.live_collector = LiveDataCollector()
        self._live_data_merged = False
        self._load_data()

    def _generate_initial_trends(self) -> Dict:
        return {
            "Generative AI": {"demand": 95, "growth": 280, "category": "AI/ML", "trend": "soaring", "source": "simulated"},
            "MLOps": {"demand": 82, "growth": 123, "category": "MLOps", "trend": "rising", "source": "simulated"},
            "LangChain": {"demand": 78, "growth": 210, "category": "AI/ML", "trend": "soaring", "source": "simulated"},
            "RAG": {"demand": 75, "growth": 195, "category": "AI/ML", "trend": "soaring", "source": "simulated"},
            "AWS": {"demand": 88, "growth": 15, "category": "DevOps/Cloud", "trend": "stable", "source": "simulated"},
            "Python": {"demand": 92, "growth": 8, "category": "Programming", "trend": "stable", "source": "simulated"},
            "SQL": {"demand": 85, "growth": 5, "category": "Data Management", "trend": "stable", "source": "simulated"},
            "Docker": {"demand": 76, "growth": 22, "category": "DevOps/Cloud", "trend": "rising", "source": "simulated"},
            "Kubernetes": {"demand": 72, "growth": 35, "category": "DevOps/Cloud", "trend": "rising", "source": "simulated"},
            "TensorFlow": {"demand": 68, "growth": -5, "category": "AI/ML", "trend": "declining", "source": "simulated"},
            "PyTorch": {"demand": 74, "growth": 45, "category": "AI/ML", "trend": "rising", "source": "simulated"},
            "Apache Spark": {"demand": 65, "growth": 10, "category": "Data Engineering", "trend": "stable", "source": "simulated"},
            "Airflow": {"demand": 60, "growth": 18, "category": "Data Engineering", "trend": "rising", "source": "simulated"},
            "FastAPI": {"demand": 55, "growth": 85, "category": "Engineering", "trend": "rising", "source": "simulated"},
            "Tableau": {"demand": 58, "growth": -8, "category": "Visualization", "trend": "declining", "source": "simulated"},
            "Power BI": {"demand": 62, "growth": 25, "category": "Visualization", "trend": "rising", "source": "simulated"},
            "Data Engineering": {"demand": 80, "growth": 40, "category": "Data Engineering", "trend": "rising", "source": "simulated"},
            "Computer Vision": {"demand": 60, "growth": 20, "category": "AI/ML", "trend": "rising", "source": "simulated"},
            "NLP": {"demand": 70, "growth": 30, "category": "AI/ML", "trend": "rising", "source": "simulated"},
            "MLflow": {"demand": 50, "growth": 95, "category": "MLOps", "trend": "soaring", "source": "simulated"},
        }

    def _save_trends(self):
        with open(self.trends_file, "w") as f:
            json.dump(self.current_trends, f, indent=2)

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def merge_live_data(self) -> Dict:
        skills = [s for s in self.current_trends if s in SKILL_TRENDING_QUERIES]
        live_growth = self.live_collector.refresh_all(skills).get("skill_growth", {})
        for skill, score in live_growth.items():
            if skill in self.current_trends and score > 0:
                base = self.current_trends[skill]
                base["live_score"] = score
                base["source"] = "live"
                base["demand"] = max(1, min(100, int(score)))
                old_growth = base["growth"]
                base["growth"] = round(score * 1.5, 1)
                if base["growth"] > 50:
                    base["trend"] = "soaring"
                elif base["growth"] > 15:
                    base["trend"] = "rising"
                elif base["growth"] < -5:
                    base["trend"] = "declining"
                else:
                    base["trend"] = "stable"
        self._live_data_merged = True
        self._save_trends()
        return self.current_trends

    def get_current_trends(self, with_live: bool = False) -> Dict:
        if with_live and not self._live_data_merged:
            try:
                self.merge_live_data()
            except Exception as e:
                logger.warning(f"Live data merge failed: {e}")
        return self.current_trends

    def get_trending_skills(self, min_growth: float = 10, top_k: int = 10) -> List[Dict]:
        trending = []
        for skill, info in self.current_trends.items():
            if info["growth"] >= min_growth:
                trending.append({"skill": skill, **info})
        trending.sort(key=lambda x: x["growth"], reverse=True)
        return trending[:top_k]

    def get_declining_skills(self, top_k: int = 5) -> List[Dict]:
        declining = []
        for skill, info in self.current_trends.items():
            if info["growth"] < 0:
                declining.append({"skill": skill, **info})
        declining.sort(key=lambda x: x["growth"])
        return declining[:top_k]

    def get_skill_trend_since(self, skill: str, weeks: int = 4) -> Dict:
        if skill not in self.current_trends:
            return {"skill": skill, "error": "Skill not found"}
        info = self.current_trends[skill]
        return {
            "skill": skill,
            "current_demand": info["demand"],
            "growth": info["growth"],
            "trend": info["trend"],
            "category": info["category"],
            "estimated_weekly_change": round(info["growth"] / 52, 1),
        }

    def simulate_weekly_update(self) -> Dict:
        changes = {}
        for skill, info in self.current_trends.items():
            noise = random.uniform(-3, 3)
            trend_boost = 0
            if info["trend"] == "soaring":
                trend_boost = random.uniform(1, 3)
            elif info["trend"] == "rising":
                trend_boost = random.uniform(0.5, 1.5)
            elif info["trend"] == "declining":
                trend_boost = random.uniform(-2, -0.5)
            change = noise + trend_boost
            info["demand"] = max(1, min(100, info["demand"] + change * 0.5))
            info["growth"] = round(info["growth"] + change * 0.2, 1)
            if info["growth"] > 50:
                info["trend"] = "soaring"
            elif info["growth"] > 15:
                info["trend"] = "rising"
            elif info["growth"] < -5:
                info["trend"] = "declining"
            else:
                info["trend"] = "stable"
            if abs(change) > 1:
                changes[skill] = round(change, 1)

        self._save_trends()
        weekly_entry = {
            "date": datetime.now().isoformat(),
            "changes": changes,
            "snapshot": {k: v["demand"] for k, v in self.current_trends.items()},
        }
        self.history["weekly"].append(weekly_entry)
        self.history["weekly"] = self.history["weekly"][-52:]
        self._save_history()
        return weekly_entry

    def get_quarterly_report(self) -> Dict:
        soaring = {s: i for s, i in self.current_trends.items() if i["trend"] == "soaring"}
        rising = {s: i for s, i in self.current_trends.items() if i["trend"] == "rising"}
        declining = {s: i for s, i in self.current_trends.items() if i["trend"] == "declining"}
        return {
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "quarter": f"Q{(datetime.now().month - 1) // 3 + 1} {datetime.now().year}",
            "summary": {
                "total_skills_tracked": len(self.current_trends),
                "soaring": len(soaring),
                "rising": len(rising),
                "stable": sum(1 for i in self.current_trends.values() if i["trend"] == "stable"),
                "declining": len(declining),
            },
            "fastest_growing": sorted(
                [{"skill": s, **i} for s, i in self.current_trends.items()],
                key=lambda x: x["growth"], reverse=True
            )[:5],
            "fastest_declining": sorted(
                [{"skill": s, **i} for s, i in self.current_trends.items()],
                key=lambda x: x["growth"]
            )[:5],
        }

    def search_trends(self, query: str) -> List[Dict]:
        query = query.lower()
        results = []
        for skill, info in self.current_trends.items():
            if query in skill.lower() or query in info["category"].lower():
                results.append({"skill": skill, **info})
        return results
