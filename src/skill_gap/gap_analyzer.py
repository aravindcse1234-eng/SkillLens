import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SkillGapAnalyzer:
    def __init__(self):
        self.market_skills: Dict[str, float] = {}
        self.skill_categories: Dict[str, str] = {}

    def load_market_data(self, df: pd.DataFrame, skill_col: str = "skill_name",
                          demand_col: str = "demand_count") -> None:
        skill_counts = df.groupby(skill_col)[demand_col].sum().sort_values(ascending=False)
        max_count = skill_counts.max() if not skill_counts.empty else 1
        self.market_skills = (skill_counts / max_count * 100).to_dict()
        logger.info(f"Loaded {len(self.market_skills)} market skills")

    def load_skill_categories(self, category_map: Dict[str, str]) -> None:
        self.skill_categories = category_map

    def analyze_gap(self, user_skills: List[str],
                    target_role: Optional[str] = None) -> Dict:
        if not self.market_skills:
            logger.warning("No market skills loaded")
            return {}

        user_skills_lower = set(s.lower() for s in user_skills)
        market_skills_lower = {k.lower(): v for k, v in self.market_skills.items()}

        matched_skills = []
        missing_skills = []
        gap_details = []

        for skill, demand_score in market_skills_lower.items():
            original_key = [k for k in self.market_skills if k.lower() == skill]
            skill_name = original_key[0] if original_key else skill

            if skill in user_skills_lower:
                matched_skills.append({
                    "skill": skill_name,
                    "market_demand": demand_score,
                    "user_has": True,
                    "priority": "maintained",
                    "gap_score": 0
                })
            else:
                gap_score = demand_score
                if gap_score > 0:
                    priority = "critical" if gap_score > 80 else ("high" if gap_score > 60 else
                                                                    ("medium" if gap_score > 40 else "low"))
                    missing_skills.append({
                        "skill": skill_name,
                        "market_demand": demand_score,
                        "user_has": False,
                        "priority": priority,
                        "gap_score": gap_score
                    })

        missing_skills.sort(key=lambda x: x["gap_score"], reverse=True)

        total_possible = len(matched_skills) + len(missing_skills)
        match_percentage = (len(matched_skills) / total_possible * 100) if total_possible > 0 else 0

        result = {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills[:30],
            "total_matched": len(matched_skills),
            "total_missing": len(missing_skills),
            "match_percentage": round(match_percentage, 1),
            "overall_gap_score": round(100 - match_percentage, 1),
            "critical_gaps": [s for s in missing_skills if s["priority"] == "critical"][:10],
            "high_priority_gaps": [s for s in missing_skills if s["priority"] == "high"][:10],
        }

        return result

    def compare_with_role(self, user_skills: List[str],
                           role_skills: List[str]) -> Dict:
        user_set = set(s.lower() for s in user_skills)
        role_set = set(s.lower() for s in role_skills)

        matched = [s for s in role_skills if s.lower() in user_set]
        missing = [s for s in role_skills if s.lower() not in user_set]
        extra = [s for s in user_skills if s.lower() not in role_set]

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills": extra,
            "match_rate": round(len(matched) / len(role_set) * 100, 1) if role_set else 0,
            "completeness_score": round(len(matched) / (len(role_set) + len(extra)) * 100, 1) if (role_set or extra) else 0,
        }
