from typing import Dict, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


CAREER_TRANSITIONS = {
    "Data Analyst": {
        "next_roles": ["Data Scientist", "Data Engineer", "Business Analyst"],
        "promotion_roles": ["Senior Data Analyst", "Lead Data Analyst", "Analytics Manager"],
        "pivot_roles": ["Business Intelligence Engineer", "Product Analyst", "Marketing Analyst"],
        "estimated_salary_growth": "15-25% per promotion",
    },
    "Data Scientist": {
        "next_roles": ["Senior Data Scientist", "ML Engineer", "AI Engineer"],
        "promotion_roles": ["Senior Data Scientist", "Staff Data Scientist", "Principal Data Scientist"],
        "pivot_roles": ["Research Scientist", "MLOps Engineer", "Data Engineer"],
        "estimated_salary_growth": "20-30% per promotion",
    },
    "Data Engineer": {
        "next_roles": ["Senior Data Engineer", "Data Architect", "MLOps Engineer"],
        "promotion_roles": ["Senior Data Engineer", "Staff Data Engineer", "Data Engineering Manager"],
        "pivot_roles": ["Data Scientist", "ML Engineer", "Software Engineer"],
        "estimated_salary_growth": "15-20% per promotion",
    },
    "ML Engineer": {
        "next_roles": ["Senior ML Engineer", "AI Engineer", "Research Scientist"],
        "promotion_roles": ["Senior ML Engineer", "Staff ML Engineer", "ML Architect"],
        "pivot_roles": ["Data Scientist", "MLOps Engineer", "Software Engineer"],
        "estimated_salary_growth": "20-35% per promotion",
    },
    "AI Engineer": {
        "next_roles": ["Senior AI Engineer", "AI Architect", "Research Scientist"],
        "promotion_roles": ["Senior AI Engineer", "Principal AI Engineer", "AI Director"],
        "pivot_roles": ["ML Engineer", "Data Scientist", "Software Engineer"],
        "estimated_salary_growth": "25-40% per promotion",
    },
    "Software Engineer": {
        "next_roles": ["Senior Software Engineer", "Tech Lead", "Software Architect"],
        "promotion_roles": ["Senior Software Engineer", "Staff Engineer", "Principal Engineer"],
        "pivot_roles": ["ML Engineer", "DevOps Engineer", "Data Engineer"],
        "estimated_salary_growth": "15-25% per promotion",
    },
    "DevOps Engineer": {
        "next_roles": ["Senior DevOps Engineer", "Cloud Architect", "SRE"],
        "promotion_roles": ["Senior DevOps Engineer", "DevOps Manager", "Infrastructure Director"],
        "pivot_roles": ["MLOps Engineer", "Software Engineer", "Data Engineer"],
        "estimated_salary_growth": "15-20% per promotion",
    },
}

ROLE_SALARY_BANDS = {
    "Data Analyst": {"entry": (55000, 75000), "mid": (75000, 100000), "senior": (100000, 130000), "lead": (130000, 160000)},
    "Data Scientist": {"entry": (80000, 110000), "mid": (110000, 145000), "senior": (145000, 185000), "lead": (185000, 230000)},
    "Data Engineer": {"entry": (75000, 100000), "mid": (100000, 135000), "senior": (135000, 175000), "lead": (175000, 215000)},
    "ML Engineer": {"entry": (90000, 120000), "mid": (120000, 160000), "senior": (160000, 210000), "lead": (210000, 260000)},
    "AI Engineer": {"entry": (100000, 135000), "mid": (135000, 175000), "senior": (175000, 225000), "lead": (225000, 280000)},
    "Software Engineer": {"entry": (70000, 95000), "mid": (95000, 130000), "senior": (130000, 170000), "lead": (170000, 210000)},
    "DevOps Engineer": {"entry": (70000, 95000), "mid": (95000, 125000), "senior": (125000, 160000), "lead": (160000, 195000)},
    "Research Scientist": {"entry": (85000, 115000), "mid": (115000, 150000), "senior": (150000, 200000), "lead": (200000, 250000)},
    "Business Analyst": {"entry": (50000, 70000), "mid": (70000, 95000), "senior": (95000, 125000), "lead": (125000, 155000)},
    "Cloud Architect": {"entry": (100000, 130000), "mid": (130000, 170000), "senior": (170000, 215000), "lead": (215000, 265000)},
}


class CareerPathAnalyzer:
    def __init__(self):
        self.transitions = CAREER_TRANSITIONS
        self.salary_bands = ROLE_SALARY_BANDS

    def get_career_paths(self, current_role: str) -> Dict:
        paths = self.transitions.get(current_role, {})
        if not paths:
            return {"current_role": current_role, "error": "No career path data available"}
        return {
            "current_role": current_role,
            "next_roles": self._enrich_role_info(paths.get("next_roles", [])),
            "promotion_roles": self._enrich_role_info(paths.get("promotion_roles", [])),
            "pivot_roles": self._enrich_role_info(paths.get("pivot_roles", [])),
            "salary_growth": paths.get("estimated_salary_growth", ""),
            "current_salary_band": self.salary_bands.get(current_role, {}),
        }

    def _enrich_role_info(self, roles: List[str]) -> List[Dict]:
        return [{"role": r, "salary_band": self.salary_bands.get(r, {})} for r in roles]

    def get_salary_band(self, role: str, level: str = "mid") -> Optional[tuple]:
        band = self.salary_bands.get(role, {})
        return band.get(level)

    def get_projection(self, current_role: str, current_salary: float,
                        years_experience: int) -> List[Dict]:
        projections = []
        bands = self.salary_bands.get(current_role, {})
        base_level = "entry" if years_experience < 2 else ("mid" if years_experience < 5 else
                                                            "senior" if years_experience < 10 else "lead")
        current_band = bands.get(base_level, (current_salary * 0.8, current_salary * 1.2))

        for y in range(0, 11, 2):
            projected = current_salary * (1 + 0.08 * y / 2)
            projections.append({
                "year": y,
                "estimated_salary": round(projected),
                "role": current_role,
                "confidence": max(0.3, 0.9 - y * 0.05),
            })

        return projections

    def get_all_roles(self) -> List[str]:
        return list(self.salary_bands.keys())
