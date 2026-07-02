from typing import Dict, List, Optional, Tuple
import numpy as np
from src.utils.logger import setup_logger
from src.skill_graph.skill_graph import SKILL_RELATIONSHIPS, ROLE_SKILL_MAP
from src.skill_graph.career_paths import ROLE_SALARY_BANDS, CAREER_TRANSITIONS

logger = setup_logger(__name__)

DIGITAL_TWIN_ROLES = list(ROLE_SKILL_MAP.keys())

SKILL_LEARNING_MONTHS = {
    "Python": 3, "SQL": 2, "Machine Learning": 4, "Deep Learning": 4,
    "NLP": 3, "Computer Vision": 4, "PyTorch": 3, "TensorFlow": 3,
    "AWS": 4, "Azure": 4, "GCP": 3, "Docker": 2, "Kubernetes": 4,
    "Git": 1, "Linux": 2, "Spark": 3, "Airflow": 2, "MLflow": 1,
    "FastAPI": 2, "Generative AI": 3, "LangChain": 2, "RAG": 2,
    "Statistics": 3, "NumPy": 1, "Pandas": 2, "Scikit-Learn": 2,
    "Tableau": 2, "Power BI": 2, "Excel": 1, "Java": 4, "JavaScript": 3,
    "React": 3, "Terraform": 3, "CI/CD": 2, "MongoDB": 2,
    "Transformers": 3, "Data Engineering": 3, "MLOps": 3,
}

EDUCATION_BASE_MAP = {
    "B.Tech": {"score": 1.0, "salary_mult": 1.0},
    "M.Tech": {"score": 1.15, "salary_mult": 1.1},
    "B.Sc": {"score": 0.9, "salary_mult": 0.9},
    "M.Sc": {"score": 1.05, "salary_mult": 1.0},
    "BCA": {"score": 0.85, "salary_mult": 0.85},
    "MCA": {"score": 1.0, "salary_mult": 0.95},
    "B.E": {"score": 1.0, "salary_mult": 1.0},
    "PhD": {"score": 1.3, "salary_mult": 1.2},
    "MBA": {"score": 1.1, "salary_mult": 1.15},
    "Diploma": {"score": 0.75, "salary_mult": 0.75},
    "Self-taught": {"score": 0.7, "salary_mult": 0.7},
}

INDIAN_ROLE_BASE_SALARIES = {
    "Data Analyst": 500000, "Data Scientist": 800000, "Data Engineer": 700000,
    "ML Engineer": 900000, "AI Engineer": 1000000, "Software Engineer": 600000,
    "DevOps Engineer": 700000, "Research Scientist": 850000,
    "Business Analyst": 450000, "Cloud Architect": 1200000,
}

INDIAN_RUPEE_SYMBOL = "\u20b9"


class CareerTwinEngine:
    def __init__(self):
        self.role_skills = ROLE_SKILL_MAP
        self.salary_bands = ROLE_SALARY_BANDS
        self.transitions = CAREER_TRANSITIONS
        self.skill_learning = SKILL_LEARNING_MONTHS
        self.edu_map = EDUCATION_BASE_MAP
        self.role_base = INDIAN_ROLE_BASE_SALARIES

    def create_profile(self, skills: List[str], education: str = "B.Tech",
                       experience_years: float = 0, current_role: str = "",
                       salary: Optional[int] = None, location: str = "Bangalore") -> Dict:
        edu_info = self.edu_map.get(education, {"score": 1.0, "salary_mult": 1.0})
        if not current_role:
            current_role = self._infer_role(skills)
        if salary is None:
            salary = self._estimate_salary(current_role, experience_years, edu_info["salary_mult"], location)
        return {
            "skills": skills,
            "education": education,
            "education_score": edu_info["score"],
            "education_salary_mult": edu_info["salary_mult"],
            "experience_years": experience_years,
            "current_role": current_role,
            "current_salary": salary,
            "location": location,
            "profile_strength": self._compute_profile_strength(skills, current_role),
        }

    def _infer_role(self, skills: List[str]) -> str:
        best_role, best_score = "Data Analyst", 0
        for role, role_skills in self.role_skills.items():
            score = sum(1 for s in skills if s in role_skills)
            if score > best_score:
                best_score, best_role = score, role
        return best_role

    def _estimate_salary(self, role: str, exp_years: float, edu_mult: float, location: str) -> int:
        base = self.role_base.get(role, 600000)
        exp_factor = 1 + (exp_years * 0.08)
        is_india = location in ("Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur")
        loc_mult = 1.0 if not is_india else 0.9
        return int(base * exp_factor * edu_mult * loc_mult)

    def _compute_profile_strength(self, skills: List[str], role: str) -> float:
        required = self.role_skills.get(role, [])
        if not required:
            return 50.0
        match = len([s for s in skills if s in required])
        skill_bonus = sum(1 for s in skills if s not in required) * 0.05
        return min(100.0, (match / len(required) * 70) + skill_bonus * 30 + 10)

    def simulate_paths(self, profile: Dict, top_k: int = 3) -> List[Dict]:
        paths = []
        current_role = profile["current_role"]
        candidates = self._get_path_candidates(current_role)
        for i, target_role in enumerate(candidates[:top_k]):
            path = self._build_path(profile, target_role)
            if path:
                path["rank"] = i + 1
                paths.append(path)
        if not paths:
            paths.append(self._build_path(profile, current_role))
        return sorted(paths, key=lambda p: p["final_salary"], reverse=True)

    def _get_path_candidates(self, role: str) -> List[str]:
        seen = set()
        candidates = []
        transitions = self.transitions.get(role, {})
        for key in ["next_roles", "promotion_roles", "pivot_roles"]:
            for r in transitions.get(key, []):
                if r not in seen and r in self.role_skills:
                    seen.add(r)
                    candidates.append(r)
        candidates += [r for r in self.role_skills if r not in seen]
        return candidates[:10]

    def _build_path(self, profile: Dict, target_role: str) -> Dict:
        current_role = profile["current_role"]
        current_skills = set(profile["skills"])
        required_skills = set(self.role_skills.get(target_role, []))
        missing_skills = [s for s in required_skills if s not in current_skills]
        mastered_skills = [s for s in required_skills if s in current_skills]
        learn_months = sum(self.skill_learning.get(s, 3) for s in missing_skills)
        current_salary = profile["current_salary"]
        target_band = self.salary_bands.get(target_role, {})
        entry_low, entry_high = target_band.get("entry", (current_salary, current_salary))
        senior_high = target_band.get("senior", (entry_high, entry_high))[1]
        transition_salary = max(current_salary, int((entry_low + entry_high) / 2))
        ten_year_growth = 1 + (0.10 if target_role in ("AI Engineer", "ML Engineer") else 0.08) * 10
        final_salary = int(transition_salary * ten_year_growth)
        projections = []
        sal = current_salary
        for y in range(11):
            growth = 0.06 + (0.04 if target_role in ("AI Engineer", "ML Engineer", "Research Scientist") else 0)
            if y == 0:
                sal = current_salary
            elif y == 1:
                sal = transition_salary
            else:
                sal = int(sal * (1 + growth))
            projections.append({"year": y, "salary": sal, "role": target_role if y >= 1 else current_role})
        gap_score = 1 - (len(missing_skills) / max(len(required_skills), 1))
        profile_strength = self._compute_profile_strength(profile["skills"], target_role)
        risk_score = max(0, min(100, int((len(missing_skills) / max(len(required_skills), 1)) * 60 + (1 - gap_score) * 40)))
        edu_factor = profile.get("education_score", 1.0)
        success_prob = min(98, int((gap_score * 50 + profile_strength * 0.3 + edu_factor * 20) / (1 + risk_score / 100)))
        return {
            "path_label": f"Path {'ABCDEFGH'[self._get_path_candidates(profile['current_role']).index(target_role) % 8]}",
            "target_role": target_role,
            "current_role": current_role,
            "missing_skills": missing_skills,
            "mastered_skills": mastered_skills,
            "total_learn_months": learn_months,
            "current_salary": current_salary,
            "transition_salary": transition_salary,
            "final_salary": final_salary,
            "salary_growth_pct": int((final_salary - current_salary) / current_salary * 100),
            "profile_strength": int(profile_strength),
            "gap_score": int(gap_score * 100),
            "risk_score": risk_score,
            "success_probability": success_prob,
            "key_skills_to_learn": missing_skills[:5],
            "projections": projections,
        }

    def year_by_year_simulation(self, profile: Dict, skills_to_learn: List[str],
                                 target_role: str, years: int = 10) -> List[Dict]:
        base = profile["current_salary"]
        role = target_role if target_role else profile["current_role"]
        target_band = self.salary_bands.get(role, {})
        entry_low, entry_high = target_band.get("entry", (base, int(base * 1.2)))
        transition_salary = max(base, int((entry_low + entry_high) / 2))
        premium = 0.02 * min(len(skills_to_learn), 5)
        simulation = []
        for y in range(years + 1):
            if y == 0:
                sim_sal = base
                sim_role = profile["current_role"]
                note = "Current"
            elif y <= 2:
                sim_sal = int(transition_salary * (1 + y * 0.08 + premium))
                sim_role = role
                note = f"Transition to {role}"
            else:
                growth = 0.06 + premium + (0.03 if y > 5 else 0)
                sim_sal = int(sim_sal * (1 + growth))
                sim_role = role
                note = "Growth phase" if y <= 5 else "Senior growth"
            simulation.append({"year": y, "salary": sim_sal, "role": sim_role, "note": note})
        return simulation


class CareerGPS:
    def __init__(self, engine: CareerTwinEngine):
        self.engine = engine

    def navigate(self, profile: Dict, target_role: str, salary_target: int,
                  timeline_months: int) -> Dict:
        required_skills = set(self.engine.role_skills.get(target_role, []))
        current_skills = set(profile["skills"])
        missing_skills = [s for s in required_skills if s not in current_skills]
        skill_sequence = self._build_skill_sequence(missing_skills, timeline_months)
        salary_proj = self._project_salary(profile, target_role, timeline_months)
        risk_score = self._compute_risk(profile, target_role, timeline_months)
        success_prob = self._compute_success_prob(profile, target_role, timeline_months)
        alt_routes = self._get_alternative_routes(profile, target_role)
        return {
            "current": {"role": profile["current_role"], "salary": profile["current_salary"],
                        "skills": profile["skills"]},
            "target": {"role": target_role, "salary_target": salary_target},
            "skill_sequence": skill_sequence,
            "salary_projections": salary_proj,
            "risk_score": risk_score,
            "success_probability": success_prob,
            "timeline_months": timeline_months,
            "alternative_routes": alt_routes,
        }

    def _build_skill_sequence(self, missing_skills: List[str], timeline_months: int) -> List[Dict]:
        sorted_skills = sorted(missing_skills, key=lambda s: self.engine.skill_learning.get(s, 3))
        total_months = sum(self.engine.skill_learning.get(s, 3) for s in sorted_skills)
        if total_months == 0:
            return []
        scale = min(1.0, timeline_months / total_months)
        sequence, month = [], 1
        for skill in sorted_skills:
            duration = max(1, int(self.engine.skill_learning.get(skill, 3) * scale))
            sequence.append({
                "skill": skill, "duration_weeks": duration * 4,
                "month": month, "total_months": duration,
            })
            month += duration
        return sequence

    def _project_salary(self, profile: Dict, target_role: str, timeline_months: int) -> List[Dict]:
        base = profile["current_salary"]
        target_band = self.engine.salary_bands.get(target_role, {})
        entry_high = target_band.get("entry", (base, int(base * 1.3)))[1]
        final_sal = max(base, entry_high)
        proj = []
        for y in range(11):
            progress = min(1.0, y / max(timeline_months / 12, 1))
            sal = int(base + (final_sal - base) * progress) if y > 0 else base
            sal = int(sal * (1 + 0.06 * y))
            proj.append({"year": y, "salary": sal, "role": target_role if y >= 1 else profile["current_role"]})
        return proj

    def _compute_risk(self, profile: Dict, target_role: str, timeline_months: int) -> int:
        required = set(self.engine.role_skills.get(target_role, []))
        current = set(profile["skills"])
        missing = required - current
        skill_risk = len(missing) / max(len(required), 1)
        time_risk = max(0, 1 - timeline_months / 24)
        next_roles = self.engine.transitions.get(profile["current_role"], {}).get("next_roles", [])
        transition_risk = 0.3 if profile["current_role"] != target_role and \
            target_role not in next_roles else 0.1
        return int(min(100, skill_risk * 50 + time_risk * 25 + transition_risk * 25))

    def _compute_success_prob(self, profile: Dict, target_role: str, timeline_months: int) -> int:
        required = set(self.engine.role_skills.get(target_role, []))
        current = set(profile["skills"])
        match = len(current & required) / max(len(required), 1)
        edu_factor = profile.get("education_score", 1.0)
        time_factor = min(1.0, timeline_months / 24)
        exp_factor = min(1.0, profile["experience_years"] / 5)
        prob = (match * 40 + time_factor * 25 + exp_factor * 20 + edu_factor * 15)
        return min(99, int(prob))

    def _get_alternative_routes(self, profile: Dict, target_role: str) -> List[Dict]:
        current = profile["current_role"]
        routes = []
        for role in self.engine.role_skills:
            if role != target_role and role != current:
                req = set(self.engine.role_skills[role])
                cur = set(profile["skills"])
                match = len(cur & req) / max(len(req), 1)
                if match > 0.3:
                    routes.append({"role": role, "match": int(match * 100)})
        return sorted(routes, key=lambda r: r["match"], reverse=True)[:3]

    def whats_if_simulation(self, profile: Dict, new_skills: List[str]) -> Dict:
        updated_skills = list(set(profile["skills"] + new_skills))
        updated_profile = self.engine.create_profile(
            updated_skills, profile["education"],
            profile["experience_years"], profile["current_role"],
            profile["current_salary"], profile["location"],
        )
        old_strength = profile["profile_strength"]
        new_strength = updated_profile["profile_strength"]
        salary_boost = int(sum(self.engine.skill_learning.get(s, 3) for s in new_skills) * 0.02 * profile["current_salary"])
        target_role = self.engine._infer_role(updated_skills)
        if target_role != profile["current_role"]:
            new_role_possible = target_role
            new_salary = self.engine._estimate_salary(
                target_role, profile["experience_years"],
                profile.get("education_salary_mult", 1.0), profile["location"])
        else:
            new_role_possible = profile["current_role"]
            new_salary = profile["current_salary"] + salary_boost
        return {
            "old_profile_strength": int(old_strength),
            "new_profile_strength": int(new_strength),
            "strength_gain": int(new_strength - old_strength),
            "new_role_possible": new_role_possible,
            "projected_salary": new_salary,
            "salary_increase": new_salary - profile["current_salary"],
            "new_skills_added": new_skills,
        }
