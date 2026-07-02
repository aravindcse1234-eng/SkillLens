from typing import Dict, List, Optional
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

GLOBAL_COUNTRIES = {
    "India": {"region": "Asia", "demand_level": "High", "salary_mult": 1.0, "growth_rate": 0.12, "tech_hub_score": 85},
    "United States": {"region": "North America", "demand_level": "Very High", "salary_mult": 2.5, "growth_rate": 0.08, "tech_hub_score": 98},
    "Germany": {"region": "Europe", "demand_level": "Very High", "salary_mult": 1.8, "growth_rate": 0.09, "tech_hub_score": 88},
    "United Kingdom": {"region": "Europe", "demand_level": "High", "salary_mult": 1.7, "growth_rate": 0.07, "tech_hub_score": 85},
    "Singapore": {"region": "Asia", "demand_level": "High", "salary_mult": 1.9, "growth_rate": 0.11, "tech_hub_score": 90},
    "UAE": {"region": "Middle East", "demand_level": "Medium", "salary_mult": 1.6, "growth_rate": 0.14, "tech_hub_score": 78},
    "Australia": {"region": "Oceania", "demand_level": "High", "salary_mult": 1.9, "growth_rate": 0.08, "tech_hub_score": 82},
    "Canada": {"region": "North America", "demand_level": "High", "salary_mult": 2.0, "growth_rate": 0.09, "tech_hub_score": 86},
    "Netherlands": {"region": "Europe", "demand_level": "High", "salary_mult": 1.8, "growth_rate": 0.10, "tech_hub_score": 84},
    "Japan": {"region": "Asia", "demand_level": "Medium", "salary_mult": 1.5, "growth_rate": 0.05, "tech_hub_score": 80},
}

EMERGING_SKILL_SIGNALS = {
    "Agentic AI": {"sources": ["Google Trends +120%", "GitHub Stars +340%", "Papers +280%"], "growth_18m": 300, "category": "AI", "maturity": "Emerging"},
    "MCP (Model Context Protocol)": {"sources": ["GitHub Stars +500%", "Anthropic adoption"], "growth_18m": 400, "category": "AI Infrastructure", "maturity": "Early"},
    "AI Security": {"sources": ["Job posts +180%", "Research papers +150%"], "growth_18m": 250, "category": "Security", "maturity": "Growing"},
    "Synthetic Data Engineering": {"sources": ["Google Trends +90%", "Venture funding +200%"], "growth_18m": 220, "category": "Data", "maturity": "Growing"},
    "Edge AI": {"sources": ["IoT adoption +60%", "TinyML +120%"], "growth_18m": 180, "category": "AI Infrastructure", "maturity": "Emerging"},
    "AI Governance": {"sources": ["Regulations +200%", "Job posts +90%"], "growth_18m": 210, "category": "Governance", "maturity": "Growing"},
    "Rust for Data": {"sources": ["GitHub Stars +110%", "Community +80%"], "growth_18m": 150, "category": "Programming", "maturity": "Growing"},
    "Causal AI": {"sources": ["Research +130%", "Industry adoption +70%"], "growth_18m": 190, "category": "AI", "maturity": "Emerging"},
}

HIRING_EVENTS = [
    {"event": "Tech Layoffs Wave", "impact": "Data Analyst hiring -12%", "type": "negative", "severity": "medium", "date": "Last 30 days"},
    {"event": "AI Investment Surge", "impact": "AI Engineer hiring +25%", "type": "positive", "severity": "high", "date": "Last 30 days"},
    {"event": "Cloud Migration Push", "impact": "Cloud Architect hiring +18%", "type": "positive", "severity": "high", "date": "Last 60 days"},
    {"event": "Remote Work Normalization", "impact": "Remote roles +8%", "type": "positive", "severity": "low", "date": "Last 90 days"},
    {"event": "GenAI Tooling Boom", "impact": "LangChain/Vector DB roles +35%", "type": "positive", "severity": "high", "date": "Last 30 days"},
]


class WorkforceIntelligence:
    def __init__(self):
        self.countries = GLOBAL_COUNTRIES
        self.emerging_skills = EMERGING_SKILL_SIGNALS
        self.hiring_events = HIRING_EVENTS

    def get_global_demand(self, role: str = "Data Scientist") -> List[Dict]:
        results = []
        for country, info in self.countries.items():
            is_india = country == "India"
            currency = "\u20b9" if is_india else "$"
            rate = 83 if is_india else 1
            base_salary = 80000 if role in ("Data Scientist", "ML Engineer", "AI Engineer") else 60000
            salary = int(base_salary * info["salary_mult"])
            results.append({
                "country": country,
                "region": info["region"],
                "demand_level": info["demand_level"],
                "avg_salary": salary,
                "avg_salary_local": f"{currency}{salary * rate:,}",
                "growth_rate": f"{info['growth_rate']*100:.0f}%",
                "tech_hub_score": info["tech_hub_score"],
            })
        return sorted(results, key=lambda x: x["avg_salary"], reverse=True)

    def get_emerging_skills(self) -> List[Dict]:
        return [{"skill": s, **info} for s, info in self.emerging_skills.items()]

    def get_hiring_alerts(self) -> List[Dict]:
        return self.hiring_events

    def compare_countries(self, role: str = "Data Scientist") -> Dict:
        data = self.get_global_demand(role)
        highest = max(data, key=lambda x: x["avg_salary"])
        best_growth = max(data, key=lambda x: float(x["growth_rate"].rstrip("%")))
        return {
            "all_countries": data,
            "highest_salary": highest["country"],
            "highest_salary_value": highest["avg_salary"],
            "best_growth": best_growth["country"],
            "best_growth_value": best_growth["growth_rate"],
        }

    def early_warning_score(self, skill: str) -> Optional[Dict]:
        signal = self.emerging_skills.get(skill)
        if not signal:
            return None
        growth = signal["growth_18m"]
        if growth >= 300:
            urgency = "Critical — Learn Now"
        elif growth >= 200:
            urgency = "High — Start within 3 months"
        elif growth >= 100:
            urgency = "Medium — Watch and prepare"
        else:
            urgency = "Low — Monitor"
        return {"skill": skill, "growth_18m": growth, "urgency": urgency, "sources": signal["sources"], "category": signal["category"], "maturity": signal["maturity"]}
