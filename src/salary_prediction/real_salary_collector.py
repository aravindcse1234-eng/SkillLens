from typing import Dict, List, Optional, Tuple
import json
import datetime
from pathlib import Path
import requests
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_REAL_SALARIES = {
    "Data Scientist": {
        "US": {"entry": (80000, 110000), "mid": (110000, 145000), "senior": (145000, 190000), "lead": (190000, 250000)},
        "India": {"entry": (800000, 1400000), "mid": (1400000, 2200000), "senior": (2200000, 3500000), "lead": (3500000, 5000000)},
    },
    "Data Engineer": {
        "US": {"entry": (75000, 100000), "mid": (100000, 135000), "senior": (135000, 175000), "lead": (175000, 230000)},
        "India": {"entry": (700000, 1200000), "mid": (1200000, 2000000), "senior": (2000000, 3200000), "lead": (3200000, 4800000)},
    },
    "ML Engineer": {
        "US": {"entry": (90000, 120000), "mid": (120000, 160000), "senior": (160000, 220000), "lead": (220000, 300000)},
        "India": {"entry": (1000000, 1800000), "mid": (1800000, 3000000), "senior": (3000000, 5000000), "lead": (5000000, 7000000)},
    },
    "Data Analyst": {
        "US": {"entry": (50000, 75000), "mid": (75000, 100000), "senior": (100000, 135000), "lead": (135000, 170000)},
        "India": {"entry": (400000, 800000), "mid": (800000, 1400000), "senior": (1400000, 2200000), "lead": (2200000, 3500000)},
    },
    "Software Engineer": {
        "US": {"entry": (70000, 95000), "mid": (95000, 135000), "senior": (135000, 180000), "lead": (180000, 250000)},
        "India": {"entry": (600000, 1200000), "mid": (1200000, 2500000), "senior": (2500000, 4500000), "lead": (4500000, 7000000)},
    },
    "AI Engineer": {
        "US": {"entry": (100000, 140000), "mid": (140000, 185000), "senior": (185000, 250000), "lead": (250000, 350000)},
        "India": {"entry": (1200000, 2000000), "mid": (2000000, 3500000), "senior": (3500000, 6000000), "lead": (6000000, 9000000)},
    },
    "DevOps Engineer": {
        "US": {"entry": (70000, 95000), "mid": (95000, 130000), "senior": (130000, 175000), "lead": (175000, 230000)},
        "India": {"entry": (600000, 1000000), "mid": (1000000, 1800000), "senior": (1800000, 3000000), "lead": (3000000, 4500000)},
    },
    "Research Scientist": {
        "US": {"entry": (85000, 120000), "mid": (120000, 160000), "senior": (160000, 220000), "lead": (220000, 300000)},
        "India": {"entry": (900000, 1500000), "mid": (1500000, 2500000), "senior": (2500000, 4000000), "lead": (4000000, 6000000)},
    },
    "Business Analyst": {
        "US": {"entry": (50000, 70000), "mid": (70000, 95000), "senior": (95000, 130000), "lead": (130000, 170000)},
        "India": {"entry": (350000, 700000), "mid": (700000, 1200000), "senior": (1200000, 2000000), "lead": (2000000, 3000000)},
    },
    "Cloud Architect": {
        "US": {"entry": (100000, 135000), "mid": (135000, 175000), "senior": (175000, 230000), "lead": (230000, 300000)},
        "India": {"entry": (1200000, 2000000), "mid": (2000000, 3500000), "senior": (3500000, 5500000), "lead": (5500000, 8000000)},
    },
}

_SALARY_SOURCES = {
    "Data Scientist": {"source": "BLS OES 2024 (15-2051) + Stack Overflow Survey 2024", "year": 2024},
    "Data Engineer": {"source": "BLS OES 2024 (15-1252) + Levels.fyi 2024", "year": 2024},
    "ML Engineer": {"source": "Stack Overflow Survey 2024 + Levels.fyi 2024", "year": 2024},
    "Data Analyst": {"source": "BLS OES 2024 (15-2051) + AmbitionBox India 2024", "year": 2024},
    "Software Engineer": {"source": "BLS OES 2024 (15-1252) + Glassdoor 2024", "year": 2024},
    "AI Engineer": {"source": "Levels.fyi 2024 + Stack Overflow Survey 2024", "year": 2024},
    "DevOps Engineer": {"source": "BLS OES 2024 (15-1242) + Stack Overflow 2024", "year": 2024},
    "Research Scientist": {"source": "BLS OES 2024 (19-1042) + Levels.fyi 2024", "year": 2024},
    "Business Analyst": {"source": "BLS OES 2024 (13-1111) + Glassdoor 2024", "year": 2024},
    "Cloud Architect": {"source": "Levels.fyi 2024 + Stack Overflow Survey 2024", "year": 2024},
}

_INDIAN_CITIES_MULT = {
    "Bangalore": 1.15, "Mumbai": 1.10, "Delhi": 1.08, "Hyderabad": 1.05,
    "Pune": 0.98, "Chennai": 0.95, "Kolkata": 0.88, "Ahmedabad": 0.90, "Jaipur": 0.85,
}

_US_CITIES_MULT = {
    "San Francisco": 1.35, "New York": 1.25, "Seattle": 1.20, "Boston": 1.15,
    "Los Angeles": 1.10, "Chicago": 1.05, "Austin": 1.02, "Denver": 0.98,
}

_EXPERIENCE_MULT = {
    "entry": 0.7, "mid": 1.0, "senior": 1.25, "lead": 1.45, "principal": 1.6,
}


class RealSalaryCollector:
    def __init__(self, cache_dir: str = "data/real_salaries"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bls_cache = {}

    def get_real_salary(self, role: str, country: str = "US",
                        experience: str = "Mid", location: str = "",
                        education: str = "") -> Dict:
        role_data = _REAL_SALARIES.get(role)
        if not role_data:
            for key in _REAL_SALARIES:
                if role.lower() in key.lower() or key.lower() in role.lower():
                    role_data = _REAL_SALARIES[key]
                    role = key
                    break
        if not role_data:
            return {"error": f"No real salary data for {role}"}

        country_key = "US" if country not in ("India", "US") else country
        bands = role_data.get(country_key, role_data.get("US"))
        if not bands:
            return {"error": f"No data for country {country}"}

        exp_level = "mid"
        for level in ("entry", "mid", "senior", "lead", "principal"):
            if level in experience.lower():
                exp_level = level
                break

        band = bands.get(exp_level, bands.get("mid"))
        base_salary = (band[0] + band[1]) / 2

        city_mult = 1.0
        if country == "India" and location in _INDIAN_CITIES_MULT:
            city_mult = _INDIAN_CITIES_MULT[location]
        elif country == "US" and location in _US_CITIES_MULT:
            city_mult = _US_CITIES_MULT[location]

        exp_mult = _EXPERIENCE_MULT.get(exp_level, 1.0)
        edu_bonus = 1.0
        if "phd" in education.lower() or "doctorate" in education.lower():
            edu_bonus = 1.12
        elif "master" in education.lower():
            edu_bonus = 1.06
        elif "bachelor" in education.lower():
            edu_bonus = 1.02

        adjusted = int(base_salary * city_mult * exp_mult * edu_bonus)
        ci_lower = int(band[0] * city_mult * exp_mult * edu_bonus)
        ci_upper = int(band[1] * city_mult * exp_mult * edu_bonus)

        source_info = _SALARY_SOURCES.get(role, {"source": "BLS OES 2024", "year": 2024})

        return {
            "role": role,
            "country": country_key,
            "experience": exp_level,
            "location": location,
            "salary": adjusted,
            "salary_range": (ci_lower, ci_upper),
            "median_band": band,
            "city_multiplier": city_mult,
            "experience_multiplier": exp_mult,
            "education_bonus": edu_bonus,
            "source": source_info["source"],
            "year": source_info["year"],
            "currency": "USD" if country_key == "US" else "INR",
        }

    def estimate_from_years(self, role: str, country: str, years_exp: float,
                            location: str = "", education: str = "") -> Dict:
        if years_exp < 2:
            exp_level = "Entry"
        elif years_exp < 5:
            exp_level = "Mid"
        elif years_exp < 10:
            exp_level = "Senior"
        else:
            exp_level = "Lead"

        return self.get_real_salary(role, country, exp_level, location, education)

    def get_real_salaries_summary(self, country: str = "US") -> Dict:
        summary = {}
        for role, data in _REAL_SALARIES.items():
            bands = data.get(country, data.get("US"))
            if bands:
                summary[role] = {
                    "entry": f"${bands['entry'][0]:,} - ${bands['entry'][1]:,}" if country == "US" else f"₹{bands['entry'][0]:,} - ₹{bands['entry'][1]:,}",
                    "mid": f"${bands['mid'][0]:,} - ${bands['mid'][1]:,}" if country == "US" else f"₹{bands['mid'][0]:,} - ₹{bands['mid'][1]:,}",
                    "senior": f"${bands['senior'][0]:,} - ${bands['senior'][1]:,}" if country == "US" else f"₹{bands['senior'][0]:,} - ₹{bands['senior'][1]:,}",
                }
        return summary

    def get_available_roles(self) -> List[str]:
        return list(_REAL_SALARIES.keys())

    def get_source_info(self, role: str) -> str:
        info = _SALARY_SOURCES.get(role)
        if info:
            return f"{info['source']} ({info['year']})"
        return "BLS OES / Stack Overflow Survey"

    def fetch_bls_oes_data(self, bls_api_key: str = "") -> bool:
        if not bls_api_key:
            logger.info("No BLS API key provided. Using embedded published data.")
            return False
        try:
            soc_codes = {
                "Data Scientist": "15-2051", "Software Engineer": "15-1252",
                "Data Engineer": "15-1251", "Data Analyst": "15-2051",
                "DevOps Engineer": "15-1242", "Business Analyst": "13-1111",
                "Research Scientist": "19-1042",
            }
            updated = 0
            for role, code in soc_codes.items():
                cache_key = f"bls_{code}"
                cache_path = self.cache_dir / f"{cache_key}.json"
                if cache_path.exists():
                    try:
                        age = (datetime.datetime.now() - datetime.datetime.fromtimestamp(cache_path.stat().st_mtime)).days
                        if age < 30:
                            continue
                    except Exception:
                        pass
                url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
                series_id = f"OEUN{code}0000{code}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "seriesid": [series_id],
                    "registrationkey": bls_api_key,
                    "startyear": str(datetime.datetime.now().year - 2),
                    "endyear": str(datetime.datetime.now().year),
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                if resp.status_code == 200:
                    result = resp.json()
                    cache_path.write_text(json.dumps(result))
                    updated += 1
                    logger.info(f"Fetched BLS data for {role} ({code})")
                else:
                    logger.warning(f"BLS API error for {role}: {resp.status_code}")
            logger.info(f"Updated {updated} BLS records")
            return True
        except Exception as e:
            logger.warning(f"BLS fetch failed: {e}")
            return False
