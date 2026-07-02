import re
from typing import Dict, List, Optional, Tuple
from src.nlp.skill_extractor import SkillExtractor
from src.data_cleaning.skill_standardizer import SkillStandardizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResumeAnalyzer:
    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.skill_std = SkillStandardizer()

    def extract_skills(self, text: str) -> List[str]:
        return self.skill_extractor.extract(text, method="hybrid")

    def extract_experience_years(self, text: str) -> Optional[float]:
        patterns = [
            r"(\d+)\+?\s*years?\s*(?:of)?\s*(?:experience|work)",
            r"experience\s*(?:of)?\s*(\d+)\+?\s*years?",
            r"(\d+)\+?\s*yrs?\s*(?:of)?\s*(?:experience|work)",
        ]
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([float(m) for m in matches if m])

        if years:
            total = sum(years) / len(years)
            ranges = re.findall(r"(\d+)\s*[-to]+\s*(\d+)\s*years?", text, re.IGNORECASE)
            if ranges:
                range_avg = sum((int(a) + int(b)) / 2 for a, b in ranges) / len(ranges)
                return (total + range_avg) / 2
            return total

        ranges = re.findall(r"(\d+)\s*[-to]+\s*(\d+)\s*years?", text, re.IGNORECASE)
        if ranges:
            return sum((int(a) + int(b)) / 2 for a, b in ranges) / len(ranges)

        year_mentions = re.findall(r"(?:19|20)\d{2}", text)
        if year_mentions:
            years_int = [int(y) for y in year_mentions]
            from datetime import datetime
            current = datetime.now().year
            return current - min(years_int) if years_int else None

        return None

    def extract_education_level(self, text: str) -> Optional[str]:
        edu_patterns = {
            "PhD": r"(?:ph\.?d|doctorate|doctoral|phd)",
            "Master": r"(?:master|m\.?sc|m\.?a\.?|mba|m\.?eng)",
            "Bachelor": r"(?:bachelor|b\.?sc|b\.?a\.?|b\.?eng|b\.?tech|undergraduate)",
            "Diploma": r"(?:diploma|associate|higher secondary)",
            "High School": r"(?:high school|secondary school|12th|10th)",
        }
        for level, pattern in edu_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return level
        return None

    def extract_job_titles(self, text: str) -> List[str]:
        title_patterns = [
            r"(?i)(data scientist|data engineer|ml engineer|machine learning engineer)",
            r"(?i)(data analyst|software engineer|ai engineer|devops engineer)",
            r"(?i)(research scientist|business analyst|deep learning engineer)",
            r"(?i)(product manager|project manager|tech lead|architect)",
        ]
        titles = set()
        for pattern in title_patterns:
            matches = re.findall(pattern, text)
            titles.update(m.strip() for m in matches)
        return list(titles)

    def analyze(self, resume_data: Dict) -> Dict:
        text = resume_data.get("clean_text", "")
        sections = resume_data.get("sections", {})

        skills = self.extract_skills(text)
        experience_years = self.extract_experience_years(text)
        education = self.extract_education_level(text)
        job_titles = self.extract_job_titles(text)

        skills_text = sections.get("skills", "")
        skills_from_section = self.extract_skills(skills_text)
        all_skills = list(set(skills + skills_from_section))

        result = {
            "file_name": resume_data.get("file_name", ""),
            "extracted_skills": self.skill_std.standardize_list(all_skills),
            "experience_years": experience_years,
            "education_level": education,
            "job_titles": job_titles,
            "text_length": len(text),
            "sections_found": list(sections.keys()),
        }
        return result
