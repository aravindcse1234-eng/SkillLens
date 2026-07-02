import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.utils.helpers import get_device
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ATSScorer:
    def __init__(self):
        self.device = str(get_device())
        self.transformer = SentenceTransformer(
            "all-MiniLM-L6-v2", device=self.device
        )
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    def compute_keyword_match(self, resume_text: str, job_description: str) -> float:
        try:
            corpus = [resume_text, job_description]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Keyword match failed: {e}")
            return 0.0

    def compute_semantic_similarity(self, resume_text: str, job_description: str) -> float:
        try:
            embeddings = self.transformer.encode([resume_text, job_description])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.0

    def compute_skills_match(self, resume_skills: List[str],
                              required_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        if not required_skills:
            return 1.0, [], []

        resume_lower = set(s.lower() for s in resume_skills)
        required_lower = set(s.lower() for s in required_skills)

        matched = [s for s in required_skills if s.lower() in resume_lower]
        missing = [s for s in required_skills if s.lower() not in resume_lower]

        score = len(matched) / len(required_skills) if required_skills else 0
        return score, matched, missing

    def compute_experience_match(self, resume_years: Optional[float],
                                   required_years: Optional[float]) -> float:
        if resume_years is None or required_years is None:
            return 0.5
        if resume_years >= required_years:
            return 1.0
        return max(0, resume_years / required_years)

    def compute_education_match(self, resume_education: Optional[str],
                                 required_education: Optional[str]) -> float:
        if not required_education or not resume_education:
            return 0.5
        levels = {"High School": 1, "Diploma": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
        r_level = levels.get(resume_education, 0)
        req_level = levels.get(required_education, 0)
        if r_level >= req_level:
            return 1.0
        return max(0, r_level / req_level) if req_level > 0 else 0.5

    def calculate_ats_score(self, resume_data: Dict, job_requirements: Dict,
                             weights: Optional[Dict] = None) -> Dict:
        if weights is None:
            weights = {"keyword_match": 0.25, "semantic_match": 0.20,
                       "skills_match": 0.30, "experience_match": 0.15,
                       "education_match": 0.10}

        resume_text = resume_data.get("clean_text", "")
        job_text = job_requirements.get("description", "")

        keyword_score = self.compute_keyword_match(resume_text, job_text)
        semantic_score = self.compute_semantic_similarity(resume_text, job_text)
        skills_score, matched_skills, missing_skills = self.compute_skills_match(
            resume_data.get("extracted_skills", []),
            job_requirements.get("required_skills", [])
        )
        exp_score = self.compute_experience_match(
            resume_data.get("experience_years"),
            job_requirements.get("experience_required")
        )
        edu_score = self.compute_education_match(
            resume_data.get("education_level"),
            job_requirements.get("education_required")
        )

        components = {
            "keyword_match": keyword_score,
            "semantic_match": semantic_score,
            "skills_match": skills_score,
            "experience_match": exp_score,
            "education_match": edu_score,
        }

        total_score = sum(components[k] * weights[k] for k in weights)
        total_score = min(max(total_score, 0), 1) * 100

        return {
            "ats_score": round(total_score, 1),
            "components": {k: round(v, 3) for k, v in components.items()},
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "improvement_suggestions": self._generate_suggestions(components, weights),
        }

    def _generate_suggestions(self, components: Dict, weights: Dict) -> List[str]:
        suggestions = []
        if components["skills_match"] < 0.5:
            suggestions.append("Add more relevant skills from the job description")
        if components["keyword_match"] < 0.4:
            suggestions.append("Include more keywords from the job description")
        if components["semantic_match"] < 0.4:
            suggestions.append("Better align your experience with the role requirements")
        if components["experience_match"] < 0.7:
            suggestions.append("Highlight years of experience more prominently")
        if components["education_match"] < 0.7:
            suggestions.append("Emphasize your educational qualifications")
        if not suggestions:
            suggestions.append("Your resume is well-aligned with the job requirements")
        return suggestions
