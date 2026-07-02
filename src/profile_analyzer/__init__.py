from typing import Dict, List, Optional
import re
import requests
import datetime
from pathlib import Path
import json
from src.utils.logger import setup_logger
from src.skill_graph.skill_graph import ROLE_SKILL_MAP

logger = setup_logger(__name__)

GITHUB_SKILL_KEYWORDS = {
    "Python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
    "JavaScript": ["javascript", "react", "node", "vue", "angular", "typescript"],
    "Machine Learning": ["machine learning", "tensorflow", "pytorch", "sklearn", "scikit-learn"],
    "Deep Learning": ["deep learning", "neural network", "cnn", "rnn", "transformer"],
    "SQL": ["sql", "postgresql", "mysql", "sqlite", "database"],
    "AWS": ["aws", "s3", "lambda", "ec2", "bedrock", "sagemaker"],
    "Docker": ["docker", "container", "dockerfile"],
    "Kubernetes": ["kubernetes", "k8s", "kubectl"],
    "React": ["react", "reactjs", "redux", "nextjs"],
    "FastAPI": ["fastapi", "uvicorn"],
    "Generative AI": ["generative", "llm", "gpt", "langchain", "rag", "vector"],
    "Spark": ["spark", "pyspark", "databricks"],
    "Git": ["git", "github", "gitlab"],
    "CI/CD": ["ci/cd", "github actions", "jenkins", "gitlab ci"],
    "Java": ["java", "spring", "maven", "gradle"],
    "Go": ["golang", "go"],
    "Rust": ["rust"],
    "TypeScript": ["typescript", "ts"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch"],
    "NLP": ["nlp", "natural language", "spacy", "nltk", "huggingface"],
    "Computer Vision": ["computer vision", "opencv", "yolo", "image"],
    "MLOps": ["mlops", "mlflow", "kubeflow", "wandb"],
    "Data Engineering": ["data engineering", "etl", "data pipeline", "airflow", "kafka"],
    "Linux": ["linux", "bash", "shell", "unix"],
}

PORTFOLIO_WEIGHT = {
    "project_count": 0.15, "tech_diversity": 0.20, "recency": 0.10,
    "skill_relevance": 0.30, "resume_match": 0.25,
}


def extract_github_username(url: str) -> Optional[str]:
    patterns = [
        r"github\.com/([a-zA-Z0-9_-]+)", r"github\.com:([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def extract_linkedin_username(url: str) -> Optional[str]:
    m = re.search(r"linkedin\.com/in/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def fetch_github_repos(username: str, cache_dir: str = "data/live_cache") -> List[Dict]:
    cache_path = Path(cache_dir) / f"github_user_{username}.json"
    if cache_path.exists():
        try:
            age = (datetime.datetime.now() - datetime.datetime.fromtimestamp(cache_path.stat().st_mtime)).seconds
            if age < 1800:
                return json.loads(cache_path.read_text())
        except Exception:
            pass
    try:
        url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=30&type=public"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            repos = resp.json()
            parsed = []
            for r in repos:
                parsed.append({
                    "name": r.get("name", ""),
                    "description": r.get("description") or "",
                    "topics": r.get("topics", []),
                    "language": r.get("language") or "",
                    "stars": r.get("stargazers_count", 0),
                    "forks": r.get("forks_count", 0),
                    "url": r.get("html_url", ""),
                    "updated_at": r.get("updated_at", ""),
                })
            cache_path.write_text(json.dumps(parsed))
            logger.info(f"Fetched {len(parsed)} repos for {username}")
            return parsed
        logger.warning(f"GitHub API error for {username}: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"GitHub fetch failed for {username}: {e}")
    return []


def fetch_readme_keywords(username: str, repo_name: str) -> str:
    try:
        url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
        resp = requests.get(url, timeout=5, headers={"Accept": "application/vnd.github.v3.raw"})
        if resp.status_code == 200:
            return resp.text[:3000]
    except Exception:
        pass
    return ""


class ProfileAnalyzer:
    def __init__(self):
        self.skill_keywords = GITHUB_SKILL_KEYWORDS
        self.role_skills = ROLE_SKILL_MAP
        self.weights = PORTFOLIO_WEIGHT

    def analyze_from_urls(self, github_url: str = "", linkedin_url: str = "", portfolio_url: str = "") -> Dict:
        github_username = extract_github_username(github_url) if github_url else None
        linkedin_username = extract_linkedin_username(linkedin_url) if linkedin_url else None
        results = {"github": None, "linkedin": None, "portfolio": None}
        if github_username:
            repos = fetch_github_repos(github_username)
            results["github"] = self.analyze_github(repos)
            results["github"]["username"] = github_username
            results["github"]["repo_count"] = len(repos)
            results["github"]["top_repos"] = sorted(repos, key=lambda r: r["stars"], reverse=True)[:5]
        if linkedin_username:
            results["linkedin"] = {
                "username": linkedin_username,
                "profile_url": f"https://linkedin.com/in/{linkedin_username}",
                "has_profile": True,
            }
        if portfolio_url and not github_url and not linkedin_url:
            results["portfolio"] = {"url": portfolio_url, "analyzed": False}
        return results

    def analyze_github(self, repos: List[Dict]) -> Dict:
        if not repos:
            return self._empty_github()
        all_topics = set()
        all_languages = {}
        total_stars = 0
        for repo in repos:
            for t in repo.get("topics", []):
                all_topics.add(t.lower())
            lang = repo.get("language")
            if lang:
                all_languages[lang] = all_languages.get(lang, 0) + 1
            total_stars += repo.get("stars", repo.get("stargazers_count", 0))
        combined_text = " ".join(list(all_topics) + list(all_languages.keys())).lower()
        detected_skills = set()
        for skill, keywords in self.skill_keywords.items():
            if any(kw in combined_text for kw in keywords):
                detected_skills.add(skill)
        tech_diversity = len(detected_skills)
        score = min(100, int(
            len(repos) * 4 +
            tech_diversity * 8 +
            min(total_stars, 100) * 0.3
        ))
        return {
            "repos_count": len(repos),
            "languages": all_languages,
            "detected_skills": sorted(detected_skills),
            "tech_diversity": tech_diversity,
            "total_stars": total_stars,
            "score": score,
            "strength": "Strong" if score >= 70 else "Moderate" if score >= 40 else "Needs Work",
        }

    def _empty_github(self) -> Dict:
        return {"repos_count": 0, "languages": {}, "detected_skills": [], "tech_diversity": 0, "total_stars": 0, "score": 0, "strength": "No data"}

    def combined_portfolio_score(self, resume_skills: List[str], github_data: Dict) -> Dict:
        resume_score = min(100, len(resume_skills) * 8)
        github_score = github_data.get("score", 0)
        combined = int(resume_score * 0.6 + github_score * 0.4)
        return {
            "resume_skills_score": resume_score,
            "github_score": github_score,
            "combined_score": combined,
            "percentile": self._estimate_percentile(combined),
        }

    def _estimate_percentile(self, score: int) -> int:
        return min(99, int(50 + (score - 50) * 0.8))

    def benchmark_against_market(self, skills: List[str], role: str) -> Dict:
        required = set(self.role_skills.get(role, []))
        profile_skills = set(skills)
        has = profile_skills & required
        missing = required - profile_skills
        extra = profile_skills - required
        match_rate = len(has) / max(len(required), 1) * 100
        percentile = int(50 + (match_rate - 50) * 0.6)
        return {
            "role": role,
            "match_rate": round(match_rate, 1),
            "skills_matched": sorted(has),
            "skills_missing": sorted(missing),
            "extra_skills": sorted(extra),
            "percentile_rank": min(99, max(1, percentile)),
            "verdict": f"Stronger than {min(99, max(1, percentile))}% of candidates" if match_rate >= 60 else f"Weaker than {100 - min(99, max(1, percentile))}% of candidates",
        }

    def suggest_resume_improvements(self, skills: List[str], target_role: str) -> List[Dict]:
        required = set(self.role_skills.get(target_role, []))
        profile_skills = set(skills)
        missing = required - profile_skills
        suggestions = []
        for s in missing:
            suggestions.append({"type": "add", "skill": s, "reason": f"Required for {target_role}", "priority": "high"})
        for s in profile_skills - required:
            suggestions.append({"type": "highlight", "skill": s, "reason": "Differentiator — keep visible", "priority": "medium"})
        return suggestions[:8]

    def resume_rebuilder(self, skills: List[str], experience: int, education: str, target_style: str = "ATS") -> Dict:
        styles = {
            "ATS": {"format": "Single column, keyword-optimized", "sections": ["Summary", "Technical Skills", "Experience", "Education", "Certifications"], "tips": ["Use exact skill names from job description", "Avoid tables/graphics", "Use standard section headers"]},
            "FAANG": {"format": "Results-driven, STAR format", "sections": ["Summary", "Impact Highlights", "Experience (STAR)", "Skills", "Education"], "tips": ["Quantify achievements", "Show scale (millions of users)", "Include preparation for behavioral rounds"]},
            "Startup": {"format": "Versatile, full-stack emphasis", "sections": ["Profile", "Versatility Highlights", "Projects", "Skills", "Experience"], "tips": ["Highlight full-stack capabilities", "Show ownership and initiative", "Demonstrate fast learning"]},
        }
        style = styles.get(target_style, styles["ATS"])
        return {
            "style": target_style, "format": style["format"],
            "suggested_sections": style["sections"], "tips": style["tips"],
            "generated_summary": f"{'Experienced' if experience > 3 else 'Aspiring'} professional with skills in {', '.join(skills[:5])}. {'Proven track record' if experience > 3 else 'Strong foundation'} in delivering data-driven solutions.",
        }
