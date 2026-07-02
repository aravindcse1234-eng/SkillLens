from typing import Dict, List, Optional
import requests
import json
import datetime
import time
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CACHE_DIR = Path("data/live_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 3600

SKILL_TRENDING_QUERIES = {
    "Python": "python", "Machine Learning": "machine+learning", "Deep Learning": "deep+learning",
    "NLP": "nlp", "Computer Vision": "computer+vision", "AWS": "aws", "Azure": "azure",
    "Docker": "docker", "Kubernetes": "kubernetes", "Generative AI": "generative+ai+llm",
    "PyTorch": "pytorch", "TensorFlow": "tensorflow", "Spark": "apache+spark",
    "Data Engineering": "data+engineering", "MLOps": "mlops", "LangChain": "langchain",
    "RAG": "rag+retrieval", "SQL": "sql", "Git": "git", "Linux": "linux",
}


def _cached_get(url: str, cache_key: str, headers: Optional[Dict] = None,
                params: Optional[Dict] = None, ttl: int = CACHE_TTL) -> Optional[Dict]:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < ttl:
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cache_path.write_text(json.dumps(data))
            return data
        logger.warning(f"HTTP {resp.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Request failed for {url}: {e}")
    return None


def fetch_github_skill_trends(skills: List[str]) -> Dict[str, int]:
    results = {}
    for skill, query in SKILL_TRENDING_QUERIES.items():
        if skill not in skills:
            continue
        cache_key = f"github_{skill.lower().replace(' ','_')}"
        url = f"https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 1}
        data = _cached_get(url, cache_key, params=params)
        if data and "total_count" in data:
            results[skill] = data["total_count"]
        else:
            results[skill] = 0
    return results


def fetch_stackoverflow_trends(skills: List[str]) -> Dict[str, int]:
    results = {}
    for skill in skills:
        tag = skill.lower().replace(" ", "-").replace("+", "-")
        cache_key = f"so_{tag}"
        url = f"https://api.stackexchange.com/2.3/tags/{tag}/info"
        params = {"site": "stackoverflow"}
        data = _cached_get(url, cache_key, params=params)
        if data and "items" in data and data["items"]:
            results[skill] = data["items"][0].get("count", 0)
        else:
            results[skill] = 0
    return results


def fetch_github_trending_repos() -> List[Dict]:
    urls = [
        "https://api.github.com/search/repositories?q=created:>30+language:python&sort=stars&order=desc&per_page=5",
        "https://api.github.com/search/repositories?q=created:>30+topic:ai&sort=stars&order=desc&per_page=5",
        "https://api.github.com/search/repositories?q=created:>30+topic:machine-learning&sort=stars&order=desc&per_page=5",
    ]
    repos = []
    for url in urls:
        cache_key = "github_trending_" + str(hash(url))[-8:]
        data = _cached_get(url, cache_key)
        if data and "items" in data:
            for item in data["items"][:3]:
                repos.append({
                    "name": item.get("full_name", ""),
                    "stars": item.get("stargazers_count", 0),
                    "description": (item.get("description") or "")[:120],
                    "topics": item.get("topics", []),
                    "url": item.get("html_url", ""),
                })
    return repos


def fetch_hackernews_stories() -> List[Dict]:
    story_ids = _cached_get("https://hacker-news.firebaseio.com/v0/topstories.json", "hn_top_ids", ttl=1800)
    if not story_ids:
        return []
    stories = []
    for sid in story_ids[:15]:
        cache_key = f"hn_item_{sid}"
        item = _cached_get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", cache_key, ttl=3600)
        if item and item.get("title"):
            title = item["title"]
            if any(kw in title.lower() for kw in ["ai", "ml", "data", "job", "hire", "career", "tech", "code", "python"]):
                stories.append({"title": title, "score": item.get("score", 0), "url": item.get("url", f"https://news.ycombinator.com/item?id={sid}")})
    return stories[:8]


def compute_skill_growth_from_sources(skills: List[str]) -> Dict[str, float]:
    github_counts = fetch_github_skill_trends(skills)
    so_counts = fetch_stackoverflow_trends(skills)
    growth = {}
    now = datetime.datetime.utcnow()
    for skill in skills:
        gh = github_counts.get(skill, 0)
        so = so_counts.get(skill, 0)
        gh_score = min(100, gh / 10000)
        so_score = min(100, so / 100000)
        combined = (gh_score * 0.5 + so_score * 0.5)
        growth[skill] = round(combined, 1)
    return growth


class LiveDataCollector:
    def __init__(self):
        self.last_refresh = None
        self.cached_data = {}

    def refresh_all(self, skills: Optional[List[str]] = None) -> Dict:
        if skills is None:
            skills = list(SKILL_TRENDING_QUERIES.keys())
        logger.info(f"Refreshing live data for {len(skills)} skills...")
        github_repos = fetch_github_trending_repos()
        hn_stories = fetch_hackernews_stories()
        growth = compute_skill_growth_from_sources(skills)
        self.cached_data = {
            "github_trending": github_repos,
            "hackernews": hn_stories,
            "skill_growth": growth,
            "refreshed_at": datetime.datetime.utcnow().isoformat(),
            "skill_count": len(skills),
            "data_sources": ["GitHub API", "Stack Overflow API", "Hacker News API"],
        }
        self.last_refresh = datetime.datetime.utcnow()
        return self.cached_data

    def get_data(self) -> Dict:
        if self.last_refresh and (datetime.datetime.utcnow() - self.last_refresh).seconds < CACHE_TTL:
            return self.cached_data
        return self.refresh_all()

    def get_trending_repos_summary(self) -> str:
        repos = self.cached_data.get("github_trending", [])
        if not repos:
            return "No trending data"
        lines = []
        for r in repos[:5]:
            stars = r.get("stars", 0)
            desc = r.get("description", "")
            lines.append(f"⭐ {r['name']} ({stars}★) — {desc[:80]}")
        return "\n".join(lines)

    def get_news_summary(self) -> str:
        stories = self.cached_data.get("hackernews", [])
        if not stories:
            return "No news data"
        lines = []
        for s in stories[:5]:
            lines.append(f"📰 {s['title']} (▲{s.get('score', 0)})")
        return "\n".join(lines)

    def get_status(self) -> Dict:
        return {
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "data_sources": self.cached_data.get("data_sources", []),
            "skills_tracked": self.cached_data.get("skill_count", 0),
            "github_repos": len(self.cached_data.get("github_trending", [])),
            "news_stories": len(self.cached_data.get("hackernews", [])),
            "has_data": bool(self.cached_data),
        }
