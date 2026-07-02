import pandas as pd
import time
import random
from typing import Optional, Dict, List
from datetime import datetime
from src.utils.helpers import save_json, ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JobScraper:
    def __init__(self, delay: float = 2.0, output_dir: str = "data/raw"):
        self.delay = delay
        self.output_dir = ensure_dir(output_dir)
        self.session = None

    def _build_session(self):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.5",
        })

    def scrape_indeed(self, query: str, location: str = "", max_pages: int = 5) -> List[Dict]:
        logger.info(f"Scraping Indeed for '{query}' in '{location}'")
        jobs = []
        for page in range(max_pages):
            try:
                time.sleep(self.delay + random.uniform(0, 1))
            except Exception as e:
                logger.error(f"Indeed scraping error: {e}")
        return jobs

    def scrape_linkedin(self, query: str, location: str = "", max_pages: int = 5) -> List[Dict]:
        logger.info(f"Scraping LinkedIn for '{query}' in '{location}'")
        jobs = []
        try:
            from linkedin_api import Linkedin
            api = Linkedin("", "")
        except ImportError:
            logger.warning("linkedin_api not installed")
        except Exception as e:
            logger.error(f"LinkedIn scraping error: {e}")
        return jobs

    def scrape_glassdoor(self, query: str, location: str = "", max_pages: int = 5) -> List[Dict]:
        logger.info(f"Scraping Glassdoor for '{query}' in '{location}'")
        jobs = []
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
        except Exception as e:
            logger.error(f"Glassdoor scraping error: {e}")
        return jobs

    def search_google_jobs(self, query: str, location: str = "") -> List[Dict]:
        logger.info(f"Searching Google Jobs for '{query}'")
        jobs = []
        try:
            from serpapi import GoogleSearch
            params = {
                "q": f"{query} jobs {location}",
                "engine": "google_jobs",
                "api_key": ""
            }
        except Exception as e:
            logger.error(f"Google Jobs error: {e}")
        return jobs

    def to_dataframe(self, jobs: List[Dict]) -> pd.DataFrame:
        if not jobs:
            return pd.DataFrame()
        df = pd.DataFrame(jobs)
        df["source"] = "web_scraped"
        df["collected_at"] = datetime.now()
        return df
