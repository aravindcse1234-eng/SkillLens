from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


LOCATION_DATA = {
    # US Cities
    "San Francisco": {"lat": 37.7749, "lng": -122.4194, "region": "West Coast", "cost_of_living": 1.8, "tech_hub_score": 98, "country": "US"},
    "New York": {"lat": 40.7128, "lng": -74.0060, "region": "East Coast", "cost_of_living": 1.6, "tech_hub_score": 95, "country": "US"},
    "Seattle": {"lat": 47.6062, "lng": -122.3321, "region": "West Coast", "cost_of_living": 1.4, "tech_hub_score": 90, "country": "US"},
    "Austin": {"lat": 30.2672, "lng": -97.7431, "region": "South", "cost_of_living": 1.1, "tech_hub_score": 85, "country": "US"},
    "Boston": {"lat": 42.3601, "lng": -71.0589, "region": "East Coast", "cost_of_living": 1.5, "tech_hub_score": 88, "country": "US"},
    "Chicago": {"lat": 41.8781, "lng": -87.6298, "region": "Midwest", "cost_of_living": 1.2, "tech_hub_score": 82, "country": "US"},
    "Los Angeles": {"lat": 34.0522, "lng": -118.2437, "region": "West Coast", "cost_of_living": 1.5, "tech_hub_score": 86, "country": "US"},
    "Denver": {"lat": 39.7392, "lng": -104.9903, "region": "Mountain", "cost_of_living": 1.15, "tech_hub_score": 78, "country": "US"},
    "Washington DC": {"lat": 38.9072, "lng": -77.0369, "region": "East Coast", "cost_of_living": 1.4, "tech_hub_score": 80, "country": "US"},
    "Atlanta": {"lat": 33.7490, "lng": -84.3880, "region": "South", "cost_of_living": 1.0, "tech_hub_score": 76, "country": "US"},
    "Dallas": {"lat": 32.7767, "lng": -96.7970, "region": "South", "cost_of_living": 1.0, "tech_hub_score": 75, "country": "US"},
    "San Diego": {"lat": 32.7157, "lng": -117.1611, "region": "West Coast", "cost_of_living": 1.4, "tech_hub_score": 74, "country": "US"},
    "Portland": {"lat": 45.5152, "lng": -122.6784, "region": "West Coast", "cost_of_living": 1.2, "tech_hub_score": 70, "country": "US"},
    "Phoenix": {"lat": 33.4484, "lng": -112.0740, "region": "Mountain", "cost_of_living": 0.95, "tech_hub_score": 68, "country": "US"},
    "Miami": {"lat": 25.7617, "lng": -80.1918, "region": "South", "cost_of_living": 1.25, "tech_hub_score": 72, "country": "US"},
    # Indian Cities
    "Bangalore": {"lat": 12.9716, "lng": 77.5946, "region": "South India", "cost_of_living": 0.35, "tech_hub_score": 95, "country": "India"},
    "Mumbai": {"lat": 19.0760, "lng": 72.8777, "region": "West India", "cost_of_living": 0.40, "tech_hub_score": 92, "country": "India"},
    "Delhi": {"lat": 28.7041, "lng": 77.1025, "region": "North India", "cost_of_living": 0.38, "tech_hub_score": 90, "country": "India"},
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867, "region": "South India", "cost_of_living": 0.32, "tech_hub_score": 88, "country": "India"},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "region": "West India", "cost_of_living": 0.30, "tech_hub_score": 85, "country": "India"},
    "Chennai": {"lat": 13.0827, "lng": 80.2707, "region": "South India", "cost_of_living": 0.30, "tech_hub_score": 82, "country": "India"},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "region": "East India", "cost_of_living": 0.28, "tech_hub_score": 75, "country": "India"},
    "Ahmedabad": {"lat": 23.0225, "lng": 72.5714, "region": "West India", "cost_of_living": 0.26, "tech_hub_score": 72, "country": "India"},
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "region": "North India", "cost_of_living": 0.25, "tech_hub_score": 68, "country": "India"},
    # Remote
    "Remote": {"lat": 0, "lng": 0, "region": "Remote", "cost_of_living": 1.0, "tech_hub_score": 50, "country": "Global"},
}


class GeographicIntelligence:
    def __init__(self):
        self.locations = LOCATION_DATA

    def get_location_info(self, city: str) -> Dict:
        return self.locations.get(city, {})

    def get_hiring_hotspots(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        if "location" not in df.columns:
            return []
        hotspots = []
        for city, count in df["location"].value_counts().head(top_k).items():
            info = self.locations.get(city, {})
            city_df = df[df["location"] == city]
            avg_salary = int(city_df["salary"].mean()) if "salary" in city_df.columns else 0
            hotspots.append({
                "city": city,
                "lat": info.get("lat", 0),
                "lng": info.get("lng", 0),
                "region": info.get("region", "Unknown"),
                "job_count": int(count),
                "avg_salary": avg_salary,
                "cost_of_living": info.get("cost_of_living", 1.0),
                "tech_hub_score": info.get("tech_hub_score", 50),
                "salary_adjusted": int(avg_salary / max(info.get("cost_of_living", 1.0), 0.5)),
            })
        hotspots.sort(key=lambda x: x["job_count"], reverse=True)
        return hotspots

    def get_salary_hotspots(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        if "location" not in df.columns or "salary" not in df.columns:
            return []
        city_salaries = df.groupby("location")["salary"].agg(["mean", "median", "std", "count"])
        city_salaries = city_salaries[city_salaries["count"] >= 3].sort_values("median", ascending=False)
        hotspots = []
        for city, row in city_salaries.head(top_k).iterrows():
            info = self.locations.get(city, {})
            hotspots.append({
                "city": city,
                "lat": info.get("lat", 0),
                "lng": info.get("lng", 0),
                "region": info.get("region", "Unknown"),
                "mean_salary": int(row["mean"]),
                "median_salary": int(row["median"]),
                "salary_std": int(row["std"]),
                "job_count": int(row["count"]),
                "cost_of_living": info.get("cost_of_living", 1.0),
                "adjusted_salary": int(row["median"] / max(info.get("cost_of_living", 1.0), 0.5)),
            })
        return hotspots

    def get_remote_work_trends(self, df: pd.DataFrame) -> Dict:
        remote_pct = 0
        hybrid_pct = 0
        onsite_pct = 0
        if "location" in df.columns:
            total = len(df)
            remote_count = len(df[df["location"].str.contains("Remote", case=False, na=False)])
            remote_pct = round(remote_count / total * 100, 1) if total else 0
        if "employment_type" in df.columns:
            types = df["employment_type"].value_counts(normalize=True).to_dict()
            remote_pct = round(types.get("Remote", types.get("remote", 0)) * 100, 1)
            hybrid_pct = round(types.get("Hybrid", types.get("hybrid", 0)) * 100, 1)
            onsite_pct = round(100 - remote_pct - hybrid_pct, 1)
        return {
            "remote_percentage": remote_pct,
            "hybrid_percentage": hybrid_pct,
            "onsite_percentage": onsite_pct,
            "trend": "increasing" if remote_pct > 20 else "stable",
            "remote_salary_premium": 1.1 if remote_pct > 20 else 1.0,
        }

    def get_region_breakdown(self, df: pd.DataFrame) -> List[Dict]:
        region_data = {}
        for city, info in self.locations.items():
            region = info["region"]
            if region == "Remote":
                continue
            city_df = df[df["location"] == city] if "location" in df.columns else pd.DataFrame()
            if region not in region_data:
                region_data[region] = {"cities": [], "total_jobs": 0, "total_salary": 0, "count": 0}
            region_data[region]["cities"].append(city)
            region_data[region]["total_jobs"] += len(city_df)
            if "salary" in city_df.columns:
                region_data[region]["total_salary"] += int(city_df["salary"].sum())
                region_data[region]["count"] += len(city_df)

        return [{
            "region": region,
            "cities": data["cities"],
            "job_count": data["total_jobs"],
            "avg_salary": int(data["total_salary"] / data["count"]) if data["count"] > 0 else 0,
        } for region, data in sorted(region_data.items(), key=lambda x: -x[1]["total_jobs"])]

    def get_map_data(self, df: pd.DataFrame) -> Dict:
        hotspots = self.get_hiring_hotspots(df)
        salary_hotspots = self.get_salary_hotspots(df)
        regions = self.get_region_breakdown(df)
        remote = self.get_remote_work_trends(df)
        return {
            "hiring_hotspots": hotspots,
            "salary_hotspots": salary_hotspots,
            "regions": regions,
            "remote_trends": remote,
            "center": {"lat": 39.8283, "lng": -98.5795},
            "zoom": 4,
        }
