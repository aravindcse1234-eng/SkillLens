import pytest
import pandas as pd
from src.geographic.geo_intelligence import GeographicIntelligence


class TestGeographicIntelligence:
    @pytest.fixture
    def geo(self):
        return GeographicIntelligence()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "location": ["San Francisco", "San Francisco", "San Francisco",
                         "New York", "New York", "New York",
                         "Seattle", "Austin", "Remote", "Boston"],
            "salary": [150000, 160000, 155000, 130000, 135000, 140000,
                       140000, 120000, 100000, 125000],
            "employment_type": ["Full-time", "Full-time", "Full-time",
                                "Full-time", "Full-time", "Full-time",
                                "Remote", "Full-time", "Remote", "Full-time"],
        })

    def test_get_location_info(self, geo):
        info = geo.get_location_info("San Francisco")
        assert info["lat"] == 37.7749
        assert info["region"] == "West Coast"

    def test_get_hiring_hotspots(self, geo, sample_df):
        hotspots = geo.get_hiring_hotspots(sample_df, top_k=5)
        assert len(hotspots) > 0
        assert hotspots[0]["city"] == "San Francisco"

    def test_get_salary_hotspots(self, geo, sample_df):
        hotspots = geo.get_salary_hotspots(sample_df, top_k=3)
        assert len(hotspots) > 0

    def test_get_remote_work_trends(self, geo, sample_df):
        trends = geo.get_remote_work_trends(sample_df)
        assert "remote_percentage" in trends
        assert "hybrid_percentage" in trends

    def test_get_region_breakdown(self, geo, sample_df):
        regions = geo.get_region_breakdown(sample_df)
        assert len(regions) > 0

    def test_get_map_data(self, geo, sample_df):
        data = geo.get_map_data(sample_df)
        assert "hiring_hotspots" in data
        assert "salary_hotspots" in data
        assert "regions" in data
        assert "remote_trends" in data
