import pytest
import pandas as pd
import numpy as np
from src.eda.analyzer import EDAnalyzer


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "title": ["Data Scientist", "Data Engineer", "ML Engineer", "Data Scientist", "Data Analyst"],
        "location": ["San Francisco", "New York", "Remote", "San Francisco", "Austin"],
        "skills": [["Python", "SQL", "ML"], ["Python", "SQL"], ["Python", "ML", "PyTorch"], ["Python", "SQL", "ML"], ["Python", "SQL", "Tableau"]],
        "salary": [120000, 110000, 130000, 125000, 85000],
        "experience_level": ["Senior", "Mid", "Senior", "Lead", "Entry"],
        "industry": ["Technology", "Finance", "Technology", "Technology", "Healthcare"],
        "education_required": ["Master", "Bachelor", "PhD", "Master", "Bachelor"],
        "years_experience": [5.0, 3.0, 7.0, 8.0, 1.0],
    })


class TestEDAnalyzer:
    def test_basic_stats(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        stats = analyzer.basic_stats()
        assert stats["total_rows"] == 5
        assert "missing_values" in stats
        assert "duplicates" in stats

    def test_analyze_job_titles(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        titles = analyzer.analyze_job_titles()
        assert not titles.empty
        assert "Data Scientist" in titles["title"].values

    def test_analyze_locations(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        locs = analyzer.analyze_locations()
        assert not locs.empty
        assert "San Francisco" in locs["location"].values

    def test_analyze_skills_demand(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        skills = analyzer.analyze_skills_demand()
        assert skills["Python"] == 5
        assert "SQL" in skills

    def test_analyze_salary_distribution(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        stats = analyzer.analyze_salary_distribution()
        assert stats["mean"] > 0
        assert stats["median"] > 0
        assert stats["min"] <= stats["max"]

    def test_analyze_experience_levels(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        exp = analyzer.analyze_experience_levels()
        assert not exp.empty
        assert set(exp["level"]).issubset({"Entry", "Mid", "Senior", "Lead"})

    def test_analyze_industry_trends(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        ind = analyzer.analyze_industry_trends()
        assert "Technology" in ind["industry"].values

    def test_analyze_education_requirements(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        edu = analyzer.analyze_education_requirements()
        assert set(edu["education"]).issubset({"Bachelor", "Master", "PhD"})

    def test_correlation_analysis(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        corr = analyzer.correlation_analysis()
        assert isinstance(corr, pd.DataFrame)
        assert not corr.empty

    def test_comprehensive_analysis(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        insights = analyzer.comprehensive_analysis()
        assert "basic_stats" in insights
        assert "salary_stats" in insights
        assert "top_skills" in insights

    def test_missing_column_returns_empty(self, sample_df):
        analyzer = EDAnalyzer(sample_df)
        result = analyzer.analyze_job_titles(col="nonexistent")
        assert result.empty

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": []})
        analyzer = EDAnalyzer(df)
        stats = analyzer.basic_stats()
        assert stats["total_rows"] == 0
