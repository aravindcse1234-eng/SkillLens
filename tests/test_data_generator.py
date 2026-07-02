import pytest
import pandas as pd
import numpy as np
from src.data.data_generator import generate_market_data, generate_skill_timeseries


class TestMarketDataGenerator:
    def test_generates_correct_shape(self):
        df = generate_market_data(n_samples=100, seed=42)
        assert len(df) == 100
        assert "salary" in df.columns
        assert "title" in df.columns
        assert "skills" in df.columns

    def test_salary_has_positive_values(self):
        df = generate_market_data(n_samples=500, seed=42)
        assert (df["salary"] > 0).all()
        assert df["salary"].between(30000, 500000).all()

    def test_experience_levels_valid(self):
        df = generate_market_data(n_samples=500, seed=42)
        valid = {"Entry", "Mid", "Senior", "Lead", "Principal"}
        assert set(df["experience_level"].unique()).issubset(valid)

    def test_experience_vs_years_consistency(self):
        df = generate_market_data(n_samples=1000, seed=42)
        senior = df[df["experience_level"] == "Senior"]
        if not senior.empty:
            assert senior["years_experience"].mean() >= 3

    def test_education_levels_valid(self):
        df = generate_market_data(n_samples=500, seed=42)
        assert set(df["education_level"].unique()).issubset({"Bachelor", "Master", "PhD"})

    def test_skills_not_empty(self):
        df = generate_market_data(n_samples=500, seed=42)
        assert df["skills"].apply(len).min() >= 2

    def test_salary_correlated_with_experience(self):
        df = generate_market_data(n_samples=2000, seed=42)
        corr = df["salary"].corr(df["years_experience"])
        assert corr > 0.2

    def test_locations_vary_salary(self):
        df = generate_market_data(n_samples=2000, seed=42)
        sf_avg = df[df["location"] == "San Francisco"]["salary"].mean()
        remote_avg = df[df["location"] == "Remote"]["salary"].mean()
        assert sf_avg > remote_avg


class TestSkillTimeseriesGenerator:
    def test_generates_correct_shape(self):
        df = generate_skill_timeseries(start_date="2020-01", periods=12, seed=42)
        assert len(df) > 0
        assert "skill_name" in df.columns
        assert "date" in df.columns
        assert "demand_score" in df.columns

    def test_multiple_skills(self):
        df = generate_skill_timeseries(periods=12, seed=42)
        assert df["skill_name"].nunique() >= 5

    def test_demand_scores_positive(self):
        df = generate_skill_timeseries(periods=24, seed=42)
        assert (df["demand_score"] > 0).all()

    def test_increasing_trend(self):
        df = generate_skill_timeseries(periods=36, seed=42)
        gen_ai = df[df["skill_name"] == "Generative AI"].sort_values("date")
        assert gen_ai["demand_score"].iloc[-1] > gen_ai["demand_score"].iloc[0]
