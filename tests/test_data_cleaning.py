import pytest
import pandas as pd
import numpy as np
from src.data_cleaning.cleaner import DataCleaner, JobPostingCleaner
from src.data_cleaning.skill_standardizer import SkillStandardizer


class TestDataCleaner:
    @pytest.fixture
    def cleaner(self):
        return DataCleaner()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "A": [1, 2, 2, 3, np.nan],
            "B": ["x", "y", "y", "z", None],
            "C": [10, 20, 30, 40, 50],
        })

    def test_remove_duplicates(self, cleaner, sample_df):
        df_with_dup = sample_df.copy()
        df_with_dup.loc[4] = [2, "y", 20]
        result = cleaner.remove_duplicates(df_with_dup)
        assert len(result) == 4
        assert "duplicates_removed" in cleaner.cleaning_report
        assert cleaner.cleaning_report["duplicates_removed"] == 1

    def test_handle_missing_values(self, cleaner, sample_df):
        result = cleaner.handle_missing_values(sample_df)
        assert result["A"].isna().sum() == 0
        assert result["B"].isna().sum() == 0

    def test_standardize_text(self, cleaner):
        assert cleaner.standardize_text("  Python  ") == "python"
        assert cleaner.standardize_text("Machine Learning!") == "machine learning"
        assert cleaner.standardize_text("") == ""


class TestSkillStandardizer:
    @pytest.fixture
    def std(self):
        return SkillStandardizer()

    def test_standardize_python(self, std):
        assert std.standardize("Python") == "Python"
        assert std.standardize("python") == "Python"
        assert std.standardize("PYTHON") == "Python"

    def test_standardize_list(self, std):
        result = std.standardize_list(["python", "tensorflow", "sklearn"])
        assert result == ["Python", "TensorFlow", "Scikit-Learn"]

    def test_extract_skills(self, std):
        text = "I know Python and Machine Learning"
        skills = std.extract_skills_from_text(text)
        assert "Python" in skills
        assert "Machine Learning" in skills


class TestJobPostingCleaner:
    @pytest.fixture
    def cleaner(self):
        return JobPostingCleaner()

    @pytest.fixture
    def job_df(self):
        return pd.DataFrame({
            "title": ["Data Scientist", "data engineer", "ML Engineer"],
            "salary": [120000, 110000, 130000],
            "description": ["python sql ml", "python spark aws", "python dl nlp"],
            "location": ["SF", "NYC", "SEA"],
        })

    def test_clean_job_titles(self, cleaner, job_df):
        result = cleaner.clean_job_titles(job_df)
        assert "Data Scientist" in result["title"].values
        assert "Data Engineer" in result["title"].values

    def test_parse_salary(self, cleaner, job_df):
        result = cleaner.parse_salary(job_df)
        assert result["salary"].dtype in ["float64", "int64"]
