import pytest
from src.resume_analyzer.analyzer import ResumeAnalyzer
from src.resume_analyzer.ats_scorer import ATSScorer


class TestResumeAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ResumeAnalyzer()

    def test_extract_experience_years(self, analyzer):
        text = "5 years of experience in data science"
        years = analyzer.extract_experience_years(text)
        assert years is not None
        assert years == 5.0

    def test_extract_experience_range(self, analyzer):
        text = "3-5 years of experience"
        years = analyzer.extract_experience_years(text)
        assert years is not None
        assert 3.5 <= years <= 4.5

    def test_extract_education_level(self, analyzer):
        assert analyzer.extract_education_level("PhD in Computer Science") == "PhD"
        assert analyzer.extract_education_level("Master of Science") == "Master"
        assert analyzer.extract_education_level("Bachelor of Technology") == "Bachelor"

    def test_extract_job_titles(self, analyzer):
        text = "Worked as a Data Scientist and Machine Learning Engineer"
        titles = analyzer.extract_job_titles(text)
        assert "Data Scientist" in titles

    def test_extract_skills(self, analyzer):
        text = "Proficient in Python, SQL, and Machine Learning"
        skills = analyzer.extract_skills(text)
        assert len(skills) > 0


class TestATSScorer:
    @pytest.fixture
    def scorer(self):
        return ATSScorer()

    def test_skills_match(self, scorer):
        score, matched, missing = scorer.compute_skills_match(
            ["Python", "SQL", "ML"], ["Python", "SQL", "AWS"]
        )
        assert score == 2 / 3
        assert "AWS" in missing
        assert "Python" in matched

    def test_experience_match(self, scorer):
        assert scorer.compute_experience_match(5, 3) == 1.0
        assert scorer.compute_experience_match(2, 5) == 0.4

    def test_education_match(self, scorer):
        assert scorer.compute_education_match("PhD", "Bachelor") == 1.0
        assert scorer.compute_education_match("Bachelor", "Master") < 1.0
