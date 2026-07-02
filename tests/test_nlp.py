import pytest
from src.nlp.skill_extractor import SkillExtractor
from src.nlp.text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_clean_text(self, preprocessor):
        cleaned = preprocessor.clean_text("Hello! Visit https://example.com")
        assert "https" not in cleaned

    def test_tokenize(self, preprocessor):
        tokens = preprocessor.tokenize("Python is great for data science")
        assert len(tokens) > 0
        assert "Python" in tokens


class TestSkillExtractor:
    @pytest.fixture
    def extractor(self):
        return SkillExtractor()

    def test_regex_extraction(self, extractor):
        text = "Looking for a Data Scientist with Python, TensorFlow, and AWS experience"
        skills = extractor.extract_with_regex(text)
        assert "Python" in skills or "python" in skills
        assert any("tensorflow" in s.lower() for s in skills)
        assert "AWS" in skills or "Aws" in skills or "aws" in skills

    def test_extract_method_hybrid(self, extractor):
        text = "Required skills: Python, SQL, Machine Learning"
        skills = extractor.extract(text, method="hybrid")
        assert len(skills) > 0
        assert skills == sorted(skills)

    def test_empty_text(self, extractor):
        assert extractor.extract("") == []
        assert extractor.extract(None) == []

    def test_skill_patterns_comprehensive(self, extractor):
        categories = extractor.SKILL_PATTERNS
        assert "programming" in categories
        assert "ml_dl" in categories
        assert "cloud" in categories
        assert "frameworks" in categories
        assert "devops" in categories
        assert "data" in categories
