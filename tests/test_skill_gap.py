import pytest
import pandas as pd
from src.skill_gap.gap_analyzer import SkillGapAnalyzer
from src.skill_gap.roadmap_generator import LearningRoadmapGenerator


class TestSkillGapAnalyzer:
    @pytest.fixture
    def analyzer(self):
        a = SkillGapAnalyzer()
        df = pd.DataFrame({
            "skill_name": ["Python", "SQL", "ML", "AWS", "Docker"],
            "demand_count": [100, 80, 90, 70, 50],
        })
        a.load_market_data(df)
        return a

    def test_analyze_gap_full_match(self, analyzer):
        result = analyzer.analyze_gap(["Python", "SQL", "ML", "AWS", "Docker"])
        assert result["total_matched"] == 5
        assert result["total_missing"] == 0
        assert result["match_percentage"] == 100

    def test_analyze_gap_partial(self, analyzer):
        result = analyzer.analyze_gap(["Python"])
        assert result["total_matched"] == 1
        assert result["total_missing"] == 4

    def test_gap_empty_skills(self, analyzer):
        result = analyzer.analyze_gap([])
        assert result["total_matched"] == 0
        assert result["total_missing"] == 5


class TestRoadmapGenerator:
    @pytest.fixture
    def generator(self):
        return LearningRoadmapGenerator()

    def test_generate_roadmap(self, generator):
        missing = [
            {"skill": "AWS", "priority": "high", "market_demand": 90, "gap_score": 80},
            {"skill": "Docker", "priority": "medium", "market_demand": 70, "gap_score": 60},
        ]
        result = generator.generate(missing)
        assert result["total_skills_to_learn"] == 2
        assert "roadmap" in result

    def test_roadmap_steps(self, generator):
        missing = [{"skill": "Kubernetes", "priority": "high", "market_demand": 85, "gap_score": 75}]
        result = generator.generate(missing)
        assert result["roadmap"][0]["skill"] == "Kubernetes"
        assert len(result["roadmap"][0]["resources"]) > 0
